//! Stage-ordered generation of a chunk rectangle into an unused chunk map.

use std::sync::{Arc, atomic::Ordering};

use futures::future::{BoxFuture, FutureExt, Shared, join_all};
use rustc_hash::FxHashMap;
use steel_utils::ChunkPos;
use thiserror::Error;

use super::ChunkMap;
use crate::chunk::{
    chunk_generation_task::StaticCache2D,
    chunk_holder::ChunkHolder,
    chunk_pyramid::{ChunkStep, GENERATION_PYRAMID},
    chunk_ticket_manager::ticket_level_for_status,
    light::LIGHT_CACHE_RADIUS,
    status::ChunkStatus,
};

const FULL_STEP: &ChunkStep = GENERATION_PYRAMID.get_step_to(ChunkStatus::Full);

type RegionTask = Shared<BoxFuture<'static, Result<(), RegionGenerationError>>>;

/// Why [`ChunkMap::generate_region`] could not generate a region.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
pub enum RegionGenerationError {
    /// The map already has holders or its generation refill loop is running.
    #[error("region generation needs an empty chunk map with its refill loop stopped")]
    MapInUse,
    /// `min` exceeds `max` on an axis, or the region is too close to the coordinate limits.
    #[error("invalid region bounds {min:?}..={max:?}")]
    InvalidBounds {
        /// Requested minimum corner.
        min: ChunkPos,
        /// Requested maximum corner.
        max: ChunkPos,
    },
    /// A step was disallowed, cancelled or panicked.
    #[error("{status:?} generation failed at {pos:?}")]
    StepFailed {
        /// Status the failed step targeted.
        status: ChunkStatus,
        /// Chunk the step ran on.
        pos: ChunkPos,
    },
}

impl ChunkMap {
    /// Generates every chunk in `min..=max` to Full and returns their holders in X-then-Z order.
    ///
    /// Each status is finished across the region and its dependency halo before the next
    /// starts, so shared dependencies run once rather than once per requested chunk. Chunks
    /// whose step areas overlap run in ascending X-then-Z order and others run in parallel, so
    /// the result equals generating one chunk at a time and does not depend on thread scheduling.
    ///
    /// The map must be empty with its refill loop stopped, and nothing else may drive it until
    /// this returns. The holders have no tickets, so the map never unloads them.
    pub async fn generate_region(
        self: &Arc<Self>,
        min: ChunkPos,
        max: ChunkPos,
    ) -> Result<Vec<Arc<ChunkHolder>>, RegionGenerationError> {
        let halo = FULL_STEP.get_accumulated_radius_of(ChunkStatus::Empty) as i32;
        // Leaves room for generation to address another halo of blocks past the region's halo.
        let limit = (i32::MAX >> 4) - halo * 2;
        let in_range = |low: i32, high: i32| low <= high && low >= -limit && high <= limit;
        if !in_range(min.0.x, max.0.x) || !in_range(min.0.y, max.0.y) {
            return Err(RegionGenerationError::InvalidBounds { min, max });
        }
        self.insert_region_holders(min, max, halo)?;

        let mut status = Some(ChunkStatus::Empty);
        while let Some(current) = status {
            let radius = FULL_STEP.get_accumulated_radius_of(current) as i32;
            self.generate_region_stage(
                GENERATION_PYRAMID.get_step_to(current),
                region_positions(min, max, radius),
            )
            .await?;
            status = current.next();
        }

        Ok(region_positions(min, max, 0)
            .into_iter()
            .map(|pos| self.region_holder(pos))
            .collect())
    }

    /// Installs a holder for every region and halo chunk, allowing only the status it needs.
    fn insert_region_holders(
        &self,
        min: ChunkPos,
        max: ChunkPos,
        halo: i32,
    ) -> Result<(), RegionGenerationError> {
        let _source_phase_guard = self.source_phase_guard.lock();
        if !self.generation_refill_stopped.load(Ordering::Acquire) || !self.chunks.is_empty() {
            return Err(RegionGenerationError::MapInUse);
        }

        for pos in region_positions(min, max, halo) {
            let dx = (min.0.x - pos.0.x).max(pos.0.x - max.0.x).max(0);
            let dz = (min.0.y - pos.0.y).max(pos.0.y - max.0.y).max(0);
            let distance = dx.max(dz) as usize;
            let needed_status = if distance == 0 {
                ChunkStatus::Full
            } else {
                let Some(status) = FULL_STEP.accumulated_dependencies.get(distance) else {
                    panic!("Full dependencies must cover their own radius");
                };
                status
            };
            let holder = Arc::new(ChunkHolder::new_with_full_publications(
                pos,
                ticket_level_for_status(needed_status),
                None,
                self.world_gen_context.min_y(),
                self.world_gen_context.height(),
                Arc::downgrade(&self.full_publications),
            ));
            if self.chunks.insert_sync(pos, holder).is_err() {
                return Err(RegionGenerationError::MapInUse);
            }
        }
        Ok(())
    }

    /// Applies `step` to `positions`, listed in ascending X-then-Z order.
    ///
    /// A chunk waits only for earlier chunks whose step areas overlap its own or, for Features,
    /// that place pieces from the same structure starts. The result therefore equals applying
    /// the step to one chunk at a time in that order.
    async fn generate_region_stage(
        self: &Arc<Self>,
        step: &'static ChunkStep,
        positions: Vec<ChunkPos>,
    ) -> Result<(), RegionGenerationError> {
        let reach = region_step_reach(step);
        let cache_radius = (step.direct_dependencies.get_radius() as i32).max(reach);
        // Areas of radius `reach` overlap within twice that distance.
        let span = reach * 2;
        let mut tasks = FxHashMap::<ChunkPos, RegionTask>::default();
        // Latest task that updates each structure-start chunk.
        let mut start_users = FxHashMap::<ChunkPos, RegionTask>::default();
        for pos in positions {
            let mut earlier = (pos.0.x - span..=pos.0.x)
                .flat_map(|x| (pos.0.y - span..=pos.0.y + span).map(move |z| ChunkPos::new(x, z)))
                .filter(|other| other.0.x < pos.0.x || other.0.y < pos.0.y)
                .filter_map(|other| tasks.get(&other).cloned())
                .collect::<Vec<_>>();
            let starts = if step.target_status == ChunkStatus::Features {
                self.referenced_structure_starts(pos)
            } else {
                Vec::new()
            };
            earlier.extend(
                starts
                    .iter()
                    .filter_map(|start| start_users.get(start).cloned()),
            );
            let chunk_map = Arc::clone(self);
            let task = async move {
                for task in earlier {
                    task.await?;
                }
                chunk_map.apply_region_step(step, pos, cache_radius).await
            }
            .boxed()
            .shared();
            for start in starts {
                start_users.insert(start, task.clone());
            }
            tasks.insert(pos, task);
        }
        join_all(tasks.into_values()).await.into_iter().collect()
    }

    /// Runs one step on one chunk through the same path as ticket-driven generation.
    async fn apply_region_step(
        self: Arc<Self>,
        step: &'static ChunkStep,
        pos: ChunkPos,
        cache_radius: i32,
    ) -> Result<(), RegionGenerationError> {
        let chunk_map = Arc::clone(&self);
        let cache = Arc::new(StaticCache2D::create(
            pos.0.x,
            pos.0.y,
            cache_radius,
            move |x, z| chunk_map.region_holder(ChunkPos::new(x, z)),
        ));
        let holder = cache.get(pos.0.x, pos.0.y);
        let ready = holder.apply_step(step, &self, &cache, Arc::clone(&self.generation_pool));
        let failed = RegionGenerationError::StepFailed {
            status: step.target_status,
            pos,
        };
        match ready {
            Some(ready) => ready.await.ok_or(failed),
            None => Err(failed),
        }
    }

    /// Returns the chunks holding structure starts that Features at `pos` places pieces from.
    fn referenced_structure_starts(&self, pos: ChunkPos) -> Vec<ChunkPos> {
        let holder = self.region_holder(pos);
        let Some(chunk) = holder.try_chunk(ChunkStatus::StructureReferences) else {
            panic!("Features dependencies must have structure references at {pos:?}");
        };
        let references = chunk.structure_references();
        references
            .values()
            .flat_map(|sources| sources.iter().copied())
            .collect()
    }

    /// Returns a holder installed by [`Self::insert_region_holders`].
    fn region_holder(&self, pos: ChunkPos) -> Arc<ChunkHolder> {
        let Some(holder) = self.chunks.read_sync(&pos, |_, holder| Arc::clone(holder)) else {
            panic!("region generation installs a holder for every dependency, missing {pos:?}");
        };
        holder
    }
}

/// Returns how far around its center a step that modifies neighbors may read or write chunks.
///
/// Block reads count only where the parent status guarantees generated blocks. Light worksets
/// span a cache window wider than the Light step's declared dependencies.
const fn region_step_reach(step: &ChunkStep) -> i32 {
    let parent_radius = match step.target_status.parent() {
        Some(parent) => step.direct_dependencies.get_radius_of(parent) as i32,
        None => 0,
    };
    match step.target_status {
        ChunkStatus::Light => LIGHT_CACHE_RADIUS,
        _ if step.block_state_write_radius > 0 => {
            if step.block_state_write_radius > parent_radius {
                step.block_state_write_radius
            } else {
                parent_radius
            }
        }
        _ => 0,
    }
}

/// Lists the rectangle expanded by `radius` in ascending X-then-Z order.
fn region_positions(min: ChunkPos, max: ChunkPos, radius: i32) -> Vec<ChunkPos> {
    (min.0.x - radius..=max.0.x + radius)
        .flat_map(|x| (min.0.y - radius..=max.0.y + radius).map(move |z| ChunkPos::new(x, z)))
        .collect()
}
