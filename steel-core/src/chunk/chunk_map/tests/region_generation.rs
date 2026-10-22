use std::time::Duration;

use rustc_hash::FxHashMap;
use steel_utils::Identifier;
use steel_utils::types::{Difficulty, GameType};
use steel_worldgen::biomes::BiomeSourceKind;
use steel_worldgen::density_functions::overworld::OverworldNoiseSettings;
use tokio::time::sleep;
use toml::map::Map;

use super::*;
use crate::bootstrap::init_globals;
use crate::chunk::chunk_map::RegionGenerationError;
use crate::chunk::chunk_pyramid::GENERATION_PYRAMID;
use crate::chunk::chunk_request::{ChunkRequest, ChunkRequestHandle, ChunkTicketKind, ReadyChunks};
use crate::entity::damage::DamageHistory;
use crate::level_data::{GameTimeSource, WorldGenerationSettings};
use crate::world::{WorldConfig, WorldStorageConfig};
use crate::worldgen::{ChunkGeneratorType, OverworldGenerator};

const SEED: i64 = 13_579;

struct GeneratedWorld {
    world: Arc<World>,
    _damage_history: Arc<DamageHistory>,
}

/// Every block state and biome of a Full chunk, in section order.
#[derive(PartialEq, Eq)]
struct ChunkContents {
    blocks: Vec<u16>,
    biomes: Vec<u16>,
}

impl ChunkContents {
    fn of(holder: &ChunkHolder) -> Self {
        let Some(chunk) = holder.try_chunk(ChunkStatus::Full) else {
            panic!("{:?} should be Full", holder.get_pos());
        };
        let mut blocks = Vec::new();
        let mut biomes = Vec::new();
        for section in &chunk.sections.sections {
            let section = section.read();
            for y in 0..16 {
                for z in 0..16 {
                    for x in 0..16 {
                        blocks.push(section.states.get(x, y, z).0);
                    }
                }
            }
            for y in 0..4 {
                for z in 0..4 {
                    for x in 0..4 {
                        biomes.push(section.biomes.get(x, y, z));
                    }
                }
            }
        }
        Self { blocks, biomes }
    }
}

fn overworld(key: &'static str, threads: usize) -> GeneratedWorld {
    init_globals();
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("region generation test pool should build"),
    );
    let runtime = Arc::new(Runtime::new().expect("region generation test runtime should build"));
    let generator = Arc::new(ChunkGeneratorType::Overworld(OverworldGenerator::new(
        None,
        BiomeSourceKind::overworld(SEED as u64),
        SEED as u64,
        &pool,
    )));
    let damage_history = Arc::new(DamageHistory::default());
    let config = WorldConfig {
        damage_history: Arc::clone(&damage_history),
        game_time_source: GameTimeSource::Primary,
        storage: WorldStorageConfig::RamOnly,
        level_data_path: None,
        generator,
        generation_settings: WorldGenerationSettings::from_generator_config(
            Identifier::vanilla_static("overworld"),
            &toml::Value::Table(Map::new()),
            OVERWORLD.key.clone(),
            OVERWORLD.min_y,
            OVERWORLD.height,
        ),
        view_distance: 2,
        simulation_distance: 2,
        max_chained_neighbor_updates: 1_000_000,
        compression: None,
        is_flat: false,
        sea_level: OverworldNoiseSettings::SEA_LEVEL,
        default_gamemode: GameType::Survival,
        difficulty: Difficulty::Normal,
    };
    let world = runtime
        .block_on(World::new_with_config(
            Arc::clone(&runtime),
            Identifier::new_static("region_generation", key),
            &OVERWORLD,
            SEED,
            config,
            pool,
        ))
        .expect("region generation test world should build");
    GeneratedWorld {
        world,
        _damage_history: damage_history,
    }
}

fn rectangle(min: ChunkPos, max: ChunkPos) -> Vec<ChunkPos> {
    (min.0.x..=max.0.x)
        .flat_map(|x| (min.0.y..=max.0.y).map(move |z| ChunkPos::new(x, z)))
        .collect()
}

async fn await_request(chunk_map: &Arc<ChunkMap>, request: &ChunkRequestHandle) -> ReadyChunks {
    loop {
        chunk_map.advance_scheduling();
        if let Some(ready) = request.ready_chunks() {
            return ready;
        }
        sleep(Duration::from_millis(1)).await;
    }
}

/// Generates through tickets, decorating one chunk at a time in ascending X-then-Z order.
///
/// Ticket generation otherwise runs neighboring Features steps in scheduling order, and
/// overlapping decorations then differ between runs.
fn generate_with_tickets(
    world: &Arc<World>,
    min: ChunkPos,
    max: ChunkPos,
) -> FxHashMap<ChunkPos, ChunkContents> {
    let chunk_map = &world.chunk_map;
    let feature_radius = GENERATION_PYRAMID
        .get_step_to(ChunkStatus::Full)
        .get_accumulated_radius_of(ChunkStatus::Features) as i32;
    let contents = chunk_map.chunk_runtime.block_on(async {
        let mut decorated = Vec::new();
        for pos in rectangle(
            ChunkPos::new(min.0.x - feature_radius, min.0.y - feature_radius),
            ChunkPos::new(max.0.x + feature_radius, max.0.y + feature_radius),
        ) {
            let request =
                chunk_map.request_chunk(pos, ChunkStatus::Features, ChunkTicketKind::Command);
            await_request(chunk_map, &request).await;
            decorated.push(request);
        }
        let request = chunk_map.request_chunks(ChunkRequest {
            status: ChunkStatus::Full,
            positions: rectangle(min, max),
            ticket_kind: ChunkTicketKind::Command,
        });
        await_request(chunk_map, &request)
            .await
            .holders
            .iter()
            .map(|holder| (holder.get_pos(), ChunkContents::of(holder)))
            .collect()
    });
    stop_chunk_tasks(world);
    contents
}

fn generate_region(
    world: &Arc<World>,
    min: ChunkPos,
    max: ChunkPos,
) -> FxHashMap<ChunkPos, ChunkContents> {
    stop_chunk_tasks(world);
    let holders = match world
        .chunk_map
        .chunk_runtime
        .block_on(world.chunk_map.generate_region(min, max))
    {
        Ok(holders) => holders,
        Err(error) => panic!("region generation failed: {error}"),
    };
    assert_eq!(
        holders
            .iter()
            .map(|holder| holder.get_pos())
            .collect::<Vec<_>>(),
        rectangle(min, max)
    );
    holders
        .iter()
        .map(|holder| (holder.get_pos(), ChunkContents::of(holder)))
        .collect()
}

#[test]
fn region_generation_matches_ticket_generation() {
    let min = ChunkPos::new(-1, 2);
    let max = ChunkPos::new(1, 3);

    let ticketed = overworld("ticketed", 4);
    let expected = generate_with_tickets(&ticketed.world, min, max);
    let region = overworld("region", 4);
    let actual = generate_region(&region.world, min, max);

    for pos in rectangle(min, max) {
        assert!(
            expected[&pos] == actual[&pos],
            "region generation diverged from ticket generation at {pos:?}"
        );
    }
}

#[test]
fn region_generation_requires_an_unused_map() {
    let fixture = overworld("unused_map", 1);
    let chunk_map = &fixture.world.chunk_map;
    let pos = ChunkPos::new(0, 0);

    let running = chunk_map
        .chunk_runtime
        .block_on(chunk_map.generate_region(pos, pos));
    assert_eq!(running.err(), Some(RegionGenerationError::MapInUse));

    stop_chunk_tasks(&fixture.world);
    let inverted = chunk_map
        .chunk_runtime
        .block_on(chunk_map.generate_region(ChunkPos::new(1, 0), pos));
    assert!(matches!(
        inverted,
        Err(RegionGenerationError::InvalidBounds { .. })
    ));
    assert!(chunk_map.chunks.is_empty());
}
