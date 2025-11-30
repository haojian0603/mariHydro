// src-tauri/src/marihydro/infra/mod.rs
pub mod config;
pub mod constants;
pub mod db;
pub mod logger;
pub mod perf;
pub mod storage;
pub mod time;

pub use config::ProjectConfig;
pub use constants::{physics, validation, defaults};
pub use db::{Database, JobRecord, ProjectRecord};
pub use logger::{init_logging, FrontendLogEntry};
pub use perf::{
    PerfStats, PerfTimer, ChunkIter, CACHE_BLOCK_SIZE, PARALLEL_THRESHOLD,
    parallel_max, parallel_min, parallel_sum, should_parallelize,
};
pub use storage::{MemoryStorage, WorkflowStorage};
pub use time::{TimeManager, TimezoneConfig};
