// crates/mh_io/src/lib.rs

//! MariHydro IO 模块
//!
//! 提供数据输入输出功能，包括异步管道、检查点、VTU导出等。
//!
//! # 模块
//!
//! - `drivers`: 数据读取驱动 (GDAL, NetCDF)
//! - `exporters`: 数据导出 (VTU, Shapefile)
//! - `infra`: 基础设施 (配置、日志、时间)
//! - `import`: 数据导入
//! - `pipeline`: 异步 IO 管道
//! - `snapshot`: 网格和状态快照
//! - `checkpoint`: 检查点保存/恢复

pub mod drivers;
pub mod exporters;
pub mod import;
pub mod infra;
pub mod error;
pub mod project;

// 核心功能模块
pub mod checkpoint;
pub mod pipeline;
pub mod snapshot;

// VTU 内部实现（不对外暴露）
mod vtu;

// 重导出常用类型
pub use drivers::{GdalDriver, GdalError, NetCdfDriver, NetCdfError, RasterMetadata};
pub use exporters::{VtuExporter, VtuMesh, VtuState};
pub use error::{IoError, IoResult};

// 类型别名
pub type Result<T> = IoResult<T>;

// 重导出核心功能
pub use checkpoint::{Checkpoint, CheckpointError, CheckpointManager};
pub use pipeline::{IoPipeline, OutputRequest, PipelineConfig, PipelineStats, PvdEntry};
pub use snapshot::{MeshSnapshot, SnapshotMeta, StateSnapshot, StateSnapshotMeta, StateStatistics};