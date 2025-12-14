// crates/mh_io/src/lib.rs

//! MariHydro IO 模块
//!
//! 提供数据输入输出功能。
//!
//! # 模块
//!
//! - [`drivers`]: 数据读取驱动 (GDAL, NetCDF)
//! - [`exporters`]: 数据导出 (VTU, Shapefile)
//! - [`infra`]: 基础设施 (配置、日志、时间)
//! - [`import`]: 数据导入
//! - [`pipeline`]: 异步 IO 管道
//! - [`snapshot`]: 网格和状态快照
//! - [`checkpoint`]: 检查点保存/恢复
//!
//! # 可选依赖
//!
//! - `gdal`: 启用 GDAL 栅格驱动
//! - `netcdf`: 启用 NetCDF 驱动
//!
//! # 使用示例
//!
//! ## 异步 IO 管道
//!
//! ```rust,ignore
//! use mh_io::pipeline::IoPipeline;
//! use mh_io::snapshot::{MeshSnapshot, StateSnapshot};
//!
//! let pipeline = IoPipeline::new();
//! pipeline.write_vtu_ascii("output.vtu", mesh_snap, state_snap, 0.0)?;
//! pipeline.wait_for_completion(Duration::from_secs(30));
//! ```
//!
//! ## 检查点
//!
//! ```rust,ignore
//! use mh_io::checkpoint::Checkpoint;
//! use mh_io::snapshot::StateSnapshot;
//!
//! let checkpoint = Checkpoint::new(100.0, 1000, state_snap);
//! checkpoint.save(Path::new("checkpoint.mhck"))?;
//! ```

pub mod drivers;
pub mod exporters;
pub mod import;
pub mod error;
pub mod project;

// 计划四新增模块
pub mod checkpoint;
pub mod pipeline;
pub mod snapshot;

// 重导出常用类型
pub use drivers::{GdalDriver, GdalError, NetCdfDriver, NetCdfError, RasterMetadata};
pub use exporters::{VtuExporter, VtuMesh, VtuState};
pub use error::{IoError, IoResult};
/// 类型别名简化
pub type Result<T> = IoResult<T>;

// 重导出计划四新增类型
pub use checkpoint::{Checkpoint, CheckpointError, CheckpointManager};
pub use pipeline::{IoPipeline, OutputRequest, PipelineConfig, PipelineStats, PvdEntry};
pub use snapshot::{MeshSnapshot, SnapshotMeta, StateSnapshot, StateSnapshotMeta, StateStatistics};