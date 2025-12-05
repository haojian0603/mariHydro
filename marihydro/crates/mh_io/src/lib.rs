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
//!
//! # 可选依赖
//!
//! - `gdal`: 启用 GDAL 栅格驱动
//! - `netcdf`: 启用 NetCDF 驱动

pub mod drivers;
pub mod exporters;
pub mod import;
pub mod infra;
pub mod project;

// 重导出常用类型
pub use drivers::{GdalDriver, GdalError, NetCdfDriver, NetCdfError, RasterMetadata};
pub use exporters::{VtuExporter, VtuMesh, VtuState};
pub use project::*;