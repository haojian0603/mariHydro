// crates/mh_io/src/drivers/mod.rs

//! 数据驱动模块
//!
//! 提供读取各种地理数据格式的驱动程序。

pub mod gdal;
pub mod netcdf;
pub mod raster;

// 重导出
pub use self::gdal::{GdalDriver, GdalError, RasterBand, RasterMetadata};
pub use self::netcdf::{Dimension, NetCdfDriver, NetCdfError, Variable, VariableInfo};
pub use raster::*;