// crates/mh_io/src/drivers/mod.rs

//! 数据驱动模块
//!
//! 提供读取各种地理数据格式的驱动程序。
//! DRIVER_METADATA_SCOPE: 公开驱动接口允许缺失的可选元数据返回 `None`，但一旦驱动对象存在且属性读取、首波段查询或字符串元数据解码失败，就必须显式报错，不能伪装成“属性不存在”。

pub mod gdal;
pub mod netcdf;
pub mod raster;

// 重导出
pub use self::gdal::{GdalDriver, GdalError, RasterBand, RasterMetadata};
pub use self::netcdf::{Dimension, NetCdfDriver, NetCdfError, Variable, VariableInfo};
pub use raster::*;
