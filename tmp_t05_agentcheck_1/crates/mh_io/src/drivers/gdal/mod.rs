// crates/mh_io/src/drivers/gdal/mod.rs

//! GDAL 栅格驱动模块
//!
//! 提供读取 GeoTIFF 等栅格格式的功能。
//!
//! # 功能
//!
//! - 读取栅格元数据
//! - 读取波段数据
//! - 获取投影信息
//! - 双线性插值
//!
//! # 依赖
//!
//! 需要启用 `gdal` feature 并安装 GDAL 库。

mod driver;
mod error;

pub use driver::*;
pub use error::*;
