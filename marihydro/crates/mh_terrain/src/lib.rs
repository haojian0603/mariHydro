// marihydro\crates\mh_terrain\src/lib.rs

//! 地形数据管理
//!
//! 提供地形数据的存储、插值和访问功能。
//!
//! # 模块
//!
//! - `interpolation`: 空间和时间插值
//! - `raster`: 栅格数据管理
//! - `tin`: TIN 三角网
//! - `tiled`: 分块地形
//! - `provider`: 数据提供者

pub mod interpolation;
pub mod provider;
pub mod raster;
pub mod tiled;
pub mod tin;

// 重导出常用类型
pub use interpolation::{
    GeoTransform, InterpolationMethod, InterpolatorConfig, NoDataStrategy, SpatialInterpolator,
    TemporalInterpolator, TemporalMethod, TimeFrame,
};
