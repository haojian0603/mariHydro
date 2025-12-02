// marihydro\crates\mh_terrain\src/interpolation/mod.rs

//! 插值模块
//!
//! 提供空间和时间插值功能。

pub mod spatial;
pub mod temporal;

// 地形专用插值（占位）
pub mod idw;
pub mod kriging;
pub mod natural_neighbor;

pub use spatial::{
    GeoTransform, InterpolationMethod, InterpolatorConfig, NoDataStrategy, SpatialInterpolator,
};
pub use temporal::{DoubleBufferTimeFrame, TemporalInterpolator, TemporalMethod, TimeFrame};
