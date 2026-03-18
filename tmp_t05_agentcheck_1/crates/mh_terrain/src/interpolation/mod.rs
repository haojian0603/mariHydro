// marihydro\crates\mh_terrain\src/interpolation/mod.rs

//! 插值模块
//!
//! 提供多种空间和时间插值方法。
//!
//! # 空间插值方法
//!
//! - [`idw`]: 反距离加权插值 (Inverse Distance Weighting)
//! - [`kriging`]: 克里金地统计插值
//! - [`natural_neighbor`]: 自然邻居 (Sibson) 插值
//!
//! # 时间插值
//!
//! - [`temporal`]: 时间序列插值
//!
//! # 选择指南
//!
//! | 方法 | 计算复杂度 | 光滑性 | 适用场景 |
//! |------|----------|--------|---------|
//! | IDW | O(n) | 低 | 快速估算、数据密集区域 |
//! | Kriging | O(n³) | 高 | 需要误差估计、地质数据 |
//! | Natural Neighbor | O(n log n) | 高 | 自然场、避免外推 |

pub mod spatial;
pub mod temporal;

// 地形专用插值
pub mod idw;
pub mod kriging;
pub mod natural_neighbor;

pub use spatial::{
    GeoTransform, InterpolationMethod, InterpolatorConfig, NoDataStrategy, SpatialInterpolator,
};
pub use temporal::{DoubleBufferTimeFrame, TemporalInterpolator, TemporalMethod, TimeFrame};

// 重导出插值器核心类型
pub use idw::{IdwConfig, IdwInterpolator};
pub use kriging::{KrigingInterpolator, VariogramModel};
pub use natural_neighbor::{NaturalNeighborConfig, NaturalNeighborInterpolator};
