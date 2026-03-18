// marihydro\crates\mh_terrain\src/provider.rs

//! 地形数据提供者
//!
//! 抽象地形数据的获取接口。

use mh_foundation::error::MhResult;
use mh_geo::Point2D;

/// 地形数据提供者 trait
pub trait TerrainProvider {
    /// 获取单点高程
    fn elevation_at(&self, point: Point2D) -> MhResult<f64>;

    /// 批量获取高程
    fn elevations_at(&self, points: &[Point2D], output: &mut [f64]) -> MhResult<()>;
}
