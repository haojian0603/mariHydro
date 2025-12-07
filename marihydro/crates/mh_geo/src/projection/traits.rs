//! 投影 Trait 定义
//!
//! 定义统一的投影接口，支持多种投影类型的扩展。

use crate::ellipsoid::Ellipsoid;
use mh_foundation::error::MhResult;

/// 地图投影 Trait
///
/// 所有投影实现都必须实现此 trait
pub trait MapProjection: Send + Sync {
    /// 获取投影名称
    fn name(&self) -> &'static str;

    /// 获取使用的椭球体
    fn ellipsoid(&self) -> &Ellipsoid;

    /// 正向投影：地理坐标 -> 平面坐标
    ///
    /// # Arguments
    /// - `lon`: 经度 (度)
    /// - `lat`: 纬度 (度)
    ///
    /// # Returns
    /// (x, y) 平面坐标 (米)
    fn forward(&self, lon: f64, lat: f64) -> MhResult<(f64, f64)>;

    /// 逆向投影：平面坐标 -> 地理坐标
    ///
    /// # Arguments
    /// - `x`: 东向坐标 (米)
    /// - `y`: 北向坐标 (米)
    ///
    /// # Returns
    /// (lon, lat) 经度和纬度 (度)
    fn inverse(&self, x: f64, y: f64) -> MhResult<(f64, f64)>;

    /// 获取中央子午线（如适用）
    fn central_meridian(&self) -> Option<f64> {
        None
    }

    /// 获取比例因子
    fn scale_factor(&self) -> f64 {
        1.0
    }

    /// 批量正向投影
    fn forward_batch(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        points.iter().map(|&(lon, lat)| self.forward(lon, lat)).collect()
    }

    /// 批量逆向投影
    fn inverse_batch(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        points.iter().map(|&(x, y)| self.inverse(x, y)).collect()
    }
}

/// 快速投影枚举（静态分发，零成本抽象）
///
/// 使用 enum 而非 trait object 以避免动态分发开销
#[derive(Debug, Clone)]
pub enum FastProjection {
    /// 地理坐标（恒等变换）
    Geographic(Ellipsoid),
    /// 横轴墨卡托（UTM/高斯-克吕格）
    TransverseMercator(TransverseMercatorParams),
    /// Web Mercator
    WebMercator,
}

/// 横轴墨卡托投影参数
#[derive(Debug, Clone)]
pub struct TransverseMercatorParams {
    /// 椭球体
    pub ellipsoid: Ellipsoid,
    /// 中央子午线 (度)
    pub central_meridian: f64,
    /// 纬度原点 (度)
    pub lat_origin: f64,
    /// 比例因子
    pub scale_factor: f64,
    /// 假东 (米)
    pub false_easting: f64,
    /// 假北 (米)
    pub false_northing: f64,
}

impl TransverseMercatorParams {
    /// 创建 UTM 参数
    #[must_use]
    pub fn utm(zone: u8, north: bool) -> Self {
        Self::utm_with_ellipsoid(zone, north, Ellipsoid::WGS84)
    }

    /// 使用指定椭球体创建 UTM 参数
    #[must_use]
    pub fn utm_with_ellipsoid(zone: u8, north: bool, ellipsoid: Ellipsoid) -> Self {
        let central_meridian = f64::from(zone) * 6.0 - 183.0;
        Self {
            ellipsoid,
            central_meridian,
            lat_origin: 0.0,
            scale_factor: 0.9996,
            false_easting: 500_000.0,
            false_northing: if north { 0.0 } else { 10_000_000.0 },
        }
    }

    /// 创建高斯-克吕格 3 度带参数
    #[must_use]
    pub fn gauss_kruger_3(zone: u8) -> Self {
        Self::gauss_kruger_3_with_ellipsoid(zone, Ellipsoid::CGCS2000)
    }

    /// 使用指定椭球体创建高斯-克吕格 3 度带参数
    #[must_use]
    pub fn gauss_kruger_3_with_ellipsoid(zone: u8, ellipsoid: Ellipsoid) -> Self {
        let central_meridian = f64::from(zone) * 3.0;
        Self {
            ellipsoid,
            central_meridian,
            lat_origin: 0.0,
            scale_factor: 1.0,
            false_easting: 500_000.0,
            false_northing: 0.0,
        }
    }

    /// 创建高斯-克吕格 6 度带参数
    #[must_use]
    pub fn gauss_kruger_6(zone: u8) -> Self {
        Self::gauss_kruger_6_with_ellipsoid(zone, Ellipsoid::CGCS2000)
    }

    /// 使用指定椭球体创建高斯-克吕格 6 度带参数
    #[must_use]
    pub fn gauss_kruger_6_with_ellipsoid(zone: u8, ellipsoid: Ellipsoid) -> Self {
        let central_meridian = f64::from(zone) * 6.0 - 3.0;
        Self {
            ellipsoid,
            central_meridian,
            lat_origin: 0.0,
            scale_factor: 1.0,
            false_easting: 500_000.0,
            false_northing: 0.0,
        }
    }

    /// 自定义横轴墨卡托参数
    #[must_use]
    pub fn custom(
        ellipsoid: Ellipsoid,
        central_meridian: f64,
        scale_factor: f64,
        false_easting: f64,
        false_northing: f64,
    ) -> Self {
        Self {
            ellipsoid,
            central_meridian,
            lat_origin: 0.0,
            scale_factor,
            false_easting,
            false_northing,
        }
    }
}

impl FastProjection {
    /// 正向投影
    pub fn forward(&self, lon: f64, lat: f64) -> MhResult<(f64, f64)> {
        use super::transverse_mercator;
        use super::web_mercator;

        match self {
            Self::Geographic(_) => Ok((lon, lat)),
            Self::TransverseMercator(params) => {
                transverse_mercator::forward(params, lon, lat)
            }
            Self::WebMercator => web_mercator::geographic_to_web_mercator(lon, lat),
        }
    }

    /// 逆向投影
    pub fn inverse(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        use super::transverse_mercator;
        use super::web_mercator;

        match self {
            Self::Geographic(_) => Ok((x, y)),
            Self::TransverseMercator(params) => {
                transverse_mercator::inverse(params, x, y)
            }
            Self::WebMercator => web_mercator::web_mercator_to_geographic(x, y),
        }
    }

    /// 获取椭球体
    #[must_use]
    pub fn ellipsoid(&self) -> Ellipsoid {
        match self {
            Self::Geographic(e) => *e,
            Self::TransverseMercator(params) => params.ellipsoid,
            Self::WebMercator => Ellipsoid::WGS84, // Web Mercator 按球体处理
        }
    }

    /// 是否为地理坐标（恒等变换）
    #[must_use]
    pub fn is_geographic(&self) -> bool {
        matches!(self, Self::Geographic(_))
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utm_params() {
        let params = TransverseMercatorParams::utm(50, true);
        assert!((params.central_meridian - 117.0).abs() < 1e-10);
        assert!((params.scale_factor - 0.9996).abs() < 1e-10);
        assert!((params.false_easting - 500_000.0).abs() < 1e-10);
        assert!((params.false_northing - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_utm_south() {
        let params = TransverseMercatorParams::utm(50, false);
        assert!((params.false_northing - 10_000_000.0).abs() < 1e-10);
    }

    #[test]
    fn test_gk3_params() {
        let params = TransverseMercatorParams::gauss_kruger_3(39);
        assert!((params.central_meridian - 117.0).abs() < 1e-10);
        assert!((params.scale_factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gk6_params() {
        let params = TransverseMercatorParams::gauss_kruger_6(20);
        assert!((params.central_meridian - 117.0).abs() < 1e-10);
    }
}