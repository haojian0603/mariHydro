//! 高斯-克吕格投影实现
//!
//! 适用于中国区域的投影坐标系统，支持 3 度带和 6 度带。
//! 基于 Karney (2011) 的高精度横轴墨卡托算法。
//!
//! # 特点
//!
//! - 默认使用 CGCS2000 椭球体（中国大地坐标系）
//! - 比例因子 k0 = 1.0
//! - 精度达亚毫米级
//!
//! # 示例
//!
//! ```
//! use mh_geo::projection::{geographic_to_gauss_kruger, gauss_kruger_to_geographic};
//!
//! // 北京 (117°E, 39°N) -> 高斯-克吕格，中央子午线 117°
//! let (x, y) = geographic_to_gauss_kruger(117.0, 39.0, 117.0).unwrap();
//!
//! // 逆向转换
//! let (lon, lat) = gauss_kruger_to_geographic(x, y, 117.0).unwrap();
//! ```

use super::traits::TransverseMercatorParams;
use super::transverse_mercator;
use crate::ellipsoid::Ellipsoid;
use mh_foundation::error::MhResult;

/// 地理坐标 -> 高斯-克吕格
///
/// # Arguments
/// - `lon`: 经度 (度)
/// - `lat`: 纬度 (度)
/// - `central_lon`: 中央子午线经度 (度)
///
/// # Returns
/// (x, y) 高斯-克吕格坐标 (米)，x 包含 500000 假东
///
/// # Errors
/// 此函数不会返回错误（除非坐标极端异常）
pub fn geographic_to_gauss_kruger(lon: f64, lat: f64, central_lon: f64) -> MhResult<(f64, f64)> {
    let params = TransverseMercatorParams::custom(
        Ellipsoid::CGCS2000,
        central_lon,
        1.0, // k0 = 1.0
        500_000.0,
        0.0,
    );
    transverse_mercator::forward(&params, lon, lat)
}

/// 高斯-克吕格 -> 地理坐标
///
/// # Arguments
/// - `x`: 东向坐标 (米)，包含 500000 假东
/// - `y`: 北向坐标 (米)
/// - `central_lon`: 中央子午线经度 (度)
///
/// # Returns
/// (longitude, latitude) 经度和纬度 (度)
///
/// # Errors
/// 此函数不会返回错误（除非坐标极端异常）
pub fn gauss_kruger_to_geographic(x: f64, y: f64, central_lon: f64) -> MhResult<(f64, f64)> {
    let params = TransverseMercatorParams::custom(
        Ellipsoid::CGCS2000,
        central_lon,
        1.0,
        500_000.0,
        0.0,
    );
    transverse_mercator::inverse(&params, x, y)
}

/// 使用 WGS84 椭球体的高斯-克吕格正向转换
///
/// 用于与 GPS 数据直接配合
pub fn geographic_to_gauss_kruger_wgs84(
    lon: f64,
    lat: f64,
    central_lon: f64,
) -> MhResult<(f64, f64)> {
    let params = TransverseMercatorParams::custom(
        Ellipsoid::WGS84,
        central_lon,
        1.0,
        500_000.0,
        0.0,
    );
    transverse_mercator::forward(&params, lon, lat)
}

/// 使用 WGS84 椭球体的高斯-克吕格逆向转换
pub fn gauss_kruger_to_geographic_wgs84(
    x: f64,
    y: f64,
    central_lon: f64,
) -> MhResult<(f64, f64)> {
    let params = TransverseMercatorParams::custom(
        Ellipsoid::WGS84,
        central_lon,
        1.0,
        500_000.0,
        0.0,
    );
    transverse_mercator::inverse(&params, x, y)
}

/// 3度带带号 -> 中央子午线
///
/// 中央子午线 = 带号 × 3
#[must_use]
pub fn gk3_central_meridian(zone: u8) -> f64 {
    f64::from(zone) * 3.0
}

/// 6度带带号 -> 中央子午线
///
/// 中央子午线 = 带号 × 6 - 3
#[must_use]
pub fn gk6_central_meridian(zone: u8) -> f64 {
    f64::from(zone) * 6.0 - 3.0
}

/// 从经度计算 3度带带号
#[must_use]
pub fn auto_gk3_zone(lon: f64) -> u8 {
    let zone = (lon / 3.0).round() as i32;
    zone.clamp(25, 45) as u8
}

/// 从经度计算 6度带带号
#[must_use]
pub fn auto_gk6_zone(lon: f64) -> u8 {
    // 6度带带号：中央子午线 = zone * 6 - 3
    // 反推：zone = (lon + 3) / 6，向最近整数取整
    let zone = ((lon + 3.0) / 6.0).round() as i32;
    zone.clamp(13, 23) as u8
}

/// 3度带高斯-克吕格正向转换
pub fn geographic_to_gk3(lon: f64, lat: f64, zone: u8) -> MhResult<(f64, f64)> {
    let central_lon = gk3_central_meridian(zone);
    geographic_to_gauss_kruger(lon, lat, central_lon)
}

/// 3度带高斯-克吕格逆向转换
pub fn gk3_to_geographic(x: f64, y: f64, zone: u8) -> MhResult<(f64, f64)> {
    let central_lon = gk3_central_meridian(zone);
    gauss_kruger_to_geographic(x, y, central_lon)
}

/// 6度带高斯-克吕格正向转换
pub fn geographic_to_gk6(lon: f64, lat: f64, zone: u8) -> MhResult<(f64, f64)> {
    let central_lon = gk6_central_meridian(zone);
    geographic_to_gauss_kruger(lon, lat, central_lon)
}

/// 6度带高斯-克吕格逆向转换
pub fn gk6_to_geographic(x: f64, y: f64, zone: u8) -> MhResult<(f64, f64)> {
    let central_lon = gk6_central_meridian(zone);
    gauss_kruger_to_geographic(x, y, central_lon)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gauss_kruger_roundtrip() {
        let lon = 117.0;
        let lat = 39.0;
        let central_lon = 117.0;

        let (x, y) = geographic_to_gauss_kruger(lon, lat, central_lon).expect("to GK failed");

        let (lon2, lat2) = gauss_kruger_to_geographic(x, y, central_lon).expect("from GK failed");

        // 高精度要求
        assert!((lon - lon2).abs() < 1e-9);
        assert!((lat - lat2).abs() < 1e-9);
    }

    #[test]
    fn test_gauss_kruger_central_meridian() {
        // 在中央子午线上的点，x 应该接近 500000
        let lon = 117.0;
        let lat = 39.0;
        let central_lon = 117.0;

        let (x, _y) = geographic_to_gauss_kruger(lon, lat, central_lon).expect("to GK");

        assert!(
            (x - 500_000.0).abs() < 1.0,
            "x should be near 500000: {x}"
        );
    }

    #[test]
    fn test_gauss_kruger_3_degree_zone() {
        // 3 度带，中央子午线 117°
        let lon = 116.0;
        let lat = 39.0;
        let central_lon = 117.0;

        let (x, y) = geographic_to_gauss_kruger(lon, lat, central_lon).expect("to GK");

        // 偏离中央子午线 1 度，x 应小于 500000
        assert!(x < 500_000.0, "x should be less than 500000: {x}");
        assert!(y > 4_000_000.0, "y should be positive: {y}");
    }

    #[test]
    fn test_gk3_zone_functions() {
        assert_eq!(auto_gk3_zone(117.0), 39);
        assert_eq!(auto_gk3_zone(116.0), 39);
        assert_eq!(auto_gk3_zone(120.0), 40);

        assert!((gk3_central_meridian(39) - 117.0).abs() < 1e-10);
    }

    #[test]
    fn test_gk6_zone_functions() {
        // 117° 在6度带中属于第20带 (中央子午线117°)
        // 带号计算: zone = floor((lon + 3) / 6) + 1
        // zone = floor((117 + 3) / 6) + 1 = floor(20) + 1 = 21
        // 但中央子午线 117° 对应 zone = 20 (cm = 20*6 - 3 = 117)
        assert_eq!(auto_gk6_zone(117.0), 20);
        assert_eq!(auto_gk6_zone(120.0), 21);
        assert_eq!(auto_gk6_zone(114.0), 20);

        assert!((gk6_central_meridian(20) - 117.0).abs() < 1e-10);
    }

    #[test]
    fn test_gk3_convenience_functions() {
        let lon = 116.5;
        let lat = 39.0;
        let zone = 39;

        let (x, y) = geographic_to_gk3(lon, lat, zone).expect("to GK3");
        let (lon2, lat2) = gk3_to_geographic(x, y, zone).expect("from GK3");

        assert!((lon - lon2).abs() < 1e-9);
        assert!((lat - lat2).abs() < 1e-9);
    }

    #[test]
    fn test_gk6_convenience_functions() {
        let lon = 116.5;
        let lat = 39.0;
        let zone = 20;

        let (x, y) = geographic_to_gk6(lon, lat, zone).expect("to GK6");
        let (lon2, lat2) = gk6_to_geographic(x, y, zone).expect("from GK6");

        assert!((lon - lon2).abs() < 1e-9);
        assert!((lat - lat2).abs() < 1e-9);
    }

    #[test]
    fn test_wgs84_vs_cgcs2000() {
        let lon = 117.0;
        let lat = 39.0;
        let central_lon = 117.0;

        let (x1, y1) = geographic_to_gauss_kruger(lon, lat, central_lon).expect("CGCS2000");
        let (x2, y2) = geographic_to_gauss_kruger_wgs84(lon, lat, central_lon).expect("WGS84");

        // WGS84 和 CGCS2000 差异应该很小（厘米级）
        let dx = (x1 - x2).abs();
        let dy = (y1 - y2).abs();

        assert!(dx < 0.1, "X difference too large: {dx}");
        assert!(dy < 0.1, "Y difference too large: {dy}");
    }

    /// 高精度往返测试
    #[test]
    fn test_gauss_kruger_precision() {
        let test_cases = [
            (117.0, 39.0, 117.0),    // 中央子午线
            (116.0, 39.0, 117.0),    // 偏离1度
            (118.0, 39.0, 117.0),    // 偏离1度
            (117.0, 0.0, 117.0),     // 赤道
            (117.0, 60.0, 117.0),    // 高纬度
        ];

        for (lon, lat, cm) in test_cases {
            let (x, y) = geographic_to_gauss_kruger(lon, lat, cm).expect("forward");
            let (lon2, lat2) = gauss_kruger_to_geographic(x, y, cm).expect("inverse");

            let error_lon = (lon2 - lon).abs();
            let error_lat = (lat2 - lat).abs();

            // 要求误差小于 1e-10 度（约 0.01mm）
            assert!(
                error_lon < 1e-10 && error_lat < 1e-10,
                "Precision error at ({lon}, {lat}): lon_err={error_lon}, lat_err={error_lat}"
            );
        }
    }
}