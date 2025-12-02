// marihydro\crates\mh_geo\src/projection/web_mercator.rs
//! Web Mercator 投影实现 (EPSG:3857)
//!
//! 也称为 Pseudo Mercator 或 Spherical Mercator

use super::WGS84_A;
use mh_foundation::error::MhResult;
use std::f64::consts::PI;

/// Web Mercator 最大纬度 (度)
pub const WEB_MERCATOR_MAX_LAT: f64 = 85.051_128_779;

/// 地理坐标 -> Web Mercator
///
/// # Arguments
/// - `lon`: 经度 (度)
/// - `lat`: 纬度 (度)
///
/// # Returns
/// (x, y) Web Mercator 坐标 (米)
///
/// # Errors
/// 此函数不会返回错误，纬度会被自动裁剪到有效范围
#[allow(clippy::unnecessary_wraps)]
pub fn geographic_to_web_mercator(lon: f64, lat: f64) -> MhResult<(f64, f64)> {
    let lat = lat.clamp(-WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT);

    let x = WGS84_A * lon.to_radians();
    let lat_rad = lat.to_radians();
    let y = WGS84_A * ((PI / 4.0 + lat_rad / 2.0).tan()).ln();

    Ok((x, y))
}

/// Web Mercator -> 地理坐标
///
/// # Arguments
/// - `x`: Web Mercator x 坐标 (米)
/// - `y`: Web Mercator y 坐标 (米)
///
/// # Returns
/// (longitude, latitude) 经度和纬度 (度)
///
/// # Errors
/// 此函数不会返回错误
#[allow(clippy::unnecessary_wraps)]
pub fn web_mercator_to_geographic(x: f64, y: f64) -> MhResult<(f64, f64)> {
    let lon = (x / WGS84_A).to_degrees();
    let lat = (2.0 * (y / WGS84_A).exp().atan() - PI / 2.0).to_degrees();

    Ok((lon, lat))
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_mercator_roundtrip() {
        let lon = 116.0;
        let lat = 40.0;

        let (x, y) = geographic_to_web_mercator(lon, lat).expect("to WebMercator failed");

        let (lon2, lat2) = web_mercator_to_geographic(x, y).expect("from WebMercator failed");

        assert!((lon - lon2).abs() < 1e-6);
        assert!((lat - lat2).abs() < 1e-6);
    }

    #[test]
    fn test_web_mercator_origin() {
        let (x, y) = geographic_to_web_mercator(0.0, 0.0).expect("origin");
        assert!(x.abs() < 1e-6);
        assert!(y.abs() < 1e-6);
    }

    #[test]
    fn test_web_mercator_clamp_latitude() {
        // 超出范围的纬度应被裁剪
        let (_, y1) = geographic_to_web_mercator(0.0, 90.0).expect("high lat");
        let (_, y2) = geographic_to_web_mercator(0.0, WEB_MERCATOR_MAX_LAT).expect("max lat");
        assert!((y1 - y2).abs() < 1e-6);
    }

    #[test]
    fn test_web_mercator_known_values() {
        // 北京约在 116°E, 40°N
        let (x, y) = geographic_to_web_mercator(116.0, 40.0).expect("beijing");

        // Web Mercator 坐标应该在合理范围
        assert!(x > 12_900_000.0 && x < 12_950_000.0, "x out of range: {x}");
        assert!(y > 4_800_000.0 && y < 4_900_000.0, "y out of range: {y}");
    }
}
