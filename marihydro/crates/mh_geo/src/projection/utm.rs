// marihydro\crates\mh_geo\src/projection/utm.rs
//! UTM 投影实现
//!
//! 基于 Transverse Mercator 投影，使用 WGS84 椭球参数

use super::{meridian_arc, meridian_arc_factor, WGS84_A, WGS84_E2, WGS84_EP2};
use mh_foundation::error::{MhError, MhResult};

/// UTM 比例因子
pub const UTM_K0: f64 = 0.9996;

/// 地理坐标 -> UTM
///
/// # Arguments
/// - `lon`: 经度 (度)
/// - `lat`: 纬度 (度)
/// - `zone`: UTM 带号 (1-60)
/// - `north`: 是否为北半球
///
/// # Returns
/// (easting, northing) 东向坐标和北向坐标 (米)
///
/// # Errors
/// 如果纬度超出 UTM 有效范围 (-80°, 84°) 则返回错误
pub fn geographic_to_utm(lon: f64, lat: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    if !(-80.0..=84.0).contains(&lat) {
        return Err(MhError::InvalidInput {
            message: format!("Latitude {lat} out of UTM range (-80, 84)"),
        });
    }

    let lon_rad = lon.to_radians();
    let lat_rad = lat.to_radians();

    let central_lon = f64::from(zone) * 6.0 - 183.0;
    let central_lon_rad = central_lon.to_radians();

    let n = WGS84_A / (1.0 - WGS84_E2 * lat_rad.sin().powi(2)).sqrt();
    let t = lat_rad.tan().powi(2);
    let c = WGS84_EP2 * lat_rad.cos().powi(2);
    let a_coef = (lon_rad - central_lon_rad) * lat_rad.cos();

    let m = meridian_arc(lat_rad);

    let x = UTM_K0
        * n
        * (a_coef
            + (1.0 - t + c) * a_coef.powi(3) / 6.0
            + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * WGS84_EP2) * a_coef.powi(5) / 120.0);

    let y = UTM_K0
        * (m + n * lat_rad.tan()
            * (a_coef.powi(2) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a_coef.powi(4) / 24.0
                + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * WGS84_EP2) * a_coef.powi(6)
                    / 720.0));

    // 添加假东和假北
    let easting = x + 500_000.0;
    let northing = if north { y } else { y + 10_000_000.0 };

    Ok((easting, northing))
}

/// UTM -> 地理坐标
///
/// # Arguments
/// - `x`: 东向坐标 (米)
/// - `y`: 北向坐标 (米)
/// - `zone`: UTM 带号 (1-60)
/// - `north`: 是否为北半球
///
/// # Returns
/// (longitude, latitude) 经度和纬度 (度)
///
/// # Errors
/// 返回可能的转换错误
#[allow(clippy::similar_names)]
pub fn utm_to_geographic(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    let x = x - 500_000.0;
    let y = if north { y } else { y - 10_000_000.0 };

    let central_lon = f64::from(zone) * 6.0 - 183.0;

    // 底点纬度
    let mu = y / (UTM_K0 * meridian_arc_factor());

    let e1 = (1.0 - (1.0 - WGS84_E2).sqrt()) / (1.0 + (1.0 - WGS84_E2).sqrt());

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let n1 = WGS84_A / (1.0 - WGS84_E2 * phi1.sin().powi(2)).sqrt();
    let r1 = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * phi1.sin().powi(2)).powf(1.5);
    let t1 = phi1.tan().powi(2);
    let c1 = WGS84_EP2 * phi1.cos().powi(2);
    let d = x / (n1 * UTM_K0);

    let lat = phi1
        - (n1 * phi1.tan() / r1)
            * (d.powi(2) / 2.0
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * WGS84_EP2) * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * WGS84_EP2
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);

    let lon = central_lon.to_radians()
        + (d
            - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2) + 8.0 * WGS84_EP2
                + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / phi1.cos();

    Ok((lon.to_degrees(), lat.to_degrees()))
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utm_roundtrip() {
        // 北京 116°E, 40°N
        let lon = 116.0;
        let lat = 40.0;
        let zone = 50;
        let north = true;

        let (x, y) = geographic_to_utm(lon, lat, zone, north).expect("to UTM failed");

        // 验证 UTM 坐标在合理范围
        assert!(x > 400_000.0 && x < 600_000.0, "x out of range: {x}");
        assert!(y > 4_000_000.0 && y < 5_000_000.0, "y out of range: {y}");

        // 反向转换
        let (lon2, lat2) = utm_to_geographic(x, y, zone, north).expect("from UTM failed");

        assert!(
            (lon - lon2).abs() < 1e-6,
            "lon mismatch: {lon} vs {lon2}"
        );
        assert!(
            (lat - lat2).abs() < 1e-6,
            "lat mismatch: {lat} vs {lat2}"
        );
    }

    #[test]
    fn test_utm_south_hemisphere() {
        // 悉尼 151°E, -33.9°S
        let lon = 151.0;
        let lat = -33.9;
        let zone = 56;
        let north = false;

        let (x, y) = geographic_to_utm(lon, lat, zone, north).expect("to UTM failed");

        // 南半球 y 应该包含假北
        assert!(y > 6_000_000.0, "y should include false northing: {y}");

        let (lon2, lat2) = utm_to_geographic(x, y, zone, north).expect("from UTM failed");

        assert!((lon - lon2).abs() < 1e-6);
        assert!((lat - lat2).abs() < 1e-6);
    }

    #[test]
    fn test_utm_out_of_range() {
        // 纬度超出范围
        let result = geographic_to_utm(0.0, 85.0, 31, true);
        assert!(result.is_err());

        let result = geographic_to_utm(0.0, -81.0, 31, false);
        assert!(result.is_err());
    }
}
