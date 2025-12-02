// marihydro\crates\mh_geo\src/projection/gauss_kruger.rs
//! 高斯-克吕格投影实现
//!
//! 适用于中国区域的投影坐标系统，支持 3 度带和 6 度带

use super::{meridian_arc, meridian_arc_factor, WGS84_A, WGS84_E2, WGS84_EP2};
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
/// 此函数不会返回错误
#[allow(clippy::unnecessary_wraps)]
pub fn geographic_to_gauss_kruger(lon: f64, lat: f64, central_lon: f64) -> MhResult<(f64, f64)> {
    let lon_rad = lon.to_radians();
    let lat_rad = lat.to_radians();
    let l0 = central_lon.to_radians();

    let l = lon_rad - l0;
    let sin_lat = lat_rad.sin();
    let cos_lat = lat_rad.cos();
    let tan_lat = lat_rad.tan();

    let n = WGS84_A / (1.0 - WGS84_E2 * sin_lat.powi(2)).sqrt();
    let t = tan_lat.powi(2);
    let c = WGS84_EP2 * cos_lat.powi(2);
    let a = l * cos_lat;

    let m = meridian_arc(lat_rad);

    let x = m
        + n * tan_lat
            * (a.powi(2) / 2.0
                + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a.powi(4) / 24.0
                + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * WGS84_EP2) * a.powi(6)
                    / 720.0);

    let y = n
        * (a + (1.0 - t + c) * a.powi(3) / 6.0
            + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * WGS84_EP2) * a.powi(5) / 120.0);

    // 高斯-克吕格通常添加 500000 假东
    Ok((y + 500_000.0, x))
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
/// 此函数不会返回错误
#[allow(clippy::unnecessary_wraps, clippy::similar_names)]
pub fn gauss_kruger_to_geographic(x: f64, y: f64, central_lon: f64) -> MhResult<(f64, f64)> {
    let x = x - 500_000.0;

    // 计算底点纬度
    let mf = y;
    let mu = mf / meridian_arc_factor();

    let e1 = (1.0 - (1.0 - WGS84_E2).sqrt()) / (1.0 + (1.0 - WGS84_E2).sqrt());

    let phi1 = mu
        + (3.0 * e1 / 2.0 - 27.0 * e1.powi(3) / 32.0) * (2.0 * mu).sin()
        + (21.0 * e1.powi(2) / 16.0 - 55.0 * e1.powi(4) / 32.0) * (4.0 * mu).sin()
        + (151.0 * e1.powi(3) / 96.0) * (6.0 * mu).sin()
        + (1097.0 * e1.powi(4) / 512.0) * (8.0 * mu).sin();

    let sin_phi1 = phi1.sin();
    let cos_phi1 = phi1.cos();
    let tan_phi1 = phi1.tan();

    let n1 = WGS84_A / (1.0 - WGS84_E2 * sin_phi1.powi(2)).sqrt();
    let r1 = WGS84_A * (1.0 - WGS84_E2) / (1.0 - WGS84_E2 * sin_phi1.powi(2)).powf(1.5);
    let t1 = tan_phi1.powi(2);
    let c1 = WGS84_EP2 * cos_phi1.powi(2);
    let d = x / n1;

    let lat = phi1
        - (n1 * tan_phi1 / r1)
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
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2) + 8.0 * WGS84_EP2 + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / cos_phi1;

    Ok((lon.to_degrees(), lat.to_degrees()))
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

        assert!((lon - lon2).abs() < 1e-6);
        assert!((lat - lat2).abs() < 1e-6);
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
        let central_lon = 117.0; // 第 39 带

        let (x, y) = geographic_to_gauss_kruger(lon, lat, central_lon).expect("to GK");

        // 偏离中央子午线 1 度，x 应小于 500000
        assert!(x < 500_000.0, "x should be less than 500000: {x}");
        assert!(y > 4_000_000.0, "y should be positive: {y}");
    }
}
