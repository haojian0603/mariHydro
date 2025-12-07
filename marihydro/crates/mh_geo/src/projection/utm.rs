//! UTM 投影实现
//!
//! 基于 Karney (2011) 的高精度横轴墨卡托算法，精度达亚毫米级。
//!
//! # 特点
//!
//! - 使用 6 阶 Krüger 级数展开
//! - 支持 WGS84 椭球体
//! - 北半球/南半球自动处理假北
//!
//! # 示例
//!
//! ```
//! use mh_geo::projection::{geographic_to_utm, utm_to_geographic};
//!
//! // 北京 (116°E, 40°N) -> UTM 50N
//! let (x, y) = geographic_to_utm(116.0, 40.0, 50, true).unwrap();
//!
//! // 逆向转换
//! let (lon, lat) = utm_to_geographic(x, y, 50, true).unwrap();
//! ```

use super::traits::TransverseMercatorParams;
use super::transverse_mercator;
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
    // 验证纬度范围
    if !(-80.0..=84.0).contains(&lat) {
        return Err(MhError::InvalidInput {
            message: format!("Latitude {lat} out of UTM range (-80, 84)"),
        });
    }

    // 验证带号范围
    if !(1..=60).contains(&zone) {
        return Err(MhError::InvalidInput {
            message: format!("UTM zone {zone} out of range (1-60)"),
        });
    }

    let params = TransverseMercatorParams::utm(zone, north);
    transverse_mercator::forward(&params, lon, lat)
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
pub fn utm_to_geographic(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    // 验证带号范围
    if !(1..=60).contains(&zone) {
        return Err(MhError::InvalidInput {
            message: format!("UTM zone {zone} out of range (1-60)"),
        });
    }

    let params = TransverseMercatorParams::utm(zone, north);
    transverse_mercator::inverse(&params, x, y)
}

/// 从经纬度自动计算 UTM 带号
///
/// # Arguments
/// - `lon`: 经度 (度)
///
/// # Returns
/// UTM 带号 (1-60)
#[must_use]
pub fn auto_utm_zone(lon: f64) -> u8 {
    let zone = ((lon + 180.0) / 6.0).floor() as i32 + 1;
    zone.clamp(1, 60) as u8
}

/// 获取 UTM 带的中央子午线
///
/// # Arguments
/// - `zone`: UTM 带号 (1-60)
///
/// # Returns
/// 中央子午线经度 (度)
#[must_use]
pub fn utm_central_meridian(zone: u8) -> f64 {
    f64::from(zone) * 6.0 - 183.0
}

/// 计算 UTM 投影的比例因子
///
/// # Arguments
/// - `lon`: 经度 (度)
/// - `lat`: 纬度 (度)
/// - `zone`: UTM 带号
///
/// # Returns
/// 在该点的比例因子
#[must_use]
pub fn utm_scale_factor(lon: f64, lat: f64, zone: u8) -> f64 {
    let params = TransverseMercatorParams::utm(zone, lat >= 0.0);
    transverse_mercator::scale_factor_at(&params, lon, lat)
}

/// 计算 UTM 投影的子午线收敛角
///
/// # Arguments
/// - `lon`: 经度 (度)
/// - `lat`: 纬度 (度)
/// - `zone`: UTM 带号
///
/// # Returns
/// 子午线收敛角 (弧度)
#[must_use]
pub fn utm_convergence_angle(lon: f64, lat: f64, zone: u8) -> f64 {
    let params = TransverseMercatorParams::utm(zone, lat >= 0.0);
    transverse_mercator::convergence_angle(&params, lon, lat)
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

        // 高精度要求：1e-9 度约 0.1mm
        assert!(
            (lon - lon2).abs() < 1e-9,
            "lon mismatch: {lon} vs {lon2}"
        );
        assert!(
            (lat - lat2).abs() < 1e-9,
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

        assert!((lon - lon2).abs() < 1e-9);
        assert!((lat - lat2).abs() < 1e-9);
    }

    #[test]
    fn test_utm_out_of_range() {
        // 纬度超出范围
        let result = geographic_to_utm(0.0, 85.0, 31, true);
        assert!(result.is_err());

        let result = geographic_to_utm(0.0, -81.0, 31, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_auto_utm_zone() {
        assert_eq!(auto_utm_zone(116.0), 50);
        assert_eq!(auto_utm_zone(-122.0), 10);
        assert_eq!(auto_utm_zone(0.0), 31);
        assert_eq!(auto_utm_zone(-180.0), 1);
        assert_eq!(auto_utm_zone(180.0), 60);
    }

    #[test]
    fn test_utm_central_meridian() {
        assert!((utm_central_meridian(50) - 117.0).abs() < 1e-10);
        assert!((utm_central_meridian(31) - 3.0).abs() < 1e-10);
        assert!((utm_central_meridian(1) - (-177.0)).abs() < 1e-10);
    }

    /// EPSG 标准验证测试 - UTM Zone 51N
    /// 
    /// 目标精度：1mm (0.001m)
    #[test]
    fn test_utm_zone51n_epsg_validation() {
        const EPSG_TEST_CASES: &[(f64, f64, f64, f64)] = &[
            // Verified against PROJ 9 (pyproj 3.7.2, EPSG:32651)
            (121.880356, 29.887703, 391888.0637264130, 3306868.4563851040),
            (121.430427, 28.637151, 346582.4108433011, 3168793.409367069),
            (121.880772, 31.491324, 393700.3650201835, 3484597.440826551),
            (122.625275, 30.246954, 463948.3333072607, 3346209.757229396),
        ];

        let zone = 51;
        let north = true;

        // 目标精度：0.0001mm
        const TOLERANCE_METERS: f64 = 0.000001;

        println!("\n=== UTM Zone 51N EPSG 精度验证 (Karney 算法) ===");
        println!(
            "{:<25} {:<18} {:<18} {:<12} {:<12}",
            "输入(lon,lat)", "计算X", "期望X", "误差X(mm)", "误差Y(mm)"
        );
        println!("{}", "-".repeat(90));

        let mut max_error_x = 0.0_f64;
        let mut max_error_y = 0.0_f64;

        for (lon, lat, expected_x, expected_y) in EPSG_TEST_CASES {
            let (actual_x, actual_y) =
                geographic_to_utm(*lon, *lat, zone, north).expect("UTM 投影失败");

            let error_x = (actual_x - expected_x).abs();
            let error_y = (actual_y - expected_y).abs();

            max_error_x = max_error_x.max(error_x);
            max_error_y = max_error_y.max(error_y);

            println!(
                "({:>10.6}, {:>9.6}) {:>18.6} {:>18.6} {:>12.6} {:>12.6}",
                lon,
                lat,
                actual_x,
                *expected_x,
                error_x * 1000.0,
                error_y * 1000.0
            );
        }

        println!("{}", "-".repeat(90));
        println!(
            "最大误差: X={:.6}mm, Y={:.6}mm",
            max_error_x * 1000.0,
            max_error_y * 1000.0
        );
        println!("目标精度: {}mm", TOLERANCE_METERS * 1000.0);
        println!(
            "结果: {}",
            if max_error_x < TOLERANCE_METERS && max_error_y < TOLERANCE_METERS {
                "✓ 达标"
            } else {
                "✗ 未达标"
            }
        );

        assert!(
            max_error_x < TOLERANCE_METERS,
            "UTM X 坐标误差超过阈值: {:.6}m > {:.3}m",
            max_error_x,
            TOLERANCE_METERS
        );
        assert!(
            max_error_y < TOLERANCE_METERS,
            "UTM Y 坐标误差超过阈值: {:.6}m > {:.3}m",
            max_error_y,
            TOLERANCE_METERS
        );
    }

    /// 往返精度测试
    #[test]
    fn test_utm_zone51n_roundtrip_precision() {
        const TEST_POINTS: &[(f64, f64)] = &[
            (121.880356, 29.887703),
            (121.430427, 28.637151),
            (121.880772, 31.491324),
            (122.625275, 30.246954),
            (117.0, 0.0),   // 赤道
            (117.0, 84.0),  // 高纬度
            (114.0, 40.0),  // 带边缘
        ];

        let zone = 51;
        let north = true;

        println!("\n=== UTM Zone 51N 往返精度验证 ===");

        for (lon, lat) in TEST_POINTS {
            // 跳过超出带号范围的点
            let expected_zone = auto_utm_zone(*lon);
            if expected_zone != zone {
                continue;
            }

            let (x, y) = geographic_to_utm(*lon, *lat, zone, north).expect("Forward failed");
            let (lon2, lat2) = utm_to_geographic(x, y, zone, north).expect("Inverse failed");

            let error_lon = (lon2 - lon).abs();
            let error_lat = (lat2 - lat).abs();
            let max_error = error_lon.max(error_lat);

            // 往返精度要求：1e-12 度（约 0.0001mm）
            assert!(
                max_error < 1e-12,
                "往返误差过大: {:.2e}度 at ({}, {})",
                max_error,
                lon,
                lat
            );
        }

        println!("往返精度验证通过：误差 < 1e-12 度");
    }

    #[test]
    fn test_utm_scale_factor() {
        // 中央子午线处比例因子应为 k0
        let k = utm_scale_factor(117.0, 40.0, 50);
        assert!((k - 0.9996).abs() < 0.0001, "k = {k}");

        // 偏离中央子午线，比例因子应该增大
        let k_offset = utm_scale_factor(120.0, 40.0, 50);
        assert!(k_offset > k, "k_offset = {k_offset}");
    }

    #[test]
    fn test_utm_convergence_angle() {
        // 中央子午线上收敛角应接近 0
        let gamma = utm_convergence_angle(117.0, 40.0, 50);
        assert!(gamma.abs() < 0.001, "gamma = {gamma}");
    }
}