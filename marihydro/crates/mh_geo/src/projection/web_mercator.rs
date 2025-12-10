//! Web Mercator 投影实现 (EPSG:3857)
//!
//! 也称为 Pseudo Mercator 或 Spherical Mercator。
//!
//! # 注意
//!
//! Web Mercator 将地球视为正球体，不使用椭球体参数。
//! 这在高纬度地区会产生较大形变，仅适用于网页地图显示。
//! **不建议用于物理计算**，仅用于可视化和底图对齐。

use crate::ellipsoid::Ellipsoid;
use mh_foundation::error::MhResult;
use std::f64::consts::PI;

/// Web Mercator 使用的地球半径（等于 WGS84 长半轴）
pub const WEB_MERCATOR_RADIUS: f64 = Ellipsoid::WGS84.a;

/// Web Mercator 最大纬度 (度)
///
/// 对应 y = ±20037508.34... 米
pub const WEB_MERCATOR_MAX_LAT: f64 = 85.051_128_779;

/// Web Mercator 世界范围 (米)
///
/// x, y 的范围都是 [-20037508.34, 20037508.34]
pub const WEB_MERCATOR_MAX_EXTENT: f64 = PI * WEB_MERCATOR_RADIUS;

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

    let x = WEB_MERCATOR_RADIUS * lon.to_radians();
    let lat_rad = lat.to_radians();
    let y = WEB_MERCATOR_RADIUS * ((PI / 4.0 + lat_rad / 2.0).tan()).ln();

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
    let lon = (x / WEB_MERCATOR_RADIUS).to_degrees();
    let lat = (2.0 * (y / WEB_MERCATOR_RADIUS).exp().atan() - PI / 2.0).to_degrees();

    Ok((lon, lat))
}

/// 计算 Web Mercator 分辨率
///
/// 返回在指定纬度和缩放级别下，每像素对应的米数
///
/// # Arguments
/// - `lat`: 纬度 (度)
/// - `zoom`: 缩放级别 (0-22)
/// - `tile_size`: 瓦片像素大小（通常为 256）
#[must_use]
pub fn web_mercator_resolution(lat: f64, zoom: u8, tile_size: u32) -> f64 {
    let lat_rad = lat.to_radians();
    let circumference = 2.0 * PI * WEB_MERCATOR_RADIUS * lat_rad.cos();
    let total_pixels = f64::from(tile_size) * 2.0_f64.powi(i32::from(zoom));
    circumference / total_pixels
}

/// 计算 Web Mercator 比例尺分母
///
/// 返回在指定纬度和缩放级别下的比例尺分母
/// 例如返回 25000 表示 1:25000
///
/// # Arguments
/// - `lat`: 纬度 (度)
/// - `zoom`: 缩放级别
/// - `dpi`: 屏幕 DPI（通常为 96）
#[must_use]
pub fn web_mercator_scale(lat: f64, zoom: u8, dpi: f64) -> f64 {
    let resolution = web_mercator_resolution(lat, zoom, 256);
    // 1 inch = 0.0254 m
    resolution * dpi / 0.0254
}

/// 经纬度 -> 瓦片坐标
///
/// 返回在指定缩放级别下的瓦片 X, Y 坐标
///
/// # Arguments
/// - `lon`: 经度 (度)
/// - `lat`: 纬度 (度)
/// - `zoom`: 缩放级别
#[must_use]
pub fn lonlat_to_tile(lon: f64, lat: f64, zoom: u8) -> (u32, u32) {
    let n = 2.0_f64.powi(i32::from(zoom));
    let x = ((lon + 180.0) / 360.0 * n).floor() as u32;

    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / PI) / 2.0 * n).floor() as u32;

    (x, y)
}

/// 瓦片坐标 -> 瓦片左上角经纬度
///
/// # Arguments
/// - `x`: 瓦片 X 坐标
/// - `y`: 瓦片 Y 坐标
/// - `zoom`: 缩放级别
#[must_use]
pub fn tile_to_lonlat(x: u32, y: u32, zoom: u8) -> (f64, f64) {
    let n = 2.0_f64.powi(i32::from(zoom));
    let lon = f64::from(x) / n * 360.0 - 180.0;
    let lat_rad = (PI * (1.0 - 2.0 * f64::from(y) / n)).sinh().atan();
    let lat = lat_rad.to_degrees();
    (lon, lat)
}

/// 瓦片范围 -> 边界框 (Web Mercator 坐标)
///
/// 返回瓦片的 (`min_x`, `min_y`, `max_x`, `max_y`)
#[must_use]
pub fn tile_to_bbox(x: u32, y: u32, zoom: u8) -> (f64, f64, f64, f64) {
    let (lon_min, lat_max) = tile_to_lonlat(x, y, zoom);
    let (lon_max, lat_min) = tile_to_lonlat(x + 1, y + 1, zoom);

    let (x_min, y_min) = geographic_to_web_mercator(lon_min, lat_min).unwrap_or((0.0, 0.0));
    let (x_max, y_max) = geographic_to_web_mercator(lon_max, lat_max).unwrap_or((0.0, 0.0));

    (x_min, y_min, x_max, y_max)
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

        assert!((lon - lon2).abs() < 1e-9);
        assert!((lat - lat2).abs() < 1e-9);
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

    #[test]
    fn test_web_mercator_extent() {
        // 世界范围测试
        let (x_max, _) = geographic_to_web_mercator(180.0, 0.0).expect("max lon");
        assert!((x_max - WEB_MERCATOR_MAX_EXTENT).abs() < 1.0);

        let (_, y_max) = geographic_to_web_mercator(0.0, WEB_MERCATOR_MAX_LAT).expect("max lat");
        assert!((y_max - WEB_MERCATOR_MAX_EXTENT).abs() < 1.0);
    }

    #[test]
    fn test_tile_conversion() {
        // 测试瓦片坐标转换
        let lon = 116.0;
        let lat = 40.0;
        let zoom = 10;

        let (tile_x, tile_y) = lonlat_to_tile(lon, lat, zoom);

        // 验证瓦片坐标在合理范围
        let max_tile = 2_u32.pow(u32::from(zoom));
        assert!(tile_x < max_tile);
        assert!(tile_y < max_tile);

        // 瓦片左上角应该在原点西北方向
        let (lon2, lat2) = tile_to_lonlat(tile_x, tile_y, zoom);
        assert!(lon2 <= lon);
        assert!(lat2 >= lat);
    }

    #[test]
    fn test_resolution() {
        // 赤道处 zoom=0 的分辨率
        let res = web_mercator_resolution(0.0, 0, 256);
        // 应该接近 156543 米/像素
        assert!((res - 156543.0).abs() < 10.0);

        // 北京纬度 zoom=0
        let res_beijing = web_mercator_resolution(40.0, 0, 256);
        // 应该小于赤道
        assert!(res_beijing < res);
    }

    #[test]
    fn test_tile_bbox() {
        let (x_min, y_min, x_max, y_max) = tile_to_bbox(0, 0, 0);

        // zoom=0 应该覆盖整个世界
        assert!((x_min + WEB_MERCATOR_MAX_EXTENT).abs() < 1000.0);
        assert!((x_max - WEB_MERCATOR_MAX_EXTENT).abs() < 1000.0);
        assert!(y_max > y_min);
    }
}