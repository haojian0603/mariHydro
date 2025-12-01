// src-tauri/src/marihydro/geo/projection.rs
//! 纯 Rust 实现的坐标投影转换
//!
//! 支持的投影类型：
//! - WGS84 地理坐标 (EPSG:4326)
//! - UTM 投影 (EPSG:326xx/327xx)  
//! - Web Mercator (EPSG:3857)
//! - 高斯-克吕格投影 (中国常用)

use crate::marihydro::core::error::{MhError, MhResult};
use std::f64::consts::PI;

// ============================================================================
// WGS84 椭球参数
// ============================================================================

/// WGS84 长半轴 (m)
const WGS84_A: f64 = 6378137.0;

/// WGS84 扁率
const WGS84_F: f64 = 1.0 / 298.257223563;

/// WGS84 短半轴 (m)
const WGS84_B: f64 = WGS84_A * (1.0 - WGS84_F);

/// 第一偏心率的平方
const WGS84_E2: f64 = 2.0 * WGS84_F - WGS84_F * WGS84_F;

/// 第二偏心率的平方
const WGS84_EP2: f64 = WGS84_E2 / (1.0 - WGS84_E2);

/// UTM 比例因子
const UTM_K0: f64 = 0.9996;

/// Web Mercator 最大纬度
const WEB_MERCATOR_MAX_LAT: f64 = 85.051128779;

// ============================================================================
// 投影类型定义
// ============================================================================

/// 支持的投影类型
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionType {
    /// WGS84 地理坐标 (经纬度)
    Geographic,
    /// UTM 投影
    Utm { zone: u8, north: bool },
    /// Web Mercator (EPSG:3857)
    WebMercator,
    /// 高斯-克吕格 3度带
    GaussKruger3 { zone: u8 },
    /// 高斯-克吕格 6度带  
    GaussKruger6 { zone: u8 },
}

impl ProjectionType {
    /// 从 EPSG 代码解析投影类型
    pub fn from_epsg(code: u32) -> MhResult<Self> {
        match code {
            4326 => Ok(Self::Geographic),
            3857 | 900913 => Ok(Self::WebMercator),
            // UTM 北半球 32601-32660
            32601..=32660 => Ok(Self::Utm {
                zone: (code - 32600) as u8,
                north: true,
            }),
            // UTM 南半球 32701-32760
            32701..=32760 => Ok(Self::Utm {
                zone: (code - 32700) as u8,
                north: false,
            }),
            // CGCS2000 高斯-克吕格 3度带
            4534..=4554 => Ok(Self::GaussKruger3 {
                zone: (code - 4534 + 25) as u8,
            }),
            _ => Err(MhError::config(format!(
                "Unsupported EPSG code: {}. Supported: 4326, 3857, 32601-32660, 32701-32760",
                code
            ))),
        }
    }

    /// 转换为 EPSG 代码
    pub fn to_epsg(&self) -> Option<u32> {
        match self {
            Self::Geographic => Some(4326),
            Self::WebMercator => Some(3857),
            Self::Utm { zone, north } => {
                if *north {
                    Some(32600 + *zone as u32)
                } else {
                    Some(32700 + *zone as u32)
                }
            }
            Self::GaussKruger3 { zone } => {
                if *zone >= 25 && *zone <= 45 {
                    Some(4534 + (*zone - 25) as u32)
                } else {
                    None
                }
            }
            Self::GaussKruger6 { .. } => None, // 无标准 EPSG
        }
    }

    /// 是否为地理坐标系
    pub fn is_geographic(&self) -> bool {
        matches!(self, Self::Geographic)
    }

    /// 获取中央子午线
    pub fn central_meridian(&self) -> Option<f64> {
        match self {
            Self::Geographic | Self::WebMercator => None,
            Self::Utm { zone, .. } => Some((*zone as f64) * 6.0 - 183.0),
            Self::GaussKruger3 { zone } => Some((*zone as f64) * 3.0),
            Self::GaussKruger6 { zone } => Some((*zone as f64) * 6.0 - 3.0),
        }
    }

    /// 自动从经纬度确定 UTM 区域
    pub fn auto_utm(lon: f64, lat: f64) -> Self {
        let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
        let zone = zone.clamp(1, 60);
        Self::Utm {
            zone,
            north: lat >= 0.0,
        }
    }
}

// ============================================================================
// 投影转换器
// ============================================================================

/// 坐标投影转换器
#[derive(Debug, Clone)]
pub struct Projection {
    source: ProjectionType,
    target: ProjectionType,
}

impl Projection {
    /// 创建新的投影转换器
    pub fn new(source: ProjectionType, target: ProjectionType) -> Self {
        Self { source, target }
    }

    /// 从 EPSG 代码创建
    pub fn from_epsg(source_epsg: u32, target_epsg: u32) -> MhResult<Self> {
        Ok(Self {
            source: ProjectionType::from_epsg(source_epsg)?,
            target: ProjectionType::from_epsg(target_epsg)?,
        })
    }

    /// 正向投影：将源坐标转换为目标坐标
    pub fn forward(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        // 先转为地理坐标
        let (lon, lat) = self.to_geographic(x, y, &self.source)?;
        // 再投影到目标
        self.from_geographic(lon, lat, &self.target)
    }

    /// 逆向投影：将目标坐标转换回源坐标
    pub fn inverse(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        let (lon, lat) = self.to_geographic(x, y, &self.target)?;
        self.from_geographic(lon, lat, &self.source)
    }

    /// 批量正向转换
    pub fn forward_batch(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        points.iter().map(|&(x, y)| self.forward(x, y)).collect()
    }

    /// 批量逆向转换
    pub fn inverse_batch(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        points.iter().map(|&(x, y)| self.inverse(x, y)).collect()
    }

    /// 是否为恒等变换
    pub fn is_identity(&self) -> bool {
        self.source == self.target
    }

    /// 获取源投影类型
    pub fn source(&self) -> &ProjectionType {
        &self.source
    }

    /// 获取目标投影类型
    pub fn target(&self) -> &ProjectionType {
        &self.target
    }

    // ========================================================================
    // 私有方法：投影 -> 地理坐标
    // ========================================================================

    fn to_geographic(&self, x: f64, y: f64, proj: &ProjectionType) -> MhResult<(f64, f64)> {
        match proj {
            ProjectionType::Geographic => Ok((x, y)),
            ProjectionType::Utm { zone, north } => utm_to_geographic(x, y, *zone, *north),
            ProjectionType::WebMercator => web_mercator_to_geographic(x, y),
            ProjectionType::GaussKruger3 { zone } => {
                gauss_kruger_to_geographic(x, y, (*zone as f64) * 3.0)
            }
            ProjectionType::GaussKruger6 { zone } => {
                gauss_kruger_to_geographic(x, y, (*zone as f64) * 6.0 - 3.0)
            }
        }
    }

    fn from_geographic(&self, lon: f64, lat: f64, proj: &ProjectionType) -> MhResult<(f64, f64)> {
        match proj {
            ProjectionType::Geographic => Ok((lon, lat)),
            ProjectionType::Utm { zone, north } => geographic_to_utm(lon, lat, *zone, *north),
            ProjectionType::WebMercator => geographic_to_web_mercator(lon, lat),
            ProjectionType::GaussKruger3 { zone } => {
                geographic_to_gauss_kruger(lon, lat, (*zone as f64) * 3.0)
            }
            ProjectionType::GaussKruger6 { zone } => {
                geographic_to_gauss_kruger(lon, lat, (*zone as f64) * 6.0 - 3.0)
            }
        }
    }
}

// ============================================================================
// UTM 投影实现
// ============================================================================

/// 地理坐标 -> UTM
fn geographic_to_utm(lon: f64, lat: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    if lat < -80.0 || lat > 84.0 {
        return Err(MhError::config(format!(
            "Latitude {} out of UTM range (-80, 84)",
            lat
        )));
    }

    let lon_rad = lon.to_radians();
    let lat_rad = lat.to_radians();

    let central_lon = (zone as f64) * 6.0 - 183.0;
    let central_lon_rad = central_lon.to_radians();

    let n = WGS84_A / (1.0 - WGS84_E2 * lat_rad.sin().powi(2)).sqrt();
    let t = lat_rad.tan().powi(2);
    let c = WGS84_EP2 * lat_rad.cos().powi(2);
    let a_coef = (lon_rad - central_lon_rad) * lat_rad.cos();

    let m = meridian_arc(lat_rad);

    let x = UTM_K0 * n * (a_coef
        + (1.0 - t + c) * a_coef.powi(3) / 6.0
        + (5.0 - 18.0 * t + t.powi(2) + 72.0 * c - 58.0 * WGS84_EP2)
            * a_coef.powi(5)
            / 120.0);

    let y = UTM_K0
        * (m
            + n * lat_rad.tan()
                * (a_coef.powi(2) / 2.0
                    + (5.0 - t + 9.0 * c + 4.0 * c.powi(2)) * a_coef.powi(4) / 24.0
                    + (61.0 - 58.0 * t + t.powi(2) + 600.0 * c - 330.0 * WGS84_EP2)
                        * a_coef.powi(6)
                        / 720.0));

    // 添加假东和假北
    let easting = x + 500000.0;
    let northing = if north { y } else { y + 10000000.0 };

    Ok((easting, northing))
}

/// UTM -> 地理坐标
fn utm_to_geographic(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    let x = x - 500000.0;
    let y = if north { y } else { y - 10000000.0 };

    let central_lon = (zone as f64) * 6.0 - 183.0;

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
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * WGS84_EP2)
                    * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * WGS84_EP2
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);

    let lon = central_lon.to_radians()
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2) + 8.0 * WGS84_EP2 + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / phi1.cos();

    Ok((lon.to_degrees(), lat.to_degrees()))
}

// ============================================================================
// Web Mercator 投影实现
// ============================================================================

/// 地理坐标 -> Web Mercator
fn geographic_to_web_mercator(lon: f64, lat: f64) -> MhResult<(f64, f64)> {
    let lat = lat.clamp(-WEB_MERCATOR_MAX_LAT, WEB_MERCATOR_MAX_LAT);

    let x = WGS84_A * lon.to_radians();
    let lat_rad = lat.to_radians();
    let y = WGS84_A * ((PI / 4.0 + lat_rad / 2.0).tan()).ln();

    Ok((x, y))
}

/// Web Mercator -> 地理坐标
fn web_mercator_to_geographic(x: f64, y: f64) -> MhResult<(f64, f64)> {
    let lon = (x / WGS84_A).to_degrees();
    let lat = (2.0 * (y / WGS84_A).exp().atan() - PI / 2.0).to_degrees();

    Ok((lon, lat))
}

// ============================================================================
// 高斯-克吕格投影实现 (适用于中国区域)
// ============================================================================

/// 地理坐标 -> 高斯-克吕格
fn geographic_to_gauss_kruger(lon: f64, lat: f64, central_lon: f64) -> MhResult<(f64, f64)> {
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
    Ok((y + 500000.0, x))
}

/// 高斯-克吕格 -> 地理坐标
fn gauss_kruger_to_geographic(x: f64, y: f64, central_lon: f64) -> MhResult<(f64, f64)> {
    let x = x - 500000.0;

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
                - (5.0 + 3.0 * t1 + 10.0 * c1 - 4.0 * c1.powi(2) - 9.0 * WGS84_EP2)
                    * d.powi(4)
                    / 24.0
                + (61.0 + 90.0 * t1 + 298.0 * c1 + 45.0 * t1.powi(2)
                    - 252.0 * WGS84_EP2
                    - 3.0 * c1.powi(2))
                    * d.powi(6)
                    / 720.0);

    let lon = central_lon.to_radians()
        + (d - (1.0 + 2.0 * t1 + c1) * d.powi(3) / 6.0
            + (5.0 - 2.0 * c1 + 28.0 * t1 - 3.0 * c1.powi(2) + 8.0 * WGS84_EP2 + 24.0 * t1.powi(2))
                * d.powi(5)
                / 120.0)
            / cos_phi1;

    Ok((lon.to_degrees(), lat.to_degrees()))
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 计算子午线弧长
fn meridian_arc(lat: f64) -> f64 {
    let n = (WGS84_A - WGS84_B) / (WGS84_A + WGS84_B);

    let a0 = WGS84_A * (1.0 - n + (5.0 / 4.0) * (n.powi(2) - n.powi(3))
        + (81.0 / 64.0) * (n.powi(4) - n.powi(5)));

    let a2 = (3.0 / 2.0) * WGS84_A * (n - n.powi(2) + (7.0 / 8.0) * (n.powi(3) - n.powi(4))
        + (55.0 / 64.0) * n.powi(5));

    let a4 = (15.0 / 16.0) * WGS84_A * (n.powi(2) - n.powi(3) + (3.0 / 4.0) * (n.powi(4) - n.powi(5)));

    let a6 = (35.0 / 48.0) * WGS84_A * (n.powi(3) - n.powi(4) + (11.0 / 16.0) * n.powi(5));

    let a8 = (315.0 / 512.0) * WGS84_A * (n.powi(4) - n.powi(5));

    a0 * lat - a2 * (2.0 * lat).sin() + a4 * (4.0 * lat).sin() - a6 * (6.0 * lat).sin()
        + a8 * (8.0 * lat).sin()
}

/// 子午线弧长系数
fn meridian_arc_factor() -> f64 {
    let n = (WGS84_A - WGS84_B) / (WGS84_A + WGS84_B);

    WGS84_A * (1.0 - n + (5.0 / 4.0) * (n.powi(2) - n.powi(3))
        + (81.0 / 64.0) * (n.powi(4) - n.powi(5)))
}

// ============================================================================
// 便捷函数
// ============================================================================

/// WGS84 -> UTM 快捷转换
pub fn wgs84_to_utm(lon: f64, lat: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    geographic_to_utm(lon, lat, zone, north)
}

/// UTM -> WGS84 快捷转换
pub fn utm_to_wgs84(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    utm_to_geographic(x, y, zone, north)
}

/// WGS84 -> Web Mercator 快捷转换
pub fn wgs84_to_web_mercator(lon: f64, lat: f64) -> MhResult<(f64, f64)> {
    geographic_to_web_mercator(lon, lat)
}

/// Web Mercator -> WGS84 快捷转换
pub fn web_mercator_to_wgs84(x: f64, y: f64) -> MhResult<(f64, f64)> {
    web_mercator_to_geographic(x, y)
}

/// 自动确定 UTM 区域并转换
pub fn wgs84_to_auto_utm(lon: f64, lat: f64) -> MhResult<(f64, f64, u8, bool)> {
    let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
    let zone = zone.clamp(1, 60);
    let north = lat >= 0.0;
    let (x, y) = geographic_to_utm(lon, lat, zone, north)?;
    Ok((x, y, zone, north))
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const TOLERANCE: f64 = 0.01; // 1 厘米精度

    #[test]
    fn test_utm_roundtrip() {
        // 北京 116°E, 40°N
        let lon = 116.0;
        let lat = 40.0;
        let zone = 50;
        let north = true;

        let (x, y) = geographic_to_utm(lon, lat, zone, north).expect("to UTM failed");
        println!("UTM: x={}, y={}", x, y);

        // 验证 UTM 坐标在合理范围
        assert!(x > 400000.0 && x < 600000.0, "x out of range: {}", x);
        assert!(y > 4000000.0 && y < 5000000.0, "y out of range: {}", y);

        // 反向转换
        let (lon2, lat2) = utm_to_geographic(x, y, zone, north).expect("from UTM failed");
        println!("Back: lon={}, lat={}", lon2, lat2);

        assert!((lon - lon2).abs() < 1e-6, "lon mismatch: {} vs {}", lon, lon2);
        assert!((lat - lat2).abs() < 1e-6, "lat mismatch: {} vs {}", lat, lat2);
    }

    #[test]
    fn test_web_mercator_roundtrip() {
        let lon = 116.0;
        let lat = 40.0;

        let (x, y) = geographic_to_web_mercator(lon, lat).expect("to WebMercator failed");
        println!("WebMercator: x={}, y={}", x, y);

        let (lon2, lat2) = web_mercator_to_geographic(x, y).expect("from WebMercator failed");
        println!("Back: lon={}, lat={}", lon2, lat2);

        assert!((lon - lon2).abs() < 1e-6);
        assert!((lat - lat2).abs() < 1e-6);
    }

    #[test]
    fn test_gauss_kruger_roundtrip() {
        let lon = 117.0;
        let lat = 39.0;
        let central_lon = 117.0;

        let (x, y) = geographic_to_gauss_kruger(lon, lat, central_lon).expect("to GK failed");
        println!("GK: x={}, y={}", x, y);

        let (lon2, lat2) =
            gauss_kruger_to_geographic(x, y, central_lon).expect("from GK failed");
        println!("Back: lon={}, lat={}", lon2, lat2);

        assert!((lon - lon2).abs() < 1e-6);
        assert!((lat - lat2).abs() < 1e-6);
    }

    #[test]
    fn test_projection_type_from_epsg() {
        assert_eq!(
            ProjectionType::from_epsg(4326).expect("4326"),
            ProjectionType::Geographic
        );
        assert_eq!(
            ProjectionType::from_epsg(3857).expect("3857"),
            ProjectionType::WebMercator
        );
        assert_eq!(
            ProjectionType::from_epsg(32650).expect("32650"),
            ProjectionType::Utm {
                zone: 50,
                north: true
            }
        );
        assert_eq!(
            ProjectionType::from_epsg(32750).expect("32750"),
            ProjectionType::Utm {
                zone: 50,
                north: false
            }
        );
    }

    #[test]
    fn test_projection_transform() {
        let proj = Projection::from_epsg(4326, 32650).expect("create projection");

        let (x, y) = proj.forward(116.0, 40.0).expect("forward");
        assert!(x > 400000.0 && x < 600000.0);
        assert!(y > 4000000.0 && y < 5000000.0);

        let (lon, lat) = proj.inverse(x, y).expect("inverse");
        assert!((lon - 116.0).abs() < 1e-6);
        assert!((lat - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_auto_utm() {
        let proj_type = ProjectionType::auto_utm(116.0, 40.0);
        assert_eq!(
            proj_type,
            ProjectionType::Utm {
                zone: 50,
                north: true
            }
        );

        let proj_type = ProjectionType::auto_utm(-122.0, 37.0); // 旧金山
        assert_eq!(
            proj_type,
            ProjectionType::Utm {
                zone: 10,
                north: true
            }
        );
    }
}
