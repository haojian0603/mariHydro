//! 纯 Rust 实现的坐标投影转换
//!
//! 支持的投影类型：
//! - WGS84 地理坐标 (EPSG:4326)
//! - UTM 投影 (EPSG:326xx/327xx)
//! - Web Mercator (EPSG:3857)
//! - 高斯-克吕格投影 (中国常用)
//!
//! # 算法特点
//!
//! - UTM/高斯-克吕格使用 Karney (2011) 算法，精度达亚毫米级
//! - 支持多种椭球体（WGS84、CGCS2000、GRS80 等）
//! - 零外部依赖，纯 Rust 实现
//!
//! # 示例
//!
//! ```
//! use mh_geo::projection::{ProjectionType, Projection};
//!
//! // 从 EPSG 代码创建投影
//! let proj = Projection::from_epsg(4326, 32650).unwrap();
//!
//! // 正向投影：WGS84 -> UTM
//! let (x, y) = proj.forward(116.0, 40.0).unwrap();
//!
//! // 逆向投影：UTM -> WGS84
//! let (lon, lat) = proj.inverse(x, y).unwrap();
//! ```

mod gauss_kruger;
mod traits;
mod math_utils;
pub mod transverse_mercator;
mod utm;
mod web_mercator;

pub use gauss_kruger::*;
pub use traits::{FastProjection, MapProjection, TransverseMercatorParams};
pub use utm::*;
pub use web_mercator::*;

use crate::ellipsoid::Ellipsoid;
use mh_foundation::error::{MhError, MhResult};

// ============================================================================
// 投影类型定义
// ============================================================================

/// 支持的投影类型
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionType {
    /// WGS84 地理坐标 (经纬度)
    Geographic,
    /// UTM 投影
    Utm {
        /// UTM 带号 (1-60)
        zone: u8,
        /// 是否为北半球
        north: bool,
    },
    /// Web Mercator (EPSG:3857)
    WebMercator,
    /// 高斯-克吕格 3度带
    GaussKruger3 {
        /// 带号
        zone: u8,
    },
    /// 高斯-克吕格 6度带
    GaussKruger6 {
        /// 带号
        zone: u8,
    },
}

impl ProjectionType {
    /// 从 EPSG 代码解析投影类型
    ///
    /// # Errors
    /// 如果 EPSG 代码不支持则返回错误
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
            // CGCS2000 高斯-克吕格 3度带 (EPSG:4534-4554)
            4534..=4554 => Ok(Self::GaussKruger3 {
                zone: (code - 4534 + 25) as u8,
            }),
            // CGCS2000 高斯-克吕格 6度带 (EPSG:4502-4512)
            4502..=4512 => Ok(Self::GaussKruger6 {
                zone: (code - 4502 + 13) as u8,
            }),
            _ => Err(MhError::Config {
                message: format!(
                    "Unsupported EPSG code: {code}. Supported: 4326, 3857, 32601-32660, 32701-32760, 4502-4554"
                ),
            }),
        }
    }

    /// 转换为 EPSG 代码
    #[must_use]
    pub fn to_epsg(&self) -> Option<u32> {
        match self {
            Self::Geographic => Some(4326),
            Self::WebMercator => Some(3857),
            Self::Utm { zone, north } => {
                if *north {
                    Some(32600 + u32::from(*zone))
                } else {
                    Some(32700 + u32::from(*zone))
                }
            }
            Self::GaussKruger3 { zone } => {
                if *zone >= 25 && *zone <= 45 {
                    Some(4534 + u32::from(*zone - 25))
                } else {
                    None
                }
            }
            Self::GaussKruger6 { zone } => {
                if *zone >= 13 && *zone <= 23 {
                    Some(4502 + u32::from(*zone - 13))
                } else {
                    None
                }
            }
        }
    }

    /// 是否为地理坐标系
    #[must_use]
    pub fn is_geographic(&self) -> bool {
        matches!(self, Self::Geographic)
    }

    /// 获取中央子午线
    #[must_use]
    pub fn central_meridian(&self) -> Option<f64> {
        match self {
            Self::Geographic | Self::WebMercator => None,
            Self::Utm { zone, .. } => Some(f64::from(*zone) * 6.0 - 183.0),
            Self::GaussKruger3 { zone } => Some(f64::from(*zone) * 3.0),
            Self::GaussKruger6 { zone } => Some(f64::from(*zone) * 6.0 - 3.0),
        }
    }

    /// 自动从经纬度确定 UTM 区域
    #[must_use]
    pub fn auto_utm(lon: f64, lat: f64) -> Self {
        let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
        let zone = zone.clamp(1, 60);
        Self::Utm {
            zone,
            north: lat >= 0.0,
        }
    }

    /// 自动从经度确定高斯-克吕格 3度带
    #[must_use]
    pub fn auto_gk3(lon: f64) -> Self {
        let zone = (lon / 3.0).round() as u8;
        let zone = zone.clamp(25, 45);
        Self::GaussKruger3 { zone }
    }

    /// 自动从经度确定高斯-克吕格 6度带
    #[must_use]
    pub fn auto_gk6(lon: f64) -> Self {
        let zone = ((lon + 3.0) / 6.0).floor() as u8 + 1;
        let zone = zone.clamp(13, 23);
        Self::GaussKruger6 { zone }
    }

    /// 转换为快速投影枚举（用于性能关键路径）
    #[must_use]
    pub fn to_fast_projection(&self) -> FastProjection {
        match self {
            Self::Geographic => FastProjection::Geographic(Ellipsoid::WGS84),
            Self::Utm { zone, north } => {
                FastProjection::TransverseMercator(TransverseMercatorParams::utm(*zone, *north))
            }
            Self::WebMercator => FastProjection::WebMercator,
            Self::GaussKruger3 { zone } => {
                FastProjection::TransverseMercator(TransverseMercatorParams::gauss_kruger_3(*zone))
            }
            Self::GaussKruger6 { zone } => {
                FastProjection::TransverseMercator(TransverseMercatorParams::gauss_kruger_6(*zone))
            }
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
    source_fast: FastProjection,
    target_fast: FastProjection,
}

impl Projection {
    /// 创建新的投影转换器
    #[must_use]
    pub fn new(source: ProjectionType, target: ProjectionType) -> Self {
        let source_fast = source.to_fast_projection();
        let target_fast = target.to_fast_projection();
        Self {
            source,
            target,
            source_fast,
            target_fast,
        }
    }

    /// 从 EPSG 代码创建
    ///
    /// # Errors
    /// 如果 EPSG 代码无效则返回错误
    pub fn from_epsg(source_epsg: u32, target_epsg: u32) -> MhResult<Self> {
        Ok(Self::new(
            ProjectionType::from_epsg(source_epsg)?,
            ProjectionType::from_epsg(target_epsg)?,
        ))
    }

    /// 正向投影：将源坐标转换为目标坐标
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    #[inline]
    pub fn forward(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        // 先转为地理坐标
        let (lon, lat) = self.source_fast.inverse(x, y)?;
        // 再投影到目标
        self.target_fast.forward(lon, lat)
    }

    /// 逆向投影：将目标坐标转换回源坐标
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    #[inline]
    pub fn inverse(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        let (lon, lat) = self.target_fast.inverse(x, y)?;
        self.source_fast.forward(lon, lat)
    }

    /// 批量正向转换
    ///
    /// # Errors
    /// 如果任意坐标超出有效范围则返回错误
    pub fn forward_batch(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        points.iter().map(|&(x, y)| self.forward(x, y)).collect()
    }

    /// 批量逆向转换
    ///
    /// # Errors
    /// 如果任意坐标超出有效范围则返回错误
    pub fn inverse_batch(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        points.iter().map(|&(x, y)| self.inverse(x, y)).collect()
    }

    /// 是否为恒等变换
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.source == self.target
    }

    /// 获取源投影类型
    #[must_use]
    pub fn source(&self) -> &ProjectionType {
        &self.source
    }

    /// 获取目标投影类型
    #[must_use]
    pub fn target(&self) -> &ProjectionType {
        &self.target
    }
}

// ============================================================================
// 便捷函数
// ============================================================================

/// WGS84 -> UTM 快捷转换
///
/// # Errors
/// 如果坐标超出有效范围则返回错误
pub fn wgs84_to_utm(lon: f64, lat: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    geographic_to_utm(lon, lat, zone, north)
}

/// UTM -> WGS84 快捷转换
///
/// # Errors
/// 如果坐标超出有效范围则返回错误
pub fn utm_to_wgs84(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
    utm_to_geographic(x, y, zone, north)
}

/// WGS84 -> Web Mercator 快捷转换
///
/// # Errors
/// 如果纬度超出有效范围则返回错误
pub fn wgs84_to_web_mercator(lon: f64, lat: f64) -> MhResult<(f64, f64)> {
    geographic_to_web_mercator(lon, lat)
}

/// Web Mercator -> WGS84 快捷转换
///
/// # Errors
/// 返回可能的错误
pub fn web_mercator_to_wgs84(x: f64, y: f64) -> MhResult<(f64, f64)> {
    web_mercator_to_geographic(x, y)
}

/// 自动确定 UTM 区域并转换
///
/// # Errors
/// 如果坐标超出有效范围则返回错误
pub fn wgs84_to_auto_utm(lon: f64, lat: f64) -> MhResult<(f64, f64, u8, bool)> {
    let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
    let zone = zone.clamp(1, 60);
    let north = lat >= 0.0;
    let (x, y) = geographic_to_utm(lon, lat, zone, north)?;
    Ok((x, y, zone, north))
}

/// WGS84 -> 高斯-克吕格 3度带快捷转换
///
/// # Errors
/// 如果坐标超出有效范围则返回错误
pub fn wgs84_to_gk3(lon: f64, lat: f64, zone: u8) -> MhResult<(f64, f64)> {
    geographic_to_gauss_kruger(lon, lat, f64::from(zone) * 3.0)
}

/// 高斯-克吕格 3度带 -> WGS84 快捷转换
///
/// # Errors
/// 如果坐标超出有效范围则返回错误
pub fn gk3_to_wgs84(x: f64, y: f64, zone: u8) -> MhResult<(f64, f64)> {
    gauss_kruger_to_geographic(x, y, f64::from(zone) * 3.0)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_projection_type_to_epsg() {
        assert_eq!(ProjectionType::Geographic.to_epsg(), Some(4326));
        assert_eq!(ProjectionType::WebMercator.to_epsg(), Some(3857));
        assert_eq!(
            ProjectionType::Utm {
                zone: 50,
                north: true
            }
            .to_epsg(),
            Some(32650)
        );
    }

    #[test]
    fn test_projection_transform() {
        let proj = Projection::from_epsg(4326, 32650).expect("create projection");

        let (x, y) = proj.forward(116.0, 40.0).expect("forward");
        assert!(x > 400_000.0 && x < 600_000.0);
        assert!(y > 4_000_000.0 && y < 5_000_000.0);

        let (lon, lat) = proj.inverse(x, y).expect("inverse");
        assert!((lon - 116.0).abs() < 1e-9);
        assert!((lat - 40.0).abs() < 1e-9);
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

        let proj_type = ProjectionType::auto_utm(-122.0, 37.0);
        assert_eq!(
            proj_type,
            ProjectionType::Utm {
                zone: 10,
                north: true
            }
        );
    }

    #[test]
    fn test_auto_gk3() {
        let proj_type = ProjectionType::auto_gk3(117.0);
        assert_eq!(proj_type, ProjectionType::GaussKruger3 { zone: 39 });
    }

    #[test]
    fn test_identity_projection() {
        let proj = Projection::new(ProjectionType::Geographic, ProjectionType::Geographic);
        assert!(proj.is_identity());

        let (x, y) = proj.forward(116.0, 40.0).expect("forward");
        assert!((x - 116.0).abs() < 1e-10);
        assert!((y - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_projection() {
        let fast = ProjectionType::Utm {
            zone: 50,
            north: true,
        }
        .to_fast_projection();

        let (x, y) = fast.forward(116.0, 40.0).expect("forward");
        let (lon, lat) = fast.inverse(x, y).expect("inverse");

        assert!((lon - 116.0).abs() < 1e-9);
        assert!((lat - 40.0).abs() < 1e-9);
    }
}