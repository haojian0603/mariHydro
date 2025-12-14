// marihydro\crates\mh_geo\src\crs.rs
//! 坐标参考系统 (CRS) 定义和解析
//!
//! 支持 EPSG 代码、PROJ4 字符串和 WKT 格式
//!
//! # 示例
//!
//! ```
//! use mh_geo::crs::{Crs, CrsDefinition};
//!
//! // 创建 WGS84 CRS
//! let wgs84 = Crs::wgs84();
//! assert!(wgs84.is_geographic());
//!
//! // 从 EPSG 代码创建 UTM
//! let utm = Crs::from_epsg(32650).unwrap();
//! assert!(utm.is_projected());
//! ```

use crate::ellipsoid::Ellipsoid;
use crate::projection::{FastProjection, ProjectionType};
use mh_foundation::error::MhResult;
use serde::{Deserialize, Serialize};

// ============================================================================
// CRS 策略配置
// ============================================================================

/// CRS 策略配置（用于配置文件）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", content = "value")]
#[derive(Default)]
pub enum CrsStrategy {
    /// 手动指定 CRS 定义
    Manual(String),
    /// 从第一个文件自动检测
    #[default]
    FromFirstFile,
    /// 强制使用 WGS84
    ForceWGS84,
}


// ============================================================================
// CRS 定义类型
// ============================================================================

/// CRS 定义类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrsDefinition {
    /// EPSG 代码（如 4326 = WGS84）
    Epsg(u32),
    /// PROJ4 字符串
    Proj4(String),
    /// WKT 格式
    Wkt(String),
}

impl CrsDefinition {
    /// 转换为 PROJ 兼容的字符串
    #[must_use]
    pub fn to_proj_string(&self) -> String {
        match self {
            CrsDefinition::Epsg(code) => format!("EPSG:{code}"),
            CrsDefinition::Proj4(s) | CrsDefinition::Wkt(s) => s.clone(),
        }
    }

    /// WGS84 地理坐标系
    #[must_use]
    pub fn wgs84() -> Self {
        CrsDefinition::Epsg(4326)
    }

    /// UTM 区域投影
    ///
    /// # Arguments
    /// - `zone`: 1-60
    /// - `north`: true = 北半球, false = 南半球
    #[must_use]
    pub fn utm_zone(zone: u8, north: bool) -> Self {
        let code = if north {
            32600 + u32::from(zone)
        } else {
            32700 + u32::from(zone)
        };
        CrsDefinition::Epsg(code)
    }

    /// Web Mercator (EPSG:3857)
    #[must_use]
    pub fn web_mercator() -> Self {
        CrsDefinition::Epsg(3857)
    }

    /// 从坐标自动计算 UTM 区域
    ///
    /// # Arguments
    /// - `lon`: 经度 (度)
    /// - `lat`: 纬度 (度)
    #[must_use]
    pub fn auto_utm(lon: f64, lat: f64) -> Self {
        let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
        let zone = zone.clamp(1, 60);
        let north = lat >= 0.0;
        Self::utm_zone(zone, north)
    }

    /// 高斯-克吕格 3度带
    #[must_use]
    pub fn gauss_kruger_3(zone: u8) -> Self {
        if (25..=45).contains(&zone) {
            CrsDefinition::Epsg(4534 + u32::from(zone - 25))
        } else {
            // 返回自定义 PROJ4 字符串
            let cm = f64::from(zone) * 3.0;
            CrsDefinition::Proj4(format!(
                "+proj=tmerc +lat_0=0 +lon_0={cm} +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            ))
        }
    }

    /// 高斯-克吕格 6度带
    #[must_use]
    pub fn gauss_kruger_6(zone: u8) -> Self {
        if (13..=23).contains(&zone) {
            CrsDefinition::Epsg(4502 + u32::from(zone - 13))
        } else {
            let cm = f64::from(zone) * 6.0 - 3.0;
            CrsDefinition::Proj4(format!(
                "+proj=tmerc +lat_0=0 +lon_0={cm} +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
            ))
        }
    }

    /// 获取 EPSG 代码（如果有）
    #[must_use]
    pub fn epsg_code(&self) -> Option<u32> {
        match self {
            CrsDefinition::Epsg(code) => Some(*code),
            CrsDefinition::Proj4(s) | CrsDefinition::Wkt(s) => Self::parse_epsg(s),
        }
    }

    /// 从字符串解析 EPSG 代码
    fn parse_epsg(s: &str) -> Option<u32> {
        // 尝试从 "EPSG:xxxx" 格式解析
        if let Some(suffix) = s.strip_prefix("EPSG:") {
            return suffix.trim().parse().ok();
        }
        // 尝试从 WKT 的 AUTHORITY["EPSG","xxxx"] 解析
        if let Some(pos) = s.find("AUTHORITY[\"EPSG\",\"") {
            let start = pos + 18;
            if let Some(end) = s[start..].find("\"]") {
                return s[start..start + end].parse().ok();
            }
        }
        // 尝试从 ID["EPSG",xxxx] 解析（WKT2 格式）
        if let Some(pos) = s.find("ID[\"EPSG\",") {
            let start = pos + 10;
            if let Some(end) = s[start..].find(']') {
                return s[start..start + end].trim().parse().ok();
            }
        }
        None
    }
}

// ============================================================================
// 解析后的 CRS 信息
// ============================================================================

/// 解析后的 CRS 信息
#[derive(Debug, Clone)]
pub struct ResolvedCrs {
    /// 原始定义字符串
    pub definition: String,
    /// EPSG 代码（如果可用）
    pub epsg: Option<u32>,
    /// 是否为地理坐标系（度）
    pub is_geographic: bool,
    /// 单位名称
    pub unit_name: String,
    /// 椭球体（如果可解析）
    pub ellipsoid: Ellipsoid,
}

impl ResolvedCrs {
    /// 从定义字符串创建
    ///
    /// # Errors
    /// 如果 CRS 定义无效则返回错误
    pub fn new(definition: &str) -> MhResult<Self> {
        let epsg = CrsDefinition::parse_epsg(definition);
        let is_geographic = Self::detect_geographic(definition, epsg);
        let unit_name = if is_geographic {
            "degree".to_string()
        } else {
            "metre".to_string()
        };
        let ellipsoid = Self::detect_ellipsoid(definition, epsg);

        Ok(Self {
            definition: definition.into(),
            epsg,
            is_geographic,
            unit_name,
            ellipsoid,
        })
    }

    /// 创建 WGS84 CRS
    #[must_use]
    pub fn wgs84() -> Self {
        Self {
            definition: "EPSG:4326".into(),
            epsg: Some(4326),
            is_geographic: true,
            unit_name: "degree".into(),
            ellipsoid: Ellipsoid::WGS84,
        }
    }

    /// 检测是否为地理坐标系
    fn detect_geographic(def: &str, epsg: Option<u32>) -> bool {
        // 常见地理 CRS EPSG 代码
        if let Some(code) = epsg {
            if code == 4326 || code == 4269 || code == 4267 || code == 4490 {
                return true;
            }
        }
        // 检查定义字符串
        let lower = def.to_lowercase();
        lower.contains("geogcs")
            || lower.contains("longlat")
            || lower.contains("+proj=longlat")
            || lower.contains("geographic")
    }

    /// 检测椭球体
    fn detect_ellipsoid(def: &str, epsg: Option<u32>) -> Ellipsoid {
        // 根据 EPSG 代码判断
        if let Some(code) = epsg {
            match code {
                // WGS84 相关
                4326 | 32601..=32660 | 32701..=32760 | 3857 => return Ellipsoid::WGS84,
                // CGCS2000 相关
                4490 | 4502..=4554 => return Ellipsoid::CGCS2000,
                // 北京54
                4214 => return Ellipsoid::KRASSOVSKY,
                _ => {}
            }
        }

        // 从字符串检测
        let lower = def.to_lowercase();
        if lower.contains("wgs84") || lower.contains("wgs 84") {
            Ellipsoid::WGS84
        } else if lower.contains("cgcs2000") || lower.contains("grs80") || lower.contains("grs 80")
        {
            Ellipsoid::CGCS2000
        } else if lower.contains("krassovsky") || lower.contains("krasovsky") {
            Ellipsoid::KRASSOVSKY
        } else {
            // 默认 WGS84
            Ellipsoid::WGS84
        }
    }

    /// 是否为投影坐标系（米）
    #[must_use]
    pub fn is_projected(&self) -> bool {
        !self.is_geographic
    }
}

// ============================================================================
// 坐标参考系统
// ============================================================================

/// 坐标参考系统
#[derive(Debug, Clone)]
pub struct Crs {
    /// CRS 定义字符串
    pub definition: String,
    /// 解析后的信息
    resolved: ResolvedCrs,
    /// 投影类型（用于快速转换）
    projection_type: Option<ProjectionType>,
}

impl Crs {
    /// 从定义字符串创建
    ///
    /// # Errors
    /// 如果 CRS 定义无法解析则返回错误
    pub fn new(def: &str) -> MhResult<Self> {
        let resolved = ResolvedCrs::new(def)?;
        let projection_type = resolved.epsg.and_then(|code| ProjectionType::from_epsg(code).ok());

        Ok(Self {
            definition: def.into(),
            resolved,
            projection_type,
        })
    }

    /// 从 EPSG 代码创建
    ///
    /// # Errors
    /// 如果 EPSG 代码无效则返回错误
    pub fn from_epsg(code: u32) -> MhResult<Self> {
        let def = format!("EPSG:{code}");
        Self::new(&def)
    }

    /// WGS84 地理坐标系
    #[must_use]
    pub fn wgs84() -> Self {
        Self {
            definition: "EPSG:4326".into(),
            resolved: ResolvedCrs::wgs84(),
            projection_type: Some(ProjectionType::Geographic),
        }
    }

    /// CGCS2000 地理坐标系
    #[must_use]
    pub fn cgcs2000() -> Self {
        Self {
            definition: "EPSG:4490".into(),
            resolved: ResolvedCrs {
                definition: "EPSG:4490".into(),
                epsg: Some(4490),
                is_geographic: true,
                unit_name: "degree".into(),
                ellipsoid: Ellipsoid::CGCS2000,
            },
            projection_type: Some(ProjectionType::Geographic),
        }
    }

    /// 创建 UTM 投影 CRS
    #[must_use]
    pub fn utm(zone: u8, north: bool) -> Self {
        let code = if north {
            32600 + u32::from(zone)
        } else {
            32700 + u32::from(zone)
        };
        Self::from_epsg(code).unwrap_or_else(|_| Self::wgs84())
    }

    /// 创建 Web Mercator CRS
    #[must_use]
    pub fn web_mercator() -> Self {
        Self::from_epsg(3857).unwrap_or_else(|_| Self::wgs84())
    }

    /// 是否为地理坐标系
    #[must_use]
    pub fn is_geographic(&self) -> bool {
        self.resolved.is_geographic
    }

    /// 是否为投影坐标系
    #[must_use]
    pub fn is_projected(&self) -> bool {
        self.resolved.is_projected()
    }

    /// 获取 EPSG 代码
    #[must_use]
    pub fn epsg(&self) -> Option<u32> {
        self.resolved.epsg
    }

    /// 获取单位名称
    #[must_use]
    pub fn unit_name(&self) -> &str {
        &self.resolved.unit_name
    }

    /// 获取椭球体
    #[must_use]
    pub fn ellipsoid(&self) -> &Ellipsoid {
        &self.resolved.ellipsoid
    }

    /// 获取投影类型
    #[must_use]
    pub fn projection_type(&self) -> Option<&ProjectionType> {
        self.projection_type.as_ref()
    }

    /// 转换为快速投影（用于性能关键路径）
    #[must_use]
    pub fn to_fast_projection(&self) -> FastProjection {
        match &self.projection_type {
            Some(pt) => pt.to_fast_projection(),
            None => FastProjection::Geographic(self.resolved.ellipsoid),
        }
    }

    /// 获取中央子午线（如果是投影坐标系）
    #[must_use]
    pub fn central_meridian(&self) -> Option<f64> {
        self.projection_type.as_ref().and_then(super::projection::ProjectionType::central_meridian)
    }
}

// ============================================================================
// 便捷构造函数
// ============================================================================

/// 从 EPSG 代码创建 CRS（便捷函数）
///
/// # Errors
/// 如果 EPSG 代码无效则返回错误
pub fn crs_from_epsg(code: u32) -> MhResult<Crs> {
    Crs::from_epsg(code)
}

/// 根据经纬度自动选择合适的投影 CRS
#[must_use]
pub fn auto_projected_crs(lon: f64, lat: f64) -> Crs {
    // 中国区域使用 CGCS2000 高斯-克吕格
    if (73.0..=135.0).contains(&lon) && (3.0..=54.0).contains(&lat) {
        let zone = (lon / 3.0).round() as u8;
        let code = 4534 + u32::from(zone.saturating_sub(25));
        Crs::from_epsg(code).unwrap_or_else(|_| {
            // 回退到 UTM
            Crs::utm(((lon + 180.0) / 6.0).floor() as u8 + 1, lat >= 0.0)
        })
    } else {
        // 其他区域使用 UTM
        let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
        Crs::utm(zone.clamp(1, 60), lat >= 0.0)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epsg_parsing() {
        let def = CrsDefinition::Epsg(4326);
        assert_eq!(def.to_proj_string(), "EPSG:4326");
        assert_eq!(def.epsg_code(), Some(4326));
    }

    #[test]
    fn test_utm_zone() {
        let utm = CrsDefinition::utm_zone(50, true);
        assert_eq!(utm.epsg_code(), Some(32650));

        let utm_south = CrsDefinition::utm_zone(50, false);
        assert_eq!(utm_south.epsg_code(), Some(32750));
    }

    #[test]
    fn test_auto_utm() {
        // 北京大约在 116°E, 40°N -> UTM 50N
        let utm = CrsDefinition::auto_utm(116.0, 40.0);
        assert_eq!(utm.epsg_code(), Some(32650));

        // 南半球
        let utm_south = CrsDefinition::auto_utm(116.0, -35.0);
        assert_eq!(utm_south.epsg_code(), Some(32750));
    }

    #[test]
    fn test_crs_geographic() {
        let wgs84 = Crs::wgs84();
        assert!(wgs84.is_geographic());
        assert!(!wgs84.is_projected());
        assert_eq!(wgs84.epsg(), Some(4326));
        assert_eq!(wgs84.ellipsoid().a, Ellipsoid::WGS84.a);
    }

    #[test]
    fn test_crs_from_epsg() {
        let utm = Crs::from_epsg(32650).expect("from_epsg");
        assert!(utm.is_projected());
        assert!(!utm.is_geographic());
        assert_eq!(utm.unit_name(), "metre");
    }

    #[test]
    fn test_crs_strategy_default() {
        let strategy = CrsStrategy::default();
        assert_eq!(strategy, CrsStrategy::FromFirstFile);
    }

    #[test]
    fn test_web_mercator() {
        let wm = CrsDefinition::web_mercator();
        assert_eq!(wm.epsg_code(), Some(3857));
    }

    #[test]
    fn test_parse_wkt_epsg() {
        let wkt = r#"GEOGCS["WGS 84",AUTHORITY["EPSG","4326"]]"#;
        let epsg = CrsDefinition::parse_epsg(wkt);
        assert_eq!(epsg, Some(4326));
    }

    #[test]
    fn test_gauss_kruger() {
        let gk3 = CrsDefinition::gauss_kruger_3(39);
        assert_eq!(gk3.epsg_code(), Some(4548)); // 4534 + 14

        let gk6 = CrsDefinition::gauss_kruger_6(20);
        assert_eq!(gk6.epsg_code(), Some(4509)); // 4502 + 7
    }

    #[test]
    fn test_ellipsoid_detection() {
        let cgcs = Crs::from_epsg(4490).expect("CGCS2000");
        assert_eq!(cgcs.ellipsoid().a, Ellipsoid::CGCS2000.a);

        let utm = Crs::from_epsg(32650).expect("UTM");
        assert_eq!(utm.ellipsoid().a, Ellipsoid::WGS84.a);
    }

    #[test]
    fn test_auto_projected_crs() {
        // 北京 - 应该选择 CGCS2000 高斯-克吕格
        let crs = auto_projected_crs(116.4, 39.9);
        assert!(crs.is_projected());

        // 纽约 - 应该选择 UTM
        let crs_ny = auto_projected_crs(-74.0, 40.7);
        assert!(crs_ny.is_projected());
    }
}