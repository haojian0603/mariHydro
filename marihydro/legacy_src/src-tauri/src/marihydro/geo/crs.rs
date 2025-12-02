// src-tauri/src/marihydro/geo/crs.rs
//! 坐标参考系统 (CRS) 定义和解析
//! 支持 EPSG 代码、PROJ4 字符串和 WKT 格式

use crate::marihydro::core::error::{MhError, MhResult};
use serde::{Deserialize, Serialize};

/// CRS 策略配置（用于配置文件）
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", content = "value")]
pub enum CrsStrategy {
    /// 手动指定 CRS 定义
    Manual(String),
    /// 从第一个文件自动检测
    FromFirstFile,
    /// 强制使用 WGS84
    ForceWGS84,
}

impl Default for CrsStrategy {
    fn default() -> Self {
        Self::FromFirstFile
    }
}

/// CRS 定义类型
#[derive(Debug, Clone, PartialEq)]
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
    pub fn to_proj_string(&self) -> String {
        match self {
            CrsDefinition::Epsg(code) => format!("EPSG:{}", code),
            CrsDefinition::Proj4(s) => s.clone(),
            CrsDefinition::Wkt(s) => s.clone(),
        }
    }

    /// WGS84 地理坐标系
    pub fn wgs84() -> Self {
        CrsDefinition::Epsg(4326)
    }

    /// UTM 区域投影
    /// - zone: 1-60
    /// - north: true = 北半球, false = 南半球
    pub fn utm_zone(zone: u8, north: bool) -> Self {
        let code = if north {
            32600 + zone as u32
        } else {
            32700 + zone as u32
        };
        CrsDefinition::Epsg(code)
    }

    /// Web Mercator (Google Maps)
    pub fn web_mercator() -> Self {
        CrsDefinition::Epsg(3857)
    }

    /// 从坐标自动计算 UTM 区域
    pub fn auto_utm(lon: f64, lat: f64) -> Self {
        let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
        let north = lat >= 0.0;
        Self::utm_zone(zone, north)
    }

    /// 获取 EPSG 代码（如果有）
    pub fn epsg_code(&self) -> Option<u32> {
        match self {
            CrsDefinition::Epsg(code) => Some(*code),
            CrsDefinition::Proj4(s) | CrsDefinition::Wkt(s) => Self::parse_epsg(s),
        }
    }

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
        None
    }
}

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
}

impl ResolvedCrs {
    /// 从定义字符串创建
    pub fn new(definition: &str) -> MhResult<Self> {
        let epsg = CrsDefinition::parse_epsg(definition);
        let is_geographic = Self::detect_geographic(definition, epsg);
        let unit_name = if is_geographic {
            "degree".to_string()
        } else {
            "metre".to_string()
        };

        Ok(Self {
            definition: definition.into(),
            epsg,
            is_geographic,
            unit_name,
        })
    }

    /// WGS84
    pub fn wgs84() -> Self {
        Self {
            definition: "EPSG:4326".into(),
            epsg: Some(4326),
            is_geographic: true,
            unit_name: "degree".into(),
        }
    }

    fn detect_geographic(def: &str, epsg: Option<u32>) -> bool {
        // 常见地理 CRS EPSG 代码
        if let Some(code) = epsg {
            if code == 4326 || code == 4269 || code == 4267 {
                return true;
            }
        }
        // 检查定义字符串
        let lower = def.to_lowercase();
        lower.contains("geogcs") 
            || lower.contains("longlat") 
            || lower.contains("+proj=longlat")
    }

    /// 是否为投影坐标系（米）
    pub fn is_projected(&self) -> bool {
        !self.is_geographic
    }
}

/// 坐标参考系统
#[derive(Debug, Clone)]
pub struct Crs {
    /// CRS 定义
    pub definition: String,
    /// 解析后的信息
    resolved: ResolvedCrs,
}

impl Crs {
    /// 从定义字符串创建
    pub fn new(def: &str) -> MhResult<Self> {
        let resolved = ResolvedCrs::new(def)?;
        Ok(Self {
            definition: def.into(),
            resolved,
        })
    }

    /// 从 EPSG 代码创建
    pub fn from_epsg(code: u32) -> MhResult<Self> {
        let def = format!("EPSG:{}", code);
        Self::new(&def)
    }

    /// WGS84 地理坐标系
    pub fn wgs84() -> Self {
        Self {
            definition: "EPSG:4326".into(),
            resolved: ResolvedCrs::wgs84(),
        }
    }

    /// 是否为地理坐标系
    pub fn is_geographic(&self) -> bool {
        self.resolved.is_geographic
    }

    /// 是否为投影坐标系
    pub fn is_projected(&self) -> bool {
        self.resolved.is_projected()
    }

    /// 获取 EPSG 代码
    pub fn epsg(&self) -> Option<u32> {
        self.resolved.epsg
    }

    /// 获取单位名称
    pub fn unit_name(&self) -> &str {
        &self.resolved.unit_name
    }
}

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
    }

    #[test]
    fn test_crs_geographic() {
        let wgs84 = Crs::wgs84();
        assert!(wgs84.is_geographic());
        assert!(!wgs84.is_projected());
    }
}

