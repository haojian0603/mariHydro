// crates/mh_io/src/import/geojson.rs
// IO_SOURCE: RFC 7946 GeoJSON geometry model; Polygon and MultiPolygon coordinates must follow linear-ring structure.
// IO_SCOPE: Supports Point, MultiPoint, LineString, MultiLineString, Polygon, and MultiPolygon with explicit structural validation. Invalid or incomplete ring structure is rejected instead of collapsing to empty geometry.

//! GeoJSON 导入模块
//!
//! 支持读取 GeoJSON 格式的地理数据，包括：
//! - 边界条件定位（Point, LineString）
//! - 初始水深场（Polygon 属性）
//! - 摩擦系数分区（Polygon 属性）
//! - 结构物位置（Point, LineString）
//!
//! # GeoJSON 格式
//!
//! ```json
//! {
//!   "type": "FeatureCollection",
//!   "features": [
//!     {
//!       "type": "Feature",
//!       "geometry": {
//!         "type": "Point",
//!         "coordinates": [100.0, 0.5]
//!       },
//!       "properties": {
//!         "name": "inlet_1",
//!         "bc_type": "discharge",
//!         "value": 100.0
//!       }
//!     }
//!   ]
//! }
//! ```
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_io::import::geojson::{GeoJsonReader, GeometryData};
//!
//! let reader = GeoJsonReader::from_file("boundaries.geojson")?;
//! for feature in reader.features() {
//!     match &feature.geometry {
//!         GeometryData::Point { x, y } => println!("Point: ({}, {})", x, y),
//!         GeometryData::LineString { coords } => println!("Line: {} points", coords.len()),
//!         _ => {}
//!     }
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// GeoJSON 读取错误
#[derive(Debug)]
pub enum GeoJsonError {
    /// IO 错误
    Io(std::io::Error),
    /// JSON 解析错误
    Parse(serde_json::Error),
    /// 无效的几何类型
    InvalidGeometry(String),
    /// 缺少必要属性
    MissingProperty(String),
    /// 坐标格式错误
    InvalidCoordinates,
    /// 几何结构不完整或不符合 GeoJSON 约束
    InvalidStructure(String),
}

impl std::fmt::Display for GeoJsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoJsonError::Io(e) => write!(f, "IO error: {}", e),
            GeoJsonError::Parse(e) => write!(f, "Parse error: {}", e),
            GeoJsonError::InvalidGeometry(t) => write!(f, "Invalid geometry type: {}", t),
            GeoJsonError::MissingProperty(p) => write!(f, "Missing property: {}", p),
            GeoJsonError::InvalidCoordinates => write!(f, "Invalid coordinates format"),
            GeoJsonError::InvalidStructure(msg) => write!(f, "Invalid geometry structure: {}", msg),
        }
    }
}

impl std::error::Error for GeoJsonError {}

impl From<std::io::Error> for GeoJsonError {
    fn from(e: std::io::Error) -> Self {
        GeoJsonError::Io(e)
    }
}

impl From<serde_json::Error> for GeoJsonError {
    fn from(e: serde_json::Error) -> Self {
        GeoJsonError::Parse(e)
    }
}

/// GeoJSON 几何类型
#[derive(Debug, Clone)]
pub enum GeometryData {
    /// 点
    Point { x: f64, y: f64 },
    /// 多点
    MultiPoint { coords: Vec<(f64, f64)> },
    /// 线串
    LineString { coords: Vec<(f64, f64)> },
    /// 多线串
    MultiLineString { lines: Vec<Vec<(f64, f64)>> },
    /// 多边形（外环 + 内环列表）
    Polygon {
        exterior: Vec<(f64, f64)>,
        holes: Vec<Vec<(f64, f64)>>,
    },
    /// 多多边形
    MultiPolygon {
        polygons: Vec<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)>,
    },
}

impl GeometryData {
    /// 获取几何类型名称
    pub fn type_name(&self) -> &'static str {
        match self {
            GeometryData::Point { .. } => "Point",
            GeometryData::MultiPoint { .. } => "MultiPoint",
            GeometryData::LineString { .. } => "LineString",
            GeometryData::MultiLineString { .. } => "MultiLineString",
            GeometryData::Polygon { .. } => "Polygon",
            GeometryData::MultiPolygon { .. } => "MultiPolygon",
        }
    }

    /// 计算边界框 [min_x, min_y, max_x, max_y]
    pub fn bounding_box(&self) -> Option<[f64; 4]> {
        let coords = self.all_coordinates();
        if coords.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for (x, y) in coords {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        Some([min_x, min_y, max_x, max_y])
    }

    /// 获取所有坐标点
    pub fn all_coordinates(&self) -> Vec<(f64, f64)> {
        match self {
            GeometryData::Point { x, y } => vec![(*x, *y)],
            GeometryData::MultiPoint { coords } => coords.clone(),
            GeometryData::LineString { coords } => coords.clone(),
            GeometryData::MultiLineString { lines } => lines.iter().flatten().copied().collect(),
            GeometryData::Polygon { exterior, holes } => {
                let mut all = exterior.clone();
                for hole in holes {
                    all.extend(hole);
                }
                all
            }
            GeometryData::MultiPolygon { polygons } => {
                let mut all = Vec::new();
                for (ext, holes) in polygons {
                    all.extend(ext);
                    for hole in holes {
                        all.extend(hole);
                    }
                }
                all
            }
        }
    }

    /// 计算质心（简化版）
    pub fn centroid(&self) -> (f64, f64) {
        let coords = self.all_coordinates();
        if coords.is_empty() {
            return (0.0, 0.0);
        }

        let sum_x: f64 = coords.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = coords.iter().map(|(_, y)| y).sum();
        let n = coords.len() as f64;

        (sum_x / n, sum_y / n)
    }
}

/// GeoJSON 属性值
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PropertyValue {
    /// 空值
    Null,
    /// 布尔值
    Bool(bool),
    /// 整数
    Integer(i64),
    /// 浮点数
    Float(f64),
    /// 字符串
    String(String),
    /// 数组
    Array(Vec<PropertyValue>),
    /// 对象
    Object(HashMap<String, PropertyValue>),
}

impl PropertyValue {
    /// 转换为 f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            PropertyValue::Float(f) => Some(*f),
            PropertyValue::Integer(i) => Some(*i as f64),
            PropertyValue::String(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// 转换为字符串
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PropertyValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// 转换为 bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// GeoJSON Feature
#[derive(Debug, Clone)]
pub struct Feature {
    /// 唯一标识符（可选）
    pub id: Option<String>,
    /// 几何数据
    pub geometry: GeometryData,
    /// 属性
    pub properties: HashMap<String, PropertyValue>,
}

impl Feature {
    /// 获取字符串属性
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(|v| v.as_str())
    }

    /// 获取数值属性
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(|v| v.as_f64())
    }

    /// 获取布尔属性
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.properties.get(key).and_then(|v| v.as_bool())
    }
}

/// GeoJSON 读取器
pub struct GeoJsonReader {
    /// 特征列表
    features: Vec<Feature>,
    /// CRS 信息（可选）
    crs: Option<String>,
}

impl GeoJsonReader {
    /// 从文件读取
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, GeoJsonError> {
        let content = fs::read_to_string(path)?;
        Self::from_str(&content)
    }

    /// 从字符串解析
    pub fn from_str(json: &str) -> Result<Self, GeoJsonError> {
        let doc: RawGeoJson = serde_json::from_str(json)?;
        Self::from_raw(doc)
    }

    /// 从原始 JSON 解析
    fn from_raw(doc: RawGeoJson) -> Result<Self, GeoJsonError> {
        let mut features = Vec::new();

        match doc.r#type.as_str() {
            "FeatureCollection" => {
                if let Some(raw_features) = doc.features {
                    for rf in raw_features {
                        if let Some(f) = Self::parse_feature(rf)? {
                            features.push(f);
                        }
                    }
                }
            }
            "Feature" => {
                if let Some(f) = Self::parse_raw_feature(&doc)? {
                    features.push(f);
                }
            }
            t => return Err(GeoJsonError::InvalidGeometry(t.to_string())),
        }

        // 提取 CRS
        let crs = doc.crs.and_then(|c| {
            c.get("properties")
                .and_then(|p| p.get("name"))
                .and_then(|n| n.as_str())
                .map(|s| s.to_string())
        });

        Ok(Self { features, crs })
    }

    /// 解析单个 feature
    fn parse_feature(rf: RawFeature) -> Result<Option<Feature>, GeoJsonError> {
        let geometry = match rf.geometry {
            Some(g) => Self::parse_geometry(g)?,
            None => return Ok(None),
        };

        let properties = rf.properties.unwrap_or_default();
        let id = rf.id.map(|v| match v {
            serde_json::Value::String(s) => s,
            serde_json::Value::Number(n) => n.to_string(),
            _ => String::new(),
        });

        Ok(Some(Feature {
            id,
            geometry,
            properties,
        }))
    }

    /// 从顶层文档解析 feature（当 type == "Feature"）
    fn parse_raw_feature(doc: &RawGeoJson) -> Result<Option<Feature>, GeoJsonError> {
        let geometry = match &doc.geometry {
            Some(g) => Self::parse_geometry(g.clone())?,
            None => return Ok(None),
        };

        let properties = doc.properties.clone().unwrap_or_default();

        Ok(Some(Feature {
            id: None,
            geometry,
            properties,
        }))
    }

    /// 解析几何
    fn parse_geometry(g: RawGeometry) -> Result<GeometryData, GeoJsonError> {
        match g.r#type.as_str() {
            "Point" => {
                let coords = Self::parse_point(&g.coordinates)?;
                Ok(GeometryData::Point {
                    x: coords.0,
                    y: coords.1,
                })
            }
            "MultiPoint" => {
                let coords = Self::parse_coord_array(&g.coordinates)?;
                Ok(GeometryData::MultiPoint { coords })
            }
            "LineString" => {
                let coords = Self::parse_coord_array(&g.coordinates)?;
                Ok(GeometryData::LineString { coords })
            }
            "MultiLineString" => {
                let lines = Self::parse_coord_array_array(&g.coordinates)?;
                Ok(GeometryData::MultiLineString { lines })
            }
            "Polygon" => {
                let (exterior, holes) = Self::parse_polygon(&g.coordinates, "Polygon")?;
                Ok(GeometryData::Polygon { exterior, holes })
            }
            "MultiPolygon" => {
                let polygons = Self::parse_multi_polygon(&g.coordinates)?;
                Ok(GeometryData::MultiPolygon { polygons })
            }
            t => Err(GeoJsonError::InvalidGeometry(t.to_string())),
        }
    }

    /// 解析单点坐标
    fn parse_point(value: &serde_json::Value) -> Result<(f64, f64), GeoJsonError> {
        let arr = value.as_array().ok_or(GeoJsonError::InvalidCoordinates)?;
        if arr.len() < 2 {
            return Err(GeoJsonError::InvalidCoordinates);
        }
        let x = arr[0].as_f64().ok_or(GeoJsonError::InvalidCoordinates)?;
        let y = arr[1].as_f64().ok_or(GeoJsonError::InvalidCoordinates)?;
        if !x.is_finite() || !y.is_finite() {
            return Err(GeoJsonError::InvalidCoordinates);
        }
        Ok((x, y))
    }

    /// 解析坐标数组
    fn parse_coord_array(value: &serde_json::Value) -> Result<Vec<(f64, f64)>, GeoJsonError> {
        let arr = value.as_array().ok_or(GeoJsonError::InvalidCoordinates)?;
        arr.iter().map(Self::parse_point).collect()
    }

    /// 解析二维坐标数组
    fn parse_coord_array_array(
        value: &serde_json::Value,
    ) -> Result<Vec<Vec<(f64, f64)>>, GeoJsonError> {
        let arr = value.as_array().ok_or(GeoJsonError::InvalidCoordinates)?;
        arr.iter().map(Self::parse_coord_array).collect()
    }

    fn parse_polygon(
        value: &serde_json::Value,
        geometry_type: &'static str,
    ) -> Result<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>), GeoJsonError> {
        let rings = value.as_array().ok_or(GeoJsonError::InvalidCoordinates)?;
        if rings.is_empty() {
            return Err(GeoJsonError::InvalidStructure(format!(
                "{geometry_type} must contain at least one linear ring"
            )));
        }

        let exterior = Self::parse_linear_ring(rings[0].clone(), geometry_type, "exterior")?;
        let mut holes = Vec::with_capacity(rings.len().saturating_sub(1));
        for ring in rings.iter().skip(1) {
            holes.push(Self::parse_linear_ring(
                ring.clone(),
                geometry_type,
                "interior",
            )?);
        }

        Ok((exterior, holes))
    }

    fn parse_linear_ring(
        value: serde_json::Value,
        geometry_type: &'static str,
        ring_role: &'static str,
    ) -> Result<Vec<(f64, f64)>, GeoJsonError> {
        let coords = Self::parse_coord_array(&value)?;
        if coords.len() < 4 {
            return Err(GeoJsonError::InvalidStructure(format!(
                "{geometry_type} {ring_role} ring must contain at least 4 positions"
            )));
        }

        let first = coords.first().copied();
        let last = coords.last().copied();
        if first != last {
            return Err(GeoJsonError::InvalidStructure(format!(
                "{geometry_type} {ring_role} ring must be closed"
            )));
        }

        Ok(coords)
    }

    /// 解析多多边形
    fn parse_multi_polygon(
        value: &serde_json::Value,
    ) -> Result<Vec<(Vec<(f64, f64)>, Vec<Vec<(f64, f64)>>)>, GeoJsonError> {
        let arr = value.as_array().ok_or(GeoJsonError::InvalidCoordinates)?;
        if arr.is_empty() {
            return Err(GeoJsonError::InvalidStructure(
                "MultiPolygon must contain at least one polygon".to_string(),
            ));
        }
        let mut result = Vec::new();
        for poly_val in arr {
            result.push(Self::parse_polygon(poly_val, "MultiPolygon")?);
        }
        Ok(result)
    }

    /// 获取所有特征
    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    /// 获取特征数量
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// 获取 CRS 信息
    pub fn crs(&self) -> Option<&str> {
        self.crs.as_deref()
    }

    /// 按属性过滤
    pub fn filter_by_property<F>(&self, predicate: F) -> Vec<&Feature>
    where
        F: Fn(&HashMap<String, PropertyValue>) -> bool,
    {
        self.features
            .iter()
            .filter(|f| predicate(&f.properties))
            .collect()
    }

    /// 按几何类型过滤
    pub fn filter_by_type(&self, type_name: &str) -> Vec<&Feature> {
        self.features
            .iter()
            .filter(|f| f.geometry.type_name() == type_name)
            .collect()
    }

    /// 获取边界条件位置
    ///
    /// 查找包含 "bc_type" 属性的 Point 或 LineString 特征
    pub fn boundary_conditions(&self) -> Vec<BoundaryConditionLocation> {
        let mut result = Vec::new();

        for feature in &self.features {
            let bc_type = match feature.get_string("bc_type") {
                Some(t) => t.to_string(),
                None => continue,
            };

            let name = feature.get_string("name").unwrap_or("unnamed").to_string();
            let value = feature.get_f64("value");

            let location = match &feature.geometry {
                GeometryData::Point { x, y } => BcLocation::Point(*x, *y),
                GeometryData::LineString { coords } => BcLocation::Line(coords.clone()),
                _ => continue,
            };

            result.push(BoundaryConditionLocation {
                name,
                bc_type,
                location,
                value,
                properties: feature.properties.clone(),
            });
        }

        result
    }

    /// 获取区域属性（多边形）
    ///
    /// 用于初始水深、摩擦系数等分区数据
    pub fn zone_properties(&self) -> Vec<ZoneProperties> {
        let mut result = Vec::new();

        for feature in &self.features {
            let name = feature.get_string("name").unwrap_or("zone").to_string();
            match &feature.geometry {
                GeometryData::Polygon { exterior, holes } => {
                    result.push(ZoneProperties {
                        name,
                        exterior: exterior.clone(),
                        holes: holes.clone(),
                        properties: feature.properties.clone(),
                    });
                }
                GeometryData::MultiPolygon { polygons } => {
                    for (idx, (exterior, holes)) in polygons.iter().enumerate() {
                        let zonename = format!("{}_{}", name, idx + 1);
                        result.push(ZoneProperties {
                            name: zonename,
                            exterior: exterior.clone(),
                            holes: holes.clone(),
                            properties: feature.properties.clone(),
                        });
                    }
                }
                _ => continue,
            }
        }

        result
    }
}

/// 原始 GeoJSON 结构（用于解析）
#[derive(Debug, Deserialize)]
struct RawGeoJson {
    r#type: String,
    #[serde(default)]
    features: Option<Vec<RawFeature>>,
    #[serde(default)]
    geometry: Option<RawGeometry>,
    #[serde(default)]
    properties: Option<HashMap<String, PropertyValue>>,
    #[serde(default)]
    crs: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct RawFeature {
    #[allow(dead_code)]
    r#type: String,
    #[serde(default)]
    id: Option<serde_json::Value>,
    geometry: Option<RawGeometry>,
    #[serde(default)]
    properties: Option<HashMap<String, PropertyValue>>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawGeometry {
    r#type: String,
    coordinates: serde_json::Value,
}

/// 边界条件位置
#[derive(Debug, Clone)]
pub struct BoundaryConditionLocation {
    /// 名称
    pub name: String,
    /// 边界条件类型（discharge, water_level, velocity 等）
    pub bc_type: String,
    /// 位置
    pub location: BcLocation,
    /// 数值（可选，可能需要从时间序列加载）
    pub value: Option<f64>,
    /// 其他属性
    pub properties: HashMap<String, PropertyValue>,
}

/// 边界条件位置类型
#[derive(Debug, Clone)]
pub enum BcLocation {
    /// 点位置
    Point(f64, f64),
    /// 线段
    Line(Vec<(f64, f64)>),
}

/// 区域属性
#[derive(Debug, Clone)]
pub struct ZoneProperties {
    /// 名称
    pub name: String,
    /// 外边界
    pub exterior: Vec<(f64, f64)>,
    /// 孔洞
    pub holes: Vec<Vec<(f64, f64)>>,
    /// 属性
    pub properties: HashMap<String, PropertyValue>,
}

impl ZoneProperties {
    /// 获取数值属性
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.properties.get(key).and_then(|v| v.as_f64())
    }

    /// 获取字符串属性
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.properties.get(key).and_then(|v| v.as_str())
    }

    /// 检查点是否在多边形内（射线法）
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        // 检查外边界
        if !point_in_polygon(x, y, &self.exterior) {
            return false;
        }
        // 检查是否在孔洞内
        for hole in &self.holes {
            if point_in_polygon(x, y, hole) {
                return false;
            }
        }
        true
    }
}

/// 射线法判断点是否在多边形内
fn point_in_polygon(x: f64, y: f64, polygon: &[(f64, f64)]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;

    for i in 0..n {
        let (xi, yi) = polygon[i];
        let (xj, yj) = polygon[j];

        if ((yi > y) != (yj > y)) && (yj - yi).abs() > 1e-12 {
            let x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi;
            if x < x_intersect {
                inside = !inside;
            }
        } else if ((yi > y) != (yj > y)) && (yj - yi).abs() <= 1e-12 {
            inside = !inside;
        }
        j = i;
    }

    inside
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_point() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [100.0, 0.5]
            },
            "properties": {
                "name": "test_point",
                "value": 42.0
            }
        }"#;

        let reader = GeoJsonReader::from_str(json).unwrap();
        assert_eq!(reader.len(), 1);

        let f = &reader.features()[0];
        assert!(matches!(
            &f.geometry,
            GeometryData::Point { x, y }
                if (*x - 100.0).abs() < 1e-10 && (*y - 0.5).abs() < 1e-10
        ));

        assert_eq!(f.get_string("name"), Some("test_point"));
        assert_eq!(f.get_f64("value"), Some(42.0));
    }

    #[test]
    fn test_parse_feature_collection() {
        let json = r#"{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0, 0]
                    },
                    "properties": {"bc_type": "discharge", "name": "inlet", "value": 100}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0, 0], [1, 1], [2, 0]]
                    },
                    "properties": {"bc_type": "water_level", "name": "outlet"}
                }
            ]
        }"#;

        let reader = GeoJsonReader::from_str(json).unwrap();
        assert_eq!(reader.len(), 2);

        let bcs = reader.boundary_conditions();
        assert_eq!(bcs.len(), 2);
        assert_eq!(bcs[0].name, "inlet");
        assert_eq!(bcs[0].bc_type, "discharge");
        assert_eq!(bcs[1].name, "outlet");
    }

    #[test]
    fn test_polygon() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
                ]
            },
            "properties": {"name": "zone1", "manning_n": 0.035}
        }"#;

        let reader = GeoJsonReader::from_str(json).unwrap();
        let zones = reader.zone_properties();
        assert_eq!(zones.len(), 1);
        assert_eq!(zones[0].name, "zone1");
        assert_eq!(zones[0].get_f64("manning_n"), Some(0.035));
    }

    #[test]
    fn test_polygon_requires_exterior_ring() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": []
            },
            "properties": {}
        }"#;

        let err = match GeoJsonReader::from_str(json) {
            Ok(_) => panic!("empty polygon rings must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GeoJsonError::InvalidStructure(msg)
            if msg.contains("Polygon must contain at least one linear ring")
        ));
    }

    #[test]
    fn test_polygon_requires_closed_ring() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 0], [10, 0], [10, 10], [0, 10]]
                ]
            },
            "properties": {}
        }"#;

        let err = match GeoJsonReader::from_str(json) {
            Ok(_) => panic!("open polygon ring must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GeoJsonError::InvalidStructure(msg)
            if msg.contains("Polygon exterior ring must be closed")
        ));
    }

    #[test]
    fn test_multi_polygon_requires_polygon_rings() {
        let json = r#"{
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    []
                ]
            },
            "properties": {}
        }"#;

        let err = match GeoJsonReader::from_str(json) {
            Ok(_) => panic!("multipolygon without rings must be rejected"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            GeoJsonError::InvalidStructure(msg)
            if msg.contains("MultiPolygon must contain at least one linear ring")
        ));
    }

    #[test]
    fn test_point_in_polygon() {
        let polygon = vec![
            (0.0, 0.0),
            (10.0, 0.0),
            (10.0, 10.0),
            (0.0, 10.0),
            (0.0, 0.0),
        ];

        assert!(point_in_polygon(5.0, 5.0, &polygon));
        assert!(!point_in_polygon(15.0, 5.0, &polygon));
        assert!(!point_in_polygon(-1.0, 5.0, &polygon));
    }

    #[test]
    fn test_bounding_box() {
        let geom = GeometryData::LineString {
            coords: vec![(0.0, 1.0), (5.0, 3.0), (2.0, 7.0)],
        };

        let bbox = geom.bounding_box().unwrap();
        assert!((bbox[0] - 0.0).abs() < 1e-10); // min_x
        assert!((bbox[1] - 1.0).abs() < 1e-10); // min_y
        assert!((bbox[2] - 5.0).abs() < 1e-10); // max_x
        assert!((bbox[3] - 7.0).abs() < 1e-10); // max_y
    }
}
