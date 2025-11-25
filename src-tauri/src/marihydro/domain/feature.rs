// src-tauri/src/marihydro/domain/feature.rs

use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

use crate::marihydro::domain::boundary::BcKind;

/// 边界条件行为模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryMode {
    Tide,
    Flow,
    Radiation,
    Wall,
    Open,
}

impl fmt::Display for BoundaryMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tide => write!(f, "Tide"),
            Self::Flow => write!(f, "Flow"),
            Self::Radiation => write!(f, "Radiation"),
            Self::Wall => write!(f, "Wall"),
            Self::Open => write!(f, "Open"),
        }
    }
}

impl From<BoundaryMode> for BcKind {
    fn from(mode: BoundaryMode) -> Self {
        match mode {
            BoundaryMode::Tide => BcKind::Tide,
            BoundaryMode::Flow => BcKind::Flow,
            BoundaryMode::Radiation => BcKind::Radiation,
            BoundaryMode::Wall => BcKind::Wall,
            BoundaryMode::Open => BcKind::Open,
        }
    }
}

/// 几何类型简易分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryType {
    Point,
    LineString,
    Polygon,
    Unknown,
}

/// 几何特征类型定义
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "properties")]
pub enum FeatureType {
    /// 内部阻水/透水结构
    Structure {
        crest_elevation: f64,
        width: Option<f64>,
        permeability: f64,
    },

    /// 开放边界
    Boundary {
        mode: BoundaryMode,
        source_id: Option<Uuid>,
    },

    /// 初始条件区域
    InitialCondition { variable: String, value: f64 },

    /// 虚拟观测站
    Station {
        output_frequency_steps: usize,
        #[serde(default)]
        output_variables: Vec<String>,
    },

    /// 注释
    Annotation { description: String },
}

/// 通用几何特征对象
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricFeature {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,

    pub name: String,

    #[serde(default = "default_true")]
    pub enabled: bool,

    pub feature_type: FeatureType,

    pub geometry_wkt: String,
}

fn default_true() -> bool {
    true
}

impl GeometricFeature {
    pub fn new(name: impl Into<String>, feature_type: FeatureType, wkt: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            enabled: true,
            feature_type,
            geometry_wkt: wkt.into(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err(format!("Feature ID {} 名称不能为空", self.id));
        }

        if self.geometry_wkt.trim().is_empty() {
            return Err(format!("Feature '{}' 缺少几何数据", self.name));
        }

        match &self.feature_type {
            FeatureType::Structure {
                width,
                permeability,
                ..
            } => {
                if let Some(w) = width {
                    if *w <= 0.0 {
                        return Err(format!("结构物 '{}' 宽度必须大于 0", self.name));
                    }
                }
                if *permeability < 0.0 || *permeability > 1.0 {
                    return Err(format!("结构物 '{}' 渗透率必须在 [0, 1] 之间", self.name));
                }
            }
            FeatureType::Boundary { mode, source_id } => {
                if (*mode == BoundaryMode::Tide || *mode == BoundaryMode::Flow)
                    && source_id.is_none()
                {
                    // 逻辑校验，暂不阻断
                }
            }
            FeatureType::InitialCondition { variable, .. } => {
                if variable.trim().is_empty() {
                    return Err(format!("初始条件 '{}' 未指定目标变量", self.name));
                }
            }
            FeatureType::Station {
                output_frequency_steps,
                ..
            } => {
                if *output_frequency_steps == 0 {
                    return Err(format!("观测站 '{}' 输出频率不能为 0", self.name));
                }
            }
            FeatureType::Annotation { .. } => {}
        }

        Ok(())
    }

    pub fn guess_geometry_type(&self) -> GeometryType {
        let wkt_upper = self.geometry_wkt.trim_start().to_uppercase();
        if wkt_upper.starts_with("POINT") {
            GeometryType::Point
        } else if wkt_upper.starts_with("LINESTRING") || wkt_upper.starts_with("MULTILINESTRING") {
            GeometryType::LineString
        } else if wkt_upper.starts_with("POLYGON") || wkt_upper.starts_with("MULTIPOLYGON") {
            GeometryType::Polygon
        } else {
            GeometryType::Unknown
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_validation() {
        let feat = GeometricFeature::new(
            "Bad Wall",
            FeatureType::Structure {
                crest_elevation: 5.0,
                width: Some(-1.0),
                permeability: 1.5,
            },
            "LINESTRING(0 0, 1 1)",
        );
        assert!(feat.validate().is_err());
    }

    #[test]
    fn test_station_validation() {
        let feat = GeometricFeature::new(
            "Good Station",
            FeatureType::Station {
                output_frequency_steps: 10,
                output_variables: vec![],
            },
            "POINT(10 10)",
        );
        assert!(feat.validate().is_ok());
        assert_eq!(feat.guess_geometry_type(), GeometryType::Point);
    }
}
