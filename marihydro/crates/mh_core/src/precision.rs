// crates/mh_core/src/precision.rs

//! 运行时精度选择
//!
//! 提供App层唯一接触的精度类型，用于运行时决定使用f32还是f64。

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// 运行时精度选择（App层唯一接触的精度类型）
///
/// # 精度语义
///
/// ## F32模式
/// - 内存减半，适合大规模网格（>1M单元）
/// - GPU计算的原生精度
/// - 数值容差放宽：h_min=1e-4, epsilon=1e-5
/// - 适用场景：实时预览、AI训练、性能基准
///
/// ## F64模式
/// - 数值稳定性优先
/// - 科学验证和论文结果复现
/// - 数值容差严格：h_min=1e-9, epsilon=1e-12
/// - 适用场景：验证计算、长期积分、复杂边界
///
/// # 示例
///
/// ```
/// use mh_core::Precision;
///
/// let precision = Precision::default(); // F32
/// assert_eq!(precision, Precision::F32);
///
/// let precision: Precision = "f64".parse().unwrap();
/// assert_eq!(precision, Precision::F64);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// 单精度浮点（32位）
    #[default]
    F32,
    /// 双精度浮点（64位）
    F64,
}

impl Precision {
    /// 获取精度名称
    pub fn name(&self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        }
    }

    /// 获取字节大小
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F64 => 8,
        }
    }

    /// 是否为单精度
    pub fn is_f32(&self) -> bool {
        matches!(self, Precision::F32)
    }

    /// 是否为双精度
    pub fn is_f64(&self) -> bool {
        matches!(self, Precision::F64)
    }

    /// 获取机器epsilon
    pub fn epsilon(&self) -> f64 {
        match self {
            Precision::F32 => f32::EPSILON as f64,
            Precision::F64 => f64::EPSILON,
        }
    }

    /// 获取推荐的最小水深
    pub fn recommended_h_min(&self) -> f64 {
        match self {
            Precision::F32 => 1e-4,
            Precision::F64 => 1e-9,
        }
    }

    /// 获取推荐的干单元阈值
    pub fn recommended_h_dry(&self) -> f64 {
        match self {
            Precision::F32 => 1e-3,
            Precision::F64 => 1e-6,
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for Precision {
    type Err = PrecisionParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" | "float" | "single" => Ok(Precision::F32),
            "f64" | "double" => Ok(Precision::F64),
            _ => Err(PrecisionParseError(s.to_string())),
        }
    }
}

/// 精度解析错误
#[derive(Debug, Clone)]
pub struct PrecisionParseError(pub String);

impl fmt::Display for PrecisionParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "无效的精度值: '{}', 期望 'f32' 或 'f64'", self.0)
    }
}

impl std::error::Error for PrecisionParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_precision() {
        assert_eq!(Precision::default(), Precision::F32);
    }

    #[test]
    fn test_precision_parse() {
        assert_eq!("f32".parse::<Precision>().unwrap(), Precision::F32);
        assert_eq!("f64".parse::<Precision>().unwrap(), Precision::F64);
        assert_eq!("F32".parse::<Precision>().unwrap(), Precision::F32);
        assert_eq!("double".parse::<Precision>().unwrap(), Precision::F64);
    }

    #[test]
    fn test_precision_size() {
        assert_eq!(Precision::F32.size_bytes(), 4);
        assert_eq!(Precision::F64.size_bytes(), 8);
    }

    #[test]
    fn test_serde() {
        let p = Precision::F64;
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(json, "\"f64\"");
        
        let parsed: Precision = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Precision::F64);
    }
}
