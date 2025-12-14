// crates/mh_config/src/precision.rs

//! 运行时精度选择
//!
//! 提供 `Precision` 枚举用于在应用层选择计算精度，
//! 无需在配置层引入泛型参数。

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// 运行时精度枚举
///
/// 用于在 Layer 4/5 选择计算精度，通过此枚举实现运行时分发。
///
/// # 示例
///
/// ```rust
/// use mh_config::Precision;
///
/// let precision = Precision::F32;
/// println!("Using {} precision, {} bytes per scalar", precision, precision.size_bytes());
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// 单精度浮点 (f32)
    /// 
    /// 适用于大规模模拟（>1M 单元），GPU 加速时内存占用减半。
    F32,
    /// 双精度浮点 (f64)
    /// 
    /// 默认精度，适用于科学验证和论文复现。
    F64,
}

impl Precision {
    /// 获取精度名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
        }
    }

    /// 每个标量占用的字节数
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// 是否为单精度
    #[inline]
    pub fn is_f32(&self) -> bool { 
        matches!(self, Self::F32) 
    }
    
    /// 是否为双精度
    #[inline]
    pub fn is_f64(&self) -> bool { 
        matches!(self, Self::F64) 
    }

    /// 获取典型的机器精度
    pub fn epsilon(&self) -> f64 {
        match self {
            Self::F32 => f32::EPSILON as f64,
            Self::F64 => f64::EPSILON,
        }
    }
}

impl Default for Precision {
    fn default() -> Self { 
        Self::F64 
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// 精度解析错误
#[derive(Debug, Clone)]
pub struct PrecisionParseError(String);

impl FromStr for Precision {
    type Err = PrecisionParseError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" | "float" | "single" | "float32" => Ok(Self::F32),
            "f64" | "double" | "float64" => Ok(Self::F64),
            _ => Err(PrecisionParseError(s.to_string())),
        }
    }
}

impl std::fmt::Display for PrecisionParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "无效的精度值: '{}', 期望 'f32' 或 'f64'", self.0)
    }
}

impl std::error::Error for PrecisionParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_default() {
        assert_eq!(Precision::default(), Precision::F64);
    }

    #[test]
    fn test_precision_parse() {
        assert_eq!("f32".parse::<Precision>().unwrap(), Precision::F32);
        assert_eq!("F64".parse::<Precision>().unwrap(), Precision::F64);
        assert_eq!("double".parse::<Precision>().unwrap(), Precision::F64);
        assert!("invalid".parse::<Precision>().is_err());
    }

    #[test]
    fn test_precision_size() {
        assert_eq!(Precision::F32.size_bytes(), 4);
        assert_eq!(Precision::F64.size_bytes(), 8);
    }
}
