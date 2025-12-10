// marihydro\crates\mh_physics\src\core\scalar.rs
//! 标量类型抽象
//!
//! 提供 f32/f64 的统一接口，支持编译期精度选择。

use bytemuck::Pod;
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

/// 标量类型约束
pub trait Scalar:
    Float
    + Pod
    + Default
    + Debug
    + Display
    + Send
    + Sync
    + NumAssign
    + FromPrimitive
    + Sum
    + 'static
{
    /// 类型名称
    fn type_name() -> &'static str;
    
    /// 机器精度
    fn epsilon() -> Self;
    
    /// 最小正规数
    fn min_positive() -> Self;
    
    /// 从 f64 转换
    fn from_f64(v: f64) -> Self;
    
    /// 转换为 f64
    fn to_f64(self) -> f64;
    
    /// 平方根
    fn sqrt(self) -> Self;
    
    /// 绝对值
    fn abs(self) -> Self;
    
    /// 最大值
    fn max(self, other: Self) -> Self;
    
    /// 最小值
    fn min(self, other: Self) -> Self;
    
    /// 钳位
    fn clamp(self, min: Self, max: Self) -> Self;
}

impl Scalar for f32 {
    fn type_name() -> &'static str { "f32" }
    fn epsilon() -> Self { f32::EPSILON }
    fn min_positive() -> Self { f32::MIN_POSITIVE }
    fn from_f64(v: f64) -> Self { v as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn abs(self) -> Self { f32::abs(self) }
    fn max(self, other: Self) -> Self { f32::max(self, other) }
    fn min(self, other: Self) -> Self { f32::min(self, other) }
    fn clamp(self, min: Self, max: Self) -> Self { f32::clamp(self, min, max) }
}

impl Scalar for f64 {
    fn type_name() -> &'static str { "f64" }
    fn epsilon() -> Self { f64::EPSILON }
    fn min_positive() -> Self { f64::MIN_POSITIVE }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn abs(self) -> Self { f64::abs(self) }
    fn max(self, other: Self) -> Self { f64::max(self, other) }
    fn min(self, other: Self) -> Self { f64::min(self, other) }
    fn clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_f32() {
        let x: f32 = Scalar::from_f64(3.14);
        assert!((x - 3.14f32).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_f64() {
        let x: f64 = Scalar::from_f64(3.14);
        assert!((x - 3.14f64).abs() < 1e-14);
    }
}
