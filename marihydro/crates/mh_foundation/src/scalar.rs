//! 统一标量类型系统
//!
//! 通过 feature 控制精度，为 GPU 和混合精度预留接口。
//!
//! # 用法
//!
//! ```
//! use mh_foundation::scalar::{Scalar, ScalarOps};
//!
//! let x: Scalar = 1.5;
//! let y = x.abs();
//! ```
//!
//! # Feature 控制
//!
//! - 默认: `Scalar = f64`
//! - `gpu-f32` feature: `Scalar = f32`

use std::ops::{Add, Sub, Mul, Div, Neg};

/// 计算用标量类型（默认 f64，启用 gpu-f32 feature 时为 f32）
#[cfg(not(feature = "gpu-f32"))]
pub type Scalar = f64;

#[cfg(feature = "gpu-f32")]
pub type Scalar = f32;

/// 标量 trait：所有物理量必须满足的约束
pub trait ScalarOps: 
    Copy + Clone + Default + PartialOrd +
    Add<Output = Self> + Sub<Output = Self> + 
    Mul<Output = Self> + Div<Output = Self> +
    Neg<Output = Self> +
    Sized
{
    /// 零值常量
    const ZERO: Self;
    /// 单位值常量
    const ONE: Self;
    /// 机器精度
    const EPSILON: Self;
    /// 最小正值
    const MIN_POSITIVE: Self;
    /// 最大值
    const MAX: Self;
    
    /// 绝对值
    fn abs(self) -> Self;
    /// 平方根
    fn sqrt(self) -> Self;
    /// 取较大值
    fn max(self, other: Self) -> Self;
    /// 取较小值
    fn min(self, other: Self) -> Self;
    /// 幂运算
    fn powf(self, n: Self) -> Self;
    /// 是否为有限数
    fn is_finite(self) -> bool;
    /// 是否为 NaN
    fn is_nan(self) -> bool;
    /// 限制到范围
    fn clamp(self, min: Self, max: Self) -> Self;
    /// 从 f64 转换
    fn from_f64(v: f64) -> Self;
    /// 转换为 f64
    fn to_f64(self) -> f64;
}

impl ScalarOps for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = 1e-12;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;
    const MAX: Self = f64::MAX;
    
    #[inline] fn abs(self) -> Self { f64::abs(self) }
    #[inline] fn sqrt(self) -> Self { f64::sqrt(self) }
    #[inline] fn max(self, other: Self) -> Self { f64::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f64::min(self, other) }
    #[inline] fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    #[inline] fn is_finite(self) -> bool { f64::is_finite(self) }
    #[inline] fn is_nan(self) -> bool { f64::is_nan(self) }
    #[inline] fn clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }
    #[inline] fn from_f64(v: f64) -> Self { v }
    #[inline] fn to_f64(self) -> f64 { self }
}

impl ScalarOps for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPSILON: Self = 1e-6;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;
    const MAX: Self = f32::MAX;
    
    #[inline] fn abs(self) -> Self { f32::abs(self) }
    #[inline] fn sqrt(self) -> Self { f32::sqrt(self) }
    #[inline] fn max(self, other: Self) -> Self { f32::max(self, other) }
    #[inline] fn min(self, other: Self) -> Self { f32::min(self, other) }
    #[inline] fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    #[inline] fn is_finite(self) -> bool { f32::is_finite(self) }
    #[inline] fn is_nan(self) -> bool { f32::is_nan(self) }
    #[inline] fn clamp(self, min: Self, max: Self) -> Self { f32::clamp(self, min, max) }
    #[inline] fn from_f64(v: f64) -> Self { v as f32 }
    #[inline] fn to_f64(self) -> f64 { self as f64 }
}

/// 物理常量（自动适配精度）
pub mod constants {
    use super::Scalar;
    
    /// 重力加速度 (m/s²)
    pub const GRAVITY: Scalar = 9.81;
    /// 水密度 (kg/m³)
    pub const WATER_DENSITY: Scalar = 1000.0;
    /// 运动粘度 (m²/s)
    pub const KINEMATIC_VISCOSITY: Scalar = 1.0e-6;
    /// 冯卡门常数
    pub const VON_KARMAN: Scalar = 0.41;
    /// 圆周率
    pub const PI: Scalar = std::f64::consts::PI as Scalar;
}

/// 精度转换辅助函数
pub mod convert {
    use super::Scalar;
    
    /// 高精度 -> 计算精度（可能有精度损失）
    #[inline]
    pub fn from_f64(v: f64) -> Scalar {
        v as Scalar
    }
    
    /// 计算精度 -> 高精度（无损）
    #[inline]
    pub fn to_f64(v: Scalar) -> f64 {
        v as f64
    }
    
    /// 批量转换为 f32（用于可视化）
    pub fn slice_to_f32(src: &[Scalar], dst: &mut [f32]) {
        debug_assert_eq!(src.len(), dst.len());
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = *s as f32;
        }
    }
    
    /// 批量从 f64 转换
    pub fn slice_from_f64(src: &[f64], dst: &mut [Scalar]) {
        debug_assert_eq!(src.len(), dst.len());
        for (s, d) in src.iter().zip(dst.iter_mut()) {
            *d = *s as Scalar;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scalar_ops() {
        let x: Scalar = 4.0;
        assert!((x.sqrt() - 2.0).abs() < 1e-10);
        assert!(((-3.0 as Scalar).abs() - 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_scalar_constants() {
        assert!((constants::GRAVITY - 9.81).abs() < 1e-10);
        assert!((constants::PI - 3.14159).abs() < 0.001);
    }
    
    #[test]
    fn test_convert() {
        let v = convert::from_f64(1.5);
        assert!((convert::to_f64(v) - 1.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_clamp() {
        let x: Scalar = 5.0;
        assert!((x.clamp(0.0, 3.0) - 3.0).abs() < 1e-10);
        assert!((x.clamp(6.0, 10.0) - 6.0).abs() < 1e-10);
        assert!((x.clamp(0.0, 10.0) - 5.0).abs() < 1e-10);
    }
}
