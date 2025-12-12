// crates/mh_core/src/scalar.rs

//! 统一标量类型抽象
//!
//! 提供编译期精度选择的唯一接口，支持物理算法在f32和f64之间零成本切换。
//!
//! # 设计原则
//!
//! 1. **单一职责**: 仅解决精度切换问题，不定义物理常量
//! 2. **零成本抽象**: `#[inline]` + 编译期单态化
//! 3. **密封trait**: 只有f32和f64可以实现
//!
//! # 使用示例
//!
//! ```
//! use mh_core::Scalar;
//!
//! fn compute_wave_speed<S: Scalar>(depth: S, g: S) -> S {
//!     (g * depth).sqrt()
//! }
//!
//! let speed_f32 = compute_wave_speed(2.0f32, 9.81f32);
//! let speed_f64 = compute_wave_speed(2.0f64, 9.81f64);
//! ```

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{Float, FromPrimitive, NumAssign};

// 密封trait，禁止外部实现
mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// 统一标量类型约束
///
/// 所有物理计算必须使用此trait作为泛型边界。
///
/// # 架构约束
///
/// - **必须**: 作为泛型约束使用，如 `<S: Scalar>`
/// - **禁止**: 作为trait对象使用，如 `&dyn Scalar`
///
/// # 实现类型
///
/// - `f32`: GPU加速模式，内存占用减半
/// - `f64`: CPU高精度模式（默认）
pub trait Scalar:
    private::Sealed
    + Float
    + FromPrimitive
    + NumAssign
    + Copy
    + Clone
    + Debug
    + Display
    + Send
    + Sync
    + Sum
    + Default
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// 零值: `0.0`
    const ZERO: Self;

    /// 单位值: `1.0`
    const ONE: Self;

    /// 二: `2.0`
    const TWO: Self;

    /// 一半: `0.5`
    const HALF: Self;

    /// 机器精度（Machine epsilon）
    const EPSILON: Self;

    /// 最小正值
    const MIN_POSITIVE: Self;

    /// 最大有限值
    const MAX: Self;

    /// 最小有限值
    const MIN: Self;

    /// 从**配置层**f64转换到**运行层**S（可能丢失精度）
    ///
    /// # 说明
    /// 此方法用于从 f64 配置值转换到运行时标量类型。
    /// 对于 f32 目标类型，可能会丢失精度
    /// TODO(重构-2024Q1): 这是临时方案，应删除此方法改用FromPrimitive
    /// 原因: 与num_traits::FromPrimitive::from_f64冲突，增加API混乱
    /// 正确做法: 所有调用处改为 S::from_f64(v).unwrap_or(S::ZERO)
    fn from_f64_lossless(v: f64) -> Self;

    /// 转换回f64（用于输出或跨模块接口）
    fn to_f64(self) -> f64;

    /// 安全除法（防止除零）
    #[inline]
    fn safe_div(self, rhs: Self, fallback: Self) -> Self {
        if rhs.abs() < Self::MIN_POSITIVE {
            fallback
        } else {
            self / rhs
        }
    }

    /// 检查是否有限（非NaN/Inf）
    #[inline]
    fn is_safe(self) -> bool {
        self.is_finite()
    }

    /// 钳制到正值
    #[inline]
    fn clamp_positive(self) -> Self {
        if self < Self::ZERO {
            Self::ZERO
        } else {
            self
        }
    }

    /// 钳制到范围
    #[inline]
    fn clamp_range(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }
}

// ============================================================================
// f32 实现
// ============================================================================

impl Scalar for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const TWO: f32 = 2.0;
    const HALF: f32 = 0.5;
    const EPSILON: f32 = f32::EPSILON;
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    const MAX: f32 = f32::MAX;
    const MIN: f32 = f32::MIN;

    #[inline]
    fn from_f64_lossless(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// ============================================================================
// f64 实现
// ============================================================================

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const HALF: f64 = 0.5;
    const EPSILON: f64 = f64::EPSILON;
    const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    const MAX: f64 = f64::MAX;
    const MIN: f64 = f64::MIN;

    #[inline]
    fn from_f64_lossless(v: f64) -> Self {
        v
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_constants() {
        assert_eq!(f32::ZERO, 0.0f32);
        assert_eq!(f64::ONE, 1.0f64);
    }

    #[test]
    fn test_from_f64_lossless() {
        let v: f32 = Scalar::from_f64_lossless(3.14159265358979);
        assert!((v - 3.1415927).abs() < 1e-6);

        let v: f64 = Scalar::from_f64_lossless(3.14159265358979);
        assert!((v - 3.14159265358979).abs() < 1e-14);
    }

    #[test]
    fn test_safe_div() {
        let a: f64 = 1.0;
        let b: f64 = 0.0;
        assert_eq!(a.safe_div(b, 999.0), 999.0);

        let c: f64 = 2.0;
        assert!((a.safe_div(c, 999.0) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_clamp_positive() {
        assert_eq!((-1.0f64).clamp_positive(), 0.0);
        assert_eq!((1.0f64).clamp_positive(), 1.0);
    }

    fn generic_function<S: Scalar>(x: S) -> S {
        x * S::TWO + S::ONE
    }

    #[test]
    fn test_generic_usage() {
        let result_f32 = generic_function(1.0f32);
        assert!((result_f32 - 3.0f32).abs() < 1e-6);

        let result_f64 = generic_function(1.0f64);
        assert!((result_f64 - 3.0f64).abs() < 1e-14);
    }
}
