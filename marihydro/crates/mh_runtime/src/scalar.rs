// crates/mh_runtime/src/scalar.rs

//! RuntimeScalar - 密封的标量类型抽象
//!
//! 提供编译期精度选择的唯一接口，支持物理算法在 f32 和 f64 之间零成本切换。
//!
//! # 设计原则
//!
//! 1. **密封 Trait**: 只有 f32 和 f64 可以实现（通过 private::Sealed）
//! 2. **零成本抽象**: `#[inline]` + 编译期单态化
//! 3. **从配置转换**: `from_config(f64)` 用于从配置层（全 f64）转换
//!
//! # 使用规范
//!
//! ```rust
//! use mh_runtime::RuntimeScalar;
//!
//! // ✅ 正确：Layer 3 引擎层使用泛型
//! fn compute_flux<S: RuntimeScalar>(h: S, g: S) -> S {
//!     (h * g).sqrt()
//! }
//!
//! // ❌ 错误：Layer 5 应用层禁止使用泛型
//! // fn run_sim<S: RuntimeScalar>(config: Config) { ... }
//! ```

use std::fmt::{Debug, Display};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use bytemuck::Pod;
use num_traits::{Float, FromPrimitive, NumAssign};

/// 密封模块，禁止外部实现
mod private {
    /// 密封 trait
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// 运行时标量类型（密封，仅 f32/f64 可实现）
///
/// 所有 Layer 3 引擎层组件必须使用此 trait 作为泛型边界，
/// 确保计算核心层可在 f32 和 f64 之间零成本切换。
///
/// # 架构约束
///
/// - **允许**: 在 Layer 3 引擎层作为泛型约束 `<S: RuntimeScalar>`
/// - **禁止**: 在 Layer 4/5 应用层使用任何泛型参数
/// - **禁止**: 作为 trait 对象使用 `&dyn RuntimeScalar`
///
/// # 实现类型
///
/// - `f32`: GPU 加速模式，内存占用减半，适合 >1M 单元的大规模模拟
/// - `f64`: CPU 高精度模式（默认），适合科学验证和论文复现
pub trait RuntimeScalar:
    private::Sealed
    + Pod
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
    /// 零值
    const ZERO: Self;
    /// 一
    const ONE: Self;
    /// 二
    const TWO: Self;
    /// 二分之一
    const HALF: Self;
    /// 机器精度
    const EPSILON: Self;
    /// 最小正值
    const MIN_POSITIVE: Self;
    /// 最大值
    const MAX: Self;
    /// 最小值
    const MIN: Self;

    /// 安全除法
    ///
    /// 当除数绝对值小于 MIN_POSITIVE 时返回 fallback
    #[inline]
    fn safe_div(self, rhs: Self, fallback: Self) -> Self {
        if rhs.abs() < Self::MIN_POSITIVE {
            fallback
        } else {
            self / rhs
        }
    }

    /// 检查是否有限（非 NaN、非 Inf）
    #[inline]
    fn is_safe(self) -> bool {
        self.is_finite()
    }

    /// 带阈值的安全除法
    #[inline]
    fn safe_div_eps(self, rhs: Self, eps: Self, fallback: Self) -> Self {
        if rhs.abs() < eps {
            fallback
        } else {
            self / rhs
        }
    }

    /// 限制在范围内
    #[inline]
    fn clamp_value(self, min: Self, max: Self) -> Self {
        if self < min {
            min
        } else if self > max {
            max
        } else {
            self
        }
    }

    /// 安全平方根（负数返回 0）
    #[inline]
    fn safe_sqrt(self) -> Self {
        if self < Self::ZERO {
            Self::ZERO
        } else {
            self.sqrt()
        }
    }

    /// 安全幂运算（负数非整数次幂返回 0）
    #[inline]
    fn safe_powf(self, exp: Self) -> Self {
        if self < Self::ZERO && (exp.fract() != Self::ZERO) {
            Self::ZERO
        } else {
            self.powf(exp)
        }
    }

    /// 安全自然对数（非正数返回 0）
    #[inline]
    fn safe_ln(self) -> Self {
        if self <= Self::ZERO {
            Self::ZERO
        } else {
            self.ln()
        }
    }

    /// 安全正弦（大数周期归约）
    #[inline]
    fn sin_safe(self) -> Self {
        let threshold = Self::from_f64(1e15).unwrap_or(Self::MAX);
        if self.abs() > threshold {
            let reduced = self % Self::from_f64(2.0 * std::f64::consts::PI).unwrap_or(Self::MAX);
            reduced.sin()
        } else {
            self.sin()
        }
    }

    /// 安全余弦（大数周期归约）
    #[inline]
    fn cos_safe(self) -> Self {
        let threshold = Self::from_f64(1e15).unwrap_or(Self::MAX);
        if self.abs() > threshold {
            let reduced = self % Self::from_f64(2.0 * std::f64::consts::PI).unwrap_or(Self::MAX);
            reduced.cos()
        } else {
            self.cos()
        }
    }

    /// 安全正弦余弦对（保证 sin² + cos² ≈ 1）
    #[inline]
    fn sin_cos_safe(self) -> (Self, Self) {
        let threshold = Self::from_f64(1e15).unwrap_or(Self::MAX);
        let reduced = if self.abs() > threshold {
            self % Self::from_f64(2.0 * std::f64::consts::PI).unwrap_or(Self::MAX)
        } else {
            self
        };
        reduced.sin_cos()
    }

    /// 近似相等判断
    #[inline]
    fn approx_eq(self, other: Self, epsilon: Self) -> bool {
        (self - other).abs() < epsilon
    }

    /// 检查是否接近零
    #[inline]
    fn is_near_zero(self, epsilon: Self) -> bool {
        self.abs() < epsilon
    }

    /// 批量验证切片中所有值是否有限
    fn validate_slice(data: &[Self]) -> Result<(), (usize, Self)> {
        for (i, &v) in data.iter().enumerate() {
            if !v.is_safe() {
                return Err((i, v));
            }
        }
        Ok(())
    }
}

// =============================================================================
// f32 实现
// =============================================================================

impl RuntimeScalar for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const TWO: f32 = 2.0;
    const HALF: f32 = 0.5;
    const EPSILON: f32 = f32::EPSILON;
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    const MAX: f32 = f32::MAX;
    const MIN: f32 = f32::MIN;
}

// =============================================================================
// f64 实现
// =============================================================================

impl RuntimeScalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const HALF: f64 = 0.5;
    const EPSILON: f64 = f64::EPSILON;
    const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    const MAX: f64 = f64::MAX;
    const MIN: f64 = f64::MIN;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_constants() {
        assert_eq!(f32::ZERO, 0.0f32);
        assert_eq!(f32::ONE, 1.0f32);
        assert_eq!(f32::TWO, 2.0f32);
        assert_eq!(f32::HALF, 0.5f32);
    }

    #[test]
    fn test_f64_constants() {
        assert_eq!(f64::ZERO, 0.0f64);
        assert_eq!(f64::ONE, 1.0f64);
        assert_eq!(f64::TWO, 2.0f64);
        assert_eq!(f64::HALF, 0.5f64);
    }

    #[test]
    fn test_from_config() {
        let v = 9.81f64;
        assert_eq!(f32::from_f64(v), Some(9.81f32));
        assert_eq!(f64::from_f64(v), Some(9.81f64));
    }

    #[test]
    fn test_safe_div() {
        let x = 1.0f64;
        let y = 0.0f64;
        assert_eq!(x.safe_div(y, 999.0), 999.0);
        assert_eq!(x.safe_div(2.0, 999.0), 0.5);
    }

    #[test]
    fn test_safe_sqrt() {
        let x = 16.0f64;
        let y = -4.0f64;
        assert_eq!(x.safe_sqrt(), 4.0);
        assert_eq!(y.safe_sqrt(), 0.0);
    }

    #[test]
    fn test_safe_ln() {
        let x = std::f64::consts::E;
        let y = -1.0f64;
        assert!((x.safe_ln() - 1.0).abs() < 1e-10);
        assert_eq!(y.safe_ln(), 0.0);
    }

    #[test]
    fn test_validate_slice() {
        let data = vec![1.0f64, 2.0, 3.0];
        assert!(f64::validate_slice(&data).is_ok());
        
        let bad_data = vec![1.0f64, f64::NAN, 3.0];
        assert!(f64::validate_slice(&bad_data).is_err());
    }

    #[test]
    fn test_sin_cos_safe() {
        let x = std::f64::consts::PI / 4.0;
        let (sin, cos) = x.sin_cos_safe();
        assert!((sin - cos).abs() < 1e-10); // sin(π/4) == cos(π/4)
    }

    #[test]
    fn test_is_near_zero() {
        let x = 1e-15f64;
        assert!(x.is_near_zero(1e-14));
        assert!(!x.is_near_zero(1e-16));
    }

    #[test]
    fn test_approx_eq() {
        let a = 1.0f64;
        let b = 1.0 + 1e-15;
        assert!(a.approx_eq(b, 1e-14));
        assert!(!a.approx_eq(b, 1e-16));
    }
}