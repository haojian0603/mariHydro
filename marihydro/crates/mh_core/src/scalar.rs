// crates/mh_core/src/scalar.rs

//! 统一标量类型抽象（RuntimeScalar）
//!
//! 提供编译期精度选择的唯一接口，支持物理算法在f32和f64之间零成本切换。
//! 这是Layer 2核心抽象层的基础trait，所有Layer 3引擎层组件必须使用此trait。
//!
//! # 架构定位
//!
//! - **层级**: Layer 2 - Core Abstractions
//! - **依赖**: 仅依赖num-traits标准库
//! - **约束**: 禁止在Layer 4/5直接使用，必须通过Backend访问
//!
//! # 设计原则
//!
//! 1. **单一职责**: 仅解决精度切换问题，不定义物理常量
//! 2. **零成本抽象**: `#[inline]` + 编译期单态化
//! 3. **密封trait**: 只有f32和f64可以实现（确保可预测性）
//!
//! # 使用规范
//!
//! ```rust
//! // ✅ 正确：Layer 3引擎层使用泛型
//! fn compute_flux<S: RuntimeScalar>(h: &[S]) -> S {
//!     let g = S::from_f64(9.81).unwrap_or(S::ZERO);
//!     h[0] * g
//! }
//!
//! // ❌ 错误：Layer 5应用层禁止使用泛型
//! // fn run_sim<S: RuntimeScalar>(config: SolverConfig) { ... }
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

/// 统一标量类型约束（RuntimeScalar）
///
/// 所有Layer 3引擎层组件必须使用此trait作为泛型边界，
/// 确保计算核心层可在f32和f64之间零成本切换。
///
/// # 架构约束
///
/// - **允许**: 在Layer 3引擎层作为泛型约束 `<S: RuntimeScalar>`
/// - **禁止**: 在Layer 4/5应用层使用任何泛型参数
/// - **禁止**: 作为trait对象使用 `&dyn RuntimeScalar`
///
/// # 实现类型
///
/// - `f32`: GPU加速模式，内存占用减半，适合>1M单元的大规模模拟
/// - `f64`: CPU高精度模式（默认），适合科学验证和论文复现
pub trait RuntimeScalar:
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
    /// # 架构说明
    ///
    /// 配置层（Layer 4/5）使用f64存储用户输入，
    /// 运行层（Layer 3）使用泛型S进行计算。
    ///
    /// # 实现要求
    ///
    /// - f32实现: `v as f32`（可能丢失精度，但配置层已验证范围）
    /// - f64实现: `v`（无损失）
    ///
    /// # 与FromPrimitive的关系
    ///
    /// 此方法是对`FromPrimitive::from_f64`的封装，明确其用途为配置转换。
    /// 返回值是`Option<Self>`，调用处必须使用`.unwrap_or(S::ZERO)`处理失败情况。
    fn from_config(v: f64) -> Option<Self>;

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

impl RuntimeScalar for f32 {
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const TWO: f32 = 2.0;
    const HALF: f32 = 0.5;
    const EPSILON: f32 = f32::EPSILON;
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    const MAX: f32 = f32::MAX;
    const MIN: f32 = f32::MIN;

    #[inline]
    fn from_config(v: f64) -> Option<Self> {
        <Self as FromPrimitive>::from_f64(v)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

// ============================================================================
// f64 实现
// ============================================================================

impl RuntimeScalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const HALF: f64 = 0.5;
    const EPSILON: f64 = f64::EPSILON;
    const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    const MAX: f64 = f64::MAX;
    const MIN: f64 = f64::MIN;

    #[inline]
    fn from_config(v: f64) -> Option<Self> {
        <Self as FromPrimitive>::from_f64(v)
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
    fn test_from_config() {
        // f32可能丢失精度
        let v: Option<f32> = RuntimeScalar::from_config(3.14159265358979);
        assert!(v.is_some());
        assert!((v.unwrap() - 3.1415927).abs() < 1e-6);

        // f64无损失
        let v: Option<f64> = RuntimeScalar::from_config(3.14159265358979);
        assert!(v.is_some());
        assert!((v.unwrap() - 3.14159265358979).abs() < 1e-14);
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

    fn generic_function<S: RuntimeScalar>(x: S) -> S {
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