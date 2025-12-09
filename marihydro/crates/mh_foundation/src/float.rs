// marihydro\crates\mh_foundation\src/float.rs

//! 安全浮点数类型和数值常量
//!
//! 提供 `SafeF64` 类型保证浮点数有限性，以及数值计算相关的常量。
//!
//! # 设计目标
//!
//! 1. **数值安全**: 保证浮点数非NaN、非Inf
//! 2. **安全运算**: 提供防止溢出的数学运算
//! 3. **零开销**: 在release模式下最小化运行时检查
//!
//! # 示例
//!
//! ```
//! use mh_foundation::float::SafeF64;
//!
//! let x = SafeF64::new(1.0).unwrap();
//! let y = SafeF64::new(2.0).unwrap();
//! let z = x + y;
//! assert_eq!(z.get(), 3.0);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// 数值常量
// ============================================================================

/// 浮点数相等性比较的默认容差
pub const DEFAULT_EPSILON: f64 = 1e-14;

/// 安全除法的最小分母阈值
pub const SAFE_DIV_EPSILON: f64 = 1e-14;

/// 最小允许面积 (m²)
pub const MIN_AREA: f64 = 1e-12;

/// 矩阵条件数警告阈值
pub const CONDITION_NUMBER_WARNING: f64 = 1e10;

/// 矩阵条件数错误阈值
pub const CONDITION_NUMBER_ERROR: f64 = 1e14;

/// 迭代求解器的默认最大迭代次数
pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// 迭代求解器的默认收敛容差
pub const DEFAULT_CONVERGENCE_TOL: f64 = 1e-8;

// ============================================================================
// 安全浮点数类型
// ============================================================================

/// 非有限值错误
#[derive(Debug, Clone)]
pub struct NonFiniteError {
    /// 非法的浮点值（NaN 或 Inf）
    pub value: f64,
}

impl fmt::Display for NonFiniteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.value.is_nan() {
            write!(f, "值为 NaN")
        } else if self.value.is_infinite() {
            write!(f, "值为无穷大: {}", self.value)
        } else {
            write!(f, "非有限浮点值: {}", self.value)
        }
    }
}

impl std::error::Error for NonFiniteError {}

/// 保证有限的浮点数（非NaN、非Inf）
///
/// # 用途
///
/// 用于确保计算结果的有效性，防止 NaN 或无穷大值污染整个计算域。
///
/// # 内存布局
///
/// 使用 `#[repr(transparent)]` 保证与 f64 相同的内存布局。
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
pub struct SafeF64(f64);

impl SafeF64 {
    /// 零值
    pub const ZERO: Self = Self(0.0);

    /// 单位值
    pub const ONE: Self = Self(1.0);

    /// 机器精度
    pub const EPSILON: Self = Self(f64::EPSILON);

    /// 尝试从 f64 创建
    #[inline]
    pub fn new(value: f64) -> Option<Self> {
        if value.is_finite() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// 从 f64 创建，非有限值返回错误
    #[inline]
    pub fn try_new(value: f64) -> Result<Self, NonFiniteError> {
        if value.is_finite() {
            Ok(Self(value))
        } else {
            Err(NonFiniteError { value })
        }
    }

    /// 从 f64 创建，非有限值替换为默认值
    #[inline]
    pub fn new_or(value: f64, default: f64) -> Self {
        if value.is_finite() {
            Self(value)
        } else {
            // 如果 default 也不是有限值，使用 0.0
            Self(if default.is_finite() { default } else { 0.0 })
        }
    }

    /// 从 f64 创建，非有限值替换为零
    #[inline]
    pub fn new_or_zero(value: f64) -> Self {
        Self::new_or(value, 0.0)
    }

    /// 常量创建（编译期）
    ///
    /// # 注意
    ///
    /// 调用者必须确保传入的值是有限的。
    #[inline]
    pub const fn from_const(value: f64) -> Self {
        Self(value)
    }

    /// 获取内部值
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// 安全除法
    #[inline]
    pub fn safe_div(self, other: Self, fallback: f64) -> Self {
        if other.0.abs() < SAFE_DIV_EPSILON {
            Self::new_or(fallback, 0.0)
        } else {
            Self::new_or(self.0 / other.0, fallback)
        }
    }

    /// 安全除法（使用阈值）
    #[inline]
    pub fn safe_div_with_threshold(self, other: f64, threshold: f64, fallback: f64) -> Self {
        if other.abs() < threshold {
            Self::new_or(fallback, 0.0)
        } else {
            Self::new_or(self.0 / other, fallback)
        }
    }

    /// 安全平方根
    #[inline]
    pub fn safe_sqrt(self) -> Self {
        Self::new_or(self.0.max(0.0).sqrt(), 0.0)
    }

    /// 安全幂运算
    #[inline]
    pub fn safe_powf(self, exp: f64) -> Self {
        if self.0 < 0.0 && (exp.fract() != 0.0) {
            // 负数的非整数次幂
            Self(0.0)
        } else {
            Self::new_or(self.0.abs().powf(exp), 0.0)
        }
    }

    /// 安全对数
    #[inline]
    pub fn safe_ln(self) -> Self {
        if self.0 <= 0.0 {
            Self::ZERO
        } else {
            Self::new_or(self.0.ln(), 0.0)
        }
    }

    /// 绝对值
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// 最大值
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// 最小值
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// 限制在范围内
    #[inline]
    pub fn clamp(self, min: f64, max: f64) -> Self {
        Self(self.0.clamp(min, max))
    }

    /// 符号函数
    #[inline]
    pub fn signum(self) -> Self {
        Self(self.0.signum())
    }

    /// 正弦
    #[inline]
    pub fn sin(self) -> Self {
        Self::new_or(self.0.sin(), 0.0)
    }

    /// 余弦
    #[inline]
    pub fn cos(self) -> Self {
        Self::new_or(self.0.cos(), 1.0)
    }

    /// 正切
    #[inline]
    pub fn tan(self) -> Self {
        Self::new_or(self.0.tan(), 0.0)
    }

    /// 反正切
    #[inline]
    pub fn atan(self) -> Self {
        Self::new_or(self.0.atan(), 0.0)
    }

    /// 双参数反正切
    #[inline]
    pub fn atan2(self, other: Self) -> Self {
        Self::new_or(self.0.atan2(other.0), 0.0)
    }

    /// 地板函数
    #[inline]
    pub fn floor(self) -> Self {
        Self(self.0.floor())
    }

    /// 天花板函数
    #[inline]
    pub fn ceil(self) -> Self {
        Self(self.0.ceil())
    }

    /// 四舍五入
    #[inline]
    pub fn round(self) -> Self {
        Self(self.0.round())
    }

    /// 检查是否接近零
    #[inline]
    pub fn is_near_zero(self, epsilon: f64) -> bool {
        self.0.abs() < epsilon
    }

    /// 检查两个值是否近似相等
    #[inline]
    pub fn approx_eq(self, other: Self, epsilon: f64) -> bool {
        (self.0 - other.0).abs() < epsilon
    }
}

impl Default for SafeF64 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<SafeF64> for f64 {
    #[inline]
    fn from(v: SafeF64) -> f64 {
        v.0
    }
}

impl fmt::Display for SafeF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ============================================================================
// 算术运算实现
// ============================================================================

impl Add for SafeF64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new_or(self.0 + rhs.0, 0.0)
    }
}

impl Sub for SafeF64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new_or(self.0 - rhs.0, 0.0)
    }
}

impl Mul for SafeF64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new_or(self.0 * rhs.0, 0.0)
    }
}

impl Div for SafeF64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.safe_div(rhs, 0.0)
    }
}

impl Mul<f64> for SafeF64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new_or(self.0 * rhs, 0.0)
    }
}

impl Div<f64> for SafeF64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        self.safe_div_with_threshold(rhs, SAFE_DIV_EPSILON, 0.0)
    }
}

impl AddAssign for SafeF64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for SafeF64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for SafeF64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for SafeF64 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for SafeF64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 安全除法（直接操作 f64）
#[inline]
pub fn safe_div(a: f64, b: f64, fallback: f64) -> f64 {
    if b.abs() < SAFE_DIV_EPSILON {
        fallback
    } else {
        let result = a / b;
        if result.is_finite() {
            result
        } else {
            fallback
        }
    }
}

/// 安全平方根
#[inline]
pub fn safe_sqrt(x: f64) -> f64 {
    x.max(0.0).sqrt()
}

/// 检查浮点数是否有效（有限）
#[inline]
pub fn is_valid_f64(x: f64) -> bool {
    x.is_finite()
}

/// 限制值到有效范围
#[inline]
pub fn clamp_valid(x: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if x.is_finite() {
        x.clamp(min, max)
    } else {
        fallback
    }
}

// ============================================================================
// Kahan 求和算法
// ============================================================================

/// Kahan 求和器
///
/// 使用 Kahan 求和算法减少浮点累加误差。适用于需要高精度求和的场景，
/// 如大量小数相加或存在大数吃小数的情况。
///
/// # 算法原理
///
/// Kahan 算法通过维护一个补偿项 (compensation) 来跟踪累加过程中丢失的低位精度。
///
/// # 示例
///
/// ```
/// use mh_foundation::float::KahanSum;
///
/// // 累加大量小数
/// let mut sum = KahanSum::new();
/// for _ in 0..10000 {
///     sum.add(0.1);
/// }
/// // 期望值为 1000.0
/// let expected = 1000.0;
/// let error = (sum.value() - expected).abs();
/// assert!(error < 1e-10, "误差应该很小: {}", error);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct KahanSum {
    /// 累加和
    sum: f64,
    /// 补偿项（低位精度损失）
    compensation: f64,
}

impl KahanSum {
    /// 创建新的 Kahan 求和器
    #[inline]
    pub fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// 从初始值创建
    #[inline]
    pub fn with_initial(value: f64) -> Self {
        Self {
            sum: value,
            compensation: 0.0,
        }
    }

    /// 添加一个值
    #[inline]
    pub fn add(&mut self, value: f64) {
        // 补偿后的值
        let y = value - self.compensation;
        // 新的部分和
        let t = self.sum + y;
        // 计算新的补偿项：(t - sum) 是 y 的高位部分，
        // 减去 y 得到丢失的低位部分（取反存储）
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// 获取当前求和值
    #[inline]
    pub fn value(&self) -> f64 {
        self.sum
    }

    /// 重置求和器
    #[inline]
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
    }

    /// 从迭代器求和
    pub fn sum_iter<I: IntoIterator<Item = f64>>(iter: I) -> f64 {
        let mut kahan = Self::new();
        for v in iter {
            kahan.add(v);
        }
        kahan.value()
    }
}

impl std::iter::Sum<f64> for KahanSum {
    fn sum<I: Iterator<Item = f64>>(iter: I) -> Self {
        let mut kahan = KahanSum::new();
        for v in iter {
            kahan.add(v);
        }
        kahan
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_f64_creation() {
        assert!(SafeF64::new(1.0).is_some());
        assert!(SafeF64::new(f64::NAN).is_none());
        assert!(SafeF64::new(f64::INFINITY).is_none());
        assert!(SafeF64::new(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn test_safe_f64_try_new() {
        assert!(SafeF64::try_new(1.0).is_ok());
        assert!(SafeF64::try_new(f64::NAN).is_err());
    }

    #[test]
    fn test_safe_f64_new_or() {
        let x = SafeF64::new_or(f64::NAN, 42.0);
        assert!((x.get() - 42.0).abs() < 1e-10);

        let y = SafeF64::new_or(1.0, 42.0);
        assert!((y.get() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_new_or_zero() {
        let x = SafeF64::new_or_zero(f64::NAN);
        assert!((x.get() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_arithmetic() {
        let a = SafeF64::new(3.0).unwrap();
        let b = SafeF64::new(4.0).unwrap();

        assert!((a + b).get() - 7.0 < 1e-10);
        assert!((b - a).get() - 1.0 < 1e-10);
        assert!((a * b).get() - 12.0 < 1e-10);
    }

    #[test]
    fn test_safe_f64_safe_div() {
        let a = SafeF64::new(10.0).unwrap();
        let b = SafeF64::new(2.0).unwrap();
        let zero = SafeF64::ZERO;

        assert!((a.safe_div(b, 0.0).get() - 5.0).abs() < 1e-10);
        assert!((a.safe_div(zero, -1.0).get() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_safe_sqrt() {
        let positive = SafeF64::new(4.0).unwrap();
        let negative = SafeF64::new(-4.0).unwrap();

        assert!((positive.safe_sqrt().get() - 2.0).abs() < 1e-10);
        assert!((negative.safe_sqrt().get() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_safe_powf() {
        let base = SafeF64::new(2.0).unwrap();
        assert!((base.safe_powf(3.0).get() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_trig() {
        let zero = SafeF64::ZERO;
        assert!((zero.sin().get() - 0.0).abs() < 1e-10);
        assert!((zero.cos().get() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_clamp() {
        let x = SafeF64::new(15.0).unwrap();
        assert!((x.clamp(0.0, 10.0).get() - 10.0).abs() < 1e-10);

        let y = SafeF64::new(-5.0).unwrap();
        assert!((y.clamp(0.0, 10.0).get() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_abs() {
        let negative = SafeF64::new(-5.0).unwrap();
        assert!((negative.abs().get() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_min_max() {
        let a = SafeF64::new(3.0).unwrap();
        let b = SafeF64::new(7.0).unwrap();

        assert!((a.min(b).get() - 3.0).abs() < 1e-10);
        assert!((a.max(b).get() - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_neg() {
        let x = SafeF64::new(5.0).unwrap();
        assert!((-x).get() + 5.0 < 1e-10);
    }

    #[test]
    fn test_safe_f64_is_near_zero() {
        let small = SafeF64::new(1e-15).unwrap();
        assert!(small.is_near_zero(1e-14));
        assert!(!small.is_near_zero(1e-16));
    }

    #[test]
    fn test_safe_f64_approx_eq() {
        let a = SafeF64::new(1.0).unwrap();
        let b = SafeF64::new(1.0 + 1e-15).unwrap();
        assert!(a.approx_eq(b, 1e-14));
    }

    #[test]
    fn test_safe_f64_assign_ops() {
        let mut x = SafeF64::new(10.0).unwrap();
        x += SafeF64::new(5.0).unwrap();
        assert!((x.get() - 15.0).abs() < 1e-10);

        x -= SafeF64::new(3.0).unwrap();
        assert!((x.get() - 12.0).abs() < 1e-10);

        x *= SafeF64::new(2.0).unwrap();
        assert!((x.get() - 24.0).abs() < 1e-10);

        x /= SafeF64::new(4.0).unwrap();
        assert!((x.get() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_f64_display() {
        let x = SafeF64::new(42.5).unwrap();
        assert_eq!(format!("{}", x), "42.5");
    }

    #[test]
    fn test_safe_f64_serialization() {
        let x = SafeF64::new(42.5).unwrap();
        let json = serde_json::to_string(&x).unwrap();
        let deserialized: SafeF64 = serde_json::from_str(&json).unwrap();
        assert_eq!(x, deserialized);
    }

    #[test]
    fn test_helper_functions() {
        assert!((safe_div(10.0, 2.0, 0.0) - 5.0).abs() < 1e-10);
        assert!((safe_div(10.0, 0.0, -1.0) - (-1.0)).abs() < 1e-10);
        assert!((safe_sqrt(4.0) - 2.0).abs() < 1e-10);
        assert!((safe_sqrt(-4.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_is_valid_f64() {
        assert!(is_valid_f64(1.0));
        assert!(!is_valid_f64(f64::NAN));
        assert!(!is_valid_f64(f64::INFINITY));
    }

    #[test]
    fn test_clamp_valid() {
        assert!((clamp_valid(5.0, 0.0, 10.0, -1.0) - 5.0).abs() < 1e-10);
        assert!((clamp_valid(15.0, 0.0, 10.0, -1.0) - 10.0).abs() < 1e-10);
        assert!((clamp_valid(f64::NAN, 0.0, 10.0, -1.0) - (-1.0)).abs() < 1e-10);
    }
}
