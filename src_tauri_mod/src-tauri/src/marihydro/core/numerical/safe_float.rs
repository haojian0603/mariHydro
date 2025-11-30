//! 安全浮点数类型
//!
//! 提供 `SafeF64` 类型，保证在构造时验证数值有效性。
//! 使用 NewType 模式包装 f64，解决 NaN/Inf 污染问题。

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// 安全浮点数：保证非NaN非Inf
///
/// 使用 `#[repr(transparent)]` 保证与 f64 布局相同，实现零开销抽象。
///
/// # Example
/// ```rust
/// use marihydro::core::numerical::SafeF64;
///
/// // 正常值
/// let a = SafeF64::new(1.0).unwrap();
/// let b = SafeF64::new(2.0).unwrap();
///
/// // NaN 值返回 None
/// assert!(SafeF64::new(f64::NAN).is_none());
///
/// // 安全除法
/// let c = a.safe_div(b);
/// assert!(c.is_some());
/// ```
#[derive(Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SafeF64(f64);

impl SafeF64 {
    /// 安全除法的最小分母阈值
    pub const EPSILON: f64 = 1e-14;

    /// 零值
    pub const ZERO: SafeF64 = SafeF64(0.0);

    /// 一
    pub const ONE: SafeF64 = SafeF64(1.0);

    /// 尝试从 f64 创建，NaN/Inf 返回 None
    #[inline]
    pub fn new(value: f64) -> Option<Self> {
        if value.is_finite() {
            Some(Self(value))
        } else {
            None
        }
    }

    /// 从 f64 创建，非法值替换为默认值
    #[inline]
    pub fn new_or(value: f64, default: f64) -> Self {
        if value.is_finite() {
            Self(value)
        } else {
            debug_assert!(default.is_finite(), "default value must be finite");
            Self(default)
        }
    }

    /// 从 f64 创建，非法值替换为零
    #[inline]
    pub fn new_or_zero(value: f64) -> Self {
        Self::new_or(value, 0.0)
    }

    /// 安全除法：除零或结果非有限时返回 None
    #[inline]
    pub fn safe_div(self, rhs: Self) -> Option<Self> {
        if rhs.0.abs() < Self::EPSILON {
            None
        } else {
            Self::new(self.0 / rhs.0)
        }
    }

    /// 安全除法：除零或结果非有限时返回默认值
    #[inline]
    pub fn safe_div_or(self, rhs: Self, default: f64) -> Self {
        self.safe_div(rhs).unwrap_or(Self::new_or(default, 0.0))
    }

    /// 安全开方：负数返回 None
    #[inline]
    pub fn safe_sqrt(self) -> Option<Self> {
        if self.0 >= 0.0 {
            Self::new(self.0.sqrt())
        } else {
            None
        }
    }

    /// 安全开方：负数返回零
    #[inline]
    pub fn safe_sqrt_or_zero(self) -> Self {
        self.safe_sqrt().unwrap_or(Self::ZERO)
    }

    /// 安全对数：非正数返回 None
    #[inline]
    pub fn safe_ln(self) -> Option<Self> {
        if self.0 > 0.0 {
            Self::new(self.0.ln())
        } else {
            None
        }
    }

    /// 安全幂运算
    #[inline]
    pub fn safe_powf(self, exp: Self) -> Option<Self> {
        Self::new(self.0.powf(exp.0))
    }

    /// 限制值在范围内
    #[inline]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    /// 取绝对值
    #[inline]
    pub fn abs(self) -> Self {
        Self(self.0.abs())
    }

    /// 取最大值
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    /// 取最小值
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    /// 检查是否为正
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 > 0.0
    }

    /// 检查是否为负
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 < 0.0
    }

    /// 检查是否接近零
    #[inline]
    pub fn is_near_zero(self) -> bool {
        self.0.abs() < Self::EPSILON
    }

    /// 转换为可比较的 u64 位模式（用于原子操作）
    ///
    /// 正数直接转换，负数需要特殊处理以保持排序性
    #[inline]
    pub fn to_comparable_bits(self) -> u64 {
        let bits = self.0.to_bits();
        if self.0.is_sign_negative() {
            !bits // 翻转使负数可比较
        } else {
            bits | (1u64 << 63) // 确保正数大于负数
        }
    }

    /// 从可比较的位模式恢复
    #[inline]
    pub fn from_comparable_bits(bits: u64) -> Option<Self> {
        let value = if bits & (1u64 << 63) != 0 {
            f64::from_bits(bits & !(1u64 << 63))
        } else {
            f64::from_bits(!bits)
        };
        Self::new(value)
    }

    /// 获取内部值
    #[inline]
    pub fn get(self) -> f64 {
        self.0
    }

    /// 获取内部值引用
    #[inline]
    pub fn as_f64(&self) -> f64 {
        self.0
    }
}

// 实现 From<f64>，会在非有限值时 panic（用于已知安全的值）
impl TryFrom<f64> for SafeF64 {
    type Error = &'static str;

    fn try_from(value: f64) -> Result<Self, Self::Error> {
        Self::new(value).ok_or("SafeF64: value must be finite")
    }
}

// From SafeF64 to f64
impl From<SafeF64> for f64 {
    #[inline]
    fn from(safe: SafeF64) -> f64 {
        safe.0
    }
}

// 算术运算实现 - 返回 Option 以处理可能的溢出
impl Add for SafeF64 {
    type Output = Option<Self>;

    #[inline]
    fn add(self, rhs: Self) -> Option<Self> {
        Self::new(self.0 + rhs.0)
    }
}

impl Sub for SafeF64 {
    type Output = Option<Self>;

    #[inline]
    fn sub(self, rhs: Self) -> Option<Self> {
        Self::new(self.0 - rhs.0)
    }
}

impl Mul for SafeF64 {
    type Output = Option<Self>;

    #[inline]
    fn mul(self, rhs: Self) -> Option<Self> {
        Self::new(self.0 * rhs.0)
    }
}

impl Div for SafeF64 {
    type Output = Option<Self>;

    #[inline]
    fn div(self, rhs: Self) -> Option<Self> {
        self.safe_div(rhs)
    }
}

impl Neg for SafeF64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

// 实现 Debug 和 Display
impl fmt::Debug for SafeF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SafeF64({})", self.0)
    }
}

impl fmt::Display for SafeF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// 实现默认值
impl Default for SafeF64 {
    fn default() -> Self {
        Self::ZERO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        assert!(SafeF64::new(1.0).is_some());
        assert!(SafeF64::new(-1.0).is_some());
        assert!(SafeF64::new(0.0).is_some());
    }

    #[test]
    fn test_new_invalid() {
        assert!(SafeF64::new(f64::NAN).is_none());
        assert!(SafeF64::new(f64::INFINITY).is_none());
        assert!(SafeF64::new(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn test_safe_div() {
        let a = SafeF64::new(4.0).unwrap();
        let b = SafeF64::new(2.0).unwrap();
        let zero = SafeF64::ZERO;

        assert_eq!(a.safe_div(b).unwrap().get(), 2.0);
        assert!(a.safe_div(zero).is_none());
    }

    #[test]
    fn test_safe_sqrt() {
        let positive = SafeF64::new(4.0).unwrap();
        let negative = SafeF64::new(-4.0).unwrap();

        assert_eq!(positive.safe_sqrt().unwrap().get(), 2.0);
        assert!(negative.safe_sqrt().is_none());
    }

    #[test]
    fn test_arithmetic() {
        let a = SafeF64::new(3.0).unwrap();
        let b = SafeF64::new(2.0).unwrap();

        assert_eq!((a + b).unwrap().get(), 5.0);
        assert_eq!((a - b).unwrap().get(), 1.0);
        assert_eq!((a * b).unwrap().get(), 6.0);
        assert_eq!((a / b).unwrap().get(), 1.5);
    }
}
