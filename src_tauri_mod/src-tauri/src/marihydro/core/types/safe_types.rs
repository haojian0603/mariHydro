// src-tauri/src/marihydro/core/types/safe_types.rs

//! 安全包装类型
//!
//! 提供编译期和运行时的类型安全保障，包括：
//! - FiniteF64: 保证有限的浮点数
//! - SafeDepth: 物理安全水深
//! - SafeVelocity: 安全速度
//! - 类型安全索引: CellIndex, FaceIndex, NodeIndex, BoundaryIndex

use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================
// 第一层：有限浮点数（所有物理量的基础）
// ============================================================

/// 保证有限的浮点数（非NaN、非Inf）
///
/// # 用途
///
/// 用于确保物理计算结果的有效性，防止 NaN 或无穷大值污染整个计算域。
///
/// # 示例
///
/// ```
/// use marihydro::core::types::FiniteF64;
///
/// let x = FiniteF64::new(1.0).unwrap();
/// let y = FiniteF64::new(f64::NAN); // 返回 None
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FiniteF64(f64);

impl FiniteF64 {
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
    /// # 安全性
    ///
    /// 调用者必须确保传入的值是有限的，否则在运行时可能导致未定义行为。
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
        if other.0.abs() < f64::EPSILON {
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

    /// 零值
    pub const ZERO: Self = Self(0.0);

    /// 单位值
    pub const ONE: Self = Self(1.0);

    /// 机器精度
    pub const EPSILON: Self = Self(f64::EPSILON);
}

impl Default for FiniteF64 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<FiniteF64> for f64 {
    #[inline]
    fn from(v: FiniteF64) -> f64 {
        v.0
    }
}

impl fmt::Display for FiniteF64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// 算术运算（保持有限性）
impl Add for FiniteF64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new_or(self.0 + rhs.0, 0.0)
    }
}

impl Sub for FiniteF64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new_or(self.0 - rhs.0, 0.0)
    }
}

impl Mul for FiniteF64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new_or(self.0 * rhs.0, 0.0)
    }
}

impl Div for FiniteF64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.safe_div(rhs, 0.0)
    }
}

impl Mul<f64> for FiniteF64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self::new_or(self.0 * rhs, 0.0)
    }
}

impl Div<f64> for FiniteF64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self {
        self.safe_div_with_threshold(rhs, f64::EPSILON, 0.0)
    }
}

impl AddAssign for FiniteF64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for FiniteF64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for FiniteF64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for FiniteF64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// 非有限值错误
#[derive(Debug, Clone)]
pub struct NonFiniteError {
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

// ============================================================
// 第二层：物理安全水深
// ============================================================

/// 安全水深（保证 >= h_min）
///
/// # 用途
///
/// 在浅水方程计算中，避免除以接近零的水深导致数值不稳定。
///
/// # 物理意义
///
/// - `value`: 安全后的水深值（用于计算）
/// - `h_min`: 最小水深阈值
/// - 原始水深低于阈值时，使用阈值代替
#[derive(Debug, Clone, Copy)]
pub struct SafeDepth {
    value: f64,
    h_min: f64,
}

impl SafeDepth {
    /// 从原始水深创建
    #[inline]
    pub fn new(h: f64, h_min: f64) -> Self {
        Self {
            value: h.max(h_min),
            h_min,
        }
    }

    /// 获取安全值（用于除法等运算）
    #[inline]
    pub fn get(self) -> f64 {
        self.value
    }

    /// 获取 h_min 阈值
    #[inline]
    pub fn h_min(self) -> f64 {
        self.h_min
    }

    /// 判断原始水深是否干（低于干湿阈值）
    #[inline]
    pub fn is_originally_dry(h: f64, h_dry: f64) -> bool {
        h < h_dry
    }

    /// 安全除法
    #[inline]
    pub fn safe_divide(numerator: f64, h: f64, h_min: f64) -> f64 {
        numerator / Self::new(h, h_min).get()
    }

    /// 计算安全速度分量
    #[inline]
    pub fn velocity_component(momentum: f64, h: f64, h_min: f64) -> f64 {
        if h < h_min {
            0.0
        } else {
            momentum / h.max(h_min)
        }
    }
}

impl Default for SafeDepth {
    fn default() -> Self {
        Self {
            value: 1e-9,
            h_min: 1e-9,
        }
    }
}

// ============================================================
// 第三层：安全速度
// ============================================================

/// 安全速度（避免除零导致的无穷大）
///
/// # 用途
///
/// 从动量和水深计算速度时，处理干单元（h ≈ 0）的情况。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SafeVelocity {
    /// x 方向速度 [m/s]
    pub u: f64,
    /// y 方向速度 [m/s]
    pub v: f64,
}

impl SafeVelocity {
    /// 零速度
    pub const ZERO: Self = Self { u: 0.0, v: 0.0 };

    /// 从动量和水深计算速度
    ///
    /// # 参数
    ///
    /// - `hu`, `hv`: 动量分量 [m²/s]
    /// - `h`: 水深 [m]
    /// - `h_dry`: 干湿判断阈值 [m]
    /// - `h_min`: 最小计算水深 [m]
    #[inline]
    pub fn from_momentum(hu: f64, hv: f64, h: f64, h_dry: f64, h_min: f64) -> Self {
        if SafeDepth::is_originally_dry(h, h_dry) {
            Self::ZERO
        } else {
            let h_safe = SafeDepth::new(h, h_min);
            Self {
                u: hu / h_safe.get(),
                v: hv / h_safe.get(),
            }
        }
    }

    /// 从分量创建
    #[inline]
    pub const fn new(u: f64, v: f64) -> Self {
        Self { u, v }
    }

    /// 从 DVec2 创建
    #[inline]
    pub fn from_vec(v: DVec2) -> Self {
        Self { u: v.x, v: v.y }
    }

    /// 速度大小
    #[inline]
    pub fn speed(self) -> f64 {
        (self.u * self.u + self.v * self.v).sqrt()
    }

    /// 速度平方
    #[inline]
    pub fn speed_squared(self) -> f64 {
        self.u * self.u + self.v * self.v
    }

    /// 转换为 DVec2
    #[inline]
    pub fn as_dvec2(self) -> DVec2 {
        DVec2::new(self.u, self.v)
    }

    /// 法向分量
    #[inline]
    pub fn normal_component(self, normal: DVec2) -> f64 {
        self.u * normal.x + self.v * normal.y
    }

    /// 切向分量
    #[inline]
    pub fn tangent_component(self, normal: DVec2) -> f64 {
        -self.u * normal.y + self.v * normal.x
    }

    /// 旋转到局部坐标系（法向、切向）
    #[inline]
    pub fn to_local(self, normal: DVec2) -> (f64, f64) {
        let un = self.normal_component(normal);
        let ut = self.tangent_component(normal);
        (un, ut)
    }

    /// 从局部坐标系转换回全局
    #[inline]
    pub fn from_local(un: f64, ut: f64, normal: DVec2) -> Self {
        Self {
            u: un * normal.x - ut * normal.y,
            v: un * normal.y + ut * normal.x,
        }
    }

    /// 限制最大速度
    #[inline]
    pub fn clamp_speed(self, max_speed: f64) -> Self {
        let speed = self.speed();
        if speed > max_speed && speed > 1e-14 {
            let factor = max_speed / speed;
            Self {
                u: self.u * factor,
                v: self.v * factor,
            }
        } else {
            self
        }
    }

    /// 检查速度是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.u.is_finite() && self.v.is_finite()
    }

    /// 动能（单位质量）
    #[inline]
    pub fn kinetic_energy_per_mass(self) -> f64 {
        0.5 * self.speed_squared()
    }
}

impl Add for SafeVelocity {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            u: self.u + rhs.u,
            v: self.v + rhs.v,
        }
    }
}

impl Sub for SafeVelocity {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            u: self.u - rhs.u,
            v: self.v - rhs.v,
        }
    }
}

impl Mul<f64> for SafeVelocity {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            u: self.u * rhs,
            v: self.v * rhs,
        }
    }
}

impl fmt::Display for SafeVelocity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.4}, {:.4}) m/s", self.u, self.v)
    }
}

// ============================================================
// 第四层：类型安全索引
// ============================================================
//
// 注意：索引类型已迁移到 core/types/indices.rs
// 为保持向后兼容性，通过 mod.rs 重导出
// 请直接使用 crate::core::types::{CellIndex, FaceIndex, NodeIndex, BoundaryIndex}
//
// 已废弃的类型别名（仅用于旧代码兼容）：
// - CellIndex, FaceIndex, NodeIndex, BoundaryIndex 现在从 indices 模块导入


#[cfg(test)]
mod tests {
    use super::*;

    // ===== FiniteF64 测试 =====

    #[test]
    fn test_finite_f64_creation() {
        assert!(FiniteF64::new(1.0).is_some());
        assert!(FiniteF64::new(f64::NAN).is_none());
        assert!(FiniteF64::new(f64::INFINITY).is_none());
        assert!(FiniteF64::new(f64::NEG_INFINITY).is_none());
    }

    #[test]
    fn test_finite_f64_new_or() {
        let x = FiniteF64::new_or(f64::NAN, 42.0);
        assert!((x.get() - 42.0).abs() < 1e-10);

        let y = FiniteF64::new_or(1.0, 42.0);
        assert!((y.get() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_finite_f64_arithmetic() {
        let a = FiniteF64::new(3.0).unwrap();
        let b = FiniteF64::new(4.0).unwrap();

        assert!((a + b).get() - 7.0 < 1e-10);
        assert!((b - a).get() - 1.0 < 1e-10);
        assert!((a * b).get() - 12.0 < 1e-10);
    }

    #[test]
    fn test_finite_f64_safe_div() {
        let a = FiniteF64::new(10.0).unwrap();
        let b = FiniteF64::new(2.0).unwrap();
        let zero = FiniteF64::ZERO;

        assert!((a.safe_div(b, 0.0).get() - 5.0).abs() < 1e-10);
        assert!((a.safe_div(zero, -1.0).get() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_finite_f64_safe_sqrt() {
        let positive = FiniteF64::new(4.0).unwrap();
        let negative = FiniteF64::new(-4.0).unwrap();

        assert!((positive.safe_sqrt().get() - 2.0).abs() < 1e-10);
        assert!((negative.safe_sqrt().get() - 0.0).abs() < 1e-10);
    }

    // ===== SafeDepth 测试 =====

    #[test]
    fn test_safe_depth() {
        let h_min = 1e-6;

        let deep = SafeDepth::new(1.0, h_min);
        assert!((deep.get() - 1.0).abs() < 1e-10);

        let shallow = SafeDepth::new(1e-8, h_min);
        assert!((shallow.get() - h_min).abs() < 1e-10);
    }

    #[test]
    fn test_safe_depth_is_dry() {
        let h_dry = 1e-6;
        assert!(SafeDepth::is_originally_dry(1e-8, h_dry));
        assert!(!SafeDepth::is_originally_dry(1.0, h_dry));
    }

    // ===== SafeVelocity 测试 =====

    #[test]
    fn test_safe_velocity_from_momentum() {
        let h_dry = 1e-6;
        let h_min = 1e-9;

        // 正常情况
        let vel = SafeVelocity::from_momentum(4.0, 6.0, 2.0, h_dry, h_min);
        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);

        // 干单元
        let dry_vel = SafeVelocity::from_momentum(4.0, 6.0, 1e-8, h_dry, h_min);
        assert!((dry_vel.u - 0.0).abs() < 1e-10);
        assert!((dry_vel.v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_velocity_speed() {
        let vel = SafeVelocity::new(3.0, 4.0);
        assert!((vel.speed() - 5.0).abs() < 1e-10);
        assert!((vel.speed_squared() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_velocity_local_transform() {
        let vel = SafeVelocity::new(1.0, 0.0);
        let normal = DVec2::new(1.0, 0.0); // 沿 x 轴

        let (un, ut) = vel.to_local(normal);
        assert!((un - 1.0).abs() < 1e-10);
        assert!((ut - 0.0).abs() < 1e-10);

        let back = SafeVelocity::from_local(un, ut, normal);
        assert!((back.u - vel.u).abs() < 1e-10);
        assert!((back.v - vel.v).abs() < 1e-10);
    }

    #[test]
    fn test_safe_velocity_clamp() {
        let vel = SafeVelocity::new(30.0, 40.0);
        let clamped = vel.clamp_speed(10.0);

        assert!((clamped.speed() - 10.0).abs() < 1e-10);
        // 方向应保持不变
        assert!((clamped.u / clamped.v - 30.0 / 40.0).abs() < 1e-10);
    }

    // ===== 索引类型测试 =====
    // 注意：索引类型测试已移至 indices.rs
}
