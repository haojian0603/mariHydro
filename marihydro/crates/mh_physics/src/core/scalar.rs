// marihydro\crates\mh_physics\src\core\scalar.rs
//! 统一标量类型抽象 - 项目唯一权威定义
//!
//! 提供 f32/f64 的统一接口，支持编译期精度选择。
//!
//! # 物理常量
//!
//! Scalar trait 包含物理模拟所需的关联常量：
//! - `ZERO`, `ONE`: 基本常量
//! - `GRAVITY`: 重力加速度 (9.81 m/s²)
//! - `PI`: 圆周率
//! - `VON_KARMAN`: 冯卡门常数 (0.41)
//! - `WATER_DENSITY`: 水密度 (1000 kg/m³)
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::core::Scalar;
//!
//! fn compute_wave_speed<S: Scalar>(h: S) -> S {
//!     (S::GRAVITY * h).sqrt()
//! }
//! ```

use bytemuck::Pod;
use num_traits::{Float, FromPrimitive, NumAssign};
use std::fmt::{Debug, Display};
use std::iter::Sum;

/// 统一标量类型约束 - 项目唯一权威定义
///
/// 所有物理计算应使用此 trait 作为泛型约束。
/// 提供 f32 和 f64 的统一接口以及物理常量。
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
    // ========== 基本常量 ==========
    
    /// 零值
    const ZERO: Self;
    /// 单位值
    const ONE: Self;
    /// 机器精度常量
    const EPSILON: Self;
    /// 圆周率
    const PI: Self;
    
    // ========== 物理常量 ==========
    
    /// 重力加速度 [m/s²]
    const GRAVITY: Self;
    /// 冯卡门常数（对数速度剖面）
    const VON_KARMAN: Self;
    /// 标准水密度 [kg/m³]
    const WATER_DENSITY: Self;
    /// 标准海水密度 [kg/m³]
    const SEAWATER_DENSITY: Self;
    /// 空气密度 [kg/m³]
    const AIR_DENSITY: Self;
    
    // ========== 类型方法 ==========
    
    /// 类型名称
    fn type_name() -> &'static str;
    
    /// 机器精度（方法版本，与 EPSILON 常量相同）
    fn epsilon() -> Self;
    
    /// 最小正规数
    fn min_positive() -> Self;
    
    /// 负无穷大
    fn neg_infinity() -> Self;
    
    /// 正无穷大
    fn pos_infinity() -> Self;
    
    /// 从 f64 转换
    fn from_f64(v: f64) -> Self;
    
    /// 转换为 f64
    fn to_f64(self) -> f64;
    
    // ========== 数学运算 ==========
    
    /// 平方根
    fn sqrt(self) -> Self;
    
    /// 绝对值
    fn abs(self) -> Self;
    
    /// 最大值
    fn max(self, other: Self) -> Self;
    
    /// 最小值
    fn min(self, other: Self) -> Self;
    
    /// 钳位到范围 [min, max]
    fn clamp(self, min: Self, max: Self) -> Self;
    
    /// 幂运算 self^n
    fn powf(self, n: Self) -> Self;
    
    /// 自然指数 e^self
    fn exp(self) -> Self;
    
    /// 自然对数 ln(self)
    fn ln(self) -> Self;
    
    /// 正弦
    fn sin(self) -> Self;
    
    /// 余弦
    fn cos(self) -> Self;
    
    /// 反正切（双参数）
    fn atan2(self, other: Self) -> Self;
    
    /// 是否有限
    fn is_finite(self) -> bool;
    
    /// 是否为 NaN
    fn is_nan(self) -> bool;
    
    /// 符号函数: -1, 0, 1
    fn signum(self) -> Self;
    
    /// 向下取整
    fn floor(self) -> Self;
    
    /// 向上取整
    fn ceil(self) -> Self;
}

impl Scalar for f32 {
    // 基本常量
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const EPSILON: f32 = 1e-6;
    const PI: f32 = std::f32::consts::PI;
    
    // 物理常量
    const GRAVITY: f32 = 9.81;
    const VON_KARMAN: f32 = 0.41;
    const WATER_DENSITY: f32 = 1000.0;
    const SEAWATER_DENSITY: f32 = 1025.0;
    const AIR_DENSITY: f32 = 1.225;
    
    fn type_name() -> &'static str { "f32" }
    fn epsilon() -> Self { Self::EPSILON }
    fn min_positive() -> Self { f32::MIN_POSITIVE }
    fn neg_infinity() -> Self { f32::NEG_INFINITY }
    fn pos_infinity() -> Self { f32::INFINITY }
    fn from_f64(v: f64) -> Self { v as f32 }
    fn to_f64(self) -> f64 { self as f64 }
    fn sqrt(self) -> Self { f32::sqrt(self) }
    fn abs(self) -> Self { f32::abs(self) }
    fn max(self, other: Self) -> Self { f32::max(self, other) }
    fn min(self, other: Self) -> Self { f32::min(self, other) }
    fn clamp(self, min: Self, max: Self) -> Self { f32::clamp(self, min, max) }
    fn powf(self, n: Self) -> Self { f32::powf(self, n) }
    fn exp(self) -> Self { f32::exp(self) }
    fn ln(self) -> Self { f32::ln(self) }
    fn sin(self) -> Self { f32::sin(self) }
    fn cos(self) -> Self { f32::cos(self) }
    fn atan2(self, other: Self) -> Self { f32::atan2(self, other) }
    fn is_finite(self) -> bool { f32::is_finite(self) }
    fn is_nan(self) -> bool { f32::is_nan(self) }
    fn signum(self) -> Self { f32::signum(self) }
    fn floor(self) -> Self { f32::floor(self) }
    fn ceil(self) -> Self { f32::ceil(self) }
}

impl Scalar for f64 {
    // 基本常量
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const EPSILON: f64 = 1e-12;
    const PI: f64 = std::f64::consts::PI;
    
    // 物理常量
    const GRAVITY: f64 = 9.81;
    const VON_KARMAN: f64 = 0.41;
    const WATER_DENSITY: f64 = 1000.0;
    const SEAWATER_DENSITY: f64 = 1025.0;
    const AIR_DENSITY: f64 = 1.225;
    
    fn type_name() -> &'static str { "f64" }
    fn epsilon() -> Self { Self::EPSILON }
    fn min_positive() -> Self { f64::MIN_POSITIVE }
    fn neg_infinity() -> Self { f64::NEG_INFINITY }
    fn pos_infinity() -> Self { f64::INFINITY }
    fn from_f64(v: f64) -> Self { v }
    fn to_f64(self) -> f64 { self }
    fn sqrt(self) -> Self { f64::sqrt(self) }
    fn abs(self) -> Self { f64::abs(self) }
    fn max(self, other: Self) -> Self { f64::max(self, other) }
    fn min(self, other: Self) -> Self { f64::min(self, other) }
    fn clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }
    fn powf(self, n: Self) -> Self { f64::powf(self, n) }
    fn exp(self) -> Self { f64::exp(self) }
    fn ln(self) -> Self { f64::ln(self) }
    fn sin(self) -> Self { f64::sin(self) }
    fn cos(self) -> Self { f64::cos(self) }
    fn atan2(self, other: Self) -> Self { f64::atan2(self, other) }
    fn is_finite(self) -> bool { f64::is_finite(self) }
    fn is_nan(self) -> bool { f64::is_nan(self) }
    fn signum(self) -> Self { f64::signum(self) }
    fn floor(self) -> Self { f64::floor(self) }
    fn ceil(self) -> Self { f64::ceil(self) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_f32_basic() {
        let x: f32 = Scalar::from_f64(1.234);
        assert!((x - 1.234f32).abs() < 1e-5);
        assert!((f32::ZERO - 0.0f32).abs() < 1e-10);
        assert!((f32::ONE - 1.0f32).abs() < 1e-10);
    }

    #[test]
    fn test_scalar_f64_basic() {
        let x: f64 = Scalar::from_f64(1.234);
        assert!((x - 1.234f64).abs() < 1e-14);
        assert!((f64::ZERO - 0.0f64).abs() < 1e-14);
        assert!((f64::ONE - 1.0f64).abs() < 1e-14);
    }
    
    #[test]
    fn test_physical_constants_f64() {
        assert!((f64::GRAVITY - 9.81).abs() < 1e-10);
        assert!((f64::VON_KARMAN - 0.41).abs() < 1e-10);
        assert!((f64::WATER_DENSITY - 1000.0).abs() < 1e-10);
        assert!((f64::SEAWATER_DENSITY - 1025.0).abs() < 1e-10);
        assert!((f64::AIR_DENSITY - 1.225).abs() < 1e-10);
        assert!((f64::PI - std::f64::consts::PI).abs() < 1e-14);
    }
    
    #[test]
    fn test_physical_constants_f32() {
        assert!((f32::GRAVITY - 9.81f32).abs() < 1e-5);
        assert!((f32::VON_KARMAN - 0.41f32).abs() < 1e-5);
        assert!((f32::WATER_DENSITY - 1000.0f32).abs() < 1e-3);
    }
    
    #[test]
    fn test_math_operations() {
        // 测试 f64
        let x: f64 = 4.0;
        assert!((x.sqrt() - 2.0).abs() < 1e-14);
        assert!(((-3.0f64).abs() - 3.0).abs() < 1e-14);
        assert!((1.0f64.exp() - std::f64::consts::E).abs() < 1e-10);
        assert!((std::f64::consts::E.ln() - 1.0).abs() < 1e-14);
        
        // 测试 f32
        let y: f32 = 9.0;
        assert!((y.sqrt() - 3.0f32).abs() < 1e-5);
        assert!((2.0f32.powf(3.0f32) - 8.0f32).abs() < 1e-5);
    }
    
    #[test]
    fn test_trigonometric() {
        let angle: f64 = f64::PI / 4.0;
        let sin_val = angle.sin();
        let cos_val = angle.cos();
        assert!((sin_val - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((cos_val - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }
    
    #[test]
    fn test_special_values() {
        assert!(<f64 as Scalar>::neg_infinity() < f64::ZERO);
        assert!(<f64 as Scalar>::pos_infinity() > f64::ZERO);
        assert!(<f64 as Scalar>::neg_infinity().is_infinite());
        assert!(!f64::NAN.is_finite());
        assert!(f64::NAN.is_nan());
    }
    
    #[test]
    fn test_clamp_and_signum() {
        let x: f64 = 5.0;
        assert!((x.clamp(0.0, 3.0) - 3.0).abs() < 1e-14);
        assert!((x.clamp(6.0, 10.0) - 6.0).abs() < 1e-14);
        assert!((-3.5f64).signum() < 0.0);
        assert!((3.5f64).signum() > 0.0);
    }
}
