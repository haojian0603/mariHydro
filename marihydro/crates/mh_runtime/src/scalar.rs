// crates/mh_runtime/src/scalar.rs
#![allow(clippy::items_after_test_module)]

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
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// 目标平台必须具备基础原子指令，否则并行模块无法安全工作
#[cfg(not(any(target_has_atomic = "32", target_has_atomic = "64")))]
compile_error!("目标平台缺少 32/64 位原子支持，无法构建并行运行时");

use bytemuck::Pod;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};

/// 密封模块，禁止外部实现
mod private {
    /// 密封 trait
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// f32 对应的原子封装（基于 AtomicU32 按位存储）
#[cfg(target_has_atomic = "32")]
pub struct AtomicF32(AtomicU32);

/// f64 对应的原子封装（基于 AtomicU64 按位存储）
#[cfg(target_has_atomic = "64")]
pub struct AtomicF64(AtomicU64);

/// 浮点原子封装
///
/// 为运行时标量提供与精度一致的原子操作封装，
/// 通过位模式在 `AtomicU32/AtomicU64` 上实现无锁加法与最大值更新。
pub trait AtomicScalar<S: RuntimeScalar>: Send + Sync + 'static {
    /// 创建新原子值
    fn new(value: S) -> Self;

    /// 加载当前值
    fn load(&self, order: Ordering) -> S;

    /// 存储值
    fn store(&self, value: S, order: Ordering);

    /// 原子加法，返回旧值
    fn fetch_add(&self, value: S, order: Ordering) -> S;

    /// 原子最大值更新，返回旧值
    fn fetch_max(&self, value: S, order: Ordering) -> S;

    /// 提取内部值（用于测试或调试）
    fn into_inner(self) -> S;
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
    + ToPrimitive
    + NumAssign
    + PartialOrd
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
    /// 与标量对应的原子类型
    type Atomic: AtomicScalar<Self>;
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
        if !self.is_finite() || !rhs.is_finite() {
            return fallback;
        }
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
        if !self.is_finite() || !rhs.is_finite() || !eps.is_finite() {
            return fallback;
        }
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

    /// 返回较大值
    #[inline]
    fn max_value(self, other: Self) -> Self {
        if self > other { self } else { other }
    }

    /// 返回较小值
    #[inline]
    fn min_value(self, other: Self) -> Self {
        if self < other { self } else { other }
    }

    /// 转换为 f64（允许损失精度，用于日志/序列化）
    fn to_f64_lossy(self) -> f64;

    /// 安全平方根（负数返回 0）
    #[inline]
    fn safe_sqrt(self) -> Self {
        if !self.is_finite() || self < Self::ZERO {
            Self::ZERO
        } else {
            self.sqrt()
        }
    }

    /// 安全幂运算（负数非整数次幂返回 0）
    #[inline]
    fn safe_powf(self, exp: Self) -> Self {
        if !self.is_finite() || !exp.is_finite() {
            return Self::ZERO;
        }
        if self < Self::ZERO && (exp.fract() != Self::ZERO) {
            Self::ZERO
        } else {
            self.powf(exp)
        }
    }

    /// 安全自然对数（非正数返回 0）
    #[inline]
    fn safe_ln(self) -> Self {
        if !self.is_finite() || self <= Self::ZERO {
            Self::ZERO
        } else {
            self.ln()
        }
    }

    /// 安全正弦（大数周期归约）
    #[inline]
    fn sin_safe(self) -> Self {
        if !self.is_finite() {
            return Self::ZERO;
        }
        let threshold = Self::from_config_or_panic(1e15, "RuntimeScalar::sin_safe.threshold");
        if self.abs() > threshold {
            let reduced = self
                % Self::from_config_or_panic(
                    2.0 * std::f64::consts::PI,
                    "RuntimeScalar::sin_safe.period",
                );
            reduced.sin()
        } else {
            self.sin()
        }
    }

    /// 安全余弦（大数周期归约）
    #[inline]
    fn cos_safe(self) -> Self {
        if !self.is_finite() {
            return Self::ONE;
        }
        let threshold = Self::from_config_or_panic(1e15, "RuntimeScalar::cos_safe.threshold");
        if self.abs() > threshold {
            let reduced = self
                % Self::from_config_or_panic(
                    2.0 * std::f64::consts::PI,
                    "RuntimeScalar::cos_safe.period",
                );
            reduced.cos()
        } else {
            self.cos()
        }
    }

    /// 安全正弦余弦对（保证 sin² + cos² ≈ 1）
    #[inline]
    fn sin_cos_safe(self) -> (Self, Self) {
        if !self.is_finite() {
            return (Self::ZERO, Self::ONE);
        }
        let threshold =
            Self::from_config_or_panic(1e15, "RuntimeScalar::sin_cos_safe.threshold");
        let reduced = if self.abs() > threshold {
            self % Self::from_config_or_panic(
                2.0 * std::f64::consts::PI,
                "RuntimeScalar::sin_cos_safe.period",
            )
        } else {
            self
        };
        reduced.sin_cos()
    }

    /// 近似相等判断
    #[inline]
    fn approx_eq(self, other: Self, epsilon: Self) -> bool {
        if !self.is_finite() || !other.is_finite() || !epsilon.is_finite() {
            return false;
        }
        (self - other).abs() < epsilon
    }

    /// 检查是否接近零
    #[inline]
    fn is_near_zero(self, epsilon: Self) -> bool {
        if !self.is_finite() || !epsilon.is_finite() {
            return false;
        }
        self.abs() < epsilon
    }

    /// 从 f64 配置值转换（用于 Layer 4 配置到 Layer 3 引擎的转换）
    ///
    /// 这是 `FromPrimitive::from_f64` 的便捷包装，提供更语义化的接口。
    #[inline]
    fn from_config(value: f64) -> Option<Self> {
        Self::from_f64(value)
    }

    /// 从配置值转换，失败时直接 panic，并携带调用上下文。
    #[inline]
    #[track_caller]
    fn from_config_or_panic(value: f64, context: &'static str) -> Self {
        Self::from_config(value).unwrap_or_else(|| {
            panic!("[mh_runtime::scalar] config scalar conversion failed: context={context}, value={value}")
        })
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

    /// 将标量包装为原子值（默认使用 Relaxed，不跨线程同步时可替换）
    #[inline]
    fn to_atomic(self) -> Self::Atomic {
        <Self::Atomic as AtomicScalar<Self>>::new(self)
    }

    /// 从原子值读取（默认 Relaxed，需更严格顺序时由调用方指定）
    #[inline]
    fn from_atomic(atomic: &Self::Atomic, order: Ordering) -> Self {
        atomic.load(order)
    }
}

// =============================================================================
// f32 实现
// =============================================================================

#[cfg(target_has_atomic = "32")]
impl RuntimeScalar for f32 {
    type Atomic = AtomicF32;
    const ZERO: f32 = 0.0;
    const ONE: f32 = 1.0;
    const TWO: f32 = 2.0;
    const HALF: f32 = 0.5;
    const EPSILON: f32 = f32::EPSILON;
    const MIN_POSITIVE: f32 = f32::MIN_POSITIVE;
    const MAX: f32 = f32::MAX;
    const MIN: f32 = f32::MIN;

    #[inline]
    fn to_f64_lossy(self) -> f64 {
        self as f64
    }
}

#[cfg(not(target_has_atomic = "32"))]
compile_error!("目标平台缺少 32 位原子支持，无法启用 f32 并行标量");

// =============================================================================
// f64 实现
// =============================================================================

#[cfg(target_has_atomic = "64")]
impl RuntimeScalar for f64 {
    type Atomic = AtomicF64;
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const TWO: f64 = 2.0;
    const HALF: f64 = 0.5;
    const EPSILON: f64 = f64::EPSILON;
    const MIN_POSITIVE: f64 = f64::MIN_POSITIVE;
    const MAX: f64 = f64::MAX;
    const MIN: f64 = f64::MIN;

    #[inline]
    fn to_f64_lossy(self) -> f64 {
        self
    }
}

#[cfg(not(target_has_atomic = "64"))]
compile_error!("目标平台缺少 64 位原子支持，无法启用 f64 并行标量");

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

fn atomic_fetch_add_float<T, F>(atomic: &T, value: F, order: Ordering, to_bits: fn(F) -> u64, from_bits: fn(u64) -> F) -> F
where
    T: AtomicInteger,
    F: Copy + Add<Output = F>,
{
    let mut old = atomic.load(order);
    loop {
        let old_val = from_bits(old);
        let new_val = to_bits(old_val + value);
        match atomic.compare_exchange_weak(old, new_val, order, order) {
            Ok(prev) => return from_bits(prev),
            Err(next) => old = next,
        }
    }
}

fn atomic_fetch_max_float<T, F>(atomic: &T, value: F, order: Ordering, to_bits: fn(F) -> u64, from_bits: fn(u64) -> F) -> F
where
    T: AtomicInteger,
    F: PartialOrd + Copy,
{
    let mut old = atomic.load(order);
    loop {
        let old_val = from_bits(old);
        if let Some(ord) = value.partial_cmp(&old_val) {
            match ord {
                std::cmp::Ordering::Less | std::cmp::Ordering::Equal => return old_val,
                std::cmp::Ordering::Greater => {}
            }
        }

        let new_val = to_bits(value);
        match atomic.compare_exchange_weak(old, new_val, order, order) {
            Ok(prev) => return from_bits(prev),
            Err(next) => old = next,
        }
    }
}

/// 内部统一的整数原子接口
trait AtomicInteger {
    fn load(&self, order: Ordering) -> u64;
    fn store(&self, value: u64, order: Ordering);
    fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64>;
}

impl AtomicInteger for AtomicU32 {
    #[inline]
    fn load(&self, order: Ordering) -> u64 {
        u64::from(AtomicU32::load(self, order))
    }

    #[inline]
    fn store(&self, value: u64, order: Ordering) {
        AtomicU32::store(self, value as u32, order);
    }

    #[inline]
    fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        AtomicU32::compare_exchange_weak(self, current as u32, new as u32, success, failure)
            .map(u64::from)
            .map_err(u64::from)
    }
}

impl AtomicInteger for AtomicU64 {
    #[inline]
    fn load(&self, order: Ordering) -> u64 {
        AtomicU64::load(self, order)
    }

    #[inline]
    fn store(&self, value: u64, order: Ordering) {
        AtomicU64::store(self, value, order);
    }

    #[inline]
    fn compare_exchange_weak(
        &self,
        current: u64,
        new: u64,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u64, u64> {
        AtomicU64::compare_exchange_weak(self, current, new, success, failure)
    }
}

#[cfg(target_has_atomic = "32")]
impl AtomicScalar<f32> for AtomicF32 {
    #[inline]
    fn new(value: f32) -> Self {
        Self(AtomicU32::new(value.to_bits()))
    }

    #[inline]
    fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(AtomicInteger::load(&self.0, order) as u32)
    }

    #[inline]
    fn store(&self, value: f32, order: Ordering) {
        AtomicInteger::store(&self.0, value.to_bits() as u64, order);
    }

    #[inline]
    fn fetch_add(&self, value: f32, order: Ordering) -> f32 {
        atomic_fetch_add_float(&self.0, value, order, |v| u64::from(v.to_bits()), |b| f32::from_bits(b as u32))
    }

    #[inline]
    fn fetch_max(&self, value: f32, order: Ordering) -> f32 {
        atomic_fetch_max_float(&self.0, value, order, |v| u64::from(v.to_bits()), |b| f32::from_bits(b as u32))
    }

    #[inline]
    fn into_inner(self) -> f32 {
        f32::from_bits(self.0.into_inner())
    }
}

#[cfg(target_has_atomic = "64")]
impl AtomicScalar<f64> for AtomicF64 {
    #[inline]
    fn new(value: f64) -> Self {
        Self(AtomicU64::new(value.to_bits()))
    }

    #[inline]
    fn load(&self, order: Ordering) -> f64 {
        f64::from_bits(AtomicInteger::load(&self.0, order))
    }

    #[inline]
    fn store(&self, value: f64, order: Ordering) {
        AtomicInteger::store(&self.0, value.to_bits(), order);
    }

    #[inline]
    fn fetch_add(&self, value: f64, order: Ordering) -> f64 {
        atomic_fetch_add_float(&self.0, value, order, f64::to_bits, f64::from_bits)
    }

    #[inline]
    fn fetch_max(&self, value: f64, order: Ordering) -> f64 {
        atomic_fetch_max_float(&self.0, value, order, f64::to_bits, f64::from_bits)
    }

    #[inline]
    fn into_inner(self) -> f64 {
        f64::from_bits(self.0.into_inner())
    }
}
