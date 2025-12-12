// crates/mh_core/src/tolerance.rs

//! 泛型化容差配置
//!
//! 提供精度相关的数值容差配置，替代全局静态变量。
//!
//! # 设计原则
//!
//! 1. **无全局状态**: 容差作为参数传递
//! 2. **泛型化**: 容差值与计算精度匹配
//! 3. **精度适配**: f32和f64有不同的默认值

use crate::scalar::Scalar;
use serde::{Deserialize, Serialize};
use std::fmt;

/// 数值容差配置（泛型化）
///
/// # 示例
///
/// ```
/// use mh_core::{Tolerance, Scalar};
///
/// // 使用默认容差
/// let tol_f32 = Tolerance::<f32>::default();
/// let tol_f64 = Tolerance::<f64>::default();
///
/// // 检查水深是否为干
/// fn is_dry<S: Scalar>(h: S, tol: &Tolerance<S>) -> bool {
///     h < tol.h_dry
/// }
/// ```
#[derive(Clone)]
pub struct Tolerance<S: Scalar> {
    /// 机器epsilon的倍数
    pub epsilon: S,
    /// 最小水深 [m]
    pub h_min: S,
    /// 干单元水深阈值 [m]
    pub h_dry: S,
    /// 速度上限 [m/s]
    pub velocity_cap: S,
    /// 时间相对容差
    pub time_rel: S,
    /// 时间绝对容差
    pub time_abs: S,
    /// 空间容差
    pub spatial: S,
    /// 安全除法阈值
    pub safe_div: S,
    /// 迭代收敛容差
    pub convergence: S,
    /// 梯度计算容差
    pub gradient_eps: S,
    /// 最小面积 [m²]
    pub min_area: S,
}

impl<S: Scalar> fmt::Debug for Tolerance<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tolerance")
            .field("h_min", &self.h_min.to_f64())
            .field("h_dry", &self.h_dry.to_f64())
            .field("velocity_cap", &self.velocity_cap.to_f64())
            .finish()
    }
}

// f32 特化默认值
impl Default for Tolerance<f32> {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            h_min: 1e-4,
            h_dry: 1e-3,
            velocity_cap: 100.0,
            time_rel: 1e-6,
            time_abs: 1e-8,
            spatial: 1e-6,
            safe_div: 1e-8,
            convergence: 1e-5,
            gradient_eps: 1e-6,
            min_area: 1e-6,
        }
    }
}

// f64 特化默认值
impl Default for Tolerance<f64> {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            h_min: 1e-9,
            h_dry: 1e-6,
            velocity_cap: 1000.0,
            time_rel: 1e-12,
            time_abs: 1e-14,
            spatial: 1e-14,
            safe_div: 1e-14,
            convergence: 1e-10,
            gradient_eps: 1e-12,
            min_area: 1e-12,
        }
    }
}

impl<S: Scalar> Tolerance<S> {
    /// 创建宽松容差（用于f32或快速计算）
    pub fn relaxed() -> Self
    where
        Self: Default,
    {
        let mut tol = Self::default();
        // 放宽容差
        tol.convergence = tol.convergence * Scalar::from_f64(10.0);
        tol.h_dry = tol.h_dry * Scalar::from_f64(10.0);
        tol
    }

    /// 创建严格容差（用于高精度计算）
    pub fn strict() -> Self
    where
        Self: Default,
    {
        let mut tol = Self::default();
        // 收紧容差
        tol.convergence = tol.convergence * Scalar::from_f64(0.1);
        tol.h_dry = tol.h_dry * Scalar::from_f64(0.1);
        tol
    }

    /// 判断水深是否为干
    #[inline]
    pub fn is_dry(&self, h: S) -> bool {
        h < self.h_dry
    }

    /// 判断水深是否为湿
    #[inline]
    pub fn is_wet(&self, h: S) -> bool {
        h >= self.h_dry
    }

    /// 获取安全水深（至少为h_min）
    #[inline]
    pub fn safe_depth(&self, h: S) -> S {
        if h < self.h_min {
            self.h_min
        } else {
            h
        }
    }

    /// 安全除法
    #[inline]
    pub fn safe_divide(&self, numerator: S, denominator: S) -> S {
        if denominator.abs() < self.safe_div {
            S::ZERO
        } else {
            numerator / denominator
        }
    }

    /// 判断时间是否接近
    #[inline]
    pub fn is_time_close(&self, a: S, b: S) -> bool {
        let diff = (a - b).abs();
        let scale = a.abs().max(b.abs()).max(S::ONE);
        diff < self.time_abs || diff < self.time_rel * scale
    }

    /// 判断空间值是否接近零
    #[inline]
    pub fn is_spatial_zero(&self, x: S) -> bool {
        x.abs() < self.spatial
    }

    /// 判断迭代是否收敛
    #[inline]
    pub fn is_converged(&self, residual: S, initial: S) -> bool {
        residual < self.convergence * initial.max(S::ONE)
    }

    /// 钳制速度
    #[inline]
    pub fn clamp_velocity(&self, v: S) -> S {
        if v.abs() > self.velocity_cap {
            if v > S::ZERO {
                self.velocity_cap
            } else {
                -self.velocity_cap
            }
        } else {
            v
        }
    }
}

/// 可序列化的容差配置（使用f64）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToleranceConfig {
    /// 最小水深 [m]
    #[serde(default = "default_h_min")]
    pub h_min: f64,
    /// 干单元水深阈值 [m]
    #[serde(default = "default_h_dry")]
    pub h_dry: f64,
    /// 速度上限 [m/s]
    #[serde(default = "default_velocity_cap")]
    pub velocity_cap: f64,
    /// 收敛容差
    #[serde(default = "default_convergence")]
    pub convergence: f64,
}

fn default_h_min() -> f64 { 1e-9 }
fn default_h_dry() -> f64 { 1e-6 }
fn default_velocity_cap() -> f64 { 1000.0 }
fn default_convergence() -> f64 { 1e-10 }

impl Default for ToleranceConfig {
    fn default() -> Self {
        Self {
            h_min: default_h_min(),
            h_dry: default_h_dry(),
            velocity_cap: default_velocity_cap(),
            convergence: default_convergence(),
        }
    }
}

impl ToleranceConfig {
    /// 转换为泛型Tolerance
    pub fn to_tolerance<S: Scalar>(&self) -> Tolerance<S>
    where
        Tolerance<S>: Default,
    {
        let mut tol = Tolerance::<S>::default();
        tol.h_min = Scalar::from_f64(self.h_min);
        tol.h_dry = Scalar::from_f64(self.h_dry);
        tol.velocity_cap = Scalar::from_f64(self.velocity_cap);
        tol.convergence = Scalar::from_f64(self.convergence);
        tol
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tolerance_default() {
        let tol_f32 = Tolerance::<f32>::default();
        let tol_f64 = Tolerance::<f64>::default();

        // f32容差应该更宽松
        assert!(tol_f32.h_min > tol_f64.h_min as f32);
        assert!(tol_f32.h_dry > tol_f64.h_dry as f32);
    }

    #[test]
    fn test_is_dry() {
        let tol = Tolerance::<f64>::default();
        assert!(tol.is_dry(1e-8));
        assert!(!tol.is_dry(1e-4));
    }

    #[test]
    fn test_safe_depth() {
        let tol = Tolerance::<f64>::default();
        assert_eq!(tol.safe_depth(1e-12), tol.h_min);
        assert_eq!(tol.safe_depth(1.0), 1.0);
    }

    #[test]
    fn test_clamp_velocity() {
        let tol = Tolerance::<f64>::default();
        assert_eq!(tol.clamp_velocity(50.0), 50.0);
        assert_eq!(tol.clamp_velocity(2000.0), tol.velocity_cap);
        assert_eq!(tol.clamp_velocity(-2000.0), -tol.velocity_cap);
    }

    #[test]
    fn test_tolerance_config() {
        let config = ToleranceConfig::default();
        let tol: Tolerance<f64> = config.to_tolerance();
        assert_eq!(tol.h_min, config.h_min);
    }
}
