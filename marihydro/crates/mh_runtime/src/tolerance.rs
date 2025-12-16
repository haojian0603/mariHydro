// crates/mh_runtime/src/tolerance.rs

//! 泛型容差配置
//!
//! 提供与标量类型关联的数值容差配置，用于控制干湿判断、
//! 安全除法、速度限制等物理约束。

use crate::scalar::RuntimeScalar;

/// 数值容差配置（泛型化）
///
/// 包含所有数值计算中使用的容差阈值。
/// 通过泛型参数 `S` 支持 f32/f64 精度切换。
#[derive(Clone)]
pub struct Tolerance<S: RuntimeScalar> {
    /// 机器精度容差
    pub epsilon: S,
    /// 最小水深 [m]
    pub h_min: S,
    /// 干单元水深阈值 [m]
    pub h_dry: S,
    /// 最大速度限制 [m/s]
    pub velocity_cap: S,
    /// 安全除法阈值
    pub safe_div: S,
    /// 收敛容差
    pub convergence: S,
    /// 梯度计算容差
    pub gradient_eps: S,
    /// 面积最小值 [m²]
    pub min_area: S,
}

impl<S: RuntimeScalar> Tolerance<S> {
    /// 判断是否为干单元
    #[inline]
    pub fn is_dry(&self, h: S) -> bool { 
        h < self.h_dry 
    }

    /// 判断是否为湿单元
    #[inline]
    pub fn is_wet(&self, h: S) -> bool { 
        h >= self.h_dry 
    }

    /// 安全水深（不小于 h_min）
    #[inline]
    pub fn safe_depth(&self, h: S) -> S {
        if h < self.h_min { self.h_min } else { h }
    }

    /// 安全除法
    #[inline]
    pub fn safe_divide(&self, num: S, den: S) -> S {
        if den.abs() < self.safe_div { 
            S::ZERO 
        } else { 
            num / den 
        }
    }

    /// 限制速度范围
    #[inline]
    pub fn clamp_velocity(&self, v: S) -> S {
        if v.abs() > self.velocity_cap {
            v.signum() * self.velocity_cap
        } else {
            v
        }
    }

    /// 检查是否收敛
    #[inline]
    pub fn is_converged(&self, residual: S) -> bool {
        residual < self.convergence
    }

    /// 从 f64 配置创建
    pub fn from_config(
        epsilon: f64,
        h_min: f64,
        h_dry: f64,
        velocity_cap: f64,
        safe_div: f64,
        convergence: f64,
    ) -> Option<Self> {
        Some(Self {
            epsilon: S::from_f64(epsilon)?,
            h_min: S::from_f64(h_min)?,
            h_dry: S::from_f64(h_dry)?,
            velocity_cap: S::from_f64(velocity_cap)?,
            safe_div: S::from_f64(safe_div)?,
            convergence: S::from_f64(convergence)?,
            gradient_eps: S::from_f64(1e-12)?,
            min_area: S::from_f64(1e-12)?,
        })
    }
}

impl Default for Tolerance<f32> {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            h_min: 1e-4,
            h_dry: 1e-3,
            velocity_cap: 100.0,
            safe_div: 1e-8,
            convergence: 1e-5,
            gradient_eps: 1e-6,
            min_area: 1e-6,
        }
    }
}

impl Default for Tolerance<f64> {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            h_min: 1e-9,
            h_dry: 1e-6,
            velocity_cap: 1000.0,
            safe_div: 1e-14,
            convergence: 1e-10,
            gradient_eps: 1e-12,
            min_area: 1e-12,
        }
    }
}

impl<S: RuntimeScalar + std::fmt::Debug> std::fmt::Debug for Tolerance<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tolerance")
            .field("h_min", &self.h_min)
            .field("h_dry", &self.h_dry)
            .field("velocity_cap", &self.velocity_cap)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_f64() {
        let tol = Tolerance::<f64>::default();
        assert!(tol.h_dry > 0.0);
        assert!(tol.velocity_cap > 0.0);
    }

    #[test]
    fn test_is_dry() {
        let tol = Tolerance::<f64>::default();
        assert!(tol.is_dry(1e-9));
        assert!(!tol.is_dry(1.0));
    }

    #[test]
    fn test_safe_divide() {
        let tol = Tolerance::<f64>::default();
        assert_eq!(tol.safe_divide(1.0, 0.0), 0.0);
        assert_eq!(tol.safe_divide(1.0, 2.0), 0.5);
    }

    #[test]
    fn test_clamp_velocity() {
        let tol = Tolerance::<f64>::default();
        assert_eq!(tol.clamp_velocity(50.0), 50.0);
        assert_eq!(tol.clamp_velocity(2000.0), 1000.0);
        assert_eq!(tol.clamp_velocity(-2000.0), -1000.0);
    }
}
