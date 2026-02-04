// crates/mh_physics/src/numerics/limiter/venkatakrishnan.rs

//! Venkatakrishnan 限制器 - 泛型实现
//!
//! 光滑的梯度限制器，避免 Barth-Jespersen 的梯度突变问题。
//! 使用光滑函数替代 min 操作，提供二阶精度并保持数值稳定性。
//!
//! # 类型参数
//! - `S: RuntimeScalar` - 支持 f32/f64 精度
//!
//! # K 参数选择
//! - 0.1-0.3: 强限制，适用于激波/溃坝
//! - 0.3-1.0: 中等限制，通用场景（默认）
//! - 1.0-5.0: 弱限制，适用于光滑流动
//!
//! # 注意事项
//! 默认构造器使用 `mesh_scale=1.0`，实际使用时必须调用 `update_mesh_scale()`
//! 根据真实网格尺度更新，否则限制效果可能不符合预期。
//!
//! # 参考文献
//! Venkatakrishnan, V. (1993). "On the accuracy of limiters and convergence to steady state solutions".
//! AIAA Paper 93-0880.

use mh_runtime::Backend;
use super::traits::{LimiterContext, SlopeLimiter};

/// Venkatakrishnan 限制器
#[derive(Debug, Clone, Copy)]
pub struct Venkatakrishnan<B: Backend> {
    k: B::Scalar,
    eps_squared: B::Scalar,
    tol: B::Scalar,
}

impl<B: Backend> Venkatakrishnan<B> {
    /// 创建新的限制器
    ///
    /// # 参数
    /// - `k`: K 参数，控制限制强度
    /// - `mesh_scale`: 网格特征尺度
    #[inline]
    pub fn new(k: B::Scalar, mesh_scale: B::Scalar) -> Self {
        let scale = if mesh_scale.is_finite() && mesh_scale > B::Scalar::ZERO {
            mesh_scale
        } else {
            B::Scalar::ONE
        };
        let kh = k * scale;
        let eps_squared = kh * kh * kh;
        
        Self {
            k,
            eps_squared,
            tol: B::Scalar::MIN_POSITIVE,
        }
    }

    /// 创建具有自定义容差的限制器
    #[inline]
    pub fn with_tolerance(k: B::Scalar, mesh_scale: B::Scalar, tol: B::Scalar) -> Self {
        let scale = if mesh_scale.is_finite() && mesh_scale > B::Scalar::ZERO {
            mesh_scale
        } else {
            B::Scalar::ONE
        };
        let kh = k * scale;
        let eps_squared = kh * kh * kh;
        
        Self {
            k,
            eps_squared,
            tol,
        }
    }

    /// 获取 K 参数
    #[inline]
    pub fn k(&self) -> B::Scalar {
        self.k
    }

    /// 获取 ε² 值
    #[inline]
    pub fn eps_squared(&self) -> B::Scalar {
        self.eps_squared
    }

    /// 更新网格尺度
    #[inline]
    pub fn update_mesh_scale(&mut self, mesh_scale: B::Scalar) {
        let scale = if mesh_scale.is_finite() && mesh_scale > B::Scalar::ZERO {
            mesh_scale
        } else {
            B::Scalar::ONE
        };
        let kh = self.k * scale;
        self.eps_squared = kh * kh * kh;
    }

    /// 计算光滑限制函数
    #[inline]
    fn phi(&self, x: B::Scalar, y: B::Scalar) -> B::Scalar {
        let x2 = x * x;
        let y2 = y * y;
        let eps2 = self.eps_squared;
        
        let numerator = (y2 + eps2) * x + B::Scalar::TWO * x2 * y;
        let denominator = y2 + B::Scalar::TWO * x2 + x * y + eps2;
        
        if denominator.abs() < self.tol {
            B::Scalar::ONE
        } else {
            numerator / denominator
        }
    }
}

impl<B: Backend> SlopeLimiter<B> for Venkatakrishnan<B> {
    #[inline]
    fn compute_limiter(&self, ctx: &LimiterContext<B>) -> B::Scalar {
        if ctx.is_gradient_zero(self.tol) {
            return B::Scalar::ONE;
        }
        
        let delta = ctx.gradient;
        
        if delta > B::Scalar::ZERO {
            let delta_max = ctx.delta_max();
            if delta_max < self.tol {
                B::Scalar::ZERO
            } else {
                self.phi(delta, delta_max).min(B::Scalar::ONE)
            }
        } else {
            let delta_min = ctx.delta_min();
            if delta_min > -self.tol {
                B::Scalar::ZERO
            } else {
                self.phi(-delta, -delta_min).min(B::Scalar::ONE)
            }
        }
    }
    
    #[inline]
    fn name(&self) -> &'static str {
        "Venkatakrishnan"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    type BackendF64 = CpuBackend<f64>;
    type BackendF32 = CpuBackend<f32>;

    #[test]
    fn test_creation_f64() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        assert_eq!(limiter.k(), 5.0);
        assert!((limiter.eps_squared() - 0.125).abs() < 1e-10);
    }

    #[test]
    fn test_creation_f32() {
        let limiter = Venkatakrishnan::<BackendF32>::new(5.0f32, 0.1f32);
        assert_eq!(limiter.k(), 5.0f32);
        assert!((limiter.eps_squared() - 0.125f32).abs() < 1e-6f32);
    }

    #[test]
    fn test_with_tolerance() {
        let limiter = Venkatakrishnan::<BackendF64>::with_tolerance(5.0, 0.1, 1e-8);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }

    #[test]
    fn test_presets() {
        let shock = Venkatakrishnan::for_shock_capturing(1.0);
        assert_eq!(shock.k(), 0.1);

        let wet_dry = Venkatakrishnan::for_wetting_drying(1.0);
        assert_eq!(wet_dry.k(), 0.3);

        let smooth = Venkatakrishnan::for_smooth_flow(1.0);
        assert_eq!(smooth.k(), 2.0);

        let minimal = Venkatakrishnan::minimal_limiting(1.0);
        assert_eq!(minimal.k(), 5.0);
    }

    #[test]
    fn test_zero_gradient() {
        let limiter = Venkatakrishnan::new(5.0, 0.1);
        let ctx = LimiterContext::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }

    #[test]
    fn test_small_gradient_f64() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.1, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((0.0..=1.0).contains(&alpha));
    }

    #[test]
    fn test_small_gradient_f32() {
        let limiter = Venkatakrishnan::<BackendF32>::new(5.0f32, 0.1f32);
        let ctx = LimiterContext::<BackendF32>::new(
            1.0f32, 0.1f32, 0.5f32, 1.5f32, 0.1f32
        );
        let alpha = limiter.compute_limiter(&ctx);
        assert!((0.0f32..=1.0f32).contains(&alpha));
    }

    #[test]
    fn test_k_parameter_sensitivity() {
        let mesh_scale = 0.1;
        let limiter_k1 = Venkatakrishnan::<BackendF64>::new(1.0, mesh_scale);
        let limiter_k5 = Venkatakrishnan::<BackendF64>::new(5.0, mesh_scale);
        let limiter_k10 = Venkatakrishnan::<BackendF64>::new(10.0, mesh_scale);

        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.4, 0.5, 1.5, 0.1);
        let alpha_k1 = limiter_k1.compute_limiter(&ctx);
        let alpha_k5 = limiter_k5.compute_limiter(&ctx);
        let alpha_k10 = limiter_k10.compute_limiter(&ctx);

        assert!(alpha_k1 <= alpha_k5);
        assert!(alpha_k5 <= alpha_k10);
        assert!(limiter_k1.eps_squared() < limiter_k5.eps_squared());
        assert!(limiter_k5.eps_squared() < limiter_k10.eps_squared());
    }

    #[test]
    fn test_large_gradient() {
        let limiter = Venkatakrishnan::<BackendF64>::new(1.0, 0.01);
        let ctx = LimiterContext::<BackendF64>::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1.0);
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_update_mesh_scale() {
        let mut limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        assert!((limiter.eps_squared() - 0.125).abs() < 1e-10);

        limiter.update_mesh_scale(0.2);
        assert!((limiter.eps_squared() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothness() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let gradients: Vec<f64> = (1..=100).map(|i| i as f64 * 0.01).collect();
        let alphas: Vec<f64> = gradients
            .iter()
            .map(|&g| {
                let ctx = LimiterContext::<BackendF64>::new(1.0, g, 0.5, 1.5, 0.1);
                limiter.compute_limiter(&ctx)
            })
            .collect();

        for window in alphas.windows(2) {
            let diff = (window[1] - window[0]).abs();
            assert!(diff < 0.2, "Limiter not smooth: diff = {}", diff);
        }
    }

    #[test]
    fn test_symmetry() {
        let limiter = Venkatakrishnan::<BackendF64>::new(5.0, 0.1);
        let ctx_pos = LimiterContext::<BackendF64>::new(1.0, 0.3, 0.5, 1.5, 0.1);
        let ctx_neg = LimiterContext::<BackendF64>::new(1.0, -0.3, 0.5, 1.5, 0.1);
        assert!((limiter.compute_limiter(&ctx_pos) - limiter.compute_limiter(&ctx_neg)).abs() < 1e-10);
    }

    #[test]
    fn test_at_maximum() {
        let limiter = Venkatakrishnan::<BackendF64>::new(1.0, 0.01);
        let ctx = LimiterContext::<BackendF64>::new(1.5, 0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 0.1);
    }

    #[test]
    fn test_limiter_bounded() {
        let limiter = Venkatakrishnan::<BackendF64>::new(3.0, 0.1);
        let test_cases = vec![
            (1.0, 0.5, 0.0, 2.0),
            (1.0, -0.5, 0.0, 2.0),
            (1.0, 0.01, 0.5, 1.5),
            (1.0, -0.01, 0.5, 1.5),
        ];

        for (q, g, q_min, q_max) in test_cases {
            let ctx = LimiterContext::<BackendF64>::new(q, g, q_min, q_max, 0.1);
            let alpha = limiter.compute_limiter(&ctx);
            assert!((0.0..=1.0).contains(&alpha));
        }
    }
}