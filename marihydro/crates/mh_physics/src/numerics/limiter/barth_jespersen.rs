//! Barth-Jespersen 限制器
//!
//! 经典的 TVD 限制器，保证重构值严格在邻居范围内。
//! 
//! # 算法
//!
//! 对于每个面 f，计算限制因子:
//!
//! ```text
//!           ⎧ min(1, Δ_max / Δ_f)  如果 Δ_f > 0
//! α_f = ⎨ min(1, Δ_min / Δ_f)  如果 Δ_f < 0
//!           ⎩ 1                     如果 Δ_f = 0
//! ```
//!
//! 其中:
//! - Δ_f = ∇q · r_f (从单元中心到面的梯度投影)
//! - Δ_max = q_max - q_i
//! - Δ_min = q_min - q_i
//!
//! # 特点
//!
//! - 严格保证 TVD 性质
//! - 可能导致梯度突变（非光滑）
//! - 中等耗散性
//!
//! # 参考文献
//!
//! Barth, T.J. and Jespersen, D.C. (1989). "The design and application 
//! of upwind schemes on unstructured meshes". AIAA Paper 89-0366.

use super::traits::{LimiterContext, SlopeLimiter};

/// Barth-Jespersen 限制器
///
/// 严格 TVD 限制器，确保重构值不超过相邻单元的极值。
#[derive(Debug, Clone, Copy, Default)]
pub struct BarthJespersen {
    /// 判断梯度为零的容差
    eps: f64,
}

impl BarthJespersen {
    /// 创建新的 Barth-Jespersen 限制器
    pub fn new() -> Self {
        Self { eps: 1e-12 }
    }
    
    /// 创建具有自定义容差的限制器
    pub fn with_tolerance(eps: f64) -> Self {
        Self { eps }
    }
}

impl SlopeLimiter for BarthJespersen {
    fn compute_limiter(&self, ctx: &LimiterContext) -> f64 {
        // 如果梯度为零，不需要限制
        if ctx.is_gradient_zero(self.eps) {
            return 1.0;
        }
        
        let delta = ctx.gradient;
        
        if delta > 0.0 {
            // 正梯度：限制不超过 q_max
            let delta_max = ctx.delta_max();
            if delta_max > self.eps {
                (delta_max / delta).min(1.0)
            } else {
                0.0 // 已经在最大值，不允许增长
            }
        } else {
            // 负梯度：限制不低于 q_min
            let delta_min = ctx.delta_min();
            if delta_min < -self.eps {
                (delta_min / delta).min(1.0)
            } else {
                0.0 // 已经在最小值，不允许减少
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "BarthJespersen"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_barth_jespersen_creation() {
        let limiter = BarthJespersen::new();
        assert_eq!(limiter.eps, 1e-12);
        
        let limiter2 = BarthJespersen::with_tolerance(1e-8);
        assert_eq!(limiter2.eps, 1e-8);
    }
    
    #[test]
    fn test_barth_jespersen_name() {
        let limiter = BarthJespersen::new();
        assert_eq!(limiter.name(), "BarthJespersen");
    }
    
    #[test]
    fn test_zero_gradient() {
        let limiter = BarthJespersen::new();
        let ctx = LimiterContext::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_positive_gradient_no_limit() {
        let limiter = BarthJespersen::new();
        // q_i = 1.0, gradient = 0.2, q_max = 1.5
        // Δ_max = 0.5, gradient = 0.2, ratio = 2.5 > 1 → α = 1.0
        let ctx = LimiterContext::new(1.0, 0.2, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_positive_gradient_needs_limit() {
        let limiter = BarthJespersen::new();
        // q_i = 1.0, gradient = 0.8, q_max = 1.5
        // Δ_max = 0.5, gradient = 0.8, ratio = 0.625 < 1 → α = 0.625
        let ctx = LimiterContext::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-10);
    }
    
    #[test]
    fn test_negative_gradient_no_limit() {
        let limiter = BarthJespersen::new();
        // q_i = 1.0, gradient = -0.2, q_min = 0.5
        // Δ_min = -0.5, gradient = -0.2, ratio = 2.5 > 1 → α = 1.0
        let ctx = LimiterContext::new(1.0, -0.2, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_negative_gradient_needs_limit() {
        let limiter = BarthJespersen::new();
        // q_i = 1.0, gradient = -0.8, q_min = 0.5
        // Δ_min = -0.5, gradient = -0.8, ratio = 0.625 < 1 → α = 0.625
        let ctx = LimiterContext::new(1.0, -0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-10);
    }
    
    #[test]
    fn test_at_maximum() {
        let limiter = BarthJespersen::new();
        // 单元值已经是最大值，正梯度应该被完全限制
        let ctx = LimiterContext::new(1.5, 0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1e-10);
    }
    
    #[test]
    fn test_at_minimum() {
        let limiter = BarthJespersen::new();
        // 单元值已经是最小值，负梯度应该被完全限制
        let ctx = LimiterContext::new(0.5, -0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1e-10);
    }
    
    #[test]
    fn test_extreme_gradient() {
        let limiter = BarthJespersen::new();
        // 极大梯度
        let ctx = LimiterContext::new(1.0, 100.0, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        // Δ_max = 0.5, gradient = 100.0, ratio = 0.005
        assert!((alpha - 0.005).abs() < 1e-10);
    }
}
