//! Barth-Jespersen 限制器
//!
//! **层级**: Layer 3 - Engine Layer
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

use mh_runtime::Backend;
use std::marker::PhantomData;

use super::traits::{LimiterContext, SlopeLimiter};

// ============================================================================
// 泛型版本（核心实现）
// ============================================================================

/// Barth-Jespersen 限制器
///
/// 严格 TVD 限制器，确保重构值不超过相邻单元的极值。
#[derive(Debug, Clone, Copy)]
pub struct BarthJespersen<B: Backend> {
    /// 判断梯度为零的容差
    eps: B::Scalar,
    /// 干单元水深阈值（可选）
    h_dry: Option<B::Scalar>,
    _marker: PhantomData<B>,
}

impl<B: Backend> Default for BarthJespersen<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> BarthJespersen<B> {
    /// 创建新的 Barth-Jespersen 限制器
    pub fn new() -> Self {
        Self { 
            eps: B::Scalar::EPSILON,
            h_dry: None,
            _marker: PhantomData,
        }
    }
    
    /// 创建具有自定义容差的限制器
    pub fn with_tolerance(eps: B::Scalar) -> Self {
        Self { eps, h_dry: None, _marker: PhantomData }
    }

    /// 设置干单元阈值
    ///
    /// 当单元水深低于此阈值时，使用更保守的限制。
    pub fn with_dry_threshold(mut self, h_dry: B::Scalar) -> Self {
        self.h_dry = Some(h_dry);
        self
    }

    /// 计算对称限制器
    ///
    /// 对于面两侧的单元，使用相同的限制因子以保证通量一致性。
    /// 返回两侧限制因子的最小值。
    #[inline]
    pub fn compute_symmetric(
        &self,
        ctx_owner: &LimiterContext<B>,
        ctx_neighbor: &LimiterContext<B>,
    ) -> B::Scalar {
        let alpha_o = self.compute_limiter(ctx_owner);
        let alpha_n = self.compute_limiter(ctx_neighbor);
        if alpha_o < alpha_n { alpha_o } else { alpha_n }
    }

    /// 对干单元应用额外保守策略
    #[inline]
    fn adjust_for_dry(&self, alpha: B::Scalar, depth: B::Scalar) -> B::Scalar {
        if let Some(h_dry) = self.h_dry {
            if depth < h_dry {
                // 干单元：更激进地限制梯度
                let ratio = (depth / h_dry).clamp_value(B::Scalar::ZERO, B::Scalar::ONE);
                return alpha * ratio;
            }
        }
        alpha
    }

    /// 计算限制器（带水深参数）
    pub fn compute_limiter_with_depth(&self, ctx: &LimiterContext<B>, depth: B::Scalar) -> B::Scalar {
        let alpha = self.compute_limiter(ctx);
        self.adjust_for_dry(alpha, depth)
    }
}

impl<B: Backend> SlopeLimiter<B> for BarthJespersen<B> {
    fn compute_limiter(&self, ctx: &LimiterContext<B>) -> B::Scalar {
        // 如果梯度为零，不需要限制
        if ctx.is_gradient_zero(self.eps) {
            return B::Scalar::ONE;
        }
        
        let delta = ctx.gradient;
        
        if delta > B::Scalar::ZERO {
            // 正梯度：限制不超过 q_max
            let delta_max = ctx.delta_max();
            if delta_max > self.eps {
                let ratio = delta_max / delta;
                if ratio < B::Scalar::ONE { ratio } else { B::Scalar::ONE }
            } else {
                B::Scalar::ZERO // 已经在最大值，不允许增长
            }
        } else {
            // 负梯度：限制不低于 q_min
            let delta_min = ctx.delta_min();
            if delta_min < -self.eps {
                let ratio = delta_min / delta;
                if ratio < B::Scalar::ONE { ratio } else { B::Scalar::ONE }
            } else {
                B::Scalar::ZERO // 已经在最小值，不允许减少
            }
        }
    }
    
    fn name(&self) -> &'static str {
        "BarthJespersen"
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    type TestBackend = CpuBackend<f64>;
    
    #[test]
    fn test_barth_jespersen_creation() {
        let limiter = BarthJespersen::<TestBackend>::new();
        assert!(limiter.eps > 0.0);
        
        let limiter2 = BarthJespersen::<TestBackend>::with_tolerance(1e-8);
        assert!((limiter2.eps - 1e-8).abs() < 1e-15);
    }
    
    #[test]
    fn test_barth_jespersen_name() {
        let limiter = BarthJespersen::<TestBackend>::new();
        assert_eq!(limiter.name(), "BarthJespersen");
    }
    
    #[test]
    fn test_zero_gradient() {
        let limiter = BarthJespersen::<TestBackend>::new();
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_positive_gradient_no_limit() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // q_i = 1.0, gradient = 0.2, q_max = 1.5
        // Δ_max = 0.5, gradient = 0.2, ratio = 2.5 > 1 → α = 1.0
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.2, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_positive_gradient_needs_limit() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // q_i = 1.0, gradient = 0.8, q_max = 1.5
        // Δ_max = 0.5, gradient = 0.8, ratio = 0.625 < 1 → α = 0.625
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-10);
    }
    
    #[test]
    fn test_negative_gradient_no_limit() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // q_i = 1.0, gradient = -0.2, q_min = 0.5
        // Δ_min = -0.5, gradient = -0.2, ratio = 2.5 > 1 → α = 1.0
        let ctx = LimiterContext::<TestBackend>::new(1.0, -0.2, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_negative_gradient_needs_limit() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // q_i = 1.0, gradient = -0.8, q_min = 0.5
        // Δ_min = -0.5, gradient = -0.8, ratio = 0.625 < 1 → α = 0.625
        let ctx = LimiterContext::<TestBackend>::new(1.0, -0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-10);
    }
    
    #[test]
    fn test_at_maximum() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // 单元值已经是最大值，正梯度应该被完全限制
        let ctx = LimiterContext::<TestBackend>::new(1.5, 0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1e-10);
    }
    
    #[test]
    fn test_at_minimum() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // 单元值已经是最小值，负梯度应该被完全限制
        let ctx = LimiterContext::<TestBackend>::new(0.5, -0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1e-10);
    }
    
    #[test]
    fn test_extreme_gradient() {
        let limiter = BarthJespersen::<TestBackend>::new();
        // 极大梯度
        let ctx = LimiterContext::<TestBackend>::new(1.0, 100.0, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        // Δ_max = 0.5, gradient = 100.0, ratio = 0.005
        assert!((alpha - 0.005).abs() < 1e-10);
    }
    
    #[test]
    fn test_generic_f32() {
        // 测试 f32 版本
        let limiter: BarthJespersen<CpuBackend<f32>> = BarthJespersen::new();
        let ctx = LimiterContext::<CpuBackend<f32>>::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-5);
    }
}
