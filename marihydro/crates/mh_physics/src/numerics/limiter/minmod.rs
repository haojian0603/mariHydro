//! Minmod 限制器
//!
//! 最耗散的经典 TVD 限制器，适用于需要最大稳定性的场景。
//!
//! # 算法
//!
//! Minmod 函数选择绝对值最小的斜率:
//!
//! ```text
//!               ⎧ min(|a|, |b|) * sign(a)  如果 sign(a) = sign(b)
//! minmod(a, b) = ⎨
//!               ⎩ 0                         如果 sign(a) ≠ sign(b)
//! ```
//!
//! 对于梯度限制，比较重构斜率和向前/向后差分:
//!
//! ```text
//! α = minmod(1, Δ_max/Δ_f) 或 minmod(1, Δ_min/Δ_f)
//! ```
//!
//! # 特点
//!
//! - 最耗散的限制器
//! - 在间断处完全切换到一阶
//! - 非常稳定
//! - 光滑区域可能过度耗散
//!
//! # 适用场景
//!
//! - 强激波或水跃
//! - 干湿交界处
//! - 需要无条件稳定的情况

use mh_runtime::Backend;
use super::traits::{LimiterContext, SlopeLimiter};

/// 泛型 Minmod 限制器
///
/// 最耗散的限制器，提供最大稳定性。
#[derive(Debug, Clone, Copy)]
pub struct Minmod<B: Backend> {
    /// 判断值为零的容差
    eps: B::Scalar,
}

impl<B: Backend> Default for Minmod<B> {
    fn default() -> Self {
        Self { eps: B::Scalar::MIN_POSITIVE }
    }
}

impl<B: Backend> Minmod<B> {
    /// 创建新的 Minmod 限制器
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 创建具有自定义容差的限制器
    pub fn with_tolerance(eps: B::Scalar) -> Self {
        Self { eps }
    }
    
    /// Minmod 函数
    ///
    /// 返回绝对值最小的值，如果符号不同则返回 0
    #[inline]
    fn minmod(&self, a: B::Scalar, b: B::Scalar) -> B::Scalar {
        if a * b <= B::Scalar::ZERO {
            // 符号不同（或其中一个为零）
            B::Scalar::ZERO
        } else if a > B::Scalar::ZERO {
            // 都是正数，取较小者
            if a < b { a } else { b }
        } else {
            // 都是负数，取绝对值较小者（即较大的负数）
            if a > b { a } else { b }
        }
    }
}

impl<B: Backend> SlopeLimiter<B> for Minmod<B> {
    fn compute_limiter(&self, ctx: &LimiterContext<B>) -> B::Scalar {
        // 如果梯度为零，不需要限制
        if ctx.is_gradient_zero(self.eps) {
            return B::Scalar::ONE;
        }
        
        let delta = ctx.gradient;
        
        // 计算允许的比值
        let ratio = if delta > B::Scalar::ZERO {
            let delta_max = ctx.delta_max();
            if delta_max < self.eps {
                B::Scalar::ZERO
            } else {
                delta_max / delta
            }
        } else {
            let delta_min = ctx.delta_min();
            if delta_min > -self.eps {
                B::Scalar::ZERO
            } else {
                delta_min / delta
            }
        };
        
        // Minmod: 取 1 和 ratio 中的较小者，但必须非负
        let minmod_val = self.minmod(B::Scalar::ONE, ratio);
        if minmod_val < B::Scalar::ZERO { B::Scalar::ZERO } else { minmod_val }
    }
    
    fn name(&self) -> &'static str {
        "Minmod"
    }
}

/// 扩展的 Minmod 限制器 (Superbee 变体的基础)
///

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    type TestBackend = CpuBackend<f64>;
    
    #[test]
    fn test_minmod_creation() {
        let limiter = Minmod::<TestBackend>::new();
        assert!(limiter.eps > 0.0);
        
        let limiter2 = Minmod::<TestBackend>::with_tolerance(1e-8);
        assert_eq!(limiter2.eps, 1e-8);
    }
    
    #[test]
    fn test_minmod_name() {
        let limiter = Minmod::<TestBackend>::new();
        assert_eq!(limiter.name(), "Minmod");
    }
    
    #[test]
    fn test_minmod_function() {
        let limiter = Minmod::<TestBackend>::new();
        
        // 同号正数
        assert_eq!(limiter.minmod(2.0, 3.0), 2.0);
        assert_eq!(limiter.minmod(3.0, 2.0), 2.0);
        
        // 同号负数
        assert_eq!(limiter.minmod(-2.0, -3.0), -2.0);
        assert_eq!(limiter.minmod(-3.0, -2.0), -2.0);
        
        // 异号
        assert_eq!(limiter.minmod(2.0, -3.0), 0.0);
        assert_eq!(limiter.minmod(-2.0, 3.0), 0.0);
        
        // 包含零
        assert_eq!(limiter.minmod(0.0, 3.0), 0.0);
        assert_eq!(limiter.minmod(2.0, 0.0), 0.0);
    }
    
    #[test]
    fn test_zero_gradient() {
        let limiter = Minmod::<TestBackend>::new();
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_positive_gradient_no_limit() {
        let limiter = Minmod::<TestBackend>::new();
        // q_i = 1.0, gradient = 0.2, q_max = 1.5
        // Δ_max = 0.5, ratio = 2.5 > 1 → minmod(1, 2.5) = 1.0
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.2, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_positive_gradient_needs_limit() {
        let limiter = Minmod::<TestBackend>::new();
        // q_i = 1.0, gradient = 0.8, q_max = 1.5
        // Δ_max = 0.5, ratio = 0.625 < 1 → minmod(1, 0.625) = 0.625
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-10);
    }
    
    #[test]
    fn test_negative_gradient_no_limit() {
        let limiter = Minmod::<TestBackend>::new();
        // q_i = 1.0, gradient = -0.2, q_min = 0.5
        // Δ_min = -0.5, ratio = 2.5 > 1 → minmod(1, 2.5) = 1.0
        let ctx = LimiterContext::<TestBackend>::new(1.0, -0.2, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_negative_gradient_needs_limit() {
        let limiter = Minmod::<TestBackend>::new();
        // q_i = 1.0, gradient = -0.8, q_min = 0.5
        // Δ_min = -0.5, ratio = 0.625 < 1 → minmod(1, 0.625) = 0.625
        let ctx = LimiterContext::<TestBackend>::new(1.0, -0.8, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!((alpha - 0.625).abs() < 1e-10);
    }
    
    #[test]
    fn test_at_maximum() {
        let limiter = Minmod::<TestBackend>::new();
        // 单元值已经是最大值，正梯度应该被完全限制
        let ctx = LimiterContext::<TestBackend>::new(1.5, 0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1e-10);
    }
    
    #[test]
    fn test_at_minimum() {
        let limiter = Minmod::<TestBackend>::new();
        // 单元值已经是最小值，负梯度应该被完全限制
        let ctx = LimiterContext::<TestBackend>::new(0.5, -0.3, 0.5, 1.5, 0.1);
        let alpha = limiter.compute_limiter(&ctx);
        assert!(alpha < 1e-10);
    }
    
    #[test]
    fn test_limiter_bounded() {
        let limiter = Minmod::<TestBackend>::new();
        
        // 测试各种情况下 α ∈ [0, 1]
        let test_cases = vec![
            (1.0, 0.5, 0.0, 2.0),
            (1.0, -0.5, 0.0, 2.0),
            (1.0, 2.0, 0.0, 2.0),
            (1.0, -2.0, 0.0, 2.0),
            (1.0, 0.01, 0.5, 1.5),
            (1.0, -0.01, 0.5, 1.5),
            (0.5, 0.3, 0.5, 1.5),  // 在边界
            (1.5, -0.3, 0.5, 1.5), // 在边界
        ];
        
        for (q, g, q_min, q_max) in test_cases {
            let ctx = LimiterContext::<TestBackend>::new(q, g, q_min, q_max, 0.1);
            let alpha = limiter.compute_limiter(&ctx);
            assert!((0.0..=1.0).contains(&alpha), 
                "Alpha {} out of bounds for q={}, g={}", alpha, q, g);
        }
    }
    
    
    #[test]
    fn test_compare_with_barth_jespersen() {
        // Minmod 应该和 Barth-Jespersen 给出相同结果（在基本情况下）
        // 因为两者都是严格 TVD 限制器
        let minmod = Minmod::<TestBackend>::new();
        let bj = super::super::BarthJespersen::<TestBackend>::new();
        
        let ctx = LimiterContext::<TestBackend>::new(1.0, 0.8, 0.5, 1.5, 0.1);
        let alpha_minmod = minmod.compute_limiter(&ctx);
        let alpha_bj = bj.compute_limiter(&ctx);
        
        // 对于简单情况，结果应该相同
        assert!((alpha_minmod - alpha_bj).abs() < 1e-10);
    }
}
