//! 限制器 trait 定义和上下文结构
//!
//! 定义了所有梯度限制器的公共接口。

use std::fmt::Debug;

/// 限制器计算所需的上下文信息
///
/// 包含计算单元的限制因子所需的所有数据。
/// 限制因子 α ∈ [0, 1]，用于限制梯度重构:
///
/// ```text
/// q_face = q_cell + α * (∇q · r)
/// ```
///
/// 其中 r 是从单元中心到面中心的向量。
#[derive(Debug, Clone, Copy)]
pub struct LimiterContext {
    /// 当前单元的场值 q_i
    pub cell_value: f64,
    
    /// 在最大距离方向的梯度投影 (∇q · r_max)
    pub gradient: f64,
    
    /// 相邻单元的最小值 q_min
    pub min_neighbor: f64,
    
    /// 相邻单元的最大值 q_max
    pub max_neighbor: f64,
    
    /// 到最远邻居面中心的距离
    pub max_distance: f64,
}

impl LimiterContext {
    /// 创建新的限制器上下文
    pub fn new(
        cell_value: f64,
        gradient: f64,
        min_neighbor: f64,
        max_neighbor: f64,
        max_distance: f64,
    ) -> Self {
        Self {
            cell_value,
            gradient,
            min_neighbor,
            max_neighbor,
            max_distance,
        }
    }
    
    /// 计算允许的最大正向变化
    ///
    /// Δ_max = q_max - q_i
    #[inline]
    pub fn delta_max(&self) -> f64 {
        self.max_neighbor - self.cell_value
    }
    
    /// 计算允许的最小负向变化
    ///
    /// Δ_min = q_min - q_i
    #[inline]
    pub fn delta_min(&self) -> f64 {
        self.min_neighbor - self.cell_value
    }
    
    /// 梯度是否为零（或接近零）
    #[inline]
    pub fn is_gradient_zero(&self, eps: f64) -> bool {
        self.gradient.abs() < eps
    }
}

/// 梯度限制器 trait
///
/// 所有限制器都实现此 trait，计算限制因子 α ∈ [0, 1]。
///
/// # 限制原则
///
/// 限制器确保重构后的面值不超过相邻单元值的范围:
///
/// ```text
/// q_min ≤ q_face ≤ q_max
/// ```
///
/// 其中 q_face = q_i + α * gradient
pub trait SlopeLimiter: Debug + Send + Sync {
    /// 计算限制因子
    ///
    /// # Arguments
    /// * `ctx` - 限制器上下文，包含单元值、梯度和邻居范围
    ///
    /// # Returns
    /// 限制因子 α ∈ [0, 1]
    fn compute_limiter(&self, ctx: &LimiterContext) -> f64;
    
    /// 返回限制器名称
    fn name(&self) -> &'static str;
    
    /// 批量计算限制因子
    ///
    /// 默认实现简单迭代，子类可以提供向量化版本。
    fn compute_limiters(&self, contexts: &[LimiterContext]) -> Vec<f64> {
        contexts.iter().map(|ctx| self.compute_limiter(ctx)).collect()
    }
}

/// 无限制器（一阶精度）
///
/// 始终返回 1.0，不限制梯度。
/// 这等效于使用一阶精度，因为梯度不被使用。
///
/// # 使用场景
/// - 调试目的
/// - 与一阶方案对比
/// - 极端情况下的稳定性
#[derive(Debug, Clone, Copy, Default)]
pub struct NoLimiter;

impl SlopeLimiter for NoLimiter {
    #[inline]
    fn compute_limiter(&self, _ctx: &LimiterContext) -> f64 {
        1.0
    }
    
    fn name(&self) -> &'static str {
        "NoLimiter"
    }
    
    fn compute_limiters(&self, contexts: &[LimiterContext]) -> Vec<f64> {
        vec![1.0; contexts.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_limiter_context_creation() {
        let ctx = LimiterContext::new(1.0, 0.5, 0.5, 1.5, 0.1);
        assert_eq!(ctx.cell_value, 1.0);
        assert_eq!(ctx.gradient, 0.5);
        assert_eq!(ctx.min_neighbor, 0.5);
        assert_eq!(ctx.max_neighbor, 1.5);
        assert_eq!(ctx.max_distance, 0.1);
    }
    
    #[test]
    fn test_limiter_context_deltas() {
        let ctx = LimiterContext::new(1.0, 0.5, 0.3, 1.8, 0.1);
        assert!((ctx.delta_max() - 0.8).abs() < 1e-10);
        assert!((ctx.delta_min() - (-0.7)).abs() < 1e-10);
    }
    
    #[test]
    fn test_limiter_context_zero_gradient() {
        let ctx1 = LimiterContext::new(1.0, 0.0, 0.5, 1.5, 0.1);
        assert!(ctx1.is_gradient_zero(1e-10));
        
        let ctx2 = LimiterContext::new(1.0, 1e-15, 0.5, 1.5, 0.1);
        assert!(ctx2.is_gradient_zero(1e-10));
        
        let ctx3 = LimiterContext::new(1.0, 0.1, 0.5, 1.5, 0.1);
        assert!(!ctx3.is_gradient_zero(1e-10));
    }
    
    #[test]
    fn test_no_limiter() {
        let limiter = NoLimiter;
        
        // 应始终返回 1.0
        let ctx1 = LimiterContext::new(1.0, 0.5, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx1), 1.0);
        
        let ctx2 = LimiterContext::new(1.0, -10.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx2), 1.0);
        
        let ctx3 = LimiterContext::new(1.0, 1000.0, 0.5, 1.5, 0.1);
        assert_eq!(limiter.compute_limiter(&ctx3), 1.0);
    }
    
    #[test]
    fn test_no_limiter_name() {
        let limiter = NoLimiter;
        assert_eq!(limiter.name(), "NoLimiter");
    }
    
    #[test]
    fn test_no_limiter_batch() {
        let limiter = NoLimiter;
        let contexts = vec![
            LimiterContext::new(1.0, 0.5, 0.5, 1.5, 0.1),
            LimiterContext::new(2.0, -0.5, 1.5, 2.5, 0.1),
            LimiterContext::new(0.5, 0.0, 0.0, 1.0, 0.1),
        ];
        
        let results = limiter.compute_limiters(&contexts);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|&x| x == 1.0));
    }
}
