//! # 梯度限制器模块
//!
//! 提供梯度限制器用于控制二阶精度重构的振荡:
//!
//! - `SlopeLimiter` - 限制器 trait 定义
//! - `NoLimiter` - 无限制（一阶精度）
//! - `BarthJespersen` - Barth-Jespersen 限制器（严格 TVD）
//! - `Venkatakrishnan` - Venkatakrishnan 限制器（光滑，保单调）
//! - `Minmod` - Minmod 限制器（最耗散）
//!
//! ## 使用方式
//!
//! ```ignore
//! use mh_physics::numerics::limiter::{SlopeLimiter, Venkatakrishnan};
//!
//! let limiter = Venkatakrishnan::new(5.0, mesh_scale);
//! let alpha = limiter.compute_limiter(
//!     q_i, grad_i, q_min, q_max, delta_max,
//! );
//! // grad_limited = grad_i * alpha
//! ```
//!
//! ## 限制器选择指南
//!
//! | 限制器 | 耗散性 | 光滑性 | 适用场景 |
//! |--------|--------|--------|----------|
//! | Barth-Jespersen | 中等 | 不光滑 | 需要严格 TVD 保证 |
//! | Venkatakrishnan | 低 | 光滑 | 通用推荐，平衡精度与稳定性 |
//! | Minmod | 高 | 光滑 | 强激波，需要最大稳定性 |

mod traits;
mod barth_jespersen;
mod venkatakrishnan;
mod minmod;

pub use traits::{SlopeLimiter, LimiterContext, NoLimiter};
pub use barth_jespersen::BarthJespersen;
pub use venkatakrishnan::Venkatakrishnan;
pub use minmod::Minmod;

use crate::types::LimiterType;

/// 根据配置创建限制器实例
///
/// # Arguments
/// * `limiter_type` - 限制器类型枚举
/// * `k` - Venkatakrishnan K 参数 (对其他类型忽略)
/// * `mesh_scale` - 网格特征尺度 (对 Venkatakrishnan 使用)
pub fn create_limiter(
    limiter_type: LimiterType, 
    k: f64, 
    mesh_scale: f64,
) -> Box<dyn SlopeLimiter + Send + Sync> {
    match limiter_type {
        LimiterType::None => Box::new(NoLimiter),
        LimiterType::BarthJespersen => Box::new(BarthJespersen::new()),
        LimiterType::Venkatakrishnan => Box::new(Venkatakrishnan::new(k, mesh_scale)),
        LimiterType::Minmod => Box::new(Minmod::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_limiter_none() {
        let limiter = create_limiter(LimiterType::None, 5.0, 1.0);
        let ctx = LimiterContext {
            cell_value: 1.0,
            gradient: 0.5,
            min_neighbor: 0.5,
            max_neighbor: 1.5,
            max_distance: 1.0,
        };
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_create_limiter_barth_jespersen() {
        let limiter = create_limiter(LimiterType::BarthJespersen, 5.0, 1.0);
        let ctx = LimiterContext {
            cell_value: 1.0,
            gradient: 0.0,
            min_neighbor: 0.5,
            max_neighbor: 1.5,
            max_distance: 1.0,
        };
        // 零梯度应返回 1.0
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_create_limiter_venkatakrishnan() {
        let limiter = create_limiter(LimiterType::Venkatakrishnan, 5.0, 1.0);
        let ctx = LimiterContext {
            cell_value: 1.0,
            gradient: 0.0,
            min_neighbor: 0.5,
            max_neighbor: 1.5,
            max_distance: 1.0,
        };
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
    
    #[test]
    fn test_create_limiter_minmod() {
        let limiter = create_limiter(LimiterType::Minmod, 5.0, 1.0);
        let ctx = LimiterContext {
            cell_value: 1.0,
            gradient: 0.0,
            min_neighbor: 0.5,
            max_neighbor: 1.5,
            max_distance: 1.0,
        };
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
}
