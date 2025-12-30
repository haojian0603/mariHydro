//! # 梯度限制器模块
//!
//! 提供梯度限制器用于控制二阶精度重构的振荡:
//!
//! - `SlopeLimiterGeneric<S>` - 泛型限制器 trait
//! - `NoLimiterGeneric<S>` - 无限制（一阶精度）
//! - `BarthJespersenGeneric<S>` - Barth-Jespersen 限制器（严格 TVD）
//! - `VenkatakrishnanGeneric<S>` - Venkatakrishnan 限制器（光滑，保单调）
//! - `MinmodGeneric<S>` - Minmod 限制器（最耗散）
//!
//! ## 使用方式
//!
//! ```ignore
//! use mh_physics::numerics::limiter::{SlopeLimiterGeneric, VenkatakrishnanGeneric};
//!
//! let limiter = VenkatakrishnanGeneric::<f64>::new(5.0, mesh_scale);
//! let alpha = limiter.compute_limiter(&ctx);
//! // grad_limited = grad_i * alpha
//! ```
//!
//! ## 限制器选择指南
//!
//! | 限制器 | 耗散性 | 光滑性 | 适用场景 |
//! |--------|--------|--------|----------|
//! | BarthJespersenGeneric | 中等 | 不光滑 | 需要严格 TVD 保证 |
//! | VenkatakrishnanGeneric | 低 | 光滑 | 通用推荐，平衡精度与稳定性 |
//! | MinmodGeneric | 高 | 光滑 | 强激波，需要最大稳定性 |

mod traits;
mod barth_jespersen;
mod venkatakrishnan;
mod minmod;

use mh_runtime::RuntimeScalar;
use crate::types::LimiterType;

// ============================================================================
// 泛型 API (Layer 3) - 主要导出
// ============================================================================

pub use traits::{SlopeLimiterGeneric, LimiterContextGeneric, NoLimiterGeneric};
pub use barth_jespersen::BarthJespersenGeneric;
pub use venkatakrishnan::VenkatakrishnanGeneric;
pub use minmod::MinmodGeneric;

// ============================================================================
// 泛型限制器工厂函数
// ============================================================================

/// 根据配置创建泛型限制器实例
///
/// # Arguments
/// * `limiter_type` - 限制器类型枚举
/// * `k` - Venkatakrishnan K 参数 (对其他类型忽略)
/// * `mesh_scale` - 网格特征尺度 (对 Venkatakrishnan 使用)
///
/// # 注意
/// `k` 和 `mesh_scale` 使用 f64 输入，内部转换为 S 类型
pub fn create_limiter_generic<S: RuntimeScalar>(
    limiter_type: LimiterType, 
    k: f64,
    mesh_scale: f64,
) -> Box<dyn SlopeLimiterGeneric<S> + Send + Sync> {
    let k_s = S::from_f64(k).unwrap_or(S::ONE);
    let scale_s = S::from_f64(mesh_scale).unwrap_or(S::ONE);
    
    match limiter_type {
        LimiterType::None => Box::new(NoLimiterGeneric::<S>::new()),
        LimiterType::BarthJespersen => Box::new(BarthJespersenGeneric::<S>::new()),
        LimiterType::Venkatakrishnan => Box::new(VenkatakrishnanGeneric::new(k_s, scale_s)),
        LimiterType::Minmod => Box::new(MinmodGeneric::<S>::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_limiter_none() {
        let limiter = create_limiter_generic::<f64>(LimiterType::None, 5.0, 1.0);
        let ctx = LimiterContextGeneric::<f64> {
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
        let limiter = create_limiter_generic::<f64>(LimiterType::BarthJespersen, 5.0, 1.0);
        let ctx = LimiterContextGeneric::<f64> {
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
        let limiter = create_limiter_generic::<f64>(LimiterType::Venkatakrishnan, 5.0, 1.0);
        let ctx = LimiterContextGeneric::<f64> {
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
        let limiter = create_limiter_generic::<f64>(LimiterType::Minmod, 5.0, 1.0);
        let ctx = LimiterContextGeneric::<f64> {
            cell_value: 1.0,
            gradient: 0.0,
            min_neighbor: 0.5,
            max_neighbor: 1.5,
            max_distance: 1.0,
        };
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
}
