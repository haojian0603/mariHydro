//! # 梯度限制器模块
//!
//! 提供梯度限制器用于控制二阶精度重构的振荡:
//!
//! - `SlopeLimiter<B>` - Backend 驱动限制器 trait
//! - `NoLimiter<B>` - 无限制（一阶精度）
//! - `BarthJespersen<B>` - Barth-Jespersen 限制器（严格 TVD）
//! - `Venkatakrishnan<B>` - Venkatakrishnan 限制器（光滑，保单调）
//! - `Minmod<B>` - Minmod 限制器（最耗散）
//!
//! ## 使用方式
//!
//! ```ignore
//! use mh_physics::numerics::limiter::{SlopeLimiter, Venkatakrishnan};
//! use mh_runtime::CpuBackend;
//!
//! let backend = CpuBackend::<f64>::new();
//! let limiter = Venkatakrishnan::<CpuBackend<f64>>::new(backend.scalar_from_f64(5.0), backend.scalar_from_f64(mesh_scale));
//! let alpha = limiter.compute_limiter(&ctx);
//! // grad_limited = grad_i * alpha
//! ```
//!
//! ## 限制器选择指南
//!
//! | 限制器 | 耗散性 | 光滑性 | 适用场景 |
//! |--------|--------|--------|----------|
//! | BarthJespersen | 中等 | 不光滑 | 需要严格 TVD 保证 |
//! | Venkatakrishnan | 低 | 光滑 | 通用推荐，平衡精度与稳定性 |
//! | Minmod | 高 | 光滑 | 强激波，需要最大稳定性 |

mod traits;
mod barth_jespersen;
mod venkatakrishnan;
mod minmod;

use mh_runtime::Backend;
use crate::types::LimiterType;

// ============================================================================
// 泛型 API (Layer 3) - 主要导出
// ============================================================================

pub use traits::{LimiterContext, NoLimiter, SlopeLimiter};
pub use barth_jespersen::BarthJespersen;
pub use venkatakrishnan::Venkatakrishnan;
pub use minmod::Minmod;

/// 静态分发限制器封装
#[derive(Debug, Clone)]
pub enum LimiterAny<B: Backend> {
    /// 无限制
    None(NoLimiter<B>),
    /// Barth-Jespersen
    BarthJespersen(BarthJespersen<B>),
    /// Venkatakrishnan
    Venkatakrishnan(Venkatakrishnan<B>),
    /// Minmod
    Minmod(Minmod<B>),
}

impl<B: Backend> SlopeLimiter<B> for LimiterAny<B> {
    #[inline]
    fn compute_limiter(&self, ctx: &LimiterContext<B>) -> B::Scalar {
        match self {
            Self::None(inner) => inner.compute_limiter(ctx),
            Self::BarthJespersen(inner) => inner.compute_limiter(ctx),
            Self::Venkatakrishnan(inner) => inner.compute_limiter(ctx),
            Self::Minmod(inner) => inner.compute_limiter(ctx),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::None(inner) => inner.name(),
            Self::BarthJespersen(inner) => inner.name(),
            Self::Venkatakrishnan(inner) => inner.name(),
            Self::Minmod(inner) => inner.name(),
        }
    }

    fn compute_limiters(&self, contexts: &[LimiterContext<B>]) -> Vec<B::Scalar> {
        match self {
            Self::None(inner) => inner.compute_limiters(contexts),
            Self::BarthJespersen(inner) => inner.compute_limiters(contexts),
            Self::Venkatakrishnan(inner) => inner.compute_limiters(contexts),
            Self::Minmod(inner) => inner.compute_limiters(contexts),
        }
    }
}

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
pub fn create_limiter<B: Backend>(
    backend: &B,
    limiter_type: LimiterType,
    k: f64,
    mesh_scale: f64,
) -> LimiterAny<B> {
    let k_s = backend.scalar_from_f64(k);
    let scale_s = backend.scalar_from_f64(mesh_scale);
    
    match limiter_type {
        LimiterType::None => LimiterAny::None(NoLimiter::<B>::new()),
        LimiterType::BarthJespersen => LimiterAny::BarthJespersen(BarthJespersen::<B>::new()),
        LimiterType::Venkatakrishnan => {
            LimiterAny::Venkatakrishnan(Venkatakrishnan::<B>::new(k_s, scale_s))
        }
        LimiterType::Minmod => LimiterAny::Minmod(Minmod::<B>::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_create_limiter_none() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let limiter = create_limiter(&backend, LimiterType::None, 5.0, 1.0);
        let ctx = LimiterContext::<mh_runtime::CpuBackend<f64>> {
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
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let limiter = create_limiter(&backend, LimiterType::BarthJespersen, 5.0, 1.0);
        let ctx = LimiterContext::<mh_runtime::CpuBackend<f64>> {
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
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let limiter = create_limiter(&backend, LimiterType::Venkatakrishnan, 5.0, 1.0);
        let ctx = LimiterContext::<mh_runtime::CpuBackend<f64>> {
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
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let limiter = create_limiter(&backend, LimiterType::Minmod, 5.0, 1.0);
        let ctx = LimiterContext::<mh_runtime::CpuBackend<f64>> {
            cell_value: 1.0,
            gradient: 0.0,
            min_neighbor: 0.5,
            max_neighbor: 1.5,
            max_distance: 1.0,
        };
        assert_eq!(limiter.compute_limiter(&ctx), 1.0);
    }
}
