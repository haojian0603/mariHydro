// src-tauri/src/marihydro/physics/numerics/limiter/mod.rs

//! 梯度限制器模块
//!
//! 提供多种梯度限制器实现，用于控制重构过程中的振荡。
//!
//! # 可用限制器
//!
//! - [`NoLimiter`]: 无限制（一阶精度）
//! - [`MinmodLimiter`]: Minmod 限制器（保守）
//! - [`SuperbeeLimiter`]: Superbee 限制器（激进）
//! - [`VanLeerLimiter`]: Van Leer 限制器（平滑）
//! - [`BarthJespersenLimiter`]: Barth-Jespersen 限制器（严格单调）
//! - [`VenkatakrishnanLimiter`]: Venkatakrishnan 限制器（平滑收敛）

pub mod barth_jespersen;
pub mod minmod;
pub mod traits;
pub mod venkatakrishnan;

// 旧接口（保持兼容）
pub use barth_jespersen::BarthJespersenLimiter;
pub use venkatakrishnan::VenkatakrishnanLimiter;

// 新的统一接口
pub use minmod::{MinmodLimiter, SuperbeeLimiter, VanLeerLimiter};
pub use traits::{
    CellLimitResult, Limiter, LimiterCapabilities, LimiterConfig, LimiterContext,
    LimiterStatistics, NoLimiter,
};

use super::gradient::ScalarGradientStorage;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;

/// 限制器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LimiterType {
    /// 无限制
    None,
    /// Minmod（保守）
    Minmod,
    /// Superbee（激进）
    Superbee,
    /// Van Leer（平滑）
    VanLeer,
    /// Barth-Jespersen（严格单调）
    BarthJespersen,
    /// Venkatakrishnan（平滑收敛）
    #[default]
    Venkatakrishnan,
}

impl LimiterType {
    /// 获取限制器名称
    pub fn name(&self) -> &'static str {
        match self {
            LimiterType::None => "None",
            LimiterType::Minmod => "Minmod",
            LimiterType::Superbee => "Superbee",
            LimiterType::VanLeer => "Van Leer",
            LimiterType::BarthJespersen => "Barth-Jespersen",
            LimiterType::Venkatakrishnan => "Venkatakrishnan",
        }
    }

    /// 是否平滑（适合稳态求解）
    pub fn is_smooth(&self) -> bool {
        matches!(
            self,
            LimiterType::VanLeer | LimiterType::Venkatakrishnan
        )
    }
}

/// 旧接口（兼容性）
pub trait GradientLimiter: Send + Sync {
    fn limit<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &M,
    ) -> MhResult<()>;

    fn compute_limiters<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &ScalarGradientStorage,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()>;

    fn name(&self) -> &'static str;
}

/// 创建限制器实例
pub fn create_limiter(limiter_type: LimiterType) -> Box<dyn Limiter> {
    match limiter_type {
        LimiterType::None => Box::new(NoLimiter),
        LimiterType::Minmod => Box::new(MinmodLimiter::new()),
        LimiterType::Superbee => Box::new(SuperbeeLimiter::new()),
        LimiterType::VanLeer => Box::new(VanLeerLimiter::new()),
        LimiterType::BarthJespersen => Box::new(BarthJespersenAdapter::new()),
        LimiterType::Venkatakrishnan => Box::new(VenkatakrishnanAdapter::new(0.3)),
    }
}

/// BarthJespersen 适配器（实现新 trait）
struct BarthJespersenAdapter {
    inner: BarthJespersenLimiter,
}

impl BarthJespersenAdapter {
    fn new() -> Self {
        Self {
            inner: BarthJespersenLimiter::default(),
        }
    }
}

impl Limiter for BarthJespersenAdapter {
    fn name(&self) -> &'static str {
        "Barth-Jespersen"
    }

    fn capabilities(&self) -> LimiterCapabilities {
        LimiterCapabilities {
            parallel: true,
            smooth: false,
            monotone: true,
            order: 2,
        }
    }

    fn limit_cell<M: MeshAccess>(
        &self,
        cell: crate::marihydro::core::types::CellIndex,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> CellLimitResult {
        // 委托给原有实现的单元计算
        let mut output = vec![1.0; mesh.n_cells()];
        let _ = self.inner.compute_limiters(ctx.field, ctx.gradient, mesh, &mut output);
        CellLimitResult::new(output[cell.0])
    }
}

/// Venkatakrishnan 适配器（实现新 trait）
struct VenkatakrishnanAdapter {
    inner: VenkatakrishnanLimiter,
}

impl VenkatakrishnanAdapter {
    fn new(k: f64) -> Self {
        Self {
            inner: VenkatakrishnanLimiter::new(k),
        }
    }
}

impl Limiter for VenkatakrishnanAdapter {
    fn name(&self) -> &'static str {
        "Venkatakrishnan"
    }

    fn capabilities(&self) -> LimiterCapabilities {
        LimiterCapabilities {
            parallel: true,
            smooth: true,
            monotone: false,
            order: 2,
        }
    }

    fn limit_cell<M: MeshAccess>(
        &self,
        cell: crate::marihydro::core::types::CellIndex,
        ctx: &LimiterContext,
        mesh: &M,
    ) -> CellLimitResult {
        let mut output = vec![1.0; mesh.n_cells()];
        let _ = self.inner.compute_limiters(ctx.field, ctx.gradient, mesh, &mut output);
        CellLimitResult::new(output[cell.0])
    }
}

