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

/// 创建限制器实例（返回具体类型而非 Box<dyn>）
pub fn create_limiter(limiter_type: LimiterType) -> LimiterEnum {
    LimiterEnum::new(limiter_type)
}

/// 限制器枚举包装器 - 替代 Box<dyn Limiter>
/// 
/// 使用枚举分发避免 E0038 (trait不是dyn兼容) 问题
#[derive(Clone)]
pub struct LimiterEnum {
    kind: LimiterType,
}

impl LimiterEnum {
    /// 创建新的限制器
    pub fn new(kind: LimiterType) -> Self {
        Self { kind }
    }

    /// 限制器名称
    pub fn name(&self) -> &'static str {
        self.kind.name()
    }

    /// 获取限制器能力
    pub fn capabilities(&self) -> LimiterCapabilities {
        match self.kind {
            LimiterType::None => LimiterCapabilities {
                parallel: true,
                smooth: true,
                monotone: false,
                order: 0,
            },
            LimiterType::Minmod | LimiterType::Superbee => LimiterCapabilities {
                parallel: true,
                smooth: false,
                monotone: true,
                order: 2,
            },
            LimiterType::VanLeer => LimiterCapabilities {
                parallel: true,
                smooth: true,
                monotone: true,
                order: 2,
            },
            LimiterType::BarthJespersen => LimiterCapabilities {
                parallel: true,
                smooth: false,
                monotone: true,
                order: 2,
            },
            LimiterType::Venkatakrishnan => LimiterCapabilities {
                parallel: true,
                smooth: true,
                monotone: false,
                order: 2,
            },
        }
    }

    /// 批量计算限制因子（使用具体类型）
    pub fn compute_limiters(
        &self,
        ctx: &LimiterContext,
        mesh: &crate::marihydro::domain::mesh::UnstructuredMesh,
        output: &mut [f64],
    ) -> MhResult<()> {
        match self.kind {
            LimiterType::None => {
                output[..mesh.n_cells()].fill(1.0);
                Ok(())
            }
            LimiterType::Minmod => {
                let limiter = MinmodLimiter::new();
                limiter.compute_limiters(ctx, mesh, output)
            }
            LimiterType::Superbee => {
                let limiter = SuperbeeLimiter::new();
                limiter.compute_limiters(ctx, mesh, output)
            }
            LimiterType::VanLeer => {
                let limiter = VanLeerLimiter::new();
                limiter.compute_limiters(ctx, mesh, output)
            }
            LimiterType::BarthJespersen => {
                let limiter = BarthJespersenLimiter::default();
                limiter.compute_limiters(ctx.field, ctx.gradient, mesh, output)
            }
            LimiterType::Venkatakrishnan => {
                let limiter = VenkatakrishnanLimiter::new(0.3);
                limiter.compute_limiters(ctx.field, ctx.gradient, mesh, output)
            }
        }
    }

    /// 应用限制器到梯度（原地修改）
    pub fn apply(
        &self,
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &crate::marihydro::domain::mesh::UnstructuredMesh,
    ) -> MhResult<()> {
        let ctx = LimiterContext::new(field, gradient);
        let mut limiters = vec![1.0; mesh.n_cells()];
        self.compute_limiters(&ctx, mesh, &mut limiters)?;
        gradient.apply_limiter(&limiters);
        Ok(())
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

