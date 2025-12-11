//! 时间积分策略
//!
//! 提供显式和半隐式时间积分的统一接口。

pub mod explicit;
pub mod semi_implicit;
pub mod workspace;

use crate::core::{Backend, Scalar};
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterStateGeneric;

// workspace 模块在下面的 pub use 中进行重导出

/// 时间积分步进结果
#[derive(Debug, Clone)]
pub struct StepResult<S: Scalar> {
    /// 使用的时间步长
    pub dt_used: S,
    /// 最大波速
    pub max_wave_speed: S,
    /// 干单元数量
    pub dry_cells: usize,
    /// 被限制的单元数量
    pub limited_cells: usize,
    /// 是否收敛（半隐式）
    pub converged: bool,
    /// 迭代次数（半隐式）
    pub iterations: usize,
}

impl<S: Scalar> Default for StepResult<S> {
    fn default() -> Self {
        Self {
            dt_used: <S as Scalar>::from_f64(0.0),
            max_wave_speed: <S as Scalar>::from_f64(0.0),
            dry_cells: 0,
            limited_cells: 0,
            converged: true,
            iterations: 0,
        }
    }
}

/// 时间积分策略 trait
pub trait TimeIntegrationStrategy<B: Backend>: Send + Sync {
    /// 策略名称
    fn name(&self) -> &'static str;
    
    /// 执行单步时间积分
    fn step(
        &mut self,
        state: &mut ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopology<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar>;
    
    /// 计算稳定时间步长
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        mesh: &dyn MeshTopology<B>,
        cfl: B::Scalar,
    ) -> B::Scalar;
    
    /// 是否支持大 CFL 数
    fn supports_large_cfl(&self) -> bool {
        false
    }
    
    /// 推荐的 CFL 数
    fn recommended_cfl(&self) -> B::Scalar {
        <B::Scalar as Scalar>::from_f64(0.5)
    }
}

/// 策略类型枚举
#[derive(Debug, Clone)]
pub enum StrategyKind {
    /// 显式 Godunov
    Explicit(ExplicitConfig),
    /// 半隐式压力校正
    SemiImplicit(SemiImplicitConfig),
}

/// 显式策略配置
#[derive(Debug, Clone, Default)]
pub struct ExplicitConfig {
    /// CFL 数
    pub cfl: f64,
    /// 是否使用二阶重构
    pub second_order: bool,
    /// 是否使用静水重构
    pub hydrostatic_reconstruction: bool,
    /// 重力加速度
    pub gravity: f64,
    /// 最小水深
    pub h_dry: f64,
}

impl ExplicitConfig {
    /// 创建默认配置
    pub fn new() -> Self {
        Self {
            cfl: 0.5,
            second_order: false,
            hydrostatic_reconstruction: true,
            gravity: 9.81,
            h_dry: 1e-6,
        }
    }
}

/// 半隐式策略配置
#[derive(Debug, Clone)]
pub struct SemiImplicitConfig {
    /// 重力加速度
    pub gravity: f64,
    /// 最小水深
    pub h_min: f64,
    /// 求解器容差
    pub solver_rtol: f64,
    /// 最大迭代次数
    pub solver_max_iter: usize,
    /// 隐式因子 (0=显式, 1=全隐式)
    pub theta: f64,
}

impl Default for SemiImplicitConfig {
    fn default() -> Self {
        Self {
            gravity: 9.81,
            h_min: 1e-6,
            solver_rtol: 1e-8,
            solver_max_iter: 200,
            theta: 0.5,
        }
    }
}

pub use explicit::ExplicitStrategy;
pub use semi_implicit::SemiImplicitStrategyGeneric;
pub use workspace::SolverWorkspaceGeneric;
