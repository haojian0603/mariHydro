//! 时间积分策略
//!
//! 提供显式和半隐式时间积分的统一接口。

pub mod explicit;
pub mod semi_implicit;
pub mod workspace;
use crate::core::Backend;
use mh_runtime::RuntimeScalar as Scalar;
use crate::mesh::MeshTopology;
use crate::state::ShallowWaterState;

#[inline]
#[track_caller]
fn scalar_from_config_or_panic<S: Scalar>(value: f64, context: &'static str) -> S {
    S::from_config(value).unwrap_or_else(|| {
        panic!(
            "[mh_physics::engine::strategy] config scalar conversion failed: context={context}, value={value}"
        )
    })
}

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
            dt_used: scalar_from_config_or_panic::<S>(0.0, "StepResult.default.dt_used"),
            max_wave_speed: scalar_from_config_or_panic::<S>(
                0.0,
                "StepResult.default.max_wave_speed",
            ),
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
        state: &mut ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        workspace: &mut SolverWorkspaceGeneric<B>,
        dt: B::Scalar,
    ) -> StepResult<B::Scalar>;
    
    /// 计算稳定时间步长
    fn compute_stable_dt(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &dyn MeshTopology<B>,
        cfl: B::Scalar,
    ) -> B::Scalar;
    
    /// 是否支持大 CFL 数
    fn supports_large_cfl(&self) -> bool {
        false
    }
    
    /// 推荐的 CFL 数
    fn recommended_cfl(&self) -> B::Scalar {
        scalar_from_config_or_panic::<B::Scalar>(0.5, "TimeIntegrationStrategy.recommended_cfl")
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
    pub cfl: f64, // ALLOW_F64: Layer 4 配置参数
    /// 是否使用二阶重构
    pub second_order: bool,
    /// 是否使用静水重构
    pub hydrostatic_reconstruction: bool,
    /// 重力加速度
    pub gravity: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深
    pub h_dry: f64, // ALLOW_F64: Layer 4 配置参数
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
    pub gravity: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// 求解器容差
    pub solver_rtol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最大迭代次数
    pub solver_max_iter: usize,
    /// 隐式因子 (0=显式, 1=全隐式)
    pub theta: f64, // ALLOW_F64: Layer 4 配置参数
}

impl SemiImplicitConfig {
    /// 创建默认配置
    pub fn new() -> Self {
        Self {
            gravity: 9.81,
            h_min: 1e-6,
            solver_rtol: 1e-8,
            solver_max_iter: 200,
            theta: 0.5,
        }
    }

    /// 保守配置（高精度，收敛更稳健）
    ///
    /// - 求解器容差缩小10倍
    /// - 最大迭代次数增加50%
    /// - 隐式因子更接近隐式（更安全）
    pub fn conservative() -> Self {
        Self {
            gravity: 9.81,
            h_min: 1e-6,
            solver_rtol: 1e-9,          // 更严格
            solver_max_iter: 300,       // 更多迭代
            theta: 0.6,                 // 更隐式
        }
    }

    /// 快速配置（高性能，牺牲部分精度）
    ///
    /// - 求解器容差放宽10倍
    /// - 最大迭代次数减少50%
    /// - 隐式因子更接近显式（更快）
    pub fn fast() -> Self {
        Self {
            gravity: 9.81,
            h_min: 1e-6,
            solver_rtol: 1e-7,          // 更宽松
            solver_max_iter: 100,       // 更少迭代
            theta: 0.4,                 // 更显式
        }
    }
}

impl Default for SemiImplicitConfig {
    fn default() -> Self {
        Self::new()
    }
}

pub use explicit::ExplicitStrategy;
pub use semi_implicit::SemiImplicitStrategyGeneric;
pub use workspace::SolverWorkspaceGeneric;
