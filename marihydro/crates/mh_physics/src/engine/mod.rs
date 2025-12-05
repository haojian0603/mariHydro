// crates/mh_physics/src/engine/mod.rs

//! 物理引擎模块
//!
//! 提供求解器、时间积分器、通量累加等核心计算组件。
//!
//! # 模块结构
//!
//! - `flux_accumulator` - 通量累加器
//! - `time_integrator` - 时间积分器 (ForwardEuler, SSP-RK2, SSP-RK3)
//! - `timestep` - CFL时间步控制
//! - `solver` - 主求解器
//! - `parallel` - 并行通量计算
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/engine 迁移，保持算法不变。

pub mod flux_accumulator;
pub mod parallel;
pub mod solver;
pub mod time_integrator;
pub mod timestep;

// 重导出常用类型
pub use flux_accumulator::{FluxAccumulator, AtomicFluxAccumulator};
pub use time_integrator::{
    TimeIntegrator, TimeIntegratorKind, TimeIntegratorEnum,
    ForwardEuler, SspRk2, SspRk3, RhsComputer, create_integrator,
};
pub use timestep::{
    CflCalculator, TimeStepController, TimeStepControllerBuilder, TimeStepStats,
};
pub use solver::{
    ShallowWaterSolver, SolverConfig, SolverConfigBuilder, SolverBuilder,
    SolverStats, SolverWorkspace, HydrostaticReconstruction, HydrostaticFaceState,
    BedSlopeCorrection,
};
pub use parallel::{
    ParallelFluxCalculator, ParallelFluxConfig, ParallelFluxConfigBuilder,
    ParallelStrategy, FluxComputeMetrics,
};