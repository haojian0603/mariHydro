// src-tauri/src/marihydro/physics/engine/mod.rs

//! 物理引擎模块
//!
//! 提供求解器、时间积分器、并行计算等核心计算组件。

pub mod flux_accumulator;
pub mod parallel_v2;
pub mod perf_utils;
pub mod solver_v2;
pub mod time_integrator;
pub mod timestep_v2;

// 通量累加器
pub use flux_accumulator::FluxAccumulator;

// 并行计算器
pub use parallel_v2::{ParallelFluxConfig, UnifiedParallelCalculator, UnifiedParallelCalculatorBuilder};

// 性能工具
pub use perf_utils::{
    parallel_max, parallel_min, parallel_sum, parallel_max_velocity, parallel_max_wave_speed,
    BatchOperator, SmallVec, AlignedVec, ResultBuffer,
};

// 求解器
pub use solver_v2::{SolverConfig, UnstructuredSolverV2, SolverBuilderV2};

// 类型别名（向后兼容）
pub type ImprovedSolver = UnstructuredSolverV2;
pub type SolverV2Builder = SolverBuilderV2;

// 时间积分器
pub use time_integrator::{
    create_integrator, ForwardEuler, RhsBuffers, RhsComputer, SspRk2, SspRk3, TimeIntegrator,
    TimeIntegratorKind,
};

// 时间步控制
pub use timestep_v2::{OptimizedCflCalculator, OptimizedTimeStepController, TimeStepControllerBuilder};
