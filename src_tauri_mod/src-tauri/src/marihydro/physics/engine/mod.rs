// src-tauri/src/marihydro/physics/engine/mod.rs
pub mod flux_accumulator;
pub mod parallel;
pub mod parallel_v2;
pub mod perf_utils;
pub mod solver;
pub mod solver_v2;
pub mod time_integrator;
pub mod timestep;
pub mod timestep_v2;

pub use flux_accumulator::FluxAccumulator;
pub use parallel::{CellBasedFluxCalculator, ColoredFluxCalculator, ParallelFluxCalculator};
// 改进版并行计算器 - 集成统一策略、自适应调度、着色验证
pub use parallel_v2::{ParallelFluxConfig, UnifiedParallelCalculator, UnifiedParallelCalculatorBuilder};
// 性能工具
pub use perf_utils::{
    parallel_max, parallel_min, parallel_sum, parallel_max_velocity, parallel_max_wave_speed,
    BatchOperator, SmallVec, AlignedVec, ResultBuffer,
};
pub use solver::UnstructuredSolver;
// 新版求解器 - 集成干湿处理、自适应Riemann求解器、隐式源项
pub use solver_v2::{ImprovedSolver, SolverConfig, SolverV2Builder};
pub use time_integrator::{
    create_integrator, ForwardEuler, RhsBuffers, RhsComputer, SspRk2, SspRk3, TimeIntegrator,
    TimeIntegratorKind,
};
pub use timestep::{CflCalculator, TimeStepController};
// 优化版时间步长控制 - 预计算dx_min、自适应增长
pub use timestep_v2::{OptimizedCflCalculator, OptimizedTimeStepController, TimeStepControllerBuilder};
