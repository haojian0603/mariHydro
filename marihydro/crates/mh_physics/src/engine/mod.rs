//! 物理引擎模块
//!
//! 提供浅水方程求解器的核心计算组件，支持Backend泛型化运行时切换。
//! 本模块属于Layer 3(Engine层)，所有数值类型使用RuntimeScalar泛型参数。

pub mod flux_accumulator;
pub mod friction;
pub mod parallel;
pub mod pcg;
pub mod solver;
pub mod strategy;
pub mod time_integrator;
pub mod timestep;

// 重导出常用类型
pub use flux_accumulator::{FluxAccumulator, AtomicFluxAccumulator};
pub use time_integrator::{
    ForwardEuler, SspRk2, SspRk3, RhsComputer, create_integrator,
    TimeIntegrator, TimeIntegratorKind, TimeIntegratorEnum,
};
pub use timestep::{
    CflCalculator, TimeStepController, TimeStepControllerBuilder, TimeStepStats,
};
pub use solver::{
    ShallowWaterSolver, SolverStats, SolverWorkspaceGeneric as SolverWorkspace,
    HydrostaticReconstruction, HydrostaticFaceState, BedSlopeCorrection,
    NumericalScheme, FallbackStrategy, StabilityOptions, StabilityStatus,
    NanDetectionResult,
};
pub use parallel::{
    ParallelFluxCalculator, ParallelFluxConfig, ParallelFluxConfigBuilder,
    ParallelStrategy, FluxComputeMetrics,
};
pub use friction::{ManningFriction, FrictionConfig};
pub use pcg::{
    PcgSolver, PcgWorkspace, PcgResult, PcgConfig,
    PreconditionerType, SparseMvp, DiagonalMatrix, PoissonMatrixBuilder,
};
pub use strategy::{
    TimeIntegrationStrategy, StrategyKind, StepResult,
    ExplicitStrategy, ExplicitConfig,
    SemiImplicitStrategyGeneric, SemiImplicitConfig as SemiImplicitStrategyConfig,
    SolverWorkspaceGeneric,
};

// 从numerics模块重导出CsrMatrix
pub use crate::numerics::linear_algebra::csr::CsrMatrix;