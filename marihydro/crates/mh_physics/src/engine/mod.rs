// crates/mh_physics/src/engine/mod.rs

//! 鐗╃悊寮曟搸妯″潡
//!
//! 鎻愪緵姹傝В鍣ㄣ€佹椂闂寸Н鍒嗗櫒銆侀€氶噺绱姞绛夋牳蹇冭绠楃粍浠躲€?
//!
//! # 妯″潡缁撴瀯
//!
//! - `flux_accumulator` - 閫氶噺绱姞鍣?
//! - `time_integrator` - 鏃堕棿绉垎鍣?(ForwardEuler, SSP-RK2, SSP-RK3)
//! - `timestep` - CFL鏃堕棿姝ユ帶鍒?
//! - `solver` - 涓绘眰瑙ｅ櫒
//! - `parallel` - 骞惰閫氶噺璁＄畻
//! - `semi_implicit` - 鍗婇殣寮忔椂闂存帹杩涚瓥鐣?
//! - `strategy` - 鏃堕棿绉垎绛栫暐妯″紡
//! - `pcg` - 棰勫鐞嗗叡杞搴︽硶姹傝В鍣?
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/physics/engine 杩佺Щ锛屼繚鎸佺畻娉曚笉鍙樸€?

pub mod flux_accumulator;
pub mod friction;
pub mod parallel;
pub mod pcg;
pub mod semi_implicit;
pub mod solver;
pub mod strategy;
pub mod time_integrator;
pub mod timestep;

// 閲嶅鍑哄父鐢ㄧ被鍨?
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
    BedSlopeCorrection, NumericalScheme, FallbackStrategy, StabilityOptions,
    StabilityStatus,
};
pub use parallel::{
    ParallelFluxCalculator, ParallelFluxConfig, ParallelFluxConfigBuilder,
    ParallelStrategy, FluxComputeMetrics,
};
pub use friction::{ManningFriction, FrictionConfig};
pub use semi_implicit::{SemiImplicitConfig, SemiImplicitStats, SemiImplicitStrategy};
pub use pcg::{
    PcgSolver, PcgConfig, PcgResult, PcgWorkspace,
    PreconditionerType, SparseMvp, DiagonalMatrix, CsrMatrix,
    PoissonMatrixBuilder,
};

// 閲嶅鍑虹瓥鐣ユā寮忕被鍨?
pub use strategy::{
    TimeIntegrationStrategy, StrategyKind, StepResult,
    ExplicitStrategy, ExplicitConfig,
    SemiImplicitStrategyGeneric, SemiImplicitConfig as SemiImplicitStrategyConfig,
    SolverWorkspaceGeneric,
};
