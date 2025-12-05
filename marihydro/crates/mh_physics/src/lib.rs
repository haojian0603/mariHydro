// marihydro\crates\mh_physics\src/lib.rs

//! 物理求解器模块
//!
//! 提供浅水方程数值求解功能，包括：
//! - 网格适配层 (adapter)
//! - 核心类型定义 (types)
//! - 状态管理 (state)
//! - 数值格式 (schemes)
//! - 引擎核心 (engine) - 时间积分、通量累加、时间步控制
//! - 源项处理 (sources) - 摩擦、科氏力等
//!
//! # 迁移说明
//!
//! 本模块从 legacy_src/physics 迁移而来。
//! 策略是"迁移而非重写"，保持算法不变，只改接口。

pub mod adapter;
pub mod engine;
pub mod schemes;
pub mod state;
pub mod types;

// 待迁移模块（占位）
pub mod forcing;
pub mod gpu;
pub mod numerics;
pub mod sediment;
pub mod sources;
pub mod waves;

// 重导出常用类型
pub use adapter::PhysicsMesh;
pub use engine::{
    AtomicFluxAccumulator, CflCalculator, FluxAccumulator, ForwardEuler, RhsComputer, SspRk2,
    SspRk3, TimeIntegrator, TimeIntegratorEnum, TimeIntegratorKind, TimeStepController,
    TimeStepControllerBuilder, TimeStepStats, create_integrator,
};
pub use schemes::{
    HllcSolver, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams, WetState,
    WettingDryingConfig, WettingDryingHandler,
};
pub use state::{
    ConservedState, Flux, GradientState, RhsBuffers, ShallowWaterState, StateError,
};
pub use types::{
    BoundaryIndex, CellIndex, FaceIndex, LimiterType, NodeIndex, NumericalParams,
    NumericalParamsBuilder, ParamsValidationError, PhysicalConstants, RiemannSolverType,
    SafeDepth, SafeVelocity, SolverConfig, TimeIntegration,
};

// 重导出源项类型
pub use sources::{
    SourceContribution, SourceContext, SourceTerm, SourceHelpers,
    ManningFriction, ManningFrictionConfig, ChezyFriction, ChezyFrictionConfig,
    CoriolisConfig, CoriolisSource,
};
