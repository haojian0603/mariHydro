// marihydro\crates\mh_physics\src/lib.rs

//! 物理求解器模块
//!
//! 提供浅水方程数值求解功能，包括：
//! - 网格适配层 (adapter)
//! - 核心类型定义 (types)
//! - 状态管理 (state)
//! - 状态访问抽象 (traits)
//! - 数值格式 (schemes)
//! - 引擎核心 (engine) - 时间积分、通量累加、时间步控制
//! - 源项处理 (sources) - 摩擦、科氏力等
//!
//! # Trait 抽象
//!
//! - [`StateAccess`]: 状态只读访问接口
//! - [`StateAccessMut`]: 状态可变访问接口
//!
//! # 迁移说明
//!
//! 本模块从 legacy_src/physics 迁移而来。
//! 策略是"迁移而非重写"，保持算法不变，只改接口。

pub mod adapter;
pub mod boundary;
pub mod engine;
pub mod schemes;
pub mod state;
pub mod tracer;
pub mod traits;
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
pub use traits::{StateAccess, StateAccessExt, StateAccessMut, StateStatistics, StateView, StateViewMut};
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

// 重导出边界条件类型
pub use boundary::{
    BoundaryKind, BoundaryCondition, ExternalForcing, BoundaryParams,
    BoundaryFaceInfo, BoundaryManager, BoundaryDataProvider, ConstantForcingProvider,
    BoundaryError, GhostStateCalculator, GhostMomentumMode,
};

// 重导出示踪剂类型
pub use tracer::{
    TracerType, TracerProperties, TracerField, TracerFieldStats, TracerState, TracerError,
    TracerAdvectionScheme, TracerDiffusionConfig, TracerTransportConfig, TracerTransportSolver,
    MultiTracerSolver, FaceFlowData, TracerFaceFlux,
};
