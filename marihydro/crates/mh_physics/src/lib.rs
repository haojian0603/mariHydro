// marihydro\crates\mh_physics\src/lib.rs

//! 物理求解器模块
//!
//! 提供浅水方程数值求解功能，包括：
//! - 核心抽象层 (core) - Backend, Buffer, Scalar 抽象
//! - 构建器层 (builder) - 从无泛型配置到泛型引擎的桥梁
//! - 网格适配层 (adapter)
//! - 核心类型定义 (types)
//! - 状态管理 (state)
//! - 状态访问抽象 (traits)
//! - 数值格式 (schemes)
//! - 引擎核心 (engine) - 时间积分、通量累加、时间步控制
//! - 源项处理 (sources) - 摩擦、科氏力、湍流等
//! - 垂向剖面 (vertical) - σ坐标、分层状态
//!
//! # 架构层级
//!
//! ```text
//! Layer 5: Application (无泛型)
//!     └─> SolverConfig, Precision
//! Layer 4: Builder (枚举→泛型桥接)
//!     └─> SolverBuilder -> Box<dyn DynSolver>
//! Layer 3: Engine (全泛型)
//!     └─> ShallowWaterSolver<B>
//! ```
//!
//! # Trait 抽象
//!
//! - [`StateAccess`]: 状态只读访问接口
//! - [`StateAccessMut`]: 状态可变访问接口
//! - [`DynSolver`]: 运行时多态求解器接口
//!

// 核心抽象层
pub mod core;

// 构建器层（无泛型入口）
pub mod builder;

// 网格抽象层
pub mod mesh;

pub mod adapter;
pub mod boundary;
pub mod engine;
pub mod schemes;
pub mod state;
pub mod tracer;
pub mod assimilation;
pub mod traits;
pub mod types;
pub mod vertical;

// 待迁移模块（占位）
pub mod forcing;
pub mod numerics;
pub mod sediment;
pub mod sources;
pub mod waves;

// 新增模块：字段注册、gpu和算子抽象
pub mod fields;
pub mod gpu;
pub mod operators;

// 重导出核心抽象
pub use core::{Backend, CpuBackend, DefaultBackend, Scalar, DeviceBuffer, D2, D3};

// 重导出网格抽象
pub use mesh::{MeshTopology, MeshKind, UnstructuredMeshAdapter};

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
    ShallowWaterStateGeneric, ShallowWaterStateDefault,
};
pub use traits::{StateAccess, StateAccessExt, StateAccessMut, StateStatistics, StateView, StateViewMut};
pub use types::{
    BoundaryIndex, CellIndex, FaceIndex, LimiterType, NodeIndex, NumericalParams,
    NumericalParamsBuilder, ParamsValidationError, PhysicalConstants, RiemannSolverType,
    SafeDepth, SafeVelocity, SolverConfig, TimeIntegration,
    BoundaryValueProvider, ConstantBoundaryProvider, ZeroBoundaryProvider,
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
