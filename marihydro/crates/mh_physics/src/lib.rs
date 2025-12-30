// marihydro/crates/mh_physics/src/lib.rs

//! 物理求解器模块
//! 
//! # 架构概述
//! 
//! 本模块实现浅水方程和三维水动力学求解器，采用分层架构设计：
//! 
//! ```text
//! Layer 5 (应用层): mh_config::SolverConfig (无泛型)
//!     └─> 通过 builder 桥接
//! Layer 4 (构建层): mh_physics::builder (枚举 → 泛型)
//!     └─> 生成具体类型
//! Layer 3 (引擎层): ShallowWaterSolver<B: Backend> (全泛型)
//! ```
//! 
//! # 快速开始
//! 
//! ## 使用预设配置（推荐）
//! 
//! ```
//! use mh_config::SolverConfig;  // 从mh_config导入正确的Layer 4配置
//! use mh_physics::Layer3Config;
//! use mh_runtime::CpuBackend;
//! 
//! // 1. 创建 Layer 4 配置（无泛型，易用）
//! let layer4_config = SolverConfig::default();  // 使用默认配置
//! 
//! // 2. 转换为 Layer 3 配置（泛型，用于求解器）
//! let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&layer4_config).unwrap();
//! 
//! // 3. 创建求解器（需要网格，参见示例）
//! // let solver = ShallowWaterSolver::new(mesh, layer3_config, CpuBackend::<f64>::new());
//! 
//! // 验证配置转换成功（默认CFL为0.9）
//! assert_eq!(layer3_config.params.cfl, 0.9);
//! ```
//! 
//! ## 性能模式选择
//! 
//! ```no_run
//! // f32 模式：内存占用减半，适合GPU加速
//! use mh_config::SolverConfig;
//! use mh_physics::Layer3Config;
//! use mh_runtime::CpuBackend;
//! 
//! let config = SolverConfig::default();
//! let layer3: Layer3Config<f32> = Layer3Config::from_layer4(&config).unwrap();
//! let backend = CpuBackend::<f32>::new();
//! // let solver_f32 = ShallowWaterSolver::new(mesh, layer3, backend);
//! ```
//! 
//! ## 完整模拟流程
//! 
//! ```no_run
//! //! 这展示了完整的模拟流程（需要外部网格文件）
//! use mh_config::SolverConfig;
//! use mh_physics::{
//!     ShallowWaterSolver, ShallowWaterState,
//!     forcing::{TimeSeries, WindProvider},
//!     config_bridge::Layer3Config,
//! };
//! use mh_runtime::CpuBackend;
//! use mh_mesh::FrozenMesh;
//! 
//! // 1. 配置
//! let config = SolverConfig::default();
//! 
//! // 2. 求解器（需要网格）
//! // let mesh = FrozenMesh::default(); // 实际应从文件加载
//! // let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&config).unwrap();
//! // let mut solver = ShallowWaterSolver::new(mesh, layer3_config, CpuBackend::<f64>::new());
//! 
//! // 3. 初始状态（示例）
//! let n_cells = 100; // 实际应从网格获取
//! let backend = CpuBackend::<f64>::new();
//! let mut state = ShallowWaterState::new_with_backend(backend, n_cells);
//! // state.set_uniform_depth(1.0);  // 假设的方法
//! 
//! // 4. 外力
//! let wind = WindProvider::constant(10.0, 225.0);
//! 
//! // 5. 时间循环（示例）
//! // for step in 0..1000 {
//! //     let dt = solver.compute_dt(&state);
//! //     solver.step(&mut state, dt);
//! // }
//! ```

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

// 新增模块：字段注册、gpu、配置桥接和统一错误处理
pub mod fields;
pub mod gpu;
pub mod config_bridge;
pub mod error;

// 重导出核心运行时符号
pub use mh_runtime::{
    Backend, CpuBackend, RuntimeScalar, DeviceBuffer, MemoryLocation
};

// 重导出索引类型（仅此一处，删除所有重复导入）
pub use mh_runtime::{
    CellIndex, FaceIndex, NodeIndex, BoundaryIndex,
};

// 重导出核心抽象
pub use core::{DefaultBackend, D2, D3};

// 重导出网格抽象
pub use mesh::{MeshTopology, MeshKind, UnstructuredMeshAdapter};

// 重导出常用类型
pub use adapter::PhysicsMesh;
pub use engine::{
    AtomicFluxAccumulator, CflCalculator, FluxAccumulator, ForwardEuler, RhsComputer, SspRk2,
    SspRk3, TimeIntegrator, TimeIntegratorEnum, TimeIntegratorKind, create_integrator, NumericalScheme,
    ShallowWaterSolver,
};
pub use schemes::{
    HllcSolver, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams, WetState,
    WettingDryingConfig, WettingDryingHandler,
};
pub use state::{
    ConservedState, Flux, GradientState, RhsBuffers, ShallowWaterState, StateError,
    ShallowWaterStateGeneric,
};
pub use traits::{StateAccess, StateAccessExt, StateAccessMut, StateStatistics, StateView, StateViewMut};

// 修复SolverStats路径
pub use engine::SolverStats;

// 重导出类型（从types模块导入，不重复导入索引）
pub use types::{
    LimiterType, NumericalParams,
    ParamsValidationError, PhysicalConstants, RiemannSolverType,
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

// 重导出统一错误类型
pub use error::{PhysicsError, PhysicsResult};

// 重导出配置桥接类型（测试用）
pub use config_bridge::Layer3Config;