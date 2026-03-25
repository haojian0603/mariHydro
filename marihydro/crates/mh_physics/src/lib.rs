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
//! Layer 3 (引擎层): ShallowWaterSolver<B: Backend, S: SourceTermGeneric<B>> (全泛型)
//! ```
//! 
//! # 快速开始
//! 
//! ## 使用预设配置（推荐）
//! 
//! ```
//! use mh_config::SolverConfig;  // 从mh_config导入正确的Layer 4配置
//! use mh_physics::{Layer3Config, NoSource};
//! use mh_runtime::CpuBackend;
//! 
//! // 1. 创建 Layer 4 配置（无泛型，易用）
//! let layer4_config = SolverConfig::default();  // 使用默认配置
//! 
//! // 2. 转换为 Layer 3 配置（泛型，用于求解器）
//! let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&layer4_config).unwrap();
//! 
//! // 3. 创建求解器（需要网格，参见示例）
//! // let solver = ShallowWaterSolver::<CpuBackend<f64>, NoSource<CpuBackend<f64>>>::new(
//! //     mesh,
//! //     layer3_config,
//! //     CpuBackend::<f64>::new(),
//! // );
//! 
//! // 验证配置转换成功（默认CFL为0.9）
//! assert_eq!(layer3_config.params.cfl, 0.9);
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

// 业务功能模块（持续整理中）
pub mod forcing;
// legacy 兼容层：新主链不要继续从根 `crate::limiters` 接入
pub mod limiters;
pub mod numerics;
pub mod sediment;
pub mod sources;
pub mod conservation;
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
    CellIndex, FaceIndex, NodeIndex, INVALID_INDEX
};

// 重导出核心抽象
pub use core::{D2, D3};

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
};

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
    CoriolisConfig, CoriolisSource, NoSource, SourceContributionGeneric,
    SourceContextGeneric, SourceRegistry, SourceStiffness, SourceTermGeneric,
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
pub use sediment::SedimentError;

// 重导出配置桥接类型（测试用）
pub use config_bridge::Layer3Config;

/// 统一 Prelude 模块
pub mod prelude {
    //! 所有 mh_physics 用户必须导入的 Prelude

    // 从 mh_runtime 导入所有
    pub use mh_runtime::prelude::*;

    // 物理层核心类型
    pub use crate::{
        PhysicsError, PhysicsResult,
        Backend, CpuBackend,
    };

    // 网格与索引
    pub use crate::{
        CellIndex, FaceIndex, NodeIndex,
        MeshTopology, PhysicsMesh, UnstructuredMeshAdapter,
    };

    // 求解器
    pub use crate::{
        ShallowWaterSolver, TimeIntegrator, TimeIntegratorKind,
        RiemannSolver, HllcSolver, RiemannFlux,
    };

    // 状态
    pub use crate::{
        ShallowWaterState, ConservedState, Flux, RhsBuffers,
    };

    // 示踪剂
    pub use crate::tracer::{
        TracerError, TracerField, TracerState, TracerType, TracerProperties,
        TracerTransportSolver, TracerFaceFlux,
    };

    // 泥沙
    pub use crate::sediment::{
        SedimentError, SedimentPropertiesGeneric, SedimentStateGeneric,
        TransportFormula, MeyerPeterMullerFormula, VanRijn1984Formula,
    };

    // 边界
    pub use crate::boundary::{
        BoundaryCondition, BoundaryKind, BoundaryManager,
    };
}

