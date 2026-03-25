// marihydro/crates/mh_physics/src/lib.rs

//! 物理求解器模块
//!
//! # 架构概述
//!
//! 本模块实现浅水方程和三维水动力学求解器，采用分层架构设计：
//!
//! ```text
//! Layer 5 (应用层): mh_cli / mh_workflow
//!     └─> 选择 Precision、加载 Layer 4 配置、调用真实求解器
//! Layer 4 (配置层): mh_config::SolverConfig / mh_physics::builder::SolverConfig
//!     └─> 通过显式配置转换进入 Layer 3
//! Layer 3 (引擎层): ShallowWaterSolver<B: Backend, S: SourceTermGeneric<B>>
//! ```
//!
//! # 快速开始
//!
//! ```ignore
//! use mh_config::SolverConfig;
//! use mh_physics::{Layer3Config, NoSource, PhysicsMesh, ShallowWaterSolver};
//! use mh_runtime::CpuBackend;
//! use std::sync::Arc;
//!
//! let layer4_config = SolverConfig::default();
//! let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&layer4_config).unwrap();
//! # let mesh: Arc<PhysicsMesh> = todo!();
//! let _solver = ShallowWaterSolver::<CpuBackend<f64>, NoSource<CpuBackend<f64>>>::new(
//!     mesh,
//!     layer3_config,
//!     CpuBackend::<f64>::new(),
//! );
//! ```

pub mod core;
pub mod builder;
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

pub mod forcing;
pub mod numerics;
pub mod sediment;
pub mod sources;
pub mod conservation;
pub mod waves;

pub mod fields;
pub mod config_bridge;
pub mod error;

pub use adapter::PhysicsMesh;
pub use boundary::{
    BoundaryCondition, BoundaryDataProvider, BoundaryError, BoundaryFaceInfo, BoundaryKind,
    BoundaryManager, BoundaryParams, ConstantForcingProvider, ExternalForcing,
    GhostMomentumMode, GhostStateCalculator,
};
pub use config_bridge::Layer3Config;
pub use core::{D2, D3};
pub use engine::{
    AtomicFluxAccumulator, CflCalculator, FluxAccumulator, ForwardEuler, NumericalScheme,
    RhsComputer, ShallowWaterSolver, SolverStats, SspRk2, SspRk3, TimeIntegrator,
    TimeIntegratorEnum, TimeIntegratorKind, create_integrator,
};
pub use error::{PhysicsError, PhysicsResult};
pub use mesh::{MeshKind, MeshTopology, UnstructuredMeshAdapter};
pub use mh_runtime::{Backend, CellIndex, CpuBackend, DeviceBuffer, FaceIndex, MemoryLocation,
    NodeIndex, RuntimeScalar, INVALID_INDEX};
pub use schemes::{
    HllcSolver, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams, WetState,
    WettingDryingConfig, WettingDryingHandler,
};
pub use sediment::SedimentError;
pub use sources::{
    CoriolisConfig, CoriolisSource, NoSource, SourceContextGeneric, SourceContributionGeneric,
    SourceRegistry, SourceStiffness, SourceTermGeneric,
};
pub use state::{ConservedState, Flux, GradientState, RhsBuffers, ShallowWaterState, StateError};
pub use tracer::{
    FaceFlowData, MultiTracerSolver, TracerAdvectionScheme, TracerDiffusionConfig, TracerError,
    TracerFaceFlux, TracerField, TracerFieldStats, TracerProperties, TracerState,
    TracerTransportConfig, TracerTransportSolver, TracerType,
};
pub use types::{
    BoundaryValueProvider, ConstantBoundaryProvider, LimiterType, NumericalParams,
    ParamsValidationError, PhysicalConstants, RiemannSolverType, SafeDepth, SafeVelocity,
    SolverConfig, TimeIntegration, ZeroBoundaryProvider,
};

/// 统一 Prelude 模块
pub mod prelude {
    //! 所有 mh_physics 用户必须导入的 Prelude

    pub use crate::boundary::{BoundaryCondition, BoundaryKind, BoundaryManager};
    pub use crate::sediment::{
        MeyerPeterMullerFormula, SedimentError, SedimentPropertiesGeneric, SedimentStateGeneric,
        TransportFormula, VanRijn1984Formula,
    };
    pub use crate::tracer::{
        TracerError, TracerFaceFlux, TracerField, TracerProperties, TracerState,
        TracerTransportSolver, TracerType,
    };
    pub use crate::{
        Backend, CellIndex, ConservedState, CpuBackend, FaceIndex, Flux, HllcSolver,
        MeshTopology, NodeIndex, PhysicsError, PhysicsMesh, PhysicsResult, RhsBuffers,
        RiemannFlux, RiemannSolver, ShallowWaterSolver, ShallowWaterState, TimeIntegrator,
        TimeIntegratorKind, UnstructuredMeshAdapter,
    };
    pub use mh_runtime::prelude::*;
}
