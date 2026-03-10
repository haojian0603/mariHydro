// marihydro\crates\mh_physics\src/lib.rs

//! 鐗╃悊姹傝В鍣ㄦā鍧?
//!
//! 鎻愪緵娴呮按鏂圭▼鏁板€兼眰瑙ｅ姛鑳斤紝鍖呮嫭锛?
//! - 鏍稿績鎶借薄灞?(core) - Backend, Buffer, f64 鎶借薄
//! - 缃戞牸閫傞厤灞?(adapter)
//! - 鏍稿績绫诲瀷瀹氫箟 (types)
//! - 鐘舵€佺鐞?(state)
//! - 鐘舵€佽闂娊璞?(traits)
//! - 鏁板€兼牸寮?(schemes)
//! - 寮曟搸鏍稿績 (engine) - 鏃堕棿绉垎銆侀€氶噺绱姞銆佹椂闂存鎺у埗
//! - 婧愰」澶勭悊 (sources) - 鎽╂摝銆佺姘忓姏銆佹箥娴佺瓑
//! - 鍨傚悜鍓栭潰 (vertical) - 蟽鍧愭爣銆佸垎灞傜姸鎬?
//!
//! # Trait 鎶借薄
//!
//! - [`StateAccess`]: 鐘舵€佸彧璇昏闂帴鍙?
//! - [`StateAccessMut`]: 鐘舵€佸彲鍙樿闂帴鍙?
//!

// 鏍稿績鎶借薄灞?
pub mod core;

// 缃戞牸鎶借薄灞?
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

// 寰呰縼绉绘ā鍧楋紙鍗犱綅锛?
pub mod forcing;
pub mod numerics;
pub mod sediment;
pub mod sources;
pub mod waves;

// 鏂板妯″潡锛氬瓧娈垫敞鍐屻€乬pu鍜岀畻瀛愭娊璞?
pub mod fields;
pub mod gpu;
pub mod operators;

// 閲嶅鍑烘牳蹇冩娊璞?
pub use core::{Backend, CpuBackend, DefaultBackend, Scalar, DeviceBuffer, D2, D3};

// 閲嶅鍑虹綉鏍兼娊璞?
pub use mesh::{MeshTopology, MeshKind, UnstructuredMeshAdapter};

// 閲嶅鍑哄父鐢ㄧ被鍨?
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

// 閲嶅鍑烘簮椤圭被鍨?
pub use sources::{
    SourceContributionGeneric, SourceContextGeneric, SourceTermGeneric, SourceStiffness, SourceRegistry,
    SourceHelpers,
    ManningFriction, ManningFrictionConfig, ChezyFriction, ChezyFrictionConfig,
    CoriolisConfig, CoriolisSource,
};

// 閲嶅鍑鸿竟鐣屾潯浠剁被鍨?
pub use boundary::{
    BoundaryKind, BoundaryCondition, ExternalForcing, BoundaryParams,
    BoundaryFaceInfo, BoundaryManager, BoundaryDataProvider, ConstantForcingProvider,
    BoundaryError, GhostStateCalculator, GhostMomentumMode,
};

// 閲嶅鍑虹ず韪墏绫诲瀷
pub use tracer::{
    TracerType, TracerProperties, TracerField, TracerFieldStats, TracerState, TracerError,
    TracerAdvectionScheme, TracerDiffusionConfig, TracerTransportConfig, TracerTransportSolver,
    MultiTracerSolver, FaceFlowData, TracerFaceFlux,
};
