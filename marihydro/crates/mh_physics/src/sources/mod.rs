// crates/mh_physics/src/sources/mod.rs

//! 源项模块
//!
//! 提供浅水方程的各种物理源项：
//! - 摩擦源项 (Manning, Chezy)
//! - 科里奥利力
//! - 扩散
//! - 大气强迫（待实现）
//! - 湍流（待实现）
//! - 植被阻力（待实现）
//!
//! # 设计
//!
//! 所有源项实现 `SourceTerm` trait，提供统一的计算接口：
//! - `compute_cell()` - 计算单个单元的源项贡献
//! - `compute_all()` - 批量计算所有单元
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::sources::{ManningFriction, CoriolisSource};
//!
//! // 创建 Manning 摩擦
//! let friction = ManningFriction::new(9.81, n_cells, 0.025);
//!
//! // 创建科氏力（北纬 30 度）
//! let coriolis = CoriolisSource::from_latitude(30.0);
//! ```

pub mod traits;
pub mod friction;
pub mod coriolis;
pub mod diffusion;
pub mod implicit;
pub mod atmosphere;
pub mod turbulence;
pub mod vegetation;
pub mod inflow;

// 核心 trait 导出
pub use traits::{
    SourceContribution, SourceContext, SourceTerm, SourceHelpers,
};

// 摩擦模块导出
pub use friction::{
    ManningFriction, ManningFrictionConfig,
    ChezyFriction, ChezyFrictionConfig,
    FrictionCalculator,
};

// 科氏力导出
pub use coriolis::{
    CoriolisConfig, CoriolisSource, EARTH_ANGULAR_VELOCITY,
};

// 扩散导出
pub use diffusion::{
    DiffusionBC, DiffusionConfig, DiffusionSolver, DiffusionError,
    VariableDiffusionSolver,
    estimate_stable_dt, required_substeps,
};

// 隐式处理导出
pub use implicit::{
    ImplicitMethod, ImplicitConfig, ImplicitMomentumDecay,
    DampingCoefficient, ManningDamping, ChezyDamping,
};

// 大气源项导出
pub use atmosphere::{
    WindStressConfig, PressureGradientConfig, WindStressSource, PressureGradientSource,
    DragCoefficientMethod,
    wind_drag_coefficient_lp81, wind_drag_coefficient_wu82,
};

// 湍流模型导出
pub use turbulence::{
    TurbulenceModel, TurbulenceConfig, SmagorinskySolver,
    VelocityGradient,
    DEFAULT_SMAGORINSKY_CONSTANT, MIN_EDDY_VISCOSITY, MAX_EDDY_VISCOSITY,
};

// 植被阻力导出
pub use vegetation::{
    VegetationType, VegetationConfig, VegetationSource,
    VegetationImplicit,
};

// 入流源项导出
pub use inflow::{
    InflowType, InflowConfig, InflowSource,
    RainfallConfig, RainfallSource,
    EvaporationConfig, EvaporationSource,
};
