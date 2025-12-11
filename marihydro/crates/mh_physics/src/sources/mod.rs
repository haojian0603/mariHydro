// crates/mh_physics/src/sources/mod.rs

//! 源项模块
//!
//! 提供浅水方程和三维模型的各种物理源项：
//!
//! # 通用源项（2D/3D）
//!
//! - 摩擦源项 ([`friction`]): Manning, Chezy 底床摩擦
//! - 科氏力 ([`coriolis`]): 地球自转效应
//! - 入流/出流 ([`inflow`]): 河流入流、降雨、蒸发
//! - 隐式处理 ([`implicit`]): 刚性源项的隐式时间积分
//!
//! # 2D 专用源项
//!
//! - 大气强迫 ([`atmosphere`]): 风应力、气压梯度
//! - 植被阻力 ([`vegetation`]): 刚性/柔性植被
//! - 波浪驱动 ([`wave_forcing`]): 辐射应力梯度
//!
//! # 湍流模型 ([`turbulence`])
//!
//! - Smagorinsky 亚格子模型（2D）
//! - k-ε 模型（3D）
//!
//! # 水工结构 ([`structures`])
//!
//! - 桥墩、堰等亚网格结构
//!
//! # 模块结构 (v0.4+)
//!
//! ```text
//! sources/
//! ├── traits.rs           # SourceTerm trait 定义
//! ├── friction.rs         # 摩擦源项
//! ├── coriolis.rs         # 科氏力
//! ├── implicit.rs         # 隐式处理
//! ├── inflow.rs           # 入流/出流
//! ├── atmosphere.rs       # 大气强迫（2D）
//! ├── vegetation.rs       # 植被阻力（2D）
//! ├── wave_forcing.rs     # 波浪驱动（2D）
//! ├── turbulence/         # 湍流模型子模块
//! │   ├── smagorinsky.rs  # 2D Smagorinsky
//! │   └── k_epsilon.rs    # 3D k-ε
//! └── structures/         # 水工结构
//!     ├── bridge_pier.rs
//!     └── weir.rs
//! ```
//!
//! # 设计
//!
//! 所有源项实现 [`SourceTerm`] trait，提供统一的计算接口：
//! - `compute_cell()` - 计算单个单元的源项贡献
//! - `compute_all()` - 批量计算所有单元
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::sources::{ManningFrictionConfig, CoriolisSource};
//! use mh_physics::sources::turbulence::{SmagorinskySolver, TurbulenceModel};
//!
//! // 创建 Manning 摩擦
//! let friction = ManningFrictionConfig::new(9.81, n_cells, 0.025);
//!
//! // 创建科氏力（北纬 30 度）
//! let coriolis = CoriolisSource::from_latitude(30.0);
//!
//! // 创建 Smagorinsky 湍流模型（推荐使用常数涡粘性）
//! let turb = SmagorinskySolver::new(n_cells, TurbulenceModel::constant(1.0));
//! ```

// ==================== 核心 trait ====================
pub mod traits;
pub mod registry;

// ==================== 通用源项 ====================
pub mod friction;
pub mod coriolis;
pub mod implicit;
pub mod inflow;

// ==================== 2D 专用源项 ====================
pub mod atmosphere;
pub mod vegetation;
pub mod wave_forcing;

// ==================== 湍流模型（独立子模块） ====================
pub mod turbulence;

// ==================== 水工结构 ====================
pub mod structures;

// ==================== 核心 trait 导出 ====================
pub use traits::{
    SourceContribution, SourceContext, SourceTerm, SourceHelpers,
    SourceContributionGeneric, SourceContextGeneric, SourceTermGeneric,
    SourceStiffness, SourceRegistryGeneric,
};

// 向后兼容别名（已废弃）
#[allow(deprecated)]
pub use traits::SourceTermF64;

pub use registry::SourceRegistry;

// ==================== 摩擦模块导出 ====================
pub use friction::{
    ManningFriction, ManningFrictionConfig,
    ChezyFriction, ChezyFrictionConfig,
    FrictionCalculator,
    ManningFrictionGeneric, ManningFrictionConfigGeneric,
    ChezyFrictionGeneric, ChezyFrictionConfigGeneric,
};

// ==================== 科氏力导出 ====================
pub use coriolis::{
    CoriolisConfig, CoriolisSource, EARTH_ANGULAR_VELOCITY,
};

// ==================== 隐式处理导出 ====================
pub use implicit::{
    ImplicitMethod, ImplicitConfig, ImplicitMomentumDecay,
    DampingCoefficient, ManningDamping, ChezyDamping,
};

// ==================== 大气源项导出 ====================
pub use atmosphere::{
    WindStressConfig, PressureGradientConfig, WindStressSource, PressureGradientSource,
    DragCoefficientMethod,
    wind_drag_coefficient_lp81, wind_drag_coefficient_wu82,
};

// ==================== 湍流模型导出 ====================
pub use turbulence::{
    // Smagorinsky (2D)
    TurbulenceModel, TurbulenceConfig, SmagorinskySolver,
    VelocityGradient,
    DEFAULT_SMAGORINSKY_CONSTANT, MIN_EDDY_VISCOSITY, MAX_EDDY_VISCOSITY,
    // 通用 trait
    TurbulenceClosure,
};

// ==================== 植被阻力导出 ====================
pub use vegetation::{
    VegetationType, VegetationConfig, VegetationSource,
    VegetationImplicit,
};

// ==================== 入流源项导出 ====================
pub use inflow::{
    InflowType, InflowConfig, InflowSource,
    RainfallConfig, RainfallSource,
    EvaporationConfig, EvaporationSource,
};

// ==================== 波浪驱动源项导出 ====================
pub use wave_forcing::{WaveForcing, WaveForcingConfig};

// ==================== 结构物源项导出 ====================
pub use structures::{BridgePierDrag, WeirFlow, WeirType};

// ==================== 扩散算子（从 numerics 重导出，保持兼容） ====================
/// 扩散相关类型（已迁移至 `numerics::operators::diffusion`）
///
/// 为保持向后兼容，从 numerics 模块重导出。
/// 新代码建议直接使用 `mh_physics::numerics::operators::diffusion`。
pub use crate::numerics::operators::diffusion::{
    DiffusionBC, DiffusionConfig, DiffusionSolver, DiffusionError,
    VariableDiffusionSolver,
    estimate_stable_dt, required_substeps,
};
