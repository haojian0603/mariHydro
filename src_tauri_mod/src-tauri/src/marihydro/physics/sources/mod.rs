// src-tauri/src/marihydro/physics/sources/mod.rs

//! 源项模块
//!
//! 包含各种物理源项的实现：摩擦、扩散、科氏力、大气压等。
//! 
//! 注意: 核心源项 trait 已被枚举化重构，主要类型定义在
//! core/traits/source.rs 中。此模块保留辅助函数和便捷构造器。

pub mod atmosphere;
pub mod baroclinic;
pub mod coriolis;
pub mod diffusion;
pub mod friction;
pub mod implicit;
pub mod inflow;
pub mod tracer_transport;
pub mod turbulence;
pub mod turbulence_ke;
pub mod vegetation;

// 基础模块（包含 SourceTerm trait 实现）
pub mod base;

// atmosphere 模块导出 (便捷构造器)
pub use atmosphere::{
    wind_drag_coefficient_lp81, wind_drag_coefficient_wu82,
    WindStressSource, PressureGradientSource,
};

// coriolis 模块导出 (便捷构造器)
pub use coriolis::CoriolisSource;

// diffusion 模块导出
pub use diffusion::{
    apply_diffusion_auto_substeps, apply_diffusion_explicit, apply_diffusion_explicit_variable,
    apply_diffusion_inplace, apply_diffusion_substeps, estimate_stable_dt, required_substeps,
    DiffusionBC,
};

// friction 模块导出 (便捷构造器)
pub use friction::{ManningFriction, ChezyFriction};

// inflow 模块导出
pub use inflow::{apply_river_inflow, apply_river_inflow_with_momentum, ActiveRiverSource, PointSource};

// tracer_transport 模块导出
pub use tracer_transport::{
    solve_tracer_step, AdvectionScheme, TracerBoundaryCondition, TracerTransportSolver,
};

// turbulence 模块导出 (便捷构造器)
pub use turbulence::{compute_vorticity, SmagorinskyTurbulence};

// turbulence_ke 模块导出
pub use turbulence_ke::{
    KEpsilonBoundary, KEpsilonCoefficients, KEpsilonConfig, KEpsilonSolver, KEpsilonState,
    TurbulenceModel,
};

// baroclinic 模块导出
pub use baroclinic::{
    BaroclinicConfig, BaroclinicSolver, DensityField, EquationOfState,
};

// vegetation 模块导出
pub use vegetation::{
    VegetationDrag, VegetationField, VegetationProperties, VegetationType,
    equivalent_manning, vegetation_turbulence_production,
};

// implicit 模块导出
pub use implicit::{
    ChezyDamping, DampingCoefficient, ImplicitConfig, ImplicitDiffusion,
    ImplicitMethod, ImplicitMomentumDecay, ManningDamping,
};

