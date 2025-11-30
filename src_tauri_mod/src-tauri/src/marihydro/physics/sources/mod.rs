// src-tauri/src/marihydro/physics/sources/mod.rs

//! 源项模块
//!
//! 包含各种物理源项的实现：摩擦、扩散、科氏力、大气压等。

pub mod atmosphere;
pub mod baroclinic;
pub mod coriolis;
pub mod diffusion;
pub mod dry_wet;
pub mod friction;
pub mod implicit;
pub mod inflow;
pub mod tracer_transport;
pub mod turbulence;
pub mod turbulence_ke;
pub mod vegetation;

// 基础模块（包含 SourceTerm trait 实现）
pub mod base;

pub use atmosphere::{
    compute_pressure_gradient, compute_wind_acceleration, compute_wind_acceleration_field,
    compute_wind_stress, wind_drag_coefficient_lp81, wind_drag_coefficient_wu82,
};
pub use coriolis::{apply_coriolis_exact, is_stable, max_stable_dt};
pub use diffusion::{
    apply_diffusion_auto_substeps, apply_diffusion_explicit, apply_diffusion_explicit_variable,
    apply_diffusion_inplace, apply_diffusion_substeps, estimate_stable_dt, required_substeps,
};
pub use dry_wet::{correct_interface_depth, enforce_dry_velocity, is_wet, is_wet_simple};
pub use friction::{
    apply_friction_chezy, apply_friction_field, apply_friction_field_chezy,
    apply_friction_field_scalar, apply_friction_implicit, apply_friction_implicit_conservative,
    apply_friction_vegetation, compute_chezy_coefficient, compute_manning_coefficient,
    compute_vegetation_coefficient,
};
pub use inflow::{apply_river_inflow, apply_river_inflow_with_momentum};
pub use tracer_transport::{
    solve_tracer_step, AdvectionScheme, TracerBoundaryCondition, TracerTransportSolver,
};
pub use turbulence::{
    compute_gradient_field, compute_turbulent_kinetic_energy, compute_vorticity, SmagorinskyModel,
};
pub use turbulence_ke::{
    KEpsilonBoundary, KEpsilonCoefficients, KEpsilonConfig, KEpsilonSolver, KEpsilonState,
    TurbulenceModel,
};
pub use baroclinic::{
    BaroclinicConfig, BaroclinicSolver, DensityField, EquationOfState,
};
pub use vegetation::{
    VegetationDrag, VegetationField, VegetationProperties, VegetationType,
    equivalent_manning, vegetation_turbulence_production,
};

// 新的隐式处理导出
pub use implicit::{
    ChezyDamping, DampingCoefficient, ImplicitConfig, ImplicitDiffusion,
    ImplicitMethod, ImplicitMomentumDecay, ManningDamping,
};

