// src-tauri/src/marihydro/physics/sources/mod.rs
pub mod atmosphere;
pub mod base;
pub mod coriolis;
pub mod diffusion;
pub mod friction;
pub mod inflow;
pub mod turbulence;

pub use atmosphere::{PressureGradientSource, WindStressSource};
pub use base::{FrictionDecayCalculator, SourceHelpers};
pub use coriolis::CoriolisSource;
pub use diffusion::DiffusionSource;
pub use friction::ManningFriction;
pub use inflow::InflowSource;
pub use turbulence::SmagorinskyTurbulence;
