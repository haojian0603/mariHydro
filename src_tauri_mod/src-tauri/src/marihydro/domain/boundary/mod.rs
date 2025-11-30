// src-tauri/src/marihydro/domain/boundary/mod.rs

//! 边界条件模块

pub mod ghost;
pub mod manager;
pub mod types;

pub use ghost::GhostStateCalculator;
pub use manager::BoundaryManager;
pub use types::{BoundaryCondition, BoundaryKind, ExternalForcing};
