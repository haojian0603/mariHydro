// src-tauri/src/marihydro/domain/boundary/mod.rs

//! 边界条件模块

pub mod ghost;
pub mod manager;
pub mod structures;
pub mod types;

pub use ghost::GhostStateCalculator;
pub use manager::BoundaryManager;
pub use structures::{
    BroadCrestedWeir, Culvert, Gate, GateType, HydraulicStructure,
    PumpMode, PumpStation, StructureManager,
};
pub use types::{BoundaryCondition, BoundaryKind, ExternalForcing};

