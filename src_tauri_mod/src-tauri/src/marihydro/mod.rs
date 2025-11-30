// src-tauri/src/marihydro/mod.rs
pub mod core;
pub mod domain;
pub mod forcing;
pub mod geo;
pub mod infra;
pub mod io;
pub mod physics;
pub mod workflow;

pub use core::error::{MhError, MhResult};
pub use core::types::{CellIndex, FaceIndex, NumericalParams};
pub use domain::mesh::UnstructuredMesh;
pub use physics::engine::solver::UnstructuredSolver;
pub use workflow::WorkflowManager;
