// src-tauri/src/marihydro/physics/engine/mod.rs
pub mod flux_accumulator;
pub mod parallel;
pub mod solver;
pub mod timestep;

pub use flux_accumulator::FluxAccumulator;
pub use parallel::ParallelFluxCalculator;
pub use solver::UnstructuredSolver;
pub use timestep::{CflCalculator, TimeStepController};
