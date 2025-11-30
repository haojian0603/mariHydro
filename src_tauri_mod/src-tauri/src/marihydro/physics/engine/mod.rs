// src-tauri/src/marihydro/physics/engine/mod.rs
pub mod flux_accumulator;
pub mod parallel;
pub mod solver;
pub mod time_integrator;
pub mod timestep;

pub use flux_accumulator::FluxAccumulator;
pub use parallel::{CellBasedFluxCalculator, ColoredFluxCalculator, ParallelFluxCalculator};
pub use solver::UnstructuredSolver;
pub use time_integrator::{
    create_integrator, ForwardEuler, RhsBuffers, RhsComputer, SspRk2, SspRk3, TimeIntegrator,
    TimeIntegratorKind,
};
pub use timestep::{CflCalculator, TimeStepController};
