// src-tauri/src/marihydro/workflow/mod.rs
pub mod job;
pub mod manager;
pub mod runner;

pub use job::{JobStatus, SimulationJob};
pub use manager::WorkflowManager;
pub use runner::SimulationRunner;
