// src-tauri/src/marihydro/workflow/mod.rs
pub mod job;
pub mod job_v2;
pub mod manager;
pub mod manager_v2;
pub mod runner;

pub use job::{JobStatus, SimulationJob};
pub use manager::WorkflowManager;
pub use runner::SimulationRunner;

// 改进版导出
pub use job_v2::{
    TypedJob, DynamicJob, JobData, JobState,
    Pending, Running, Paused, Completed, Failed, Cancelled,
};
pub use manager_v2::{
    WorkflowManagerV2, WorkflowEvent, WorkflowEventListener, LoggingEventListener,
};
