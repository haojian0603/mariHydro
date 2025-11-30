// src-tauri/src/marihydro/workflow/mod.rs

//! 工作流管理模块
//!
//! 提供作业调度、状态管理、事件通知等功能。

pub mod job;
pub mod job_v2;
pub mod manager_v2;
pub mod runner;

// 基础类型导出
pub use job::{JobStatus, SimulationJob};

// 工作流管理器（使用 V2 版本）
pub use manager_v2::{
    LoggingEventListener, WorkflowEvent, WorkflowEventListener, WorkflowManagerV2,
};

// 类型状态 Job（可选使用）
pub use job_v2::{
    Cancelled, Completed, DynamicJob, Failed, JobData, JobState, Paused, Pending, Running,
    TypedJob,
};

// 运行器
pub use runner::{MockExecutor, SimulationExecutor, SimulationRunner};
