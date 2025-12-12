// crates/mh_workflow/src/lib.rs

//! MariHydro 工作流管理模块
//!
//! 提供完整的任务管理和计算调度功能。
//!
//! # 模块结构
//!
//! - [`job`]: 任务定义和状态
//! - [`events`]: 事件系统
//! - [`storage`]: 持久化存储
//! - [`manager`]: 任务管理器
//! - [`runner`]: 任务运行器
//! - [`scheduler`]: 混合调度器
//!
//! # 示例
//!
//! ```rust,ignore
//! use mh_workflow::{WorkflowManager, SimulationJob, SimulationConfig, MemoryStorage};
//!
//! // 创建工作流管理器
//! let storage = MemoryStorage::new();
//! let manager = WorkflowManager::new(storage);
//!
//! // 创建模拟任务
//! let config = SimulationConfig {
//!     project_path: "project.mhp".into(),
//!     start_time: 0.0,
//!     end_time: 3600.0,
//!     output_interval: 60.0,
//!     use_gpu: true,
//!     num_threads: 0,
//! };
//!
//! let job = SimulationJob::new("Test Simulation", config);
//!
//! // 提交任务
//! let job_id = manager.submit(job)?;
//! ```

pub mod events;
pub mod job;
pub mod manager;
pub mod runner;
pub mod scheduler;
pub mod storage;

// 重导出核心类型
pub use events::{EventDispatcher, EventListener, WorkflowEvent};
pub use job::{JobId, JobPriority, JobStatus, SimulationConfig, SimulationJob};
pub use manager::{WorkflowError, WorkflowManager};
pub use runner::{JobRunner, RunnerConfig, RunnerError};
pub use scheduler::{
    DeviceSelection, HybridConfig, HybridScheduler, 
    HybridStrategy, PerformanceStats, SchedulerDiagnostics, SelectionStats,
};
pub use storage::{FileStorage, MemoryStorage, Storage, StorageError};
