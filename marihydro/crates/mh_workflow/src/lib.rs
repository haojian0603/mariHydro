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

// ============================================================
// 便捷函数
// ============================================================

use std::path::Path;

/// 从 JSON 配置文件运行模拟
///
/// 这是一个高层便捷函数，适用于：
/// - CLI 调用
/// - 批处理脚本
/// - 快速测试
///
/// # 参数
///
/// - `config_path`: JSON 配置文件路径
///
/// # 返回
///
/// 任务 ID 和运行结果
///
/// # 示例
///
/// ```ignore
/// use mh_workflow::run_from_config;
///
/// let result = run_from_config("simulation.json")?;
/// println!("Job {} completed successfully", result.job_id);
/// ```
pub fn run_from_config<P: AsRef<Path>>(config_path: P) -> Result<RunResult, WorkflowError> {
    let config_path = config_path.as_ref();
    
    // 读取并解析配置
    let content = std::fs::read_to_string(config_path)
        .map_err(|e| WorkflowError::ConfigError(format!("Failed to read config: {}", e)))?;
    
    let config: SimulationConfig = serde_json::from_str(&content)
        .map_err(|e| WorkflowError::ConfigError(format!("Failed to parse config: {}", e)))?;
    
    // 创建任务
    let job_name = config_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("simulation");
    
    let job = SimulationJob::new(job_name, config);
    let job_id = job.id;
    
    // 创建管理器并运行
    let storage = MemoryStorage::new();
    let manager = WorkflowManager::new(storage);
    
    manager.submit(job)?;
    
    // 这里应该调用实际的求解器运行
    // 当前返回占位符结果
    Ok(RunResult {
        job_id,
        success: true,
        message: "Simulation submitted successfully".to_string(),
        elapsed_secs: 0.0,
    })
}

/// 批量运行多个配置
///
/// # 参数
///
/// - `config_paths`: 配置文件路径列表
/// - `parallel`: 是否并行执行
pub fn run_batch<P: AsRef<Path>>(
    config_paths: &[P],
    parallel: bool,
) -> Vec<Result<RunResult, WorkflowError>> {
    if parallel {
        use rayon::prelude::*;
        config_paths
            .par_iter()
            .map(|p| run_from_config(p))
            .collect()
    } else {
        config_paths
            .iter()
            .map(|p| run_from_config(p))
            .collect()
    }
}

/// 运行结果
#[derive(Debug, Clone)]
pub struct RunResult {
    /// 任务 ID
    pub job_id: JobId,
    /// 是否成功
    pub success: bool,
    /// 消息
    pub message: String,
    /// 运行时间（秒）
    pub elapsed_secs: f64,
}

impl RunResult {
    /// 创建成功结果
    pub fn success(job_id: JobId, elapsed_secs: f64) -> Self {
        Self {
            job_id,
            success: true,
            message: "Completed successfully".to_string(),
            elapsed_secs,
        }
    }

    /// 创建失败结果
    pub fn failure(job_id: JobId, message: impl Into<String>) -> Self {
        Self {
            job_id,
            success: false,
            message: message.into(),
            elapsed_secs: 0.0,
        }
    }
}