// crates/mh_workflow/src/job.rs

//! 任务定义模块
//!
//! 定义模拟任务的数据结构和状态管理。

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// 任务ID
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(Uuid);

impl JobId {
    /// 创建新的任务ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// 从UUID创建
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// 获取内部UUID
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for JobId {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// 任务状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    /// 等待中
    Pending,
    /// 排队中
    Queued,
    /// 运行中
    Running,
    /// 已暂停
    Paused,
    /// 已完成
    Completed,
    /// 失败
    Failed,
    /// 已取消
    Cancelled,
}

impl JobStatus {
    /// 是否为终止状态
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }

    /// 是否可以取消
    pub fn can_cancel(&self) -> bool {
        matches!(self, Self::Pending | Self::Queued | Self::Running | Self::Paused)
    }

    /// 是否可以暂停
    pub fn can_pause(&self) -> bool {
        matches!(self, Self::Running)
    }

    /// 是否可以恢复
    pub fn can_resume(&self) -> bool {
        matches!(self, Self::Paused)
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Pending => "Pending",
            Self::Queued => "Queued",
            Self::Running => "Running",
            Self::Paused => "Paused",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
        };
        write!(f, "{}", s)
    }
}

/// 任务优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobPriority {
    /// 低优先级
    Low = 0,
    /// 普通优先级
    Normal = 1,
    /// 高优先级
    High = 2,
    /// 紧急优先级
    Critical = 3,
}

impl Default for JobPriority {
    fn default() -> Self {
        Self::Normal
    }
}

impl std::fmt::Display for JobPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Low => "Low",
            Self::Normal => "Normal",
            Self::High => "High",
            Self::Critical => "Critical",
        };
        write!(f, "{}", s)
    }
}

/// 模拟任务配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// 项目路径
    pub project_path: PathBuf,
    /// 开始时间 (秒)
    pub start_time: f64,
    /// 结束时间 (秒)
    pub end_time: f64,
    /// 输出间隔 (秒)
    pub output_interval: f64,
    /// 使用GPU
    pub use_gpu: bool,
    /// 线程数 (0=自动)
    pub num_threads: usize,
    /// 检查点间隔 (秒, 0=禁用)
    pub checkpoint_interval: f64,
    /// 最大CFL数
    pub max_cfl: f64,
    /// 是否启用输出
    pub enable_output: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            project_path: PathBuf::new(),
            start_time: 0.0,
            end_time: 3600.0,
            output_interval: 60.0,
            use_gpu: false,
            num_threads: 0,
            checkpoint_interval: 300.0,
            max_cfl: 0.5,
            enable_output: true,
        }
    }
}

impl SimulationConfig {
    /// 创建新配置
    pub fn new(project_path: impl Into<PathBuf>) -> Self {
        Self {
            project_path: project_path.into(),
            ..Default::default()
        }
    }

    /// 设置时间范围
    pub fn with_time_range(mut self, start: f64, end: f64) -> Self {
        self.start_time = start;
        self.end_time = end;
        self
    }

    /// 设置输出间隔
    pub fn with_output_interval(mut self, interval: f64) -> Self {
        self.output_interval = interval;
        self
    }

    /// 启用GPU
    pub fn with_gpu(mut self, use_gpu: bool) -> Self {
        self.use_gpu = use_gpu;
        self
    }

    /// 设置线程数
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// 验证配置
    pub fn validate(&self) -> Result<(), String> {
        if self.end_time <= self.start_time {
            return Err("End time must be greater than start time".into());
        }
        if self.output_interval <= 0.0 {
            return Err("Output interval must be positive".into());
        }
        if self.max_cfl <= 0.0 || self.max_cfl > 1.0 {
            return Err("Max CFL must be in range (0, 1]".into());
        }
        Ok(())
    }
}

/// 模拟任务
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationJob {
    /// 任务ID
    pub id: JobId,
    /// 任务名称
    pub name: String,
    /// 任务描述
    pub description: Option<String>,
    /// 任务配置
    pub config: SimulationConfig,
    /// 任务状态
    pub status: JobStatus,
    /// 优先级
    pub priority: JobPriority,
    /// 创建时间
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// 开始时间
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    /// 完成时间
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    /// 进度 (0.0-1.0)
    pub progress: f64,
    /// 当前模拟时间
    pub current_time: f64,
    /// 已完成时间步数
    pub completed_steps: u64,
    /// 错误信息
    pub error: Option<String>,
    /// 标签
    pub tags: Vec<String>,
}

impl SimulationJob {
    /// 创建新任务
    pub fn new(name: impl Into<String>, config: SimulationConfig) -> Self {
        Self {
            id: JobId::new(),
            name: name.into(),
            description: None,
            config,
            status: JobStatus::Pending,
            priority: JobPriority::Normal,
            created_at: chrono::Utc::now(),
            started_at: None,
            completed_at: None,
            progress: 0.0,
            current_time: 0.0,
            completed_steps: 0,
            error: None,
            tags: Vec::new(),
        }
    }

    /// 设置描述
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: JobPriority) -> Self {
        self.priority = priority;
        self
    }

    /// 添加标签
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// 更新进度
    pub fn update_progress(&mut self, current_time: f64, steps: u64) {
        self.current_time = current_time;
        self.completed_steps = steps;

        let duration = self.config.end_time - self.config.start_time;
        if duration > 0.0 {
            self.progress = (current_time - self.config.start_time) / duration;
            self.progress = self.progress.clamp(0.0, 1.0);
        }
    }

    /// 标记开始
    pub fn mark_started(&mut self) {
        self.status = JobStatus::Running;
        self.started_at = Some(chrono::Utc::now());
        self.current_time = self.config.start_time;
    }

    /// 标记完成
    pub fn mark_completed(&mut self) {
        self.status = JobStatus::Completed;
        self.completed_at = Some(chrono::Utc::now());
        self.progress = 1.0;
        self.current_time = self.config.end_time;
    }

    /// 标记失败
    pub fn mark_failed(&mut self, error: impl Into<String>) {
        self.status = JobStatus::Failed;
        self.completed_at = Some(chrono::Utc::now());
        self.error = Some(error.into());
    }

    /// 标记取消
    pub fn mark_cancelled(&mut self) {
        self.status = JobStatus::Cancelled;
        self.completed_at = Some(chrono::Utc::now());
    }

    /// 标记暂停
    pub fn mark_paused(&mut self) {
        self.status = JobStatus::Paused;
    }

    /// 标记恢复
    pub fn mark_resumed(&mut self) {
        self.status = JobStatus::Running;
    }

    /// 获取运行时长
    pub fn elapsed(&self) -> Option<chrono::Duration> {
        self.started_at.map(|start| {
            let end = self.completed_at.unwrap_or_else(chrono::Utc::now);
            end - start
        })
    }

    /// 获取剩余时间估计
    pub fn estimated_remaining(&self) -> Option<chrono::Duration> {
        if self.progress <= 0.0 || self.progress >= 1.0 {
            return None;
        }

        self.elapsed().map(|elapsed| {
            let total_secs = elapsed.num_seconds() as f64 / self.progress;
            let remaining_secs = total_secs * (1.0 - self.progress);
            chrono::Duration::seconds(remaining_secs as i64)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_id() {
        let id1 = JobId::new();
        let id2 = JobId::new();
        assert_ne!(id1, id2);

        let id_str = id1.to_string();
        let id_parsed: JobId = id_str.parse().unwrap();
        assert_eq!(id1, id_parsed);
    }

    #[test]
    fn test_job_status() {
        assert!(JobStatus::Completed.is_terminal());
        assert!(JobStatus::Failed.is_terminal());
        assert!(!JobStatus::Running.is_terminal());

        assert!(JobStatus::Running.can_cancel());
        assert!(JobStatus::Running.can_pause());
        assert!(!JobStatus::Completed.can_cancel());
    }

    #[test]
    fn test_simulation_config_validation() {
        let config = SimulationConfig::default();
        assert!(config.validate().is_ok());

        let invalid = SimulationConfig {
            start_time: 100.0,
            end_time: 50.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_simulation_job() {
        let config = SimulationConfig::new("test.mhp")
            .with_time_range(0.0, 3600.0)
            .with_gpu(true);

        let mut job = SimulationJob::new("Test Job", config)
            .with_priority(JobPriority::High)
            .with_description("Test description");

        assert_eq!(job.status, JobStatus::Pending);
        assert_eq!(job.priority, JobPriority::High);
        assert_eq!(job.progress, 0.0);

        job.mark_started();
        assert_eq!(job.status, JobStatus::Running);
        assert!(job.started_at.is_some());

        job.update_progress(1800.0, 1000);
        assert!((job.progress - 0.5).abs() < 0.01);

        job.mark_completed();
        assert_eq!(job.status, JobStatus::Completed);
        assert!(job.completed_at.is_some());
        assert_eq!(job.progress, 1.0);
    }
}
