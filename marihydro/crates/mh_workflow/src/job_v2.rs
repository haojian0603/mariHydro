// crates/mh_workflow/src/job_v2.rs

//! 类型状态模式的作业管理
//!
//! 使用泛型类型参数编码作业状态，在编译时保证状态转换的正确性。
//!
//! # 设计理念
//!
//! 使用 Rust 类型系统在编译时强制执行有效的状态转换：
//!
//! ```text
//!        ┌─────────────────────────────────────────────┐
//!        │                                             │
//!        ▼                                             │
//!    ┌───────┐    start()    ┌─────────┐   complete()  │
//!    │Pending│──────────────►│ Running │─────────────► │ Completed
//!    └───┬───┘               └────┬────┘               │
//!        │                        │                    │
//!        │ cancel()          pause()│   fail()         │
//!        │                        │      │             │
//!        ▼                        ▼      ▼             │
//!    ┌─────────┐            ┌────────┐ ┌──────┐       │
//!    │Cancelled│            │ Paused │ │Failed│───────┘
//!    └─────────┘            └────────┘ └──────┘
//!                                │
//!                          resume()
//!                                │
//!                                └──► Running
//! ```
//!
//! # 使用示例
//!
//! ```ignore
//! let pending = TypedJob::new("job-1", "project-1");
//! let running = pending.start();  // Pending -> Running
//! let completed = running.complete("/path/to/result");  // Running -> Completed
//!
//! // 编译错误：不能直接从 Pending 到 Completed
//! // let invalid = pending.complete("/path");
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use thiserror::Error;

// ============================================================================
// 状态标记类型
// ============================================================================

/// 待处理状态
#[derive(Debug, Clone, Copy)]
pub struct Pending;

/// 排队中状态
#[derive(Debug, Clone, Copy)]
pub struct Queued;

/// 运行中状态
#[derive(Debug, Clone, Copy)]
pub struct Running;

/// 暂停状态
#[derive(Debug, Clone, Copy)]
pub struct Paused;

/// 已完成状态
#[derive(Debug, Clone, Copy)]
pub struct Completed;

/// 失败状态
#[derive(Debug, Clone, Copy)]
pub struct Failed;

/// 已取消状态
#[derive(Debug, Clone, Copy)]
pub struct Cancelled;

// ============================================================================
// 状态特征
// ============================================================================

/// 状态特征
///
/// 所有状态标记类型都实现此 trait，提供状态元信息。
pub trait JobState: Clone + Copy + Send + Sync + 'static {
    /// 状态名称
    fn name() -> &'static str;

    /// 是否为终态（不能再转换）
    fn is_terminal() -> bool {
        false
    }

    /// 是否可以取消
    fn can_cancel() -> bool {
        true
    }
}

impl JobState for Pending {
    fn name() -> &'static str {
        "PENDING"
    }
}

impl JobState for Queued {
    fn name() -> &'static str {
        "QUEUED"
    }
}

impl JobState for Running {
    fn name() -> &'static str {
        "RUNNING"
    }
}

impl JobState for Paused {
    fn name() -> &'static str {
        "PAUSED"
    }
}

impl JobState for Completed {
    fn name() -> &'static str {
        "COMPLETED"
    }
    fn is_terminal() -> bool {
        true
    }
    fn can_cancel() -> bool {
        false
    }
}

impl JobState for Failed {
    fn name() -> &'static str {
        "FAILED"
    }
    fn is_terminal() -> bool {
        true
    }
    fn can_cancel() -> bool {
        false
    }
}

impl JobState for Cancelled {
    fn name() -> &'static str {
        "CANCELLED"
    }
    fn is_terminal() -> bool {
        true
    }
    fn can_cancel() -> bool {
        false
    }
}

// ============================================================================
// 作业数据结构
// ============================================================================

/// 作业内部数据（与状态无关）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobData {
    /// 作业 ID
    pub id: String,
    /// 项目 ID
    pub project_id: String,
    /// 进度 (0.0 - 1.0)
    pub progress: f64,
    /// 状态消息
    pub message: Option<String>,
    /// 结果路径
    pub result_path: Option<String>,
    /// 错误消息
    pub error_message: Option<String>,
    /// 参数覆盖
    pub parameter_overrides: Option<HashMap<String, f64>>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 开始时间
    pub started_at: Option<DateTime<Utc>>,
    /// 完成时间
    pub finished_at: Option<DateTime<Utc>>,
}

impl JobData {
    /// 创建新的作业数据
    pub fn new(id: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            project_id: project_id.into(),
            progress: 0.0,
            message: Some("Pending".to_string()),
            result_path: None,
            error_message: None,
            parameter_overrides: None,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
        }
    }

    /// 计算运行时长
    pub fn elapsed(&self) -> Option<chrono::Duration> {
        match (self.started_at, self.finished_at) {
            (Some(start), Some(end)) => Some(end - start),
            (Some(start), None) => Some(Utc::now() - start),
            _ => None,
        }
    }

    /// 获取运行时长（秒）
    pub fn elapsed_secs(&self) -> Option<f64> {
        self.elapsed().map(|d| d.num_milliseconds() as f64 / 1000.0)
    }
}

// ============================================================================
// 类型状态作业
// ============================================================================

/// 带类型状态的作业
///
/// 类型参数 `S` 编码当前状态，状态转换通过方法返回不同类型来实现。
///
/// # 示例
///
/// ```ignore
/// let pending = TypedJob::new("job-1", "project-1");
/// assert_eq!(pending.state_name(), "PENDING");
///
/// let running = pending.start();
/// assert_eq!(running.state_name(), "RUNNING");
///
/// let completed = running.complete("/path/to/result");
/// assert!(completed.is_terminal());
/// ```
#[derive(Debug, Clone)]
pub struct TypedJob<S: JobState> {
    data: JobData,
    _state: PhantomData<S>,
}

impl<S: JobState> TypedJob<S> {
    /// 获取作业 ID
    pub fn id(&self) -> &str {
        &self.data.id
    }

    /// 获取项目 ID
    pub fn project_id(&self) -> &str {
        &self.data.project_id
    }

    /// 获取进度
    pub fn progress(&self) -> f64 {
        self.data.progress
    }

    /// 获取消息
    pub fn message(&self) -> Option<&str> {
        self.data.message.as_deref()
    }

    /// 获取状态名称
    pub fn state_name(&self) -> &'static str {
        S::name()
    }

    /// 是否为终态
    pub fn is_terminal(&self) -> bool {
        S::is_terminal()
    }

    /// 是否可以取消
    pub fn can_cancel(&self) -> bool {
        S::can_cancel()
    }

    /// 获取内部数据引用
    pub fn data(&self) -> &JobData {
        &self.data
    }

    /// 获取内部数据可变引用
    pub fn data_mut(&mut self) -> &mut JobData {
        &mut self.data
    }

    /// 转换为动态类型作业（用于存储/序列化）
    pub fn into_dynamic(self) -> DynamicJob {
        DynamicJob {
            data: self.data,
            state: S::name().to_string(),
        }
    }
}

// ============================================================================
// Pending 状态方法
// ============================================================================

impl TypedJob<Pending> {
    /// 创建新作业
    pub fn new(id: impl Into<String>, project_id: impl Into<String>) -> Self {
        Self {
            data: JobData::new(id, project_id),
            _state: PhantomData,
        }
    }

    /// 设置参数覆盖
    pub fn with_overrides(mut self, overrides: HashMap<String, f64>) -> Self {
        self.data.parameter_overrides = Some(overrides);
        self
    }

    /// 设置消息
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.data.message = Some(message.into());
        self
    }

    /// 排队 (Pending -> Queued)
    pub fn enqueue(self) -> TypedJob<Queued> {
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 直接开始 (Pending -> Running)
    pub fn start(mut self) -> TypedJob<Running> {
        self.data.started_at = Some(Utc::now());
        self.data.message = Some("Running".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 取消 (Pending -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled before start".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// ============================================================================
// Queued 状态方法
// ============================================================================

impl TypedJob<Queued> {
    /// 开始运行 (Queued -> Running)
    pub fn start(mut self) -> TypedJob<Running> {
        self.data.started_at = Some(Utc::now());
        self.data.message = Some("Running".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 取消 (Queued -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled from queue".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// ============================================================================
// Running 状态方法
// ============================================================================

impl TypedJob<Running> {
    /// 更新进度
    pub fn update_progress(&mut self, progress: f64, message: Option<&str>) {
        self.data.progress = progress.clamp(0.0, 1.0);
        if let Some(msg) = message {
            self.data.message = Some(msg.to_string());
        }
    }

    /// 暂停 (Running -> Paused)
    pub fn pause(mut self) -> TypedJob<Paused> {
        self.data.message = Some("Paused".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 完成 (Running -> Completed)
    pub fn complete(mut self, result_path: impl Into<String>) -> TypedJob<Completed> {
        self.data.progress = 1.0;
        self.data.finished_at = Some(Utc::now());
        self.data.result_path = Some(result_path.into());
        self.data.message = Some("Completed".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 失败 (Running -> Failed)
    pub fn fail(mut self, error: impl Into<String>) -> TypedJob<Failed> {
        self.data.finished_at = Some(Utc::now());
        self.data.error_message = Some(error.into());
        self.data.message = Some("Failed".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 取消 (Running -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled during execution".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// ============================================================================
// Paused 状态方法
// ============================================================================

impl TypedJob<Paused> {
    /// 恢复 (Paused -> Running)
    pub fn resume(mut self) -> TypedJob<Running> {
        self.data.message = Some("Resumed".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// 取消 (Paused -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled while paused".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// ============================================================================
// Completed 状态方法
// ============================================================================

impl TypedJob<Completed> {
    /// 获取结果路径
    pub fn result_path(&self) -> Option<&str> {
        self.data.result_path.as_deref()
    }
}

// ============================================================================
// Failed 状态方法
// ============================================================================

impl TypedJob<Failed> {
    /// 获取错误消息
    pub fn error_message(&self) -> Option<&str> {
        self.data.error_message.as_deref()
    }

    /// 重试 (Failed -> Pending)
    pub fn retry(mut self) -> TypedJob<Pending> {
        self.data.progress = 0.0;
        self.data.error_message = None;
        self.data.started_at = None;
        self.data.finished_at = None;
        self.data.message = Some("Retrying".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// ============================================================================
// 动态类型作业（用于存储）
// ============================================================================

/// 动态类型作业（用于存储和序列化）
///
/// 当需要将作业存储到数据库或序列化时，使用此类型。
/// 可以通过 `into_typed()` 转换回类型状态作业。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicJob {
    /// 作业数据
    pub data: JobData,
    /// 状态名称
    pub state: String,
}

impl DynamicJob {
    /// 获取作业 ID
    pub fn id(&self) -> &str {
        &self.data.id
    }

    /// 获取状态名称
    pub fn state(&self) -> &str {
        &self.state
    }

    /// 是否为终态
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state.as_str(),
            "COMPLETED" | "FAILED" | "CANCELLED"
        )
    }

    /// 尝试转换为类型状态作业
    pub fn into_typed(self) -> Result<TypedJobAny, JobError> {
        match self.state.as_str() {
            "PENDING" => Ok(TypedJobAny::Pending(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            "QUEUED" => Ok(TypedJobAny::Queued(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            "RUNNING" => Ok(TypedJobAny::Running(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            "PAUSED" => Ok(TypedJobAny::Paused(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            "COMPLETED" => Ok(TypedJobAny::Completed(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            "FAILED" => Ok(TypedJobAny::Failed(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            "CANCELLED" => Ok(TypedJobAny::Cancelled(TypedJob {
                data: self.data,
                _state: PhantomData,
            })),
            _ => Err(JobError::InvalidState(self.state)),
        }
    }
}

// ============================================================================
// 类型状态作业枚举
// ============================================================================

/// 类型状态作业的枚举包装
///
/// 用于在运行时处理不同状态的作业，同时保留类型信息。
#[derive(Debug)]
pub enum TypedJobAny {
    /// 待处理
    Pending(TypedJob<Pending>),
    /// 排队中
    Queued(TypedJob<Queued>),
    /// 运行中
    Running(TypedJob<Running>),
    /// 暂停
    Paused(TypedJob<Paused>),
    /// 已完成
    Completed(TypedJob<Completed>),
    /// 失败
    Failed(TypedJob<Failed>),
    /// 已取消
    Cancelled(TypedJob<Cancelled>),
}

impl TypedJobAny {
    /// 获取作业 ID
    pub fn id(&self) -> &str {
        match self {
            Self::Pending(j) => j.id(),
            Self::Queued(j) => j.id(),
            Self::Running(j) => j.id(),
            Self::Paused(j) => j.id(),
            Self::Completed(j) => j.id(),
            Self::Failed(j) => j.id(),
            Self::Cancelled(j) => j.id(),
        }
    }

    /// 获取项目 ID
    pub fn project_id(&self) -> &str {
        match self {
            Self::Pending(j) => j.project_id(),
            Self::Queued(j) => j.project_id(),
            Self::Running(j) => j.project_id(),
            Self::Paused(j) => j.project_id(),
            Self::Completed(j) => j.project_id(),
            Self::Failed(j) => j.project_id(),
            Self::Cancelled(j) => j.project_id(),
        }
    }

    /// 获取进度
    pub fn progress(&self) -> f64 {
        match self {
            Self::Pending(j) => j.progress(),
            Self::Queued(j) => j.progress(),
            Self::Running(j) => j.progress(),
            Self::Paused(j) => j.progress(),
            Self::Completed(j) => j.progress(),
            Self::Failed(j) => j.progress(),
            Self::Cancelled(j) => j.progress(),
        }
    }

    /// 获取状态名称
    pub fn state_name(&self) -> &'static str {
        match self {
            Self::Pending(_) => Pending::name(),
            Self::Queued(_) => Queued::name(),
            Self::Running(_) => Running::name(),
            Self::Paused(_) => Paused::name(),
            Self::Completed(_) => Completed::name(),
            Self::Failed(_) => Failed::name(),
            Self::Cancelled(_) => Cancelled::name(),
        }
    }

    /// 是否为终态
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed(_) | Self::Failed(_) | Self::Cancelled(_)
        )
    }

    /// 转换为动态类型
    pub fn into_dynamic(self) -> DynamicJob {
        match self {
            Self::Pending(j) => j.into_dynamic(),
            Self::Queued(j) => j.into_dynamic(),
            Self::Running(j) => j.into_dynamic(),
            Self::Paused(j) => j.into_dynamic(),
            Self::Completed(j) => j.into_dynamic(),
            Self::Failed(j) => j.into_dynamic(),
            Self::Cancelled(j) => j.into_dynamic(),
        }
    }
}

// ============================================================================
// 错误类型
// ============================================================================

/// Job 相关错误
#[derive(Debug, Error)]
pub enum JobError {
    /// 无效的状态
    #[error("无效的状态: {0}")]
    InvalidState(String),

    /// 状态转换错误
    #[error("状态转换错误: 无法从 {from} 转换到 {to}")]
    InvalidTransition {
        /// 源状态
        from: String,
        /// 目标状态
        to: String,
    },
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_state_transitions() {
        // Pending -> Running -> Completed
        let pending = TypedJob::new("job-1", "project-1");
        assert_eq!(pending.state_name(), "PENDING");

        let running = pending.start();
        assert_eq!(running.state_name(), "RUNNING");

        let completed = running.complete("/path/to/result");
        assert_eq!(completed.state_name(), "COMPLETED");
        assert!(completed.is_terminal());
    }

    #[test]
    fn test_pause_resume() {
        let pending = TypedJob::new("job-2", "project-1");
        let running = pending.start();
        let paused = running.pause();
        assert_eq!(paused.state_name(), "PAUSED");

        let running_again = paused.resume();
        assert_eq!(running_again.state_name(), "RUNNING");
    }

    #[test]
    fn test_failure_retry() {
        let pending = TypedJob::new("job-3", "project-1");
        let running = pending.start();
        let failed = running.fail("Test error");

        assert_eq!(failed.error_message(), Some("Test error"));
        assert!(failed.is_terminal());

        let retried = failed.retry();
        assert_eq!(retried.state_name(), "PENDING");
        assert!(!retried.is_terminal());
    }

    #[test]
    fn test_dynamic_conversion() {
        let pending = TypedJob::new("job-4", "project-1");
        let running = pending.start();

        let dynamic = running.into_dynamic();
        assert_eq!(dynamic.state, "RUNNING");

        let typed = dynamic.into_typed().unwrap();
        assert!(matches!(typed, TypedJobAny::Running(_)));
    }

    #[test]
    fn test_progress_update() {
        let pending = TypedJob::new("job-5", "project-1");
        let mut running = pending.start();

        running.update_progress(0.5, Some("Half done"));
        assert!((running.progress() - 0.5).abs() < 1e-10);
        assert_eq!(running.message(), Some("Half done"));
    }

    #[test]
    fn test_cancel_states() {
        // Pending can cancel
        let pending = TypedJob::new("job-6", "project-1");
        assert!(pending.can_cancel());
        let cancelled = pending.cancel();
        assert!(cancelled.is_terminal());
        assert!(!cancelled.can_cancel());

        // Running can cancel
        let pending2 = TypedJob::new("job-7", "project-1");
        let running = pending2.start();
        assert!(running.can_cancel());
        let cancelled2 = running.cancel();
        assert!(cancelled2.is_terminal());
    }

    #[test]
    fn test_enqueue_workflow() {
        let pending = TypedJob::new("job-8", "project-1");
        let queued = pending.enqueue();
        assert_eq!(queued.state_name(), "QUEUED");

        let running = queued.start();
        assert_eq!(running.state_name(), "RUNNING");
    }

    #[test]
    fn test_with_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert("time_step".to_string(), 0.001);
        overrides.insert("max_velocity".to_string(), 10.0);

        let pending = TypedJob::new("job-9", "project-1").with_overrides(overrides);

        assert!(pending.data().parameter_overrides.is_some());
        let params = pending.data().parameter_overrides.as_ref().unwrap();
        assert_eq!(params.get("time_step"), Some(&0.001));
    }

    #[test]
    fn test_elapsed_time() {
        let pending = TypedJob::new("job-10", "project-1");
        let running = pending.start();

        // 运行中作业应该有 elapsed 时间
        assert!(running.data().elapsed_secs().is_some());

        let completed = running.complete("/result");
        // 完成后也应该有 elapsed 时间
        assert!(completed.data().elapsed_secs().is_some());
    }

    #[test]
    fn test_typed_job_any() {
        let pending = TypedJob::new("job-11", "project-1");
        let running = pending.start();
        let dynamic = running.into_dynamic();

        let any = dynamic.into_typed().unwrap();
        assert_eq!(any.id(), "job-11");
        assert_eq!(any.project_id(), "project-1");
        assert_eq!(any.state_name(), "RUNNING");
        assert!(!any.is_terminal());

        let dynamic_again = any.into_dynamic();
        assert_eq!(dynamic_again.state, "RUNNING");
    }

    #[test]
    fn test_invalid_state_conversion() {
        let dynamic = DynamicJob {
            data: JobData::new("job-12", "project-1"),
            state: "INVALID".to_string(),
        };

        let result = dynamic.into_typed();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), JobError::InvalidState(_)));
    }
}
