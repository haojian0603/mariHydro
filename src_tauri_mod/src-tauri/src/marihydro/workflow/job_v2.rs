// src-tauri/src/marihydro/workflow/job_v2.rs
//! 类型状态模式的作业管理
//! 
//! 使用泛型类型参数编码作业状态，在编译时保证状态转换的正确性

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;

// === 状态标记类型 ===

/// 待处理状态
#[derive(Debug, Clone, Copy)]
pub struct Pending;

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

/// 状态特征 - 所有状态都必须实现
pub trait JobState: Clone + Copy {
    fn name() -> &'static str;
    fn is_terminal() -> bool { false }
}

impl JobState for Pending {
    fn name() -> &'static str { "PENDING" }
}

impl JobState for Running {
    fn name() -> &'static str { "RUNNING" }
}

impl JobState for Paused {
    fn name() -> &'static str { "PAUSED" }
}

impl JobState for Completed {
    fn name() -> &'static str { "COMPLETED" }
    fn is_terminal() -> bool { true }
}

impl JobState for Failed {
    fn name() -> &'static str { "FAILED" }
    fn is_terminal() -> bool { true }
}

impl JobState for Cancelled {
    fn name() -> &'static str { "CANCELLED" }
    fn is_terminal() -> bool { true }
}

// === 作业数据 ===

/// 作业内部数据（与状态无关）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobData {
    pub id: String,
    pub project_id: String,
    pub progress: f64,
    pub message: Option<String>,
    pub result_path: Option<String>,
    pub error_message: Option<String>,
    pub parameter_overrides: Option<HashMap<String, f64>>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl JobData {
    pub fn new(id: String, project_id: String) -> Self {
        Self {
            id,
            project_id,
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
}

// === 类型状态作业 ===

/// 带类型状态的作业
/// 
/// 状态转换在编译时检查：
/// - Pending -> Running (start)
/// - Running -> Paused (pause)
/// - Running -> Completed (complete)
/// - Running -> Failed (fail)
/// - Running -> Cancelled (cancel)
/// - Paused -> Running (resume)
/// - Paused -> Cancelled (cancel)
/// - Pending -> Cancelled (cancel)
#[derive(Debug, Clone)]
pub struct TypedJob<S: JobState> {
    data: JobData,
    _state: PhantomData<S>,
}

impl<S: JobState> TypedJob<S> {
    /// 获取作业ID
    pub fn id(&self) -> &str {
        &self.data.id
    }
    
    /// 获取项目ID
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
    
    /// 检查是否为终态
    pub fn is_terminal(&self) -> bool {
        S::is_terminal()
    }
    
    /// 获取内部数据（只读）
    pub fn data(&self) -> &JobData {
        &self.data
    }
    
    /// 转换为动态类型作业（用于存储）
    pub fn into_dynamic(self) -> DynamicJob {
        DynamicJob {
            data: self.data,
            state: S::name().to_string(),
        }
    }
}

// === Pending 状态的方法 ===

impl TypedJob<Pending> {
    /// 创建新作业
    pub fn new(id: String, project_id: String) -> Self {
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
    
    /// 开始作业 (Pending -> Running)
    pub fn start(mut self) -> TypedJob<Running> {
        self.data.started_at = Some(Utc::now());
        self.data.message = Some("Running".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
    
    /// 取消作业 (Pending -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled before start".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// === Running 状态的方法 ===

impl TypedJob<Running> {
    /// 更新进度
    pub fn update_progress(&mut self, progress: f64, message: Option<&str>) {
        self.data.progress = progress.clamp(0.0, 100.0);
        if let Some(m) = message {
            self.data.message = Some(m.to_string());
        }
    }
    
    /// 暂停作业 (Running -> Paused)
    pub fn pause(mut self) -> TypedJob<Paused> {
        self.data.message = Some("Paused".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
    
    /// 完成作业 (Running -> Completed)
    pub fn complete(mut self, result_path: &str) -> TypedJob<Completed> {
        self.data.progress = 100.0;
        self.data.result_path = Some(result_path.to_string());
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Completed successfully".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
    
    /// 失败作业 (Running -> Failed)
    pub fn fail(mut self, error: &str) -> TypedJob<Failed> {
        self.data.error_message = Some(error.to_string());
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some(format!("Failed: {}", error));
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
    
    /// 取消作业 (Running -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// === Paused 状态的方法 ===

impl TypedJob<Paused> {
    /// 恢复作业 (Paused -> Running)
    pub fn resume(mut self) -> TypedJob<Running> {
        self.data.message = Some("Resumed".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
    
    /// 取消作业 (Paused -> Cancelled)
    pub fn cancel(mut self) -> TypedJob<Cancelled> {
        self.data.finished_at = Some(Utc::now());
        self.data.message = Some("Cancelled while paused".to_string());
        TypedJob {
            data: self.data,
            _state: PhantomData,
        }
    }
}

// === 终态方法 ===

impl TypedJob<Completed> {
    /// 获取结果路径
    pub fn result_path(&self) -> Option<&str> {
        self.data.result_path.as_deref()
    }
}

impl TypedJob<Failed> {
    /// 获取错误信息
    pub fn error(&self) -> Option<&str> {
        self.data.error_message.as_deref()
    }
}

// === 动态类型作业（用于存储和传输）===

/// 动态类型作业
/// 
/// 用于 JSON 序列化和存储，可以转换为/从类型状态作业
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicJob {
    #[serde(flatten)]
    pub data: JobData,
    pub state: String,
}

impl DynamicJob {
    /// 尝试转换为 Pending 状态
    pub fn try_into_pending(self) -> Option<TypedJob<Pending>> {
        if self.state == "PENDING" {
            Some(TypedJob {
                data: self.data,
                _state: PhantomData,
            })
        } else {
            None
        }
    }
    
    /// 尝试转换为 Running 状态
    pub fn try_into_running(self) -> Option<TypedJob<Running>> {
        if self.state == "RUNNING" {
            Some(TypedJob {
                data: self.data,
                _state: PhantomData,
            })
        } else {
            None
        }
    }
    
    /// 尝试转换为 Paused 状态
    pub fn try_into_paused(self) -> Option<TypedJob<Paused>> {
        if self.state == "PAUSED" {
            Some(TypedJob {
                data: self.data,
                _state: PhantomData,
            })
        } else {
            None
        }
    }
    
    /// 检查是否为终态
    pub fn is_terminal(&self) -> bool {
        matches!(self.state.as_str(), "COMPLETED" | "FAILED" | "CANCELLED")
    }
    
    /// 获取状态
    pub fn state(&self) -> &str {
        &self.state
    }
}

// 从 SimulationJob 转换（兼容旧代码）
impl From<super::job::SimulationJob> for DynamicJob {
    fn from(job: super::job::SimulationJob) -> Self {
        DynamicJob {
            data: JobData {
                id: job.id,
                project_id: job.project_id,
                progress: job.progress,
                message: job.message,
                result_path: job.result_path,
                error_message: None,
                parameter_overrides: job.parameter_overrides,
                created_at: job.created_at,
                started_at: job.started_at,
                finished_at: job.finished_at,
            },
            state: job.status.as_str().to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_job_state_transitions() {
        // Pending -> Running -> Completed
        let job = TypedJob::<Pending>::new("job1".into(), "proj1");
        assert_eq!(job.state_name(), "PENDING");
        
        let job = job.start();
        assert_eq!(job.state_name(), "RUNNING");
        
        let job = job.complete("/result");
        assert_eq!(job.state_name(), "COMPLETED");
        assert!(job.is_terminal());
    }

    #[test]
    fn test_job_pause_resume() {
        let job = TypedJob::<Pending>::new("job1".into(), "proj1");
        let mut job = job.start();
        
        job.update_progress(50.0, Some("Halfway"));
        assert_eq!(job.progress(), 50.0);
        
        let job = job.pause();
        assert_eq!(job.state_name(), "PAUSED");
        
        let job = job.resume();
        assert_eq!(job.state_name(), "RUNNING");
    }

    #[test]
    fn test_job_cancellation() {
        // 取消 Pending 作业
        let job = TypedJob::<Pending>::new("job1".into(), "proj1");
        let job = job.cancel();
        assert_eq!(job.state_name(), "CANCELLED");
        
        // 取消 Running 作业
        let job = TypedJob::<Pending>::new("job2".into(), "proj1");
        let job = job.start();
        let job = job.cancel();
        assert_eq!(job.state_name(), "CANCELLED");
    }

    #[test]
    fn test_dynamic_conversion() {
        let job = TypedJob::<Pending>::new("job1".into(), "proj1");
        let dynamic = job.into_dynamic();
        
        assert_eq!(dynamic.state, "PENDING");
        
        let restored = dynamic.try_into_pending();
        assert!(restored.is_some());
    }

    #[test]
    fn test_job_failure() {
        let job = TypedJob::<Pending>::new("job1".into(), "proj1");
        let job = job.start();
        let job = job.fail("Out of memory");
        
        assert_eq!(job.state_name(), "FAILED");
        assert_eq!(job.error(), Some("Out of memory"));
    }
}
