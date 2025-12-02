// src-tauri/src/marihydro/workflow/manager_v2.rs
//! 泛型化工作流管理器
//! 
//! 改进：
//! - 使用 WorkflowStorage trait 抽象存储层
//! - 支持内存/SQLite/自定义存储后端
//! - 类型安全的 Job 状态管理
//! - 事件通知机制

use super::job::{JobStatus, SimulationJob};
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::infra::storage::{MemoryStorage, WorkflowStorage};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

/// 工作流事件
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// 项目创建
    ProjectCreated { project_id: String },
    /// 项目删除
    ProjectDeleted { project_id: String },
    /// 作业提交
    JobSubmitted { job_id: String, project_id: String },
    /// 作业开始
    JobStarted { job_id: String },
    /// 作业进度更新
    JobProgress { job_id: String, progress: f64, message: Option<String> },
    /// 作业完成
    JobCompleted { job_id: String, result_path: String },
    /// 作业失败
    JobFailed { job_id: String, error: String },
    /// 作业取消
    JobCancelled { job_id: String },
}

/// 事件监听器
pub trait WorkflowEventListener: Send + Sync {
    fn on_event(&self, event: &WorkflowEvent);
}

/// 泛型化工作流管理器
/// 
/// 类型参数：
/// - S: 存储后端，实现 WorkflowStorage trait
pub struct WorkflowManagerV2<S: WorkflowStorage> {
    storage: Arc<S>,
    /// 事件监听器
    listeners: RwLock<Vec<Arc<dyn WorkflowEventListener>>>,
}

impl<S: WorkflowStorage> WorkflowManagerV2<S> {
    /// 使用指定存储创建管理器
    pub fn with_storage(storage: S) -> Self {
        Self {
            storage: Arc::new(storage),
            listeners: RwLock::new(Vec::new()),
        }
    }
    
    /// 获取存储引用
    pub fn storage(&self) -> &S {
        &self.storage
    }
    
    /// 添加事件监听器
    pub fn add_listener(&self, listener: Arc<dyn WorkflowEventListener>) {
        self.listeners.write().push(listener);
    }
    
    /// 移除所有监听器
    pub fn clear_listeners(&self) {
        self.listeners.write().clear();
    }
    
    /// 发送事件到所有监听器
    fn emit_event(&self, event: WorkflowEvent) {
        let listeners = self.listeners.read();
        for listener in listeners.iter() {
            listener.on_event(&event);
        }
    }
    
    // === 项目管理 ===
    
    /// 保存项目
    pub fn save_project(&self, id: &str, manifest_json: &str) -> MhResult<()> {
        self.storage.save_project(id, manifest_json)?;
        self.emit_event(WorkflowEvent::ProjectCreated {
            project_id: id.to_string(),
        });
        Ok(())
    }
    
    /// 加载项目
    pub fn load_project(&self, id: &str) -> MhResult<String> {
        self.storage.load_project(id)?.ok_or_else(|| MhError::not_found(format!("Project {} not found", id)))
    }
    
    /// 列出所有项目
    pub fn list_projects(&self) -> Vec<String> {
        self.storage.list_project_ids().unwrap_or_default()
    }
    
    /// 删除项目
    pub fn delete_project(&self, id: &str) -> MhResult<()> {
        // 先删除关联的作业
        let jobs = self.list_jobs(id);
        for job in jobs {
            let _ = self.storage.delete_job(&job.id);
        }
        
        // 删除项目
        self.storage.delete_project(id)?;
        
        self.emit_event(WorkflowEvent::ProjectDeleted {
            project_id: id.to_string(),
        });
        
        Ok(())
    }
    
    /// 检查项目是否存在
    pub fn project_exists(&self, id: &str) -> bool {
        self.storage.load_project(id).is_ok()
    }
    
    // === 作业管理 ===
    
    /// 提交新作业
    pub fn submit_job(
        &self,
        project_id: &str,
        overrides: Option<HashMap<String, f64>>,
    ) -> MhResult<String> {
        // 检查项目存在
        if !self.project_exists(project_id) {
            return Err(MhError::config(format!("Project {} not found", project_id)));
        }
        
        // 创建作业
        let job_id = Uuid::new_v4().to_string();
        let mut job = SimulationJob::new(job_id.clone(), project_id.to_string());
        
        if let Some(o) = overrides {
            job = job.with_overrides(o);
        }
        
        // 保存作业
        self.storage.save_job(&job)?;
        
        self.emit_event(WorkflowEvent::JobSubmitted {
            job_id: job_id.clone(),
            project_id: project_id.to_string(),
        });
        
        Ok(job_id)
    }
    
    /// 获取作业
    pub fn get_job(&self, job_id: &str) -> MhResult<SimulationJob> {
        self.storage.get_job(job_id)?
            .ok_or_else(|| MhError::not_found(format!("Job {}", job_id)))
    }
    
    /// 开始作业
    pub fn start_job(&self, job_id: &str) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        
        if job.status != JobStatus::Pending {
            return Err(MhError::config(format!(
                "Job {} cannot be started, current status: {:?}",
                job_id, job.status
            )));
        }
        
        job.start();
        self.storage.save_job(&job)?;
        
        self.emit_event(WorkflowEvent::JobStarted {
            job_id: job_id.to_string(),
        });
        
        Ok(())
    }
    
    /// 更新作业进度
    pub fn update_job_progress(
        &self,
        job_id: &str,
        progress: f64,
        message: Option<&str>,
    ) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        
        // 如果还是 Pending 状态，自动开始
        if job.status == JobStatus::Pending {
            job.start();
            self.emit_event(WorkflowEvent::JobStarted {
                job_id: job_id.to_string(),
            });
        }
        
        job.update_progress(progress, message);
        self.storage.save_job(&job)?;
        
        self.emit_event(WorkflowEvent::JobProgress {
            job_id: job_id.to_string(),
            progress,
            message: message.map(String::from),
        });
        
        Ok(())
    }
    
    /// 完成作业
    pub fn complete_job(&self, job_id: &str, result_path: &str) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        job.complete(result_path);
        self.storage.save_job(&job)?;
        
        self.emit_event(WorkflowEvent::JobCompleted {
            job_id: job_id.to_string(),
            result_path: result_path.to_string(),
        });
        
        Ok(())
    }
    
    /// 失败作业
    pub fn fail_job(&self, job_id: &str, error: &str) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        job.fail(error);
        self.storage.save_job(&job)?;
        
        self.emit_event(WorkflowEvent::JobFailed {
            job_id: job_id.to_string(),
            error: error.to_string(),
        });
        
        Ok(())
    }
    
    /// 取消作业
    pub fn cancel_job(&self, job_id: &str) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        
        if job.status.is_terminal() {
            return Err(MhError::config(format!(
                "Job {} is already terminal: {:?}",
                job_id, job.status
            )));
        }
        
        job.cancel();
        self.storage.save_job(&job)?;
        
        self.emit_event(WorkflowEvent::JobCancelled {
            job_id: job_id.to_string(),
        });
        
        Ok(())
    }
    
    /// 暂停作业
    pub fn pause_job(&self, job_id: &str) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        
        if job.status != JobStatus::Running {
            return Err(MhError::config(format!(
                "Job {} is not running, cannot pause",
                job_id
            )));
        }
        
        job.status = JobStatus::Paused;
        job.message = Some("Paused".to_string());
        self.storage.save_job(&job)?;
        
        Ok(())
    }
    
    /// 恢复作业
    pub fn resume_job(&self, job_id: &str) -> MhResult<()> {
        let mut job = self.get_job(job_id)?;
        
        if job.status != JobStatus::Paused {
            return Err(MhError::config(format!(
                "Job {} is not paused, cannot resume",
                job_id
            )));
        }
        
        job.status = JobStatus::Running;
        job.message = Some("Resumed".to_string());
        self.storage.save_job(&job)?;
        
        Ok(())
    }
    
    /// 列出项目的所有作业
    pub fn list_jobs(&self, project_id: &str) -> Vec<SimulationJob> {
        self.storage.list_jobs(project_id).unwrap_or_default()
    }
    
    /// 列出待处理的作业
    pub fn list_pending_jobs(&self) -> Vec<SimulationJob> {
        self.storage.list_jobs_by_status(JobStatus::Pending).unwrap_or_default()
    }
    
    /// 列出运行中的作业
    pub fn list_running_jobs(&self) -> Vec<SimulationJob> {
        self.storage.list_jobs_by_status(JobStatus::Running).unwrap_or_default()
    }
    
    /// 删除作业
    pub fn delete_job(&self, job_id: &str) -> MhResult<()> {
        self.storage.delete_job(job_id)?;
        Ok(())
    }
}

// 为内存存储提供便捷构造
impl WorkflowManagerV2<MemoryStorage> {
    /// 创建使用内存存储的管理器
    pub fn new() -> Self {
        Self::with_storage(MemoryStorage::new())
    }
}

impl Default for WorkflowManagerV2<MemoryStorage> {
    fn default() -> Self {
        Self::new()
    }
}

/// 日志事件监听器
pub struct LoggingEventListener {
    prefix: String,
}

impl LoggingEventListener {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }
}

impl WorkflowEventListener for LoggingEventListener {
    fn on_event(&self, event: &WorkflowEvent) {
        match event {
            WorkflowEvent::ProjectCreated { project_id } => {
                log::info!("[{}] Project created: {}", self.prefix, project_id);
            }
            WorkflowEvent::ProjectDeleted { project_id } => {
                log::info!("[{}] Project deleted: {}", self.prefix, project_id);
            }
            WorkflowEvent::JobSubmitted { job_id, project_id } => {
                log::info!("[{}] Job submitted: {} (project: {})", self.prefix, job_id, project_id);
            }
            WorkflowEvent::JobStarted { job_id } => {
                log::info!("[{}] Job started: {}", self.prefix, job_id);
            }
            WorkflowEvent::JobProgress { job_id, progress, message } => {
                log::debug!(
                    "[{}] Job progress: {} - {:.1}% {:?}",
                    self.prefix, job_id, progress, message
                );
            }
            WorkflowEvent::JobCompleted { job_id, result_path } => {
                log::info!("[{}] Job completed: {} -> {}", self.prefix, job_id, result_path);
            }
            WorkflowEvent::JobFailed { job_id, error } => {
                log::error!("[{}] Job failed: {} - {}", self.prefix, job_id, error);
            }
            WorkflowEvent::JobCancelled { job_id } => {
                log::warn!("[{}] Job cancelled: {}", self.prefix, job_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = WorkflowManagerV2::new();
        assert!(manager.list_projects().is_empty());
    }

    #[test]
    fn test_project_workflow() {
        let manager = WorkflowManagerV2::new();
        
        // 保存项目
        manager.save_project("proj1", r#"{"name":"test"}"#).unwrap();
        assert!(manager.project_exists("proj1"));
        
        // 加载项目
        let manifest = manager.load_project("proj1").unwrap();
        assert!(manifest.contains("test"));
        
        // 列出项目
        let projects = manager.list_projects();
        assert_eq!(projects.len(), 1);
        
        // 删除项目
        manager.delete_project("proj1").unwrap();
        assert!(!manager.project_exists("proj1"));
    }

    #[test]
    fn test_job_workflow() {
        let manager = WorkflowManagerV2::new();
        
        // 创建项目
        manager.save_project("proj1", "{}").unwrap();
        
        // 提交作业
        let job_id = manager.submit_job("proj1", None).unwrap();
        assert!(!job_id.is_empty());
        
        // 获取作业
        let job = manager.get_job(&job_id).unwrap();
        assert_eq!(job.status, JobStatus::Pending);
        
        // 更新进度
        manager.update_job_progress(&job_id, 50.0, Some("Halfway")).unwrap();
        let job = manager.get_job(&job_id).unwrap();
        assert_eq!(job.progress, 50.0);
        assert_eq!(job.status, JobStatus::Running);
        
        // 完成作业
        manager.complete_job(&job_id, "/path/to/result").unwrap();
        let job = manager.get_job(&job_id).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
    }

    #[test]
    fn test_job_cancellation() {
        let manager = WorkflowManagerV2::new();
        manager.save_project("proj1", "{}").unwrap();
        
        let job_id = manager.submit_job("proj1", None).unwrap();
        manager.start_job(&job_id).unwrap();
        manager.cancel_job(&job_id).unwrap();
        
        let job = manager.get_job(&job_id).unwrap();
        assert_eq!(job.status, JobStatus::Cancelled);
    }
}
