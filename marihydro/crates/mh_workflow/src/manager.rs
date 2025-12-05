// crates/mh_workflow/src/manager.rs

//! 任务管理器模块
//!
//! 提供任务的提交、查询、管理功能。

use crate::events::{EventDispatcher, WorkflowEvent};
use crate::job::{JobId, JobPriority, JobStatus, SimulationJob};
use crate::storage::{Storage, StorageError};
use parking_lot::RwLock;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use thiserror::Error;

/// 工作流错误
#[derive(Debug, Error)]
pub enum WorkflowError {
    /// 存储错误
    #[error("Storage error: {0}")]
    Storage(#[from] StorageError),

    /// 任务不存在
    #[error("Job not found: {0}")]
    NotFound(JobId),

    /// 无效状态转换
    #[error("Invalid state transition: {0:?} -> {1:?}")]
    InvalidTransition(JobStatus, JobStatus),

    /// 配置无效
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// 任务已存在
    #[error("Job already exists: {0}")]
    AlreadyExists(JobId),

    /// 其他错误
    #[error("{0}")]
    Other(String),
}

/// 任务优先级队列项
#[derive(Debug, Clone, Eq, PartialEq)]
struct PriorityQueueItem {
    job_id: JobId,
    priority: JobPriority,
    created_at: chrono::DateTime<chrono::Utc>,
}

impl Ord for PriorityQueueItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // 高优先级优先，同优先级先进先出
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.created_at.cmp(&self.created_at))
    }
}

impl PartialOrd for PriorityQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// 工作流管理器配置
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// 最大并发任务数
    pub max_concurrent: usize,
    /// 是否自动启动队列中的任务
    pub auto_start: bool,
    /// 任务超时时间 (秒，0=无超时)
    pub timeout_secs: u64,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 1,
            auto_start: true,
            timeout_secs: 0,
        }
    }
}

/// 工作流管理器
pub struct WorkflowManager<S: Storage> {
    /// 存储后端
    storage: Arc<S>,
    /// 配置
    config: ManagerConfig,
    /// 活跃任务 (运行中或暂停的任务)
    active_jobs: RwLock<HashMap<JobId, SimulationJob>>,
    /// 等待队列
    queue: RwLock<BinaryHeap<PriorityQueueItem>>,
    /// 事件分发器
    events: EventDispatcher,
}

impl<S: Storage> WorkflowManager<S> {
    /// 创建管理器
    pub fn new(storage: S) -> Self {
        Self::with_config(storage, ManagerConfig::default())
    }

    /// 创建带配置的管理器
    pub fn with_config(storage: S, config: ManagerConfig) -> Self {
        Self {
            storage: Arc::new(storage),
            config,
            active_jobs: RwLock::new(HashMap::new()),
            queue: RwLock::new(BinaryHeap::new()),
            events: EventDispatcher::new(),
        }
    }

    /// 获取事件分发器
    pub fn events(&self) -> &EventDispatcher {
        &self.events
    }

    /// 获取配置
    pub fn config(&self) -> &ManagerConfig {
        &self.config
    }

    /// 提交任务
    pub fn submit(&self, job: SimulationJob) -> Result<JobId, WorkflowError> {
        // 验证配置
        if let Err(e) = job.config.validate() {
            return Err(WorkflowError::InvalidConfig(e));
        }

        let id = job.id;
        let name = job.name.clone();

        // 保存到存储
        self.storage.save_job(&job)?;

        // 添加到队列
        {
            let mut queue = self.queue.write();
            queue.push(PriorityQueueItem {
                job_id: id,
                priority: job.priority,
                created_at: job.created_at,
            });
        }

        // 发送事件
        self.events.emit(WorkflowEvent::JobSubmitted {
            job_id: id,
            name,
        });

        self.events.emit(WorkflowEvent::JobQueued {
            job_id: id,
            position: self.queue_position(id).unwrap_or(0),
        });

        tracing::info!("Job submitted: {}", id);

        Ok(id)
    }

    /// 获取任务
    pub fn get_job(&self, id: JobId) -> Result<SimulationJob, WorkflowError> {
        // 先查活跃任务
        if let Some(job) = self.active_jobs.read().get(&id) {
            return Ok(job.clone());
        }

        // 再查存储
        self.storage
            .load_job(id)?
            .ok_or(WorkflowError::NotFound(id))
    }

    /// 更新任务
    pub fn update_job(&self, job: &SimulationJob) -> Result<(), WorkflowError> {
        let old_status = self
            .get_job(job.id)
            .map(|j| j.status)
            .unwrap_or(job.status);

        // 保存到存储
        self.storage.save_job(job)?;

        // 更新活跃任务
        if let Some(active) = self.active_jobs.write().get_mut(&job.id) {
            *active = job.clone();
        }

        // 发送状态变更事件
        if old_status != job.status {
            self.events.emit(WorkflowEvent::JobStatusChanged {
                job_id: job.id,
                old_status,
                new_status: job.status,
            });
        }

        Ok(())
    }

    /// 启动任务
    pub fn start_job(&self, id: JobId) -> Result<(), WorkflowError> {
        let mut job = self.get_job(id)?;

        if job.status != JobStatus::Pending && job.status != JobStatus::Queued {
            return Err(WorkflowError::InvalidTransition(
                job.status,
                JobStatus::Running,
            ));
        }

        job.mark_started();

        // 添加到活跃任务
        self.active_jobs.write().insert(id, job.clone());

        // 从队列移除
        self.remove_from_queue(id);

        // 保存并发送事件
        self.storage.save_job(&job)?;
        self.events.emit(WorkflowEvent::JobStarted { job_id: id });

        tracing::info!("Job started: {}", id);

        Ok(())
    }

    /// 暂停任务
    pub fn pause_job(&self, id: JobId) -> Result<(), WorkflowError> {
        let mut job = self.get_job(id)?;

        if !job.status.can_pause() {
            return Err(WorkflowError::InvalidTransition(
                job.status,
                JobStatus::Paused,
            ));
        }

        job.mark_paused();
        self.update_job(&job)?;
        self.events.emit(WorkflowEvent::JobPaused { job_id: id });

        tracing::info!("Job paused: {}", id);

        Ok(())
    }

    /// 恢复任务
    pub fn resume_job(&self, id: JobId) -> Result<(), WorkflowError> {
        let mut job = self.get_job(id)?;

        if !job.status.can_resume() {
            return Err(WorkflowError::InvalidTransition(
                job.status,
                JobStatus::Running,
            ));
        }

        job.mark_resumed();
        self.update_job(&job)?;
        self.events.emit(WorkflowEvent::JobResumed { job_id: id });

        tracing::info!("Job resumed: {}", id);

        Ok(())
    }

    /// 取消任务
    pub fn cancel_job(&self, id: JobId) -> Result<(), WorkflowError> {
        let mut job = self.get_job(id)?;

        if !job.status.can_cancel() {
            return Err(WorkflowError::InvalidTransition(
                job.status,
                JobStatus::Cancelled,
            ));
        }

        job.mark_cancelled();

        // 从活跃任务移除
        self.active_jobs.write().remove(&id);

        // 从队列移除
        self.remove_from_queue(id);

        self.storage.save_job(&job)?;
        self.events.emit(WorkflowEvent::JobCancelled { job_id: id });

        tracing::info!("Job cancelled: {}", id);

        Ok(())
    }

    /// 完成任务
    pub fn complete_job(&self, id: JobId, total_steps: u64) -> Result<(), WorkflowError> {
        let mut job = self.get_job(id)?;

        let duration_secs = job.elapsed().map(|d| d.num_seconds() as f64).unwrap_or(0.0);

        job.mark_completed();

        // 从活跃任务移除
        self.active_jobs.write().remove(&id);

        self.storage.save_job(&job)?;
        self.events.emit(WorkflowEvent::JobCompleted {
            job_id: id,
            duration_secs,
            total_steps,
        });

        tracing::info!("Job completed: {} ({:.2}s)", id, duration_secs);

        Ok(())
    }

    /// 任务失败
    pub fn fail_job(&self, id: JobId, error: impl Into<String>) -> Result<(), WorkflowError> {
        let error = error.into();
        let mut job = self.get_job(id)?;

        job.mark_failed(&error);

        // 从活跃任务移除
        self.active_jobs.write().remove(&id);

        self.storage.save_job(&job)?;
        self.events.emit(WorkflowEvent::JobFailed {
            job_id: id,
            error: error.clone(),
        });

        tracing::error!("Job failed: {} - {}", id, error);

        Ok(())
    }

    /// 更新任务进度
    pub fn update_progress(
        &self,
        id: JobId,
        current_time: f64,
        completed_steps: u64,
        message: Option<String>,
    ) -> Result<(), WorkflowError> {
        let mut job = self.get_job(id)?;
        job.update_progress(current_time, completed_steps);

        self.update_job(&job)?;

        self.events.emit(WorkflowEvent::JobProgress {
            job_id: id,
            progress: job.progress,
            current_time,
            completed_steps,
            message,
        });

        Ok(())
    }

    /// 列出所有任务
    pub fn list_jobs(&self, filter: Option<JobStatus>) -> Result<Vec<SimulationJob>, WorkflowError> {
        let jobs = self.storage.list_jobs()?;

        let filtered = if let Some(status) = filter {
            jobs.into_iter().filter(|j| j.status == status).collect()
        } else {
            jobs
        };

        Ok(filtered)
    }

    /// 获取活跃任务
    pub fn active_jobs(&self) -> Vec<SimulationJob> {
        self.active_jobs.read().values().cloned().collect()
    }

    /// 获取队列中的任务
    pub fn queued_jobs(&self) -> Result<Vec<SimulationJob>, WorkflowError> {
        self.list_jobs(Some(JobStatus::Queued))
    }

    /// 获取下一个要执行的任务
    pub fn next_job(&self) -> Result<Option<SimulationJob>, WorkflowError> {
        let queue = self.queue.read();
        if let Some(item) = queue.peek() {
            self.get_job(item.job_id).map(Some)
        } else {
            Ok(None)
        }
    }

    /// 弹出下一个要执行的任务
    pub fn pop_next_job(&self) -> Result<Option<SimulationJob>, WorkflowError> {
        let item = self.queue.write().pop();
        if let Some(item) = item {
            self.get_job(item.job_id).map(Some)
        } else {
            Ok(None)
        }
    }

    /// 获取队列位置
    pub fn queue_position(&self, id: JobId) -> Option<usize> {
        let queue = self.queue.read();
        let items: Vec<_> = queue.iter().collect();
        items.iter().position(|item| item.job_id == id)
    }

    /// 队列长度
    pub fn queue_len(&self) -> usize {
        self.queue.read().len()
    }

    /// 活跃任务数
    pub fn active_count(&self) -> usize {
        self.active_jobs.read().len()
    }

    /// 清理已完成任务
    pub fn cleanup_completed(&self) -> Result<usize, WorkflowError> {
        let jobs = self.storage.list_jobs()?;
        let mut count = 0;

        for job in jobs {
            if job.status.is_terminal() {
                self.storage.delete_job(job.id)?;
                count += 1;
            }
        }

        tracing::info!("Cleaned up {} completed jobs", count);

        Ok(count)
    }

    /// 从队列移除任务
    fn remove_from_queue(&self, id: JobId) {
        let mut queue = self.queue.write();
        let items: Vec<_> = std::mem::take(&mut *queue)
            .into_iter()
            .filter(|item| item.job_id != id)
            .collect();
        *queue = items.into_iter().collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::SimulationConfig;
    use crate::storage::MemoryStorage;

    fn create_manager() -> WorkflowManager<MemoryStorage> {
        let storage = MemoryStorage::new();
        WorkflowManager::new(storage)
    }

    fn create_job(name: &str) -> SimulationJob {
        let config = SimulationConfig::new("test.mhp");
        SimulationJob::new(name, config)
    }

    #[test]
    fn test_submit_and_get_job() {
        let manager = create_manager();
        let job = create_job("Test Job");
        let id = job.id;

        manager.submit(job).unwrap();

        let loaded = manager.get_job(id).unwrap();
        assert_eq!(loaded.name, "Test Job");
        assert_eq!(loaded.status, JobStatus::Pending);
    }

    #[test]
    fn test_job_lifecycle() {
        let manager = create_manager();
        let job = create_job("Test Job");
        let id = job.id;

        manager.submit(job).unwrap();

        // 启动
        manager.start_job(id).unwrap();
        let job = manager.get_job(id).unwrap();
        assert_eq!(job.status, JobStatus::Running);

        // 暂停
        manager.pause_job(id).unwrap();
        let job = manager.get_job(id).unwrap();
        assert_eq!(job.status, JobStatus::Paused);

        // 恢复
        manager.resume_job(id).unwrap();
        let job = manager.get_job(id).unwrap();
        assert_eq!(job.status, JobStatus::Running);

        // 完成
        manager.complete_job(id, 1000).unwrap();
        let job = manager.get_job(id).unwrap();
        assert_eq!(job.status, JobStatus::Completed);
    }

    #[test]
    fn test_priority_queue() {
        let manager = create_manager();

        let low = create_job("Low").with_priority(JobPriority::Low);
        let high = create_job("High").with_priority(JobPriority::High);
        let normal = create_job("Normal").with_priority(JobPriority::Normal);

        manager.submit(low).unwrap();
        manager.submit(high).unwrap();
        manager.submit(normal).unwrap();

        // 高优先级应该先出队
        let next = manager.pop_next_job().unwrap().unwrap();
        assert_eq!(next.name, "High");

        let next = manager.pop_next_job().unwrap().unwrap();
        assert_eq!(next.name, "Normal");

        let next = manager.pop_next_job().unwrap().unwrap();
        assert_eq!(next.name, "Low");
    }
}
