// crates/mh_workflow/src/events.rs

//! 事件系统模块
//!
//! 提供工作流事件的定义和分发机制。

use crate::job::{JobId, JobStatus};
use parking_lot::RwLock;
use std::sync::Arc;

/// 工作流事件
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// 任务已提交
    JobSubmitted {
        /// 任务ID
        job_id: JobId,
        /// 任务名称
        name: String,
    },
    /// 任务已入队
    JobQueued {
        /// 任务ID
        job_id: JobId,
        /// 队列位置
        position: usize,
    },
    /// 任务已开始
    JobStarted {
        /// 任务ID
        job_id: JobId,
    },
    /// 任务进度更新
    JobProgress {
        /// 任务ID
        job_id: JobId,
        /// 进度 (0.0-1.0)
        progress: f64,
        /// 当前模拟时间
        current_time: f64,
        /// 已完成步数
        completed_steps: u64,
        /// 附加消息
        message: Option<String>,
    },
    /// 任务已完成
    JobCompleted {
        /// 任务ID
        job_id: JobId,
        /// 运行时长 (秒)
        duration_secs: f64,
        /// 总步数
        total_steps: u64,
    },
    /// 任务失败
    JobFailed {
        /// 任务ID
        job_id: JobId,
        /// 错误信息
        error: String,
    },
    /// 任务已取消
    JobCancelled {
        /// 任务ID
        job_id: JobId,
    },
    /// 任务已暂停
    JobPaused {
        /// 任务ID
        job_id: JobId,
    },
    /// 任务已恢复
    JobResumed {
        /// 任务ID
        job_id: JobId,
    },
    /// 任务状态变更
    JobStatusChanged {
        /// 任务ID
        job_id: JobId,
        /// 旧状态
        old_status: JobStatus,
        /// 新状态
        new_status: JobStatus,
    },
    /// 检查点已保存
    CheckpointSaved {
        /// 任务ID
        job_id: JobId,
        /// 检查点路径
        path: String,
    },
    /// 输出已写入
    OutputWritten {
        /// 任务ID
        job_id: JobId,
        /// 输出路径
        path: String,
        /// 模拟时间
        sim_time: f64,
    },
}

impl WorkflowEvent {
    /// 获取事件对应的任务ID
    pub fn job_id(&self) -> JobId {
        match self {
            Self::JobSubmitted { job_id, .. } => *job_id,
            Self::JobQueued { job_id, .. } => *job_id,
            Self::JobStarted { job_id } => *job_id,
            Self::JobProgress { job_id, .. } => *job_id,
            Self::JobCompleted { job_id, .. } => *job_id,
            Self::JobFailed { job_id, .. } => *job_id,
            Self::JobCancelled { job_id } => *job_id,
            Self::JobPaused { job_id } => *job_id,
            Self::JobResumed { job_id } => *job_id,
            Self::JobStatusChanged { job_id, .. } => *job_id,
            Self::CheckpointSaved { job_id, .. } => *job_id,
            Self::OutputWritten { job_id, .. } => *job_id,
        }
    }

    /// 获取事件名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::JobSubmitted { .. } => "JobSubmitted",
            Self::JobQueued { .. } => "JobQueued",
            Self::JobStarted { .. } => "JobStarted",
            Self::JobProgress { .. } => "JobProgress",
            Self::JobCompleted { .. } => "JobCompleted",
            Self::JobFailed { .. } => "JobFailed",
            Self::JobCancelled { .. } => "JobCancelled",
            Self::JobPaused { .. } => "JobPaused",
            Self::JobResumed { .. } => "JobResumed",
            Self::JobStatusChanged { .. } => "JobStatusChanged",
            Self::CheckpointSaved { .. } => "CheckpointSaved",
            Self::OutputWritten { .. } => "OutputWritten",
        }
    }
}

/// 事件监听器trait
pub trait EventListener: Send + Sync {
    /// 处理事件
    fn on_event(&self, event: &WorkflowEvent);

    /// 获取监听器名称 (用于调试)
    fn name(&self) -> &str {
        "anonymous"
    }
}

/// 函数式事件监听器
pub struct FnListener<F>
where
    F: Fn(&WorkflowEvent) + Send + Sync,
{
    name: String,
    handler: F,
}

impl<F> FnListener<F>
where
    F: Fn(&WorkflowEvent) + Send + Sync,
{
    /// 创建函数式监听器
    pub fn new(name: impl Into<String>, handler: F) -> Self {
        Self {
            name: name.into(),
            handler,
        }
    }
}

impl<F> EventListener for FnListener<F>
where
    F: Fn(&WorkflowEvent) + Send + Sync,
{
    fn on_event(&self, event: &WorkflowEvent) {
        (self.handler)(event);
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// 日志事件监听器
pub struct LoggingListener {
    /// 日志前缀
    prefix: String,
    /// 是否详细输出
    verbose: bool,
}

impl LoggingListener {
    /// 创建日志监听器
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            verbose: false,
        }
    }

    /// 设置详细模式
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl EventListener for LoggingListener {
    fn on_event(&self, event: &WorkflowEvent) {
        let msg = match event {
            WorkflowEvent::JobSubmitted { job_id, name } => {
                format!("Job '{}' (id={}) submitted", name, job_id)
            }
            WorkflowEvent::JobStarted { job_id } => {
                format!("Job {} started", job_id)
            }
            WorkflowEvent::JobProgress {
                job_id,
                progress,
                current_time,
                ..
            } => {
                format!(
                    "Job {} progress: {:.1}% (t={:.2}s)",
                    job_id,
                    progress * 100.0,
                    current_time
                )
            }
            WorkflowEvent::JobCompleted {
                job_id,
                duration_secs,
                total_steps,
            } => {
                format!(
                    "Job {} completed in {:.2}s ({} steps)",
                    job_id, duration_secs, total_steps
                )
            }
            WorkflowEvent::JobFailed { job_id, error } => {
                format!("Job {} failed: {}", job_id, error)
            }
            WorkflowEvent::JobCancelled { job_id } => {
                format!("Job {} cancelled", job_id)
            }
            WorkflowEvent::JobStatusChanged {
                job_id,
                old_status,
                new_status,
            } => {
                format!(
                    "Job {} status: {} -> {}",
                    job_id, old_status, new_status
                )
            }
            _ if self.verbose => {
                format!("{:?}", event)
            }
            _ => return,
        };

        tracing::info!("{}: {}", self.prefix, msg);
    }

    fn name(&self) -> &str {
        "LoggingListener"
    }
}

/// 事件分发器
#[derive(Default)]
pub struct EventDispatcher {
    listeners: RwLock<Vec<Arc<dyn EventListener>>>,
}

impl EventDispatcher {
    /// 创建新的事件分发器
    pub fn new() -> Self {
        Self {
            listeners: RwLock::new(Vec::new()),
        }
    }

    /// 添加监听器
    pub fn add_listener(&self, listener: Arc<dyn EventListener>) {
        let name = listener.name().to_string();
        self.listeners.write().push(listener);
        tracing::debug!("Added event listener: {}", name);
    }

    /// 添加函数式监听器
    pub fn add_fn_listener<F>(&self, name: impl Into<String>, handler: F)
    where
        F: Fn(&WorkflowEvent) + Send + Sync + 'static,
    {
        let listener = Arc::new(FnListener::new(name, handler));
        self.add_listener(listener);
    }

    /// 移除监听器
    pub fn remove_listener(&self, listener: &Arc<dyn EventListener>) {
        self.listeners
            .write()
            .retain(|l| !Arc::ptr_eq(l, listener));
    }

    /// 清除所有监听器
    pub fn clear(&self) {
        self.listeners.write().clear();
    }

    /// 分发事件
    pub fn emit(&self, event: WorkflowEvent) {
        let listeners = self.listeners.read();

        tracing::trace!("Emitting event: {}", event.name());

        for listener in listeners.iter() {
            listener.on_event(&event);
        }
    }

    /// 获取监听器数量
    pub fn listener_count(&self) -> usize {
        self.listeners.read().len()
    }
}

impl std::fmt::Debug for EventDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventDispatcher")
            .field("listener_count", &self.listener_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_event_dispatcher() {
        let dispatcher = EventDispatcher::new();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        dispatcher.add_fn_listener("test", move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        let job_id = JobId::new();
        dispatcher.emit(WorkflowEvent::JobStarted { job_id });
        dispatcher.emit(WorkflowEvent::JobCompleted {
            job_id,
            duration_secs: 10.0,
            total_steps: 100,
        });

        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_event_job_id() {
        let job_id = JobId::new();
        let event = WorkflowEvent::JobProgress {
            job_id,
            progress: 0.5,
            current_time: 100.0,
            completed_steps: 50,
            message: None,
        };

        assert_eq!(event.job_id(), job_id);
        assert_eq!(event.name(), "JobProgress");
    }
}
