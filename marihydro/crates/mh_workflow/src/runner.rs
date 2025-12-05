// crates/mh_workflow/src/runner.rs

//! 任务运行器模块
//!
//! 提供任务执行的抽象和控制。

use crate::events::WorkflowEvent;
use crate::job::{JobId, SimulationConfig, SimulationJob};
use crate::manager::{WorkflowError, WorkflowManager};
use crate::scheduler::{DeviceSelection, HybridScheduler};
use crate::storage::Storage;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// 运行器错误
#[derive(Debug, Error)]
pub enum RunnerError {
    /// 工作流错误
    #[error("Workflow error: {0}")]
    Workflow(#[from] WorkflowError),

    /// 任务已在运行
    #[error("Job {0} is already running")]
    AlreadyRunning(JobId),

    /// 任务未找到
    #[error("Job not found: {0}")]
    NotFound(JobId),

    /// 计算错误
    #[error("Computation error: {0}")]
    Computation(String),

    /// 取消
    #[error("Job was cancelled")]
    Cancelled,

    /// 超时
    #[error("Job timed out after {0} seconds")]
    Timeout(u64),

    /// 其他错误
    #[error("{0}")]
    Other(String),
}

/// 运行器配置
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// 进度更新间隔 (秒)
    pub progress_interval: f64,
    /// 检查点间隔 (秒，0=禁用)
    pub checkpoint_interval: f64,
    /// 超时时间 (秒，0=无超时)
    pub timeout_secs: u64,
    /// 是否使用GPU
    pub use_gpu: bool,
    /// 线程数
    pub num_threads: usize,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            progress_interval: 1.0,
            checkpoint_interval: 300.0,
            timeout_secs: 0,
            use_gpu: false,
            num_threads: 0,
        }
    }
}

impl From<&SimulationConfig> for RunnerConfig {
    fn from(config: &SimulationConfig) -> Self {
        Self {
            progress_interval: 1.0,
            checkpoint_interval: config.checkpoint_interval,
            timeout_secs: 0,
            use_gpu: config.use_gpu,
            num_threads: config.num_threads,
        }
    }
}

/// 运行上下文
pub struct RunContext {
    /// 任务ID
    pub job_id: JobId,
    /// 配置
    pub config: SimulationConfig,
    /// 取消标志
    cancelled: Arc<AtomicBool>,
    /// 暂停标志
    paused: Arc<AtomicBool>,
    /// 开始时间
    start_time: Instant,
    /// 当前模拟时间
    current_time: RwLock<f64>,
    /// 已完成步数
    completed_steps: RwLock<u64>,
    /// 设备选择
    device: Option<DeviceSelection>,
}

impl RunContext {
    /// 创建运行上下文
    pub fn new(job: &SimulationJob) -> Self {
        Self {
            job_id: job.id,
            config: job.config.clone(),
            cancelled: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            start_time: Instant::now(),
            current_time: RwLock::new(job.config.start_time),
            completed_steps: RwLock::new(0),
            device: None,
        }
    }

    /// 设置设备选择
    pub fn with_device(mut self, device: DeviceSelection) -> Self {
        self.device = Some(device);
        self
    }

    /// 是否已取消
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// 请求取消
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// 是否已暂停
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// 设置暂停状态
    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::SeqCst);
    }

    /// 获取当前模拟时间
    pub fn current_time(&self) -> f64 {
        *self.current_time.read()
    }

    /// 更新当前模拟时间
    pub fn set_current_time(&self, time: f64) {
        *self.current_time.write() = time;
    }

    /// 获取已完成步数
    pub fn completed_steps(&self) -> u64 {
        *self.completed_steps.read()
    }

    /// 增加步数
    pub fn increment_steps(&self, count: u64) {
        *self.completed_steps.write() += count;
    }

    /// 获取运行时长
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 获取进度 (0.0-1.0)
    pub fn progress(&self) -> f64 {
        let current = self.current_time();
        let duration = self.config.end_time - self.config.start_time;
        if duration > 0.0 {
            ((current - self.config.start_time) / duration).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// 检查是否应该输出
    pub fn should_output(&self, last_output_time: f64) -> bool {
        let current = self.current_time();
        current - last_output_time >= self.config.output_interval
    }

    /// 检查是否完成
    pub fn is_finished(&self) -> bool {
        self.current_time() >= self.config.end_time
    }
}

/// 任务句柄
pub struct JobHandle {
    /// 任务ID
    pub job_id: JobId,
    /// 取消标志
    cancel_flag: Arc<AtomicBool>,
    /// 暂停标志
    pause_flag: Arc<AtomicBool>,
}

impl JobHandle {
    /// 请求取消
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    /// 请求暂停
    pub fn pause(&self) {
        self.pause_flag.store(true, Ordering::SeqCst);
    }

    /// 请求恢复
    pub fn resume(&self) {
        self.pause_flag.store(false, Ordering::SeqCst);
    }

    /// 是否已取消
    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }

    /// 是否已暂停
    pub fn is_paused(&self) -> bool {
        self.pause_flag.load(Ordering::SeqCst)
    }
}

/// 任务运行器
pub struct JobRunner<S: Storage> {
    /// 工作流管理器
    manager: Arc<WorkflowManager<S>>,
    /// 混合调度器
    scheduler: HybridScheduler,
    /// 运行器配置
    config: RunnerConfig,
    /// 活跃任务句柄
    handles: RwLock<HashMap<JobId, JobHandle>>,
}

impl<S: Storage> JobRunner<S> {
    /// 创建任务运行器
    pub fn new(manager: Arc<WorkflowManager<S>>) -> Self {
        Self {
            manager,
            scheduler: HybridScheduler::new(Default::default()),
            config: RunnerConfig::default(),
            handles: RwLock::new(HashMap::new()),
        }
    }

    /// 设置配置
    pub fn with_config(mut self, config: RunnerConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置调度器
    pub fn with_scheduler(mut self, scheduler: HybridScheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    /// 获取管理器
    pub fn manager(&self) -> &Arc<WorkflowManager<S>> {
        &self.manager
    }

    /// 获取调度器
    pub fn scheduler(&self) -> &HybridScheduler {
        &self.scheduler
    }

    /// 启动任务（同步）
    pub fn run(&self, job_id: JobId) -> Result<(), RunnerError> {
        // 获取任务
        let job = self.manager.get_job(job_id)?;

        // 检查是否已在运行
        if self.handles.read().contains_key(&job_id) {
            return Err(RunnerError::AlreadyRunning(job_id));
        }

        // 选择计算设备
        let device = self.scheduler.select_device(1000); // 假设1000个单元

        // 创建运行上下文
        let context = RunContext::new(&job).with_device(device);

        // 创建句柄
        let handle = JobHandle {
            job_id,
            cancel_flag: context.cancelled.clone(),
            pause_flag: context.paused.clone(),
        };
        self.handles.write().insert(job_id, handle);

        // 启动任务
        self.manager.start_job(job_id)?;

        // 运行主循环
        let result = self.run_loop(&context);

        // 移除句柄
        self.handles.write().remove(&job_id);

        // 处理结果
        match result {
            Ok(()) => {
                self.manager
                    .complete_job(job_id, context.completed_steps())?;
                Ok(())
            }
            Err(RunnerError::Cancelled) => {
                self.manager.cancel_job(job_id)?;
                Err(RunnerError::Cancelled)
            }
            Err(e) => {
                self.manager.fail_job(job_id, e.to_string())?;
                Err(e)
            }
        }
    }

    /// 获取任务句柄
    pub fn get_handle(&self, job_id: JobId) -> Option<JobHandle> {
        let handles = self.handles.read();
        handles.get(&job_id).map(|h| JobHandle {
            job_id: h.job_id,
            cancel_flag: h.cancel_flag.clone(),
            pause_flag: h.pause_flag.clone(),
        })
    }

    /// 取消任务
    pub fn cancel(&self, job_id: JobId) -> Result<(), RunnerError> {
        if let Some(handle) = self.handles.read().get(&job_id) {
            handle.cancel();
            Ok(())
        } else {
            Err(RunnerError::NotFound(job_id))
        }
    }

    /// 暂停任务
    pub fn pause(&self, job_id: JobId) -> Result<(), RunnerError> {
        if let Some(handle) = self.handles.read().get(&job_id) {
            handle.pause();
            self.manager.pause_job(job_id)?;
            Ok(())
        } else {
            Err(RunnerError::NotFound(job_id))
        }
    }

    /// 恢复任务
    pub fn resume(&self, job_id: JobId) -> Result<(), RunnerError> {
        if let Some(handle) = self.handles.read().get(&job_id) {
            handle.resume();
            self.manager.resume_job(job_id)?;
            Ok(())
        } else {
            Err(RunnerError::NotFound(job_id))
        }
    }

    /// 运行主循环
    fn run_loop(&self, context: &RunContext) -> Result<(), RunnerError> {
        let mut last_progress_time = Instant::now();
        let mut last_checkpoint_time = Instant::now();
        let mut last_output_time = context.config.start_time;

        let timeout = if self.config.timeout_secs > 0 {
            Some(Duration::from_secs(self.config.timeout_secs))
        } else {
            None
        };

        tracing::info!(
            "Starting simulation: {} -> {} (dt_out={})",
            context.config.start_time,
            context.config.end_time,
            context.config.output_interval
        );

        // 模拟主循环
        while !context.is_finished() {
            // 检查取消
            if context.is_cancelled() {
                return Err(RunnerError::Cancelled);
            }

            // 检查超时
            if let Some(timeout) = timeout {
                if context.elapsed() > timeout {
                    return Err(RunnerError::Timeout(self.config.timeout_secs));
                }
            }

            // 检查暂停
            while context.is_paused() {
                std::thread::sleep(Duration::from_millis(100));
                if context.is_cancelled() {
                    return Err(RunnerError::Cancelled);
                }
            }

            // 执行一个时间步（这里是模拟，实际需要调用求解器）
            self.execute_timestep(context)?;

            // 更新进度
            if last_progress_time.elapsed().as_secs_f64() >= self.config.progress_interval {
                self.manager.update_progress(
                    context.job_id,
                    context.current_time(),
                    context.completed_steps(),
                    None,
                )?;
                last_progress_time = Instant::now();
            }

            // 检查点
            if self.config.checkpoint_interval > 0.0
                && last_checkpoint_time.elapsed().as_secs_f64() >= self.config.checkpoint_interval
            {
                self.save_checkpoint(context)?;
                last_checkpoint_time = Instant::now();
            }

            // 输出
            if context.should_output(last_output_time) {
                self.write_output(context)?;
                last_output_time = context.current_time();
            }
        }

        tracing::info!(
            "Simulation completed: {} steps in {:.2}s",
            context.completed_steps(),
            context.elapsed().as_secs_f64()
        );

        Ok(())
    }

    /// 执行一个时间步
    fn execute_timestep(&self, context: &RunContext) -> Result<(), RunnerError> {
        // TODO: 实际的求解器调用
        // 这里只是模拟时间推进

        let dt = 0.1; // 假设固定时间步长
        let current = context.current_time();
        let new_time = (current + dt).min(context.config.end_time);

        context.set_current_time(new_time);
        context.increment_steps(1);

        // 模拟计算耗时
        std::thread::sleep(Duration::from_micros(100));

        Ok(())
    }

    /// 保存检查点
    fn save_checkpoint(&self, context: &RunContext) -> Result<(), RunnerError> {
        let path = format!(
            "{}_checkpoint_{:.2}.bin",
            context.config.project_path.display(),
            context.current_time()
        );

        tracing::debug!("Saving checkpoint: {}", path);

        // TODO: 实际的检查点保存

        self.manager.events().emit(WorkflowEvent::CheckpointSaved {
            job_id: context.job_id,
            path,
        });

        Ok(())
    }

    /// 写入输出
    fn write_output(&self, context: &RunContext) -> Result<(), RunnerError> {
        let path = format!(
            "{}_output_{:.2}.vtu",
            context.config.project_path.display(),
            context.current_time()
        );

        tracing::debug!("Writing output: {}", path);

        // TODO: 实际的输出写入

        self.manager.events().emit(WorkflowEvent::OutputWritten {
            job_id: context.job_id,
            path,
            sim_time: context.current_time(),
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::SimulationConfig;
    use crate::storage::MemoryStorage;

    fn create_runner() -> JobRunner<MemoryStorage> {
        let storage = MemoryStorage::new();
        let manager = Arc::new(WorkflowManager::new(storage));
        JobRunner::new(manager)
    }

    #[test]
    fn test_run_context() {
        let config = SimulationConfig::new("test.mhp")
            .with_time_range(0.0, 100.0);
        let job = SimulationJob::new("Test", config);
        let context = RunContext::new(&job);

        assert_eq!(context.current_time(), 0.0);
        assert_eq!(context.completed_steps(), 0);
        assert!(!context.is_cancelled());
        assert!(!context.is_paused());
        assert!(!context.is_finished());

        context.set_current_time(50.0);
        assert!((context.progress() - 0.5).abs() < 0.01);

        context.set_current_time(100.0);
        assert!(context.is_finished());
    }

    #[test]
    fn test_job_handle_cancel() {
        let cancel_flag = Arc::new(AtomicBool::new(false));
        let pause_flag = Arc::new(AtomicBool::new(false));

        let handle = JobHandle {
            job_id: JobId::new(),
            cancel_flag: cancel_flag.clone(),
            pause_flag: pause_flag.clone(),
        };

        assert!(!handle.is_cancelled());
        handle.cancel();
        assert!(handle.is_cancelled());

        assert!(!handle.is_paused());
        handle.pause();
        assert!(handle.is_paused());
        handle.resume();
        assert!(!handle.is_paused());
    }
}
