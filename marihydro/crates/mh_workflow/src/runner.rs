// crates/mh_workflow/src/runner.rs

//! 任务运行器模块
//!
//! 提供任务执行的抽象和控制，实现完整的模拟工作流：
//! - 网格加载和初始条件设置
//! - 求解器初始化和时间步进
//! - 检查点保存和恢复
//! - VTU输出生成
//! - 进度跟踪和事件通知
//!
//! # 架构说明
//!
//! RunContext 持有求解器和状态的可变引用，通过内部可变性（RwLock）实现
//! 在不可变上下文中的安全修改。所有IO操作（检查点、输出）均采用原子写入
//! 避免数据损坏。

use crate::events::WorkflowEvent;
use crate::job::{JobId, SimulationConfig, SimulationJob};
use crate::manager::{WorkflowError, WorkflowManager};
use crate::scheduler::{DeviceSelection, HybridScheduler};
use crate::storage::Storage;
use num_traits::cast::ToPrimitive;
use mh_physics::{
    engine::{ShallowWaterSolverF64, SolverStats, StabilityStatus},
    state::ShallowWaterStateF64,
    adapter::PhysicsMesh,
    Layer3Config,
};
use mh_io::{
    checkpoint::{Checkpoint, CheckpointManager},
    exporters::vtu::{VtuExporter, SimpleState},
};
use mh_runtime::CpuBackend;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;

/// 运行器错误类型
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

    /// 任务被取消
    #[error("Job was cancelled")]
    Cancelled,

    /// 任务超时
    #[error("Job timed out after {0} seconds")]
    Timeout(u64),

    /// 配置错误
    #[error("Configuration error: {0}")]
    Config(String),

    /// 初始化错误
    #[error("Initialization error: {0}")]
    Initialization(String),

    /// IO错误
    #[error("IO error: {0}")]
    Io(String),

    /// 数值不稳定
    #[error("Numerical instability detected: {0}")]
    Instability(String),

    /// 其他错误
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for RunnerError {
    fn from(e: std::io::Error) -> Self {
        RunnerError::Io(format!("{}", e))
    }
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
    /// 线程数
    pub num_threads: usize,
    /// VTU输出间隔 (秒)
    pub output_interval: f64,
    /// 是否启用NaN检测
    pub nan_detection_enabled: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            progress_interval: 1.0,
            checkpoint_interval: 300.0,
            timeout_secs: 0,
            num_threads: 0,
            output_interval: 60.0,
            nan_detection_enabled: true,
        }
    }
}

impl From<&SimulationConfig> for RunnerConfig {
    fn from(config: &SimulationConfig) -> Self {
        Self {
            progress_interval: 1.0,
            checkpoint_interval: config.checkpoint_interval,
            timeout_secs: 0,
            num_threads: config.num_threads,
            output_interval: config.output_interval,
            nan_detection_enabled: true,
        }
    }
}

/// 运行上下文，持有模拟的所有运行时状态
///
/// 使用 Arc<RwLock> 实现内部可变性，允许在不可变引用下修改求解器状态。
pub struct RunContext {
    /// 任务ID
    pub job_id: JobId,
    /// 模拟配置
    pub config: SimulationConfig,
    /// 取消标志
    cancelled: Arc<AtomicBool>,
    /// 暂停标志
    paused: Arc<AtomicBool>,
    /// 开始时间
    start_time: Instant,
    /// 当前模拟时间
    current_sim_time: RwLock<f64>,
    /// 已完成步数
    completed_steps: RwLock<u64>,
    /// 设备选择
    device: Option<DeviceSelection>,
    /// 网格引用
    pub mesh: Arc<PhysicsMesh>,
    /// 求解器实例（内部可变）
    pub solver: Arc<RwLock<ShallowWaterSolverF64>>,
    /// 状态实例（内部可变）
    pub state: Arc<RwLock<ShallowWaterStateF64>>,
    /// 上次输出时间
    last_output_time: RwLock<f64>,
    /// 输出文件计数器
    output_counter: AtomicU64,
    /// 上次检查点时间
    last_checkpoint_time: RwLock<f64>,
}

impl RunContext {
    /// 创建并初始化运行上下文
    ///
    /// # 错误处理
    /// - 加载网格失败返回 Initialization 错误
    /// - 配置转换失败返回 Config 错误
    /// - 初始状态创建失败返回 Initialization 错误
    pub fn new(
        job: &SimulationJob,
        runner_config: &RunnerConfig,
    ) -> Result<Self, RunnerError> {
        // 1. 加载项目网格
        let mesh = load_mesh_from_project(&job.config.project_path)
            .map_err(|e| RunnerError::Initialization(format!("网格加载失败: {}", e)))?;
        let mesh = Arc::new(mesh);

        // 2. 从项目加载Layer4配置并转换为Layer3
        let layer4_config = load_layer4_config(&job.config.project_path)
            .map_err(|e| RunnerError::Config(format!("配置加载失败: {}", e)))?;
        
        let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&layer4_config)
            .map_err(|e| RunnerError::Config(format!("配置转换失败: {}", e)))?;

        // 3. 创建求解器
        let backend = CpuBackend::<f64>::new();
        let solver = Arc::new(RwLock::new(
            ShallowWaterSolverF64::new(mesh.clone(), layer3_config, backend)
        ));

        // 4. 创建初始状态
        let state = create_initial_state(&mesh, &job.config)
            .map_err(|e| RunnerError::Initialization(format!("初始状态创建失败: {}", e)))?;
        let state = Arc::new(RwLock::new(state));

        Ok(Self {
            job_id: job.id,
            config: job.config.clone(),
            cancelled: Arc::new(AtomicBool::new(false)),
            paused: Arc::new(AtomicBool::new(false)),
            start_time: Instant::now(),
            current_sim_time: RwLock::new(job.config.start_time),
            completed_steps: RwLock::new(0),
            device: None,
            mesh,
            solver,
            state,
            last_output_time: RwLock::new(job.config.start_time),
            output_counter: AtomicU64::new(0),
            last_checkpoint_time: RwLock::new(job.config.start_time),
        })
    }

    /// 设置设备选择
    pub fn with_device(mut self, device: DeviceSelection) -> Self {
        self.device = Some(device);
        self
    }

    /// 检查是否已取消
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// 请求取消
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// 检查是否已暂停
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    /// 设置暂停状态
    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::SeqCst);
    }

    /// 获取当前模拟时间
    pub fn current_sim_time(&self) -> f64 {
        *self.current_sim_time.read()
    }

    /// 更新当前模拟时间
    pub fn set_current_sim_time(&self, time: f64) {
        *self.current_sim_time.write() = time;
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
        let current = self.current_sim_time();
        let duration = self.config.end_time - self.config.start_time;
        if duration > 0.0 {
            ((current - self.config.start_time) / duration).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// 检查是否应该输出
    pub fn should_output(&self, now: f64) -> bool {
        now - *self.last_output_time.read() >= self.config.output_interval
    }

    /// 检查是否应该保存检查点
    pub fn should_checkpoint(&self, now: f64, config: &RunnerConfig) -> bool {
        config.checkpoint_interval > 0.0
            && now - *self.last_checkpoint_time.read() >= config.checkpoint_interval
    }

    /// 检查是否完成
    pub fn is_finished(&self) -> bool {
        self.current_sim_time() >= self.config.end_time
    }

    /// 获取求解器统计信息
    pub fn solver_stats(&self) -> SolverStats {
        self.solver.read().stats().clone()
    }
}

/// 任务句柄，提供外部控制接口
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

/// 任务运行器，管理模拟的生命周期
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
        Self::with_config(manager, RunnerConfig::default())
    }

    /// 创建带配置的任务运行器
    pub fn with_config(manager: Arc<WorkflowManager<S>>, config: RunnerConfig) -> Self {
        Self {
            manager,
            scheduler: HybridScheduler::new(Default::default()),
            config,
            handles: RwLock::new(HashMap::new()),
        }
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

    /// 启动任务（同步执行）
    ///
    /// # 错误处理
    /// - 任务已在运行返回 AlreadyRunning
    /// - 初始化失败返回 Initialization 错误
    /// - 计算过程中错误返回 Computation 错误
    pub fn run(&self, job_id: JobId) -> Result<(), RunnerError> {
        // 获取任务
        let job = self.manager.get_job(job_id)?;

        // 检查是否已在运行
        if self.handles.read().contains_key(&job_id) {
            return Err(RunnerError::AlreadyRunning(job_id));
        }

        // 选择计算设备
        let device = self.scheduler.select_device(1000); // 假设1000个单元

        // 创建并初始化运行上下文
        let context = RunContext::new(&job, &self.config)
            .map_err(|e| RunnerError::Initialization(format!("上下文化失败: {}", e)))?
            .with_device(device);

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
                let total_steps = context.completed_steps();
                self.manager
                    .complete_job(job_id, total_steps)?;
                tracing::info!(
                    "Job {} completed: {} steps in {:.2}s",
                    job_id,
                    total_steps,
                    context.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Err(RunnerError::Cancelled) => {
                self.manager.cancel_job(job_id)?;
                tracing::warn!("Job {} was cancelled", job_id);
                Err(RunnerError::Cancelled)
            }
            Err(e) => {
                self.manager.fail_job(job_id, e.to_string())?;
                tracing::error!("Job {} failed: {}", job_id, e);
                Err(e)
            }
        }
    }

    /// 获取任务句柄
    pub fn get_handle(&self, job_id: JobId) -> Option<JobHandle> {
        let handles = self.handles.read();
        handles.get(&job_id).cloned()
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

    /// 运行主循环，执行完整的时间推进
    ///
    /// # 流程
    /// 1. 检查取消/暂停标志
    /// 2. 计算时间步
    /// 3. 执行求解器步进
    /// 4. 保存检查点（如需要）
    /// 5. 写入输出文件（如需要）
    /// 6. 更新进度
    fn run_loop(&self, context: &RunContext) -> Result<(), RunnerError> {
        let mut last_progress_time = Instant::now();
        let timeout = if self.config.timeout_secs > 0 {
            Some(Duration::from_secs(self.config.timeout_secs))
        } else {
            None
        };

        tracing::info!(
            "Starting simulation {}: {:.2}s -> {:.2}s (dt_out={:.2}s, dt_chk={:.2}s)",
            context.job_id,
            context.config.start_time,
            context.config.end_time,
            context.config.output_interval,
            context.config.checkpoint_interval,
        );

        // 主循环
        while !context.is_finished() && !context.is_cancelled() {
            // 检查超时
            if let Some(timeout) = timeout {
                if context.elapsed() > timeout {
                    return Err(RunnerError::Timeout(self.config.timeout_secs));
                }
            }

            // 执行一个时间步
            self.execute_timestep(context)?;

            let current_time = context.current_sim_time();

            // 检查点
            if context.should_checkpoint(current_time, &self.config) {
                self.save_checkpoint(context)?;
                *context.last_checkpoint_time.write() = current_time;
            }

            // 输出
            if context.should_output(current_time) {
                self.write_output(context)?;
                *context.last_output_time.write() = current_time;
            }

            // 进度更新
            if last_progress_time.elapsed().as_secs_f64() >= self.config.progress_interval {
                let stats = context.solver_stats();
                self.manager.update_progress(
                    context.job_id,
                    current_time,
                    context.completed_steps(),
                    Some(format!(
                        "dt={:.4}s, wave_speed={:.2}m/s, dry_cells={}, status={}",
                        stats.dt, stats.max_wave_speed, stats.dry_cells, stats.stability_status
                    )),
                )?;
                last_progress_time = Instant::now();
            }

            // 暂停检查
            while context.is_paused() {
                std::thread::sleep(Duration::from_millis(100));
                if context.is_cancelled() {
                    break;
                }
            }
        }

        tracing::info!(
            "Main loop finished for job {}: {} steps, final time {:.2}s, stability={:?}",
            context.job_id,
            context.completed_steps(),
            context.current_sim_time(),
            context.solver_stats().stability_status,
        );

        Ok(())
    }

    /// 执行单个时间步
    ///
    /// # 步骤
    /// 1. 锁定求解器和状态
    /// 2. 计算CFL稳定时间步
    /// 3. 执行求解器步进
    /// 4. 更新模拟时间
    /// 5. NaN检测与清理
    /// 6. 稳定性验证
    fn execute_timestep(&self, context: &RunContext) -> Result<(), RunnerError> {
        // 1. 锁定资源
        let mut solver = context.solver.write();
        let mut state = context.state.write();

        // 2. 计算时间步
        let dt_computed = solver.compute_dt(&state);
        let dt_cfl = dt_computed * context.config.max_cfl;
        let dt = dt_cfl.max(1e-8).min(10.0); // 保护性限制

        // 3. 执行步进
        solver.step(&mut state, dt);

        // 4. 更新时间
        let new_time = context.current_sim_time() + dt.to_f64().unwrap();
        context.set_current_sim_time(new_time);
        context.increment_steps(1);

        // 5. NaN检测
        if self.config.nan_detection_enabled {
            let nan_result = solver.detect_and_clean_nan(&mut state);
            if nan_result.found_nan {
                tracing::warn!(
                    "NaN detected in job {} at time {}: affected cells {:?}",
                    context.job_id,
                    new_time,
                    nan_result.affected_cells
                );
            }
        }

        // 6. 稳定性检查
        let stats = solver.stats();
        if stats.stability_status == StabilityStatus::Unstable {
            return Err(RunnerError::Instability(format!(
                "求解器在第 {} 步不稳定: {}",
                context.completed_steps(),
                stats.stability_status,
            )));
        }

        Ok(())
    }

    /// 保存检查点到磁盘
    ///
    /// # 流程
    /// 1. 锁定状态获取数据
    /// 2. 创建 Checkpoint 对象
    /// 3. 添加配置和网格哈希
    /// 4. 写入文件系统
    /// 5. 发送保存成功事件
    fn save_checkpoint(&self, context: &RunContext) -> Result<(), RunnerError> {
        // 1. 获取只读锁
        let state = context.state.read();
        let solver = context.solver.read();

        // 2. 构建状态快照
        let snapshot = mh_io::snapshot::StateSnapshot::from_state_data(
            state.h_slice().to_vec(),
            state.hu_slice().to_vec(),
            state.hv_slice().to_vec(),
        ).with_bed(state.z_slice().to_vec());

        // 3. 创建检查点
        let mut checkpoint = Checkpoint::new(
            context.current_sim_time(),
            context.completed_steps(),
            snapshot,
        );
        checkpoint = checkpoint
            .with_config_hash(compute_config_hash(&solver))
            .with_mesh_snapshot(&context.mesh.clone())
            .with_mesh_hash(compute_mesh_hash(&context.mesh));

        // 4. 保存到文件
        let checkpoint_dir = context.config.project_path.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        let manager = CheckpointManager::new(checkpoint_dir, 5)
            .with_prefix(&format!("job_{}", context.job_id));
        
        let path = manager.save(&checkpoint)
            .map_err(|e| RunnerError::Other(format!("检查点保存失败: {}", e)))?;

        // 5. 发送事件
        self.manager.events().emit(WorkflowEvent::CheckpointSaved {
            job_id: context.job_id,
            path: path.display().to_string(),
        });

        tracing::debug!(
            "Checkpoint saved for job {} at time {:.2}s to {:?}",
            context.job_id,
            context.current_sim_time(),
            path
        );

        Ok(())
    }

    /// 写入VTU输出文件
    ///
    /// # 流程
    /// 1. 锁定状态获取数据
    /// 2. 创建VTU状态包装器
    /// 3. 构建输出目录
    /// 4. 生成文件名并导出
    /// 5. 发送输出事件
    fn write_output(&self, context: &RunContext) -> Result<(), RunnerError> {
        // 1. 获取只读锁
        let state = context.state.read();
        let step = context.output_counter.fetch_add(1, Ordering::SeqCst);

        // 2. 创建VTU兼容的状态包装
        let vtu_state = SimpleState::new(
            state.h_slice(),
            state.hu_slice(),
            state.hv_slice(),
        );

        // 3. 准备输出目录
        let output_dir = context.config.project_path.join("output");
        std::fs::create_dir_all(&output_dir)?;

        // 4. 生成文件路径
        let filename = format!("output_{:06}.vtu", step);
        let path = output_dir.join(&filename);

        // 5. 执行导出
        let exporter = VtuExporter::new()
            .binary(false)
            .h_dry(1e-6); // 可配置化

        exporter.export(&path, &*context.mesh, &vtu_state, context.current_sim_time())
            .map_err(|e| RunnerError::Other(format!("VTU导出失败: {}", e)))?;

        // 6. 更新最后输出时间
        *context.last_output_time.write() = context.current_sim_time();

        // 7. 发送事件
        self.manager.events().emit(WorkflowEvent::OutputWritten {
            job_id: context.job_id,
            path: path.display().to_string(),
            sim_time: context.current_sim_time(),
        });

        tracing::debug!(
            "Output written for job {}: step {}, time {:.2}s to {:?}",
            context.job_id,
            step,
            context.current_sim_time(),
            path
        );

        Ok(())
    }
}

// ============================================================
// 辅助函数
// ============================================================

/// 从项目目录加载网格文件
///
/// 假设项目目录包含：
/// - project.mhp (JSON格式)
/// - mesh/ 子目录包含网格文件
fn load_mesh_from_project(project_path: &Path) -> Result<PhysicsMesh, Box<dyn std::error::Error>> {
    let project_file = project_path.join("project.mhp");
    if !project_file.exists() {
        return Err("项目文件 project.mhp 不存在".into());
    }

    let content = std::fs::read_to_string(&project_file)?;
    let project: serde_json::Value = serde_json::from_str(&content)?;
    
    let mesh_path = project["mesh"]
        .as_str()
        .ok_or("项目文件缺少mesh字段")?;
    
    let mesh_full_path = project_path.join(mesh_path);
    if !mesh_full_path.exists() {
        return Err(format!("网格文件不存在: {:?}", mesh_full_path).into());
    }

    // 使用mh_mesh加载网格
    let frozen_mesh = mh_mesh::io::load_mhb(&mesh_full_path)
        .map_err(|e| format!("加载网格失败: {}", e))?;
    
    Ok(PhysicsMesh::from_frozen(&frozen_mesh))
}

/// 从项目目录加载Layer4配置
fn load_layer4_config(project_path: &Path) -> Result<mh_config::SolverConfig, Box<dyn std::error::Error>> {
    let config_file = project_path.join("config.json");
    if !config_file.exists() {
        tracing::warn!("配置文件不存在，使用默认配置");
        return Ok(mh_config::SolverConfig::default());
    }

    let content = std::fs::read_to_string(&config_file)?;
    let config: mh_config::SolverConfig = serde_json::from_str(&content)
        .map_err(|e| format!("解析配置文件失败: {}", e))?;
    
    Ok(config)
}

/// 创建初始状态
///
/// 优先加载initial_state.json，如果不存在则使用静水条件
fn create_initial_state(
    mesh: &PhysicsMesh,
    config: &SimulationConfig,
) -> Result<ShallowWaterStateF64, Box<dyn std::error::Error>> {
    let n_cells = mesh.n_cells();
    let backend = CpuBackend::<f64>::new();
    
    let initial_file = config.project_path.join("initial_state.json");
    if initial_file.exists() {
        tracing::info!("加载初始状态文件: {:?}", initial_file);
        let content = std::fs::read_to_string(&initial_file)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        let h = parse_f64_array(&data["h"])?;
        let hu = parse_f64_array(&data["hu"]).unwrap_or(vec![0.0; n_cells]);
        let hv = parse_f64_array(&data["hv"]).unwrap_or(vec![0.0; n_cells]);
        
        if h.len() != n_cells {
            return Err(format!("初始状态h数组长度不匹配: 期望 {}, 实际 {}", n_cells, h.len()).into());
        }
        
        Ok(ShallowWaterStateF64::from_data(
            backend,
            h,
            hu,
            hv,
            mesh.cell_z_bed().to_vec(),
        ))
    } else {
        tracing::info!("未找到初始状态文件，使用默认静水条件 (h=1.0m)");
        Ok(ShallowWaterStateF64::cold_start(backend, 1.0, &mesh.cell_z_bed()))
    }
}

/// 解析JSON中的f64数组
fn parse_f64_array(value: &serde_json::Value) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    value.as_array()
        .ok_or("期望数组".into())
        .and_then(|arr| {
            arr.iter()
                .map(|v| v.as_f64()
                    .ok_or_else(|| format!("无效的双精度浮点数: {}", v).into()))
                .collect()
        })
}

/// 计算配置哈希
fn compute_config_hash(solver: &ShallowWaterSolverF64) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    solver.config().cfl.to_bits().hash(&mut hasher);
    solver.config().params.h_dry.to_bits().hash(&mut hasher);
    hasher.finish()
}

/// 计算网格哈希
fn compute_mesh_hash(mesh: &PhysicsMesh) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    mesh.n_cells().hash(&mut hasher);
    mesh.n_nodes().hash(&mut hasher);
    
    // 添加几何特征到哈希
    let total_area: f64 = (0..mesh.n_cells())
        .filter_map(|i| mesh.cell_area(mh_runtime::CellIndex::new(i)))
        .sum();
    total_area.to_bits().hash(&mut hasher);
    
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::SimulationConfig;
    use crate::storage::MemoryStorage;
    use std::path::PathBuf;

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.progress_interval, 1.0);
        assert!(config.nan_detection_enabled);
    }

    #[test]
    fn test_run_context_creation() {
        let project_dir = tempfile::tempdir().unwrap();
        let config = SimulationConfig::new(project_dir.path())
            .with_time_range(0.0, 100.0);
        let job = SimulationJob::new("TestJob", config);

        let runner = JobRunner::new(Arc::new(WorkflowManager::new(MemoryStorage::new())));
        let context = RunContext::new(&job, &runner.config);

        assert!(context.is_ok());
        let ctx = context.unwrap();
        assert_eq!(ctx.current_sim_time(), 0.0);
        assert_eq!(ctx.completed_steps(), 0);
    }

    #[test]
    fn test_job_handle_controls() {
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