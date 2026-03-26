// crates/mh_workflow/src/runner.rs

//! 任务运行器模块
//!
//! 提供任务执行的全生命周期管理，包括网格加载、求解器初始化、
//! 时间步进、检查点保存、VTU输出生成和进度跟踪。
//!
//! # 错误处理
//!
//! 使用强类型错误枚举 `ContextError`，明确区分错误来源（Mesh/Config/State），
//! 避免 `Box<dyn Error>` 导致的信息丢失。

use crate::events::WorkflowEvent;
use crate::job::{JobId, SimulationConfig, SimulationJob};
use crate::manager::{WorkflowError, WorkflowManager};
use crate::scheduler::{DeviceSelection, HybridScheduler};
use crate::storage::Storage;
use mh_foundation::MhError;
use mh_physics::{
    engine::{ShallowWaterSolver, SolverStats, StabilityStatus},
    state::ShallowWaterState,
    adapter::PhysicsMesh,
    Layer3Config,
};
use mh_physics::{BoundaryDataProvider, ExternalForcing};
use mh_physics::forcing::{
    ForcingDataError, ForcingField, SpatialInterpolation,
    compute_interpolation_weights, InterpolationWeights,
};
use mh_physics::sources::atmosphere::{WindStressConfig, WindStressRuntimeSource};
use mh_io::{
    checkpoint::{Checkpoint, CheckpointManager},
    exporters::vtu::{VtuExporter, SimpleState},
};
use mh_mesh::structured::{StructuredMesh, StructuredMeshConfig};
use mh_runtime::{CpuBackend, RuntimeScalar, Vector2D};
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
    #[error("流程错误: {0}")]
    Workflow(#[from] WorkflowError),

    #[error("任务已在运行: {0}")]
    AlreadyRunning(JobId),

    #[error("未找到任务: {0}")]
    NotFound(JobId),

    #[error("计算错误: {0}")]
    Computation(String),

    #[error("任务已取消")]
    Cancelled,

    #[error("任务超时: {0} 秒")]
    Timeout(u64),

    #[error("配置错误: {0}")]
    Config(String),

    #[error("初始化错误: {0}")]
    Initialization(String),

    #[error("IO 错误: {0}")]
    Io(String),

    #[error("数值不稳定: {0}")]
    Instability(String),

    #[error("{0}")]
    Other(String),
}

impl From<RunnerError> for MhError {
    fn from(err: RunnerError) -> Self {
        match err {
            RunnerError::Workflow(err) => err.into(),
            RunnerError::AlreadyRunning(job_id) => {
                MhError::invalid_input(format!("任务已在运行: {job_id}"))
            }
            RunnerError::NotFound(job_id) => MhError::not_found(format!("任务:{job_id}")),
            RunnerError::Computation(message) => {
                MhError::internal(format!("计算失败: {message}"))
            }
            RunnerError::Cancelled => MhError::internal("任务已取消".to_string()),
            RunnerError::Timeout(secs) => {
                MhError::internal(format!("任务超时: {secs} 秒"))
            }
            RunnerError::Config(message) => {
                MhError::invalid_input(format!("配置错误: {message}"))
            }
            RunnerError::Initialization(message) => {
                MhError::invalid_input(format!("初始化失败: {message}"))
            }
            RunnerError::Io(message) => MhError::io(format!("IO 错误: {message}")),
            RunnerError::Instability(message) => {
                MhError::internal(format!("数值不稳定: {message}"))
            }
            RunnerError::Other(message) => MhError::internal(message),
        }
    }
}

impl From<std::io::Error> for RunnerError {
    fn from(e: std::io::Error) -> Self {
        RunnerError::Io(format!("{}", e))
    }
}

/// 运行器配置
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub progress_interval: f64,
    pub checkpoint_interval: f64,
    pub timeout_secs: u64,
    pub num_threads: usize,
    pub output_interval: f64,
    pub nan_detection_enabled: bool,
    pub dt_min: f64,
    pub dt_max: f64,
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
            dt_min: 1e-8,
            dt_max: 10.0,
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
            dt_min: 1e-8,
            dt_max: 10.0,
        }
    }
}

/// 运行上下文，持有模拟的所有运行时状态
// 手动实现Debug，因为ShallowWaterSolver没有实现Debug
pub struct RunContext {
    pub job_id: JobId,
    pub config: SimulationConfig,
    cancelled: Arc<AtomicBool>,
    paused: Arc<AtomicBool>,
    start_time: Instant,
    current_sim_time: RwLock<f64>,
    completed_steps: RwLock<u64>,
    device: Option<DeviceSelection>,
    pub mesh: Arc<PhysicsMesh>,
    pub solver: Arc<RwLock<ShallowWaterSolver<CpuBackend<f64>, WindStressRuntimeSource>>>,
    pub state: Arc<RwLock<ShallowWaterState<CpuBackend<f64>>>>,
    pub forcing_snapshot: Option<ForcingSnapshot>,
    wind_forcing: RwLock<Option<WindForcingRuntime>>,
    last_output_time: RwLock<f64>,
    output_counter: AtomicU64,
    last_checkpoint_time: RwLock<f64>,
}

// 手动实现Debug，跳过没有Debug的字段
impl std::fmt::Debug for RunContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunContext")
            .field("job_id", &self.job_id)
            .field("config", &self.config)
            .field("current_sim_time", &self.current_sim_time)
            .field("completed_steps", &self.completed_steps)
            .field("device", &self.device)
            .field("mesh", &self.mesh)
            .field("forcing_snapshot", &self.forcing_snapshot)
            .field("last_output_time", &self.last_output_time)
            .field("output_counter", &self.output_counter)
            .field("last_checkpoint_time", &self.last_checkpoint_time)
            .finish_non_exhaustive() // 表明还有未显示的字段
    }
}

impl RunContext {
    /// 创建并初始化运行上下文
    ///
    /// # 错误
    /// - 网格文件不存在或格式错误
    /// - 配置文件解析失败
    /// - 初始状态与网格拓扑不匹配
    pub fn new(
        job: &SimulationJob,
        _runner_config: &RunnerConfig,
    ) -> Result<Self, RunnerError> {
        let layer4_config = load_layer4_config(&job.config.project_path)
            .map_err(|e| RunnerError::Config(format!("配置加载失败: {}", e)))?;

        let mesh = load_mesh_from_project(&job.config.project_path, &layer4_config)
            .map_err(|e| RunnerError::Initialization(format!("网格加载失败: {}", e)))?;
        let mesh = Arc::new(mesh);

        let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&layer4_config)
            .map_err(|e| RunnerError::Config(format!("配置转换失败: {}", e)))?;

        // 验证mesh与state的拓扑一致性，避免后续索引越界
        validate_mesh_topology(&mesh)
            .map_err(|e| RunnerError::Initialization(format!("网格拓扑验证失败: {}", e)))?;

        let backend = CpuBackend::<f64>::new();
        let mut solver = ShallowWaterSolver::<CpuBackend<f64>, WindStressRuntimeSource>::new(
            mesh.clone(),
            layer3_config,
            backend,
        );
        let wind_forcing = attach_forcing_sources(
            &mut solver,
            &job.config.project_path,
            &mesh,
            job.config.start_time,
        )?;
        let solver = Arc::new(RwLock::new(solver));

        let state = create_initial_state(&mesh, &job.config)
            .map_err(|e| RunnerError::Initialization(format!("初始状态创建失败: {}", e)))?;
        let state = Arc::new(RwLock::new(state));

        let forcing_snapshot = load_forcing_snapshot(
            &job.config.project_path,
            &mesh,
            job.config.start_time,
        )?;
        if let Some(snapshot) = &forcing_snapshot {
            tracing::info!(
                "强迫数据采样: source={}, var={}, t={}, lon={}, lat={}, value={:?}",
                snapshot.source,
                snapshot.variable,
                snapshot.time,
                snapshot.sample_lon,
                snapshot.sample_lat,
                snapshot.sample_value
            );
        }

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
            forcing_snapshot,
            wind_forcing: RwLock::new(wind_forcing),
            last_output_time: RwLock::new(job.config.start_time),
            output_counter: AtomicU64::new(0),
            last_checkpoint_time: RwLock::new(job.config.start_time),
        })
    }

    pub fn with_device(mut self, device: DeviceSelection) -> Self {
        self.device = Some(device);
        self
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::SeqCst)
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::SeqCst);
    }

    pub fn current_sim_time(&self) -> f64 {
        *self.current_sim_time.read()
    }

    pub fn set_current_sim_time(&self, time: f64) {
        *self.current_sim_time.write() = time;
    }

    pub fn completed_steps(&self) -> u64 {
        *self.completed_steps.read()
    }

    pub fn increment_steps(&self, count: u64) {
        *self.completed_steps.write() += count;
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn progress(&self) -> f64 {
        let current = self.current_sim_time();
        let duration = self.config.end_time - self.config.start_time;
        if duration > 0.0 {
            ((current - self.config.start_time) / duration).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn should_output(&self, now: f64) -> bool {
        if !self.config.enable_output || self.config.output_interval <= 0.0 {
            return false;
        }
        now - *self.last_output_time.read() >= self.config.output_interval
    }

    pub fn should_checkpoint(&self, now: f64, config: &RunnerConfig) -> bool {
        config.checkpoint_interval > 0.0
            && now - *self.last_checkpoint_time.read() >= config.checkpoint_interval
    }

    pub fn is_finished(&self) -> bool {
        self.current_sim_time() >= self.config.end_time
    }

    pub fn solver_stats(&self) -> SolverStats<CpuBackend<f64>> {
        self.solver.read().stats().clone()
    }
}

/// 任务句柄，提供外部控制接口
#[derive(Debug, Clone)]
pub struct JobHandle {
    pub job_id: JobId,
    cancel_flag: Arc<AtomicBool>,
    pause_flag: Arc<AtomicBool>,
}

impl JobHandle {
    pub fn cancel(&self) {
        self.cancel_flag.store(true, Ordering::SeqCst);
    }

    pub fn pause(&self) {
        self.pause_flag.store(true, Ordering::SeqCst);
    }

    pub fn resume(&self) {
        self.pause_flag.store(false, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::SeqCst)
    }

    pub fn is_paused(&self) -> bool {
        self.pause_flag.load(Ordering::SeqCst)
    }
}

/// 任务运行器，管理模拟的生命周期
pub struct JobRunner<S: Storage> {
    manager: Arc<WorkflowManager<S>>,
    scheduler: HybridScheduler,
    config: RunnerConfig,
    handles: RwLock<HashMap<JobId, JobHandle>>,
}

impl<S: Storage> JobRunner<S> {
    pub fn new(manager: Arc<WorkflowManager<S>>) -> Self {
        Self::with_config(manager, RunnerConfig::default())
    }

    pub fn with_config(manager: Arc<WorkflowManager<S>>, config: RunnerConfig) -> Self {
        Self {
            manager,
            scheduler: HybridScheduler::new(Default::default()),
            config,
            handles: RwLock::new(HashMap::new()),
        }
    }

    pub fn with_scheduler(mut self, scheduler: HybridScheduler) -> Self {
        self.scheduler = scheduler;
        self
    }

    pub fn manager(&self) -> &Arc<WorkflowManager<S>> {
        &self.manager
    }

    pub fn scheduler(&self) -> &HybridScheduler {
        &self.scheduler
    }

    /// 启动任务（同步执行）
    pub fn run(&self, job_id: JobId) -> Result<(), RunnerError> {
        let job = self.manager.get_job(job_id)?;

        if self.handles.read().contains_key(&job_id) {
            return Err(RunnerError::AlreadyRunning(job_id));
        }

        let device = self.scheduler.select_device(1000);

        let context = RunContext::new(&job, &self.config)
            .map_err(|e| RunnerError::Initialization(format!("上下文化失败: {}", e)))?
            .with_device(device);

        let handle = JobHandle {
            job_id,
            cancel_flag: context.cancelled.clone(),
            pause_flag: context.paused.clone(),
        };
        self.handles.write().insert(job_id, handle);

        self.manager.start_job(job_id)?;

        let result = self.run_loop(&context);

        self.handles.write().remove(&job_id);

        match result {
            Ok(()) => {
                let total_steps = context.completed_steps();
                self.manager.complete_job(job_id, total_steps)?;
                tracing::info!(
                    "任务 {} 完成：{} 步，用时 {:.2} 秒",
                    job_id,
                    total_steps,
                    context.elapsed().as_secs_f64()
                );
                Ok(())
            }
            Err(RunnerError::Cancelled) => {
                self.manager.cancel_job(job_id)?;
                tracing::warn!("任务 {} 已取消", job_id);
                Err(RunnerError::Cancelled)
            }
            Err(e) => {
                self.manager.fail_job(job_id, e.to_string())?;
                tracing::error!("任务 {} 失败：{}", job_id, e);
                Err(e)
            }
        }
    }

    pub fn get_handle(&self, job_id: JobId) -> Option<JobHandle> {
        let handles = self.handles.read();
        handles.get(&job_id).cloned()
    }

    pub fn cancel(&self, job_id: JobId) -> Result<(), RunnerError> {
        if let Some(handle) = self.handles.read().get(&job_id) {
            handle.cancel();
            Ok(())
        } else {
            Err(RunnerError::NotFound(job_id))
        }
    }

    pub fn pause(&self, job_id: JobId) -> Result<(), RunnerError> {
        if let Some(handle) = self.handles.read().get(&job_id) {
            handle.pause();
            self.manager.pause_job(job_id)?;
            Ok(())
        } else {
            Err(RunnerError::NotFound(job_id))
        }
    }

    pub fn resume(&self, job_id: JobId) -> Result<(), RunnerError> {
        if let Some(handle) = self.handles.read().get(&job_id) {
            handle.resume();
            self.manager.resume_job(job_id)?;
            Ok(())
        } else {
            Err(RunnerError::NotFound(job_id))
        }
    }

    /// 执行主循环
    fn run_loop(&self, context: &RunContext) -> Result<(), RunnerError> {
        let mut last_progress_time = Instant::now();
        let timeout = if self.config.timeout_secs > 0 {
            Some(Duration::from_secs(self.config.timeout_secs))
        } else {
            None
        };

        tracing::info!(
            "开始模拟 {}：{:.2} 秒 -> {:.2} 秒（输出间隔 {:.2} 秒，检查点间隔 {:.2} 秒）",
            context.job_id,
            context.config.start_time,
            context.config.end_time,
            context.config.output_interval,
            context.config.checkpoint_interval,
        );

        while !context.is_finished() {
            if context.is_cancelled() {
                return Err(RunnerError::Cancelled);
            }

            while context.is_paused() {
                if context.is_cancelled() {
                    return Err(RunnerError::Cancelled);
                }
                std::thread::sleep(Duration::from_millis(50));
            }

            if let Some(timeout) = timeout {
                if context.elapsed() > timeout {
                    return Err(RunnerError::Timeout(self.config.timeout_secs));
                }
            }

            self.execute_timestep(context)?;

            let current_time = context.current_sim_time();

            if context.should_checkpoint(current_time, &self.config) {
                self.save_checkpoint(context)?;
                *context.last_checkpoint_time.write() = current_time;
            }

            if context.config.enable_output && context.should_output(current_time) {
                self.write_output(context)?;
                *context.last_output_time.write() = current_time;
            }

            if last_progress_time.elapsed().as_secs_f64() >= self.config.progress_interval {
                let stats = context.solver_stats();
                self.manager.update_progress(
                    context.job_id,
                    current_time,
                    context.completed_steps(),
                    Some(format!(
                        "dt={:.4} 秒，波速={:.2} m/s，干单元={}，状态={}",
                        stats.dt, stats.max_wave_speed, stats.dry_cells, stats.stability_status
                    )),
                )?;
                last_progress_time = Instant::now();
            }

        }

        tracing::info!(
            "主循环结束：任务 {}，总步数 {}，最终时间 {:.2} 秒，稳定性={:?}",
            context.job_id,
            context.completed_steps(),
            context.current_sim_time(),
            context.solver_stats().stability_status,
        );

        Ok(())
    }

    /// 执行单个时间步
    fn execute_timestep(&self, context: &RunContext) -> Result<(), RunnerError> {
        let mut solver = context.solver.write();
        let mut state = context.state.write();

        let dt_computed = solver.compute_dt(&state);
        let dt = dt_computed.max(self.config.dt_min).min(self.config.dt_max);

        {
            let mut wind_forcing = context.wind_forcing.write();
            if let Some(runtime) = wind_forcing.as_mut() {
                runtime.update(context.current_sim_time())?;
            }
        }

        solver.step_with_sources(&mut state, dt, context.current_sim_time());

        let new_time = context.current_sim_time() + dt.to_f64_lossy();
        context.set_current_sim_time(new_time);
        context.increment_steps(1);

        if self.config.nan_detection_enabled {
            let nan_result = solver.detect_and_clean_nan(&mut state);
            if nan_result.found_nan {
                tracing::warn!(
                    "检测到 NaN：任务 {}，时间 {}，受影响单元 {:?}",
                    context.job_id,
                    new_time,
                    nan_result.affected_cells
                );
            }
        }

        let stats = solver.stats().clone();
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
    fn save_checkpoint(&self, context: &RunContext) -> Result<(), RunnerError> {
        let state = context.state.read();
        let solver = context.solver.read();

        let snapshot = mh_io::snapshot::StateSnapshot::<f64>::from_state_data(
            state.h_slice().to_vec(),
            state.hu_slice().to_vec(),
            state.hv_slice().to_vec(),
        ).with_bed(state.z_slice().to_vec());

        // 使用try_from安全转换u64→usize，避免32位系统溢出
        let step_count = usize::try_from(context.completed_steps())
            .map_err(|_| RunnerError::Other("步数超出usize范围".to_string()))?;

        let mut checkpoint = Checkpoint::new(
            context.current_sim_time(),
            step_count,
            snapshot,
        );
        
        checkpoint = checkpoint
            .with_config_hash(compute_config_hash(&solver))
            .with_mesh_hash(compute_mesh_hash(&context.mesh));

        let checkpoint_dir = context.config.project_path.join("checkpoints");
        std::fs::create_dir_all(&checkpoint_dir)?;
        
        let manager = CheckpointManager::new(checkpoint_dir, 5)
            .with_prefix(&format!("job_{}", context.job_id));
        
        let path = manager.save(&checkpoint)
            .map_err(|e| RunnerError::Other(format!("检查点保存失败: {}", e)))?;

        self.manager.events().emit(WorkflowEvent::CheckpointSaved {
            job_id: context.job_id,
            path: path.display().to_string(),
        });

        tracing::debug!(
            "检查点已保存：任务 {}，时间 {:.2} 秒，路径 {:?}",
            context.job_id,
            context.current_sim_time(),
            path
        );

        Ok(())
    }

    /// 写入VTU输出文件
    fn write_output(&self, context: &RunContext) -> Result<(), RunnerError> {
        let state = context.state.read();
        let step = context.output_counter.fetch_add(1, Ordering::SeqCst);

        let vtu_state = SimpleState::new(
            state.h_slice(),
            state.hu_slice(),
            state.hv_slice(),
        )
        .map_err(|e| RunnerError::Other(format!("VTU状态构造失败: {}", e)))?;

        let output_dir = context.config.project_path.join("output");
        std::fs::create_dir_all(&output_dir)?;

        let filename = format!("output_{:06}.vtu", step);
        let path = output_dir.join(&filename);

        let exporter = VtuExporter::new()
            .binary(false)
            .h_dry(1e-6);

        exporter.export(&path, &*context.mesh, &vtu_state, context.current_sim_time())
            .map_err(|e| RunnerError::Other(format!("VTU导出失败: {}", e)))?;

        *context.last_output_time.write() = context.current_sim_time();

        self.manager.events().emit(WorkflowEvent::OutputWritten {
            job_id: context.job_id,
            path: path.display().to_string(),
            sim_time: context.current_sim_time(),
        });

        tracing::debug!(
            "输出已写入：任务 {}，步数 {}，时间 {:.2} 秒，路径 {:?}",
            context.job_id,
            step,
            context.current_sim_time(),
            path
        );

        Ok(())
    }
}

/// 验证mesh拓扑一致性，确保所有索引在有效范围内
fn validate_mesh_topology(mesh: &PhysicsMesh) -> Result<(), String> {
    // 验证cell索引范围
    let n_cells = mesh.cell_count();
    for i in 0..n_cells {
        // cell_z_bed 直接返回 f64，不需要错误处理
        let _z = mesh.cell_z_bed(mh_runtime::CellIndex::new(i));
        // 检查是否能获取面积（验证索引有效性）
        if mesh.cell_area(mh_runtime::CellIndex::new(i)).is_none() {
            return Err(format!("单元索引 {} 无效: 无法获取面积", i));
        }
    }

    // 验证face的owner/neighbor索引
    for face_idx in 0..mesh.face_count() {
        let face = mh_runtime::FaceIndex::new(face_idx);
        let _owner = mesh.face_owner(face); // 直接返回值，不需要map_err
        // 如果需要验证，检查owner索引是否在有效范围内
        let owner_idx = _owner.get();
        if owner_idx >= mesh.cell_count() {
            return Err(format!("面 {} 的owner索引 {} 超出范围", face_idx, owner_idx));
        }
    }

    Ok(())
}

/// 从项目目录加载网格文件
fn load_mesh_from_project(
    project_path: &Path,
    layer4_config: &mh_config::SolverConfig,
) -> Result<PhysicsMesh, String> {
    let project_file = project_path.join("project.mhp");
    if !project_file.exists() {
        return Err("项目文件 project.mhp 不存在".to_string());
    }

    let content = std::fs::read_to_string(&project_file)
        .map_err(|e| format!("读取项目文件失败: {}", e))?;

    let project: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("解析项目文件失败: {}", e))?;

    if let Some(structured) = project.get("structured") {
        let nx = structured["nx"].as_u64().ok_or("结构化网格缺少 nx")? as usize;
        let ny = structured["ny"].as_u64().ok_or("结构化网格缺少 ny")? as usize;
        let dx = structured["dx"].as_f64().ok_or("结构化网格缺少 dx")?;
        let dy = structured["dy"].as_f64().ok_or("结构化网格缺少 dy")?;
        let origin = structured
            .get("origin")
            .and_then(|v| v.as_array())
            .and_then(|v| {
                if v.len() == 2 {
                    Some((v[0].as_f64()?, v[1].as_f64()?))
                } else {
                    None
                }
            })
            .unwrap_or((0.0, 0.0));

        let config = StructuredMeshConfig {
            nx,
            ny,
            dx,
            dy,
            origin,
        };
        let mut mesh = StructuredMesh::new(config);

        if let Some(bed) = structured.get("bed_elevation") {
            let elevation = parse_f64_array(bed)?;
            if elevation.len() != nx * ny {
                return Err(format!(
                    "结构化网格 bed_elevation 长度不匹配: 期望 {}, 实际 {}",
                    nx * ny,
                    elevation.len()
                ));
            }
            mesh.set_bed_elevation(elevation);
        }

        let frozen_mesh = mesh
            .freeze()
            .map_err(|e| format!("结构化网格冻结失败: {}", e))?;
        return Ok(PhysicsMesh::from_frozen(&frozen_mesh));
    }

    let mesh_path = project
        .get("mesh")
        .and_then(|v| v.as_str())
        .map(|path| project_path.join(path))
        .or_else(|| {
            let path = project_path.join(&layer4_config.mesh.file);
            if layer4_config.mesh.file.as_os_str().is_empty() {
                None
            } else {
                Some(path)
            }
        })
        .ok_or("项目文件缺少 mesh 字段且配置未提供 mesh.file")?;

    if !mesh_path.exists() {
        return Err(format!("网格文件不存在: {:?}", mesh_path));
    }

    let frozen_mesh = mh_mesh::io::load_mhb(&mesh_path)
        .map_err(|e| format!("加载网格失败: {}", e))?;

    Ok(PhysicsMesh::from_frozen(&frozen_mesh))
}

/// 从项目目录加载Layer4配置
fn load_layer4_config(project_path: &Path) -> Result<mh_config::SolverConfig, String> {
    let config_file = project_path.join("config.json");
    if !config_file.exists() {
        tracing::warn!("配置文件不存在，使用默认配置");
        return Ok(mh_config::SolverConfig::default());
    }

    let content = std::fs::read_to_string(&config_file)
        .map_err(|e| format!("读取配置文件失败: {}", e))?;
    
    let config: mh_config::SolverConfig = serde_json::from_str(&content)
        .map_err(|e| format!("解析配置文件失败: {}", e))?;
    
    Ok(config)
}

fn load_forcing_snapshot(
    project_path: &Path,
    mesh: &PhysicsMesh,
    default_time: f64,
) -> Result<Option<ForcingSnapshot>, RunnerError> {
    let project_file = project_path.join("project.mhp");
    let content = std::fs::read_to_string(&project_file)
        .map_err(|e| RunnerError::Io(format!("读取项目文件失败: {e}")))?;
    let project: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| RunnerError::Config(format!("解析项目文件失败: {e}")))?;

    let forcing = match project.get("forcing") {
        Some(value) => value,
        None => return Ok(None),
    };

    if let Some(inline_wind) = forcing.get("inline_wind") {
        let inline = inline_wind
            .get("u")
            .ok_or_else(|| RunnerError::Config("forcing.inline_wind 缺少 u".to_string()))?;
        let field = parse_inline_forcing(inline)
            .map_err(|e| RunnerError::Initialization(format!("强迫数据解析失败: {e}")))?;
        let variable = "wind_u".to_string();
        let time = inline
            .get("time")
            .and_then(|v| v.as_f64())
            .unwrap_or(default_time);
        let (sample_lon, sample_lat) = inline_wind
            .get("sample")
            .and_then(|v| v.as_array())
            .and_then(|v| {
                if v.len() == 2 {
                    Some((v[0].as_f64()?, v[1].as_f64()?))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                match mesh.cell_center_generic::<CpuBackend<f64>>(mh_runtime::CellIndex::new(0)) {
                    Ok(center) => (center.x(), center.y()),
                    Err(_) => (0.0, 0.0),
                }
            });

        let sample_value = field.interpolate_bilinear(sample_lon, sample_lat);
        return Ok(Some(ForcingSnapshot {
            source: "inline_wind".to_string(),
            variable,
            time,
            sample_lon,
            sample_lat,
            sample_value,
        }));
    }

    if let Some(inline) = forcing.get("inline") {
        let field = parse_inline_forcing(inline)
            .map_err(|e| RunnerError::Initialization(format!("强迫数据解析失败: {e}")))?;
        let variable = inline
            .get("variable")
            .and_then(|v| v.as_str())
            .unwrap_or("inline")
            .to_string();
        let time = inline
            .get("time")
            .and_then(|v| v.as_f64())
            .unwrap_or(default_time);
        let (sample_lon, sample_lat) = inline
            .get("sample")
            .and_then(|v| v.as_array())
            .and_then(|v| {
                if v.len() == 2 {
                    Some((v[0].as_f64()?, v[1].as_f64()?))
                } else {
                    None
                }
            })
            .unwrap_or_else(|| {
                match mesh.cell_center_generic::<CpuBackend<f64>>(mh_runtime::CellIndex::new(0)) {
                    Ok(center) => (center.x(), center.y()),
                    Err(_) => (0.0, 0.0),
                }
            });

        let sample_value = field.interpolate_bilinear(sample_lon, sample_lat);
        return Ok(Some(ForcingSnapshot {
            source: "inline".to_string(),
            variable,
            time,
            sample_lon,
            sample_lat,
            sample_value,
        }));
    }

    #[cfg(feature = "netcdf")]
    {
        if let Some(netcdf) = forcing.get("netcdf") {
            let file = netcdf
                .get("file")
                .and_then(|v| v.as_str())
                .ok_or_else(|| RunnerError::Config("forcing.netcdf 缺少 file".to_string()))?;
            let variable = netcdf
                .get("variable")
                .and_then(|v| v.as_str())
                .ok_or_else(|| RunnerError::Config("forcing.netcdf 缺少 variable".to_string()))?;
            let time = netcdf
                .get("time")
                .and_then(|v| v.as_f64())
                .unwrap_or(default_time);
            let (sample_lon, sample_lat) = netcdf
                .get("sample")
                .and_then(|v| v.as_array())
                .and_then(|v| {
                    if v.len() == 2 {
                        Some((v[0].as_f64()?, v[1].as_f64()?))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| {
                    match mesh.cell_center_generic::<CpuBackend<f64>>(mh_runtime::CellIndex::new(0)) {
                        Ok(center) => (center.x(), center.y()),
                        Err(_) => (0.0, 0.0),
                    }
                });

            let reader = mh_physics::forcing::data::NetCdfReader::open(
                project_path.join(file),
                variable,
            )
            .map_err(|e| RunnerError::Initialization(format!("强迫数据读取失败: {e}")))?;
            let field = reader
                .read_field_at_time(time)
                .map_err(|e| RunnerError::Initialization(format!("强迫数据读取失败: {e}")))?;
            let sample_value = field.interpolate_bilinear(sample_lon, sample_lat);

            return Ok(Some(ForcingSnapshot {
                source: file.to_string(),
                variable: variable.to_string(),
                time,
                sample_lon,
                sample_lat,
                sample_value,
            }));
        }
    }

    #[cfg(not(feature = "netcdf"))]
    if forcing.get("netcdf").is_some() {
        return Err(RunnerError::Config(
            "netcdf 特性未启用，无法读取 forcing.netcdf".to_string(),
        ));
    }

    Ok(None)
}

fn attach_forcing_sources(
    solver: &mut ShallowWaterSolver<CpuBackend<f64>, WindStressRuntimeSource>,
    project_path: &Path,
    mesh: &PhysicsMesh,
    default_time: f64,
) -> Result<Option<WindForcingRuntime>, RunnerError> {
    let project_file = project_path.join("project.mhp");
    let content = std::fs::read_to_string(&project_file)
        .map_err(|e| RunnerError::Io(format!("读取项目文件失败: {e}")))?;
    let project: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| RunnerError::Config(format!("解析项目文件失败: {e}")))?;

    let forcing = match project.get("forcing") {
        Some(value) => value,
        None => return Ok(None),
    };

    if let Some(inline_boundary) = forcing.get("inline_boundary") {
        let series = parse_boundary_series(inline_boundary)?;
        let provider = std::sync::Arc::new(UniformBoundaryProvider::new(series));
        solver.set_boundary_provider(provider);
    }

    if let Some(inline_wind) = forcing.get("inline_wind") {
        let series_u = parse_forcing_series(inline_wind.get("u"), default_time)
            .map_err(|e| RunnerError::Initialization(format!("风场 U 解析失败: {e}")))?;
        let series_v = parse_forcing_series(inline_wind.get("v"), default_time)
            .map_err(|e| RunnerError::Initialization(format!("风场 V 解析失败: {e}")))?;

        let (positions, _) = flatten_forcing_field(&series_u[0])?;
        let weights = compute_interpolation_weights(
            mesh,
            SpatialInterpolation::default(),
            &positions,
        );

        let wind_config = std::sync::Arc::new(std::sync::RwLock::new(
            WindStressConfig::default_config(mesh.cell_count()),
        ));
        let wind_source = WindStressRuntimeSource::new(wind_config.clone());
        solver.register_source(wind_source);

        tracing::info!("风应力源项已接入: {} 单元", mesh.cell_count());

        return Ok(Some(WindForcingRuntime {
            source: wind_config,
            weights,
            source_positions: positions,
            series_u,
            series_v,
            mesh: std::sync::Arc::new(mesh.clone()),
        }));
    }

    Ok(None)
}

/// 创建初始状态
fn create_initial_state(
    mesh: &PhysicsMesh,
    config: &SimulationConfig,
) -> Result<ShallowWaterState<CpuBackend<f64>>, String> {
    let backend = CpuBackend::<f64>::new();
    
    let initial_file = config.project_path.join("initial_state.json");
    if initial_file.exists() {
        tracing::info!("加载初始状态文件: {:?}", initial_file);
        let content = std::fs::read_to_string(&initial_file)
            .map_err(|e| format!("读取初始状态文件失败: {}", e))?;
        
        let data: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| format!("解析初始状态文件失败: {}", e))?;
        
        let h = parse_f64_array(&data["h"])
            .map_err(|e| format!("解析h字段失败: {}", e))?;
        
        if h.len() != mesh.cell_count() {
            return Err(format!("初始状态h数组长度不匹配: 期望 {}, 实际 {}", mesh.cell_count(), h.len()));
        }
        
        let n_cells = h.len();
        let hu = parse_f64_array(&data["hu"]).unwrap_or(vec![0.0; n_cells]);
        let hv = parse_f64_array(&data["hv"]).unwrap_or(vec![0.0; n_cells]);
        
        let z_bed: Vec<f64> = (0..mesh.cell_count())
            .map(|i| mesh.cell_z_bed(mh_runtime::CellIndex::new(i)))
            .collect();
        
        ShallowWaterState::<CpuBackend<f64>>::from_data(
            backend,
            h,
            hu,
            hv,
            z_bed,
        )
        .map_err(|e| format!("初始状态数据无效: {}", e))
    } else {
        tracing::info!("未找到初始状态文件，使用默认静水条件 (h=1.0m)");
        let z_bed: Vec<f64> = (0..mesh.cell_count())
            .map(|i| mesh.cell_z_bed(mh_runtime::CellIndex::new(i)))
            .collect();
        Ok(ShallowWaterState::<CpuBackend<f64>>::cold_start(backend, 1.0, &z_bed))
    }
}

fn parse_f64_array(value: &serde_json::Value) -> Result<Vec<f64>, String> {
    value.as_array()
        .ok_or("期望数组".to_string())
        .and_then(|arr| {
            arr.iter()
                .map(|v| v.as_f64()
                    .ok_or_else(|| format!("无效的双精度浮点数: {}", v)))
                .collect()
        })
}

fn parse_f64_matrix(value: &serde_json::Value) -> Result<Vec<Vec<f64>>, ForcingDataError> {
    let rows = value
        .as_array()
        .ok_or(ForcingDataError::SpatialMismatch)?;

    let mut matrix = Vec::with_capacity(rows.len());
    for row in rows {
        let cols = row.as_array().ok_or(ForcingDataError::SpatialMismatch)?;
        let mut values = Vec::with_capacity(cols.len());
        for v in cols {
            let val = v.as_f64().ok_or(ForcingDataError::SpatialMismatch)?;
            values.push(val);
        }
        matrix.push(values);
    }
    Ok(matrix)
}

fn parse_inline_forcing(value: &serde_json::Value) -> Result<ForcingField, ForcingDataError> {
    let lons = parse_f64_array(value.get("lons").ok_or(ForcingDataError::SpatialMismatch)?)
        .map_err(|_| ForcingDataError::SpatialMismatch)?;
    let lats = parse_f64_array(value.get("lats").ok_or(ForcingDataError::SpatialMismatch)?)
        .map_err(|_| ForcingDataError::SpatialMismatch)?;
    let values = parse_f64_matrix(value.get("values").ok_or(ForcingDataError::SpatialMismatch)?)?;

    if values.len() != lats.len() {
        return Err(ForcingDataError::SpatialMismatch);
    }
    if values.iter().any(|row| row.len() != lons.len()) {
        return Err(ForcingDataError::SpatialMismatch);
    }

    let time = value.get("time").and_then(|v| v.as_f64()).unwrap_or(0.0);
    let fill_value = value
        .get("fill_value")
        .and_then(|v| v.as_f64())
        .unwrap_or(-9999.0);

    Ok(ForcingField {
        values,
        lons,
        lats,
        time,
        fill_value,
    })
}

fn parse_forcing_series(
    value: Option<&serde_json::Value>,
    default_time: f64,
) -> Result<Vec<ForcingField>, ForcingDataError> {
    let value = value.ok_or(ForcingDataError::ReadError("缺少强迫数据".to_string()))?;
    let mut fields = Vec::new();

    if let Some(array) = value.as_array() {
        for item in array {
            fields.push(parse_inline_forcing(item)?);
        }
    } else {
        let mut field = parse_inline_forcing(value)?;
        if field.time == 0.0 {
            field.time = default_time;
        }
        fields.push(field);
    }

    fields.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());
    Ok(fields)
}

fn select_forcing_field(series: &[ForcingField], time: f64) -> Result<ForcingField, RunnerError> {
    if series.is_empty() {
        return Err(RunnerError::Initialization("强迫数据为空".to_string()));
    }
    if series.len() == 1 {
        return Ok(series[0].clone());
    }

    if time <= series[0].time {
        return Ok(series[0].clone());
    }
    if time >= series[series.len() - 1].time {
        return Ok(series[series.len() - 1].clone());
    }

    for i in 0..series.len() - 1 {
        let t0 = series[i].time;
        let t1 = series[i + 1].time;
        if t0 <= time && time <= t1 {
            let frac = if (t1 - t0).abs() < 1e-14 { 0.0 } else { (time - t0) / (t1 - t0) };
            return blend_forcing_fields(&series[i], &series[i + 1], frac);
        }
    }

    Ok(series[0].clone())
}

fn blend_forcing_fields(a: &ForcingField, b: &ForcingField, frac: f64) -> Result<ForcingField, RunnerError> {
    if a.lons != b.lons || a.lats != b.lats {
        return Err(RunnerError::Initialization("强迫数据网格不一致".to_string()));
    }

    if a.values.len() != b.values.len() {
        return Err(RunnerError::Initialization("强迫数据维度不一致".to_string()));
    }

    let mut values = Vec::with_capacity(a.values.len());
    for (row_a, row_b) in a.values.iter().zip(b.values.iter()) {
        if row_a.len() != row_b.len() {
            return Err(RunnerError::Initialization("强迫数据列维度不一致".to_string()));
        }
        let mut row = Vec::with_capacity(row_a.len());
        for (&va, &vb) in row_a.iter().zip(row_b.iter()) {
            row.push(va * (1.0 - frac) + vb * frac);
        }
        values.push(row);
    }

    Ok(ForcingField {
        values,
        lons: a.lons.clone(),
        lats: a.lats.clone(),
        time: a.time * (1.0 - frac) + b.time * frac,
        fill_value: a.fill_value,
    })
}

fn parse_boundary_series(value: &serde_json::Value) -> Result<Vec<(f64, ExternalForcing)>, RunnerError> {
    let mut series = Vec::new();
    if let Some(frames) = value.get("frames").and_then(|v| v.as_array()) {
        for frame in frames {
            let time = frame.get("time").and_then(|v| v.as_f64()).ok_or_else(|| {
                RunnerError::Config("边界强迫缺少 time".to_string())
            })?;
            let eta = frame.get("eta").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let u = frame.get("u").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let v = frame.get("v").and_then(|v| v.as_f64()).unwrap_or(0.0);
            series.push((time, ExternalForcing::new(eta, u, v)));
        }
    } else {
        let time = value.get("time").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let eta = value.get("eta").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let u = value.get("u").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let v = value.get("v").and_then(|v| v.as_f64()).unwrap_or(0.0);
        series.push((time, ExternalForcing::new(eta, u, v)));
    }

    if series.is_empty() {
        return Err(RunnerError::Config("边界强迫数据为空".to_string()));
    }
    Ok(series)
}

fn flatten_forcing_field(field: &ForcingField) -> Result<(Vec<(f64, f64)>, Vec<f64>), RunnerError> {
    if field.lons.is_empty() || field.lats.is_empty() {
        return Err(RunnerError::Initialization("强迫数据经纬度为空".to_string()));
    }
    if field.values.len() != field.lats.len() {
        return Err(RunnerError::Initialization(
            "强迫数据纬度维度与 values 行数不一致".to_string(),
        ));
    }

    let mut positions = Vec::with_capacity(field.lons.len() * field.lats.len());
    let mut values = Vec::with_capacity(field.lons.len() * field.lats.len());

    for (j, lat) in field.lats.iter().copied().enumerate() {
        let row = field.values.get(j).ok_or_else(|| {
            RunnerError::Initialization("强迫数据 values 行访问失败".to_string())
        })?;
        if row.len() != field.lons.len() {
            return Err(RunnerError::Initialization(
                "强迫数据经度维度与 values 列数不一致".to_string(),
            ));
        }
        for (i, lon) in field.lons.iter().copied().enumerate() {
            positions.push((lon, lat));
            let mut v = row[i];
            if v == field.fill_value {
                v = 0.0;
            }
            values.push(v);
        }
    }

    Ok((positions, values))
}

fn compute_config_hash(solver: &ShallowWaterSolver<CpuBackend<f64>, WindStressRuntimeSource>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    let config = solver.config();

    // 尽量覆盖关键数值与开关，避免误判兼容性
    config.params.cfl.to_bits().hash(&mut hasher);
    config.params.h_dry.to_bits().hash(&mut hasher);
    config.params.h_min.to_bits().hash(&mut hasher);
    config.params.h_wet.to_bits().hash(&mut hasher);
    config.params.h_friction.to_bits().hash(&mut hasher);
    config.params.vel_max.to_bits().hash(&mut hasher);
    config.params.vel_min.to_bits().hash(&mut hasher);
    config.params.flux_eps.to_bits().hash(&mut hasher);
    config.params.entropy_ratio.to_bits().hash(&mut hasher);
    config.params.min_wave_speed.to_bits().hash(&mut hasher);
    config.params.det_min.to_bits().hash(&mut hasher);
    config.params.limiter_k.to_bits().hash(&mut hasher);
    config.params.nu_min.to_bits().hash(&mut hasher);
    config.params.nu_max.to_bits().hash(&mut hasher);
    config.params.dt_min.to_bits().hash(&mut hasher);
    config.params.dt_max.to_bits().hash(&mut hasher);
    config.params.eta_tolerance.to_bits().hash(&mut hasher);
    config.params.flux_tolerance.to_bits().hash(&mut hasher);
    config.params.conservation_tolerance.to_bits().hash(&mut hasher);

    let scheme_id: u8 = match config.scheme {
        mh_physics::NumericalScheme::FirstOrder => 0,
        mh_physics::NumericalScheme::SecondOrderMuscl => 1,
        mh_physics::NumericalScheme::SecondOrderWeno => 2,
    };
    scheme_id.hash(&mut hasher);

    let riemann_id: u8 = match config.riemann_solver {
        mh_config::solver_config::RiemannSolverType::Hllc => 0,
        mh_config::solver_config::RiemannSolverType::Roe => 1,
        mh_config::solver_config::RiemannSolverType::Rusanov => 2,
        mh_config::solver_config::RiemannSolverType::Central => 3,
    };
    riemann_id.hash(&mut hasher);

    let integrator_id: u8 = match config.integrator {
        mh_physics::engine::solver::TimeIntegrator::Explicit => 0,
        mh_physics::engine::solver::TimeIntegrator::SemiImplicit => 1,
    };
    integrator_id.hash(&mut hasher);

    let fallback_id: u8 = match config.fallback {
        mh_physics::engine::solver::FallbackStrategy::NoFallback => 0,
        mh_physics::engine::solver::FallbackStrategy::FallbackToFirstOrder => 1,
        mh_physics::engine::solver::FallbackStrategy::ReduceTimestep => 2,
        mh_physics::engine::solver::FallbackStrategy::Progressive => 3,
    };
    fallback_id.hash(&mut hasher);

    let time_integrator_id: u8 = match config.time_integrator_kind {
        mh_physics::engine::TimeIntegratorKind::ForwardEuler => 0,
        mh_physics::engine::TimeIntegratorKind::SspRk2 => 1,
        mh_physics::engine::TimeIntegratorKind::SspRk3 => 2,
    };
    time_integrator_id.hash(&mut hasher);

    config.gravity.to_bits().hash(&mut hasher);
    config.default_manning_n.to_bits().hash(&mut hasher);
    config.use_hydrostatic_reconstruction.hash(&mut hasher);
    config.implicit_friction.hash(&mut hasher);
    config.parallel_threshold.hash(&mut hasher);
    config.max_fallback_attempts.hash(&mut hasher);
    config.timestep_reduction_factor.to_bits().hash(&mut hasher);

    config.stability.check_nan.hash(&mut hasher);
    config.stability.check_negative_depth.hash(&mut hasher);
    config.stability.check_extreme_velocity.hash(&mut hasher);
    config.stability.velocity_limit.to_bits().hash(&mut hasher);
    config.stability.depth_limit.to_bits().hash(&mut hasher);

    hasher.finish()
}

fn compute_mesh_hash(mesh: &PhysicsMesh) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    mesh.cell_count().hash(&mut hasher);
    mesh.node_count().hash(&mut hasher);
    
    let total_area: f64 = (0..mesh.cell_count())
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
    use serde_json::json;

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.progress_interval, 1.0);
        assert!(config.nan_detection_enabled);
    }

    #[test]
    fn test_run_context_creation() {
        let project_dir = tempfile::tempdir().unwrap();
        
        // 创建最小有效项目结构
        let project_file = project_dir.path().join("project.mhp");
        let mesh_content = r#"{"mesh": "test.mhb"}"#;
        std::fs::write(&project_file, mesh_content).unwrap();
        
        let config = SimulationConfig::new(project_dir.path())
            .with_time_range(0.0, 100.0);
        let job = SimulationJob::new("TestJob", config);

        let runner = JobRunner::new(Arc::new(WorkflowManager::new(MemoryStorage::new())));
        
        // 由于缺少实际网格文件，此测试主要验证错误处理路径
        let result = RunContext::new(&job, &runner.config);
        
        // 期望失败，因为mesh文件不存在，但不应panic
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err,
            RunnerError::Initialization(ref msg)
                if msg.contains("网格加载失败") || msg.contains("网格文件不存在")
        ));
    }

    #[test]
    fn test_run_context_structured_with_inline_forcing() {
        let project_dir = tempfile::tempdir().unwrap();

        let project_file = project_dir.path().join("project.mhp");
        let content = r#"{
            "structured": {"nx": 2, "ny": 2, "dx": 1.0, "dy": 1.0, "origin": [0.0, 0.0]},
            "forcing": {
                "inline_wind": {
                    "u": {
                        "lons": [0.0, 1.0],
                        "lats": [0.0, 1.0],
                        "values": [[1.0, 2.0], [3.0, 4.0]],
                        "time": 0.0,
                        "fill_value": -9999.0
                    },
                    "v": {
                        "lons": [0.0, 1.0],
                        "lats": [0.0, 1.0],
                        "values": [[-1.0, -2.0], [-3.0, -4.0]],
                        "time": 0.0,
                        "fill_value": -9999.0
                    },
                    "sample": [0.5, 0.5]
                }
            }
        }"#;
        std::fs::write(&project_file, content).unwrap();

        let config = SimulationConfig::new(project_dir.path())
            .with_time_range(0.0, 10.0);
        let job = SimulationJob::new("StructuredWithForcing", config);
        let runner = JobRunner::new(Arc::new(WorkflowManager::new(MemoryStorage::new())));

        let context = RunContext::new(&job, &runner.config).expect("上下文创建失败");
        let snapshot = context.forcing_snapshot.expect("强迫数据未加载");
        assert_eq!(snapshot.source, "inline_wind");
        assert_eq!(snapshot.variable, "wind_u");
        assert!(snapshot.sample_value.is_some());

        let solver = context.solver.read();
        assert!(solver.source_count() >= 1);
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

    #[test]
    fn test_parse_forcing_series_default_time() {
        let value = json!({
            "lons": [0.0],
            "lats": [0.0],
            "values": [[1.0]]
        });
        let series = parse_forcing_series(Some(&value), 12.0).expect("解析失败");
        assert_eq!(series.len(), 1);
        assert!((series[0].time - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_select_forcing_field_interpolation() {
        let field_a = ForcingField {
            values: vec![vec![0.0]],
            lons: vec![0.0],
            lats: vec![0.0],
            time: 0.0,
            fill_value: -9999.0,
        };
        let field_b = ForcingField {
            values: vec![vec![10.0]],
            lons: vec![0.0],
            lats: vec![0.0],
            time: 10.0,
            fill_value: -9999.0,
        };
        let series = vec![field_a, field_b];
        let blended = select_forcing_field(&series, 5.0).expect("插值失败");
        assert!((blended.values[0][0] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_select_forcing_field_grid_mismatch() {
        let field_a = ForcingField {
            values: vec![vec![1.0]],
            lons: vec![0.0],
            lats: vec![0.0],
            time: 0.0,
            fill_value: -9999.0,
        };
        let field_b = ForcingField {
            values: vec![vec![2.0, 3.0]],
            lons: vec![0.0, 1.0],
            lats: vec![0.0],
            time: 10.0,
            fill_value: -9999.0,
        };
        let series = vec![field_a, field_b];
        let err = select_forcing_field(&series, 5.0).unwrap_err();
        assert!(matches!(err, RunnerError::Initialization(_)));
    }

    #[test]
    fn test_parse_boundary_series_missing_time() {
        let value = json!({
            "frames": [{"eta": 1.0}]
        });
        let err = parse_boundary_series(&value).unwrap_err();
        assert!(matches!(err, RunnerError::Config(_)));
    }

    #[test]
    fn test_uniform_boundary_provider_interpolation() {
        let series = vec![
            (0.0, ExternalForcing::new(1.0, 2.0, 3.0)),
            (10.0, ExternalForcing::new(3.0, 4.0, 5.0)),
        ];
        let provider = UniformBoundaryProvider::new(series);

        let early = provider.interpolate(-1.0).expect("插值失败");
        assert!((early.eta - 1.0).abs() < 1e-12);

        let mid = provider.interpolate(5.0).expect("插值失败");
        assert!((mid.eta - 2.0).abs() < 1e-12);
        assert!((mid.u() - 3.0).abs() < 1e-12);
        assert!((mid.v() - 4.0).abs() < 1e-12);

        let late = provider.interpolate(20.0).expect("插值失败");
        assert!((late.eta - 3.0).abs() < 1e-12);
    }
}

/// 强迫数据采样快照
#[derive(Debug, Clone)]
pub struct ForcingSnapshot {
    pub source: String,
    pub variable: String,
    pub time: f64,
    pub sample_lon: f64,
    pub sample_lat: f64,
    pub sample_value: Option<f64>,
}

struct WindForcingRuntime {
    source: std::sync::Arc<std::sync::RwLock<WindStressConfig>>,
    weights: InterpolationWeights,
    source_positions: Vec<(f64, f64)>,
    series_u: Vec<ForcingField>,
    series_v: Vec<ForcingField>,
    mesh: std::sync::Arc<PhysicsMesh>,
}

impl WindForcingRuntime {
    fn update(&mut self, time: f64) -> Result<(), RunnerError> {
        let field_u = select_forcing_field(&self.series_u, time)?;
        let field_v = select_forcing_field(&self.series_v, time)?;

        let (positions_u, values_u) = flatten_forcing_field(&field_u)?;
        let (positions_v, values_v) = flatten_forcing_field(&field_v)?;

        if positions_u.len() != positions_v.len() || positions_u != positions_v {
            return Err(RunnerError::Initialization(
                "风场 U/V 网格不一致".to_string(),
            ));
        }

        if positions_u != self.source_positions {
            self.source_positions = positions_u;
            self.weights = compute_interpolation_weights(
                &self.mesh,
                SpatialInterpolation::default(),
                &self.source_positions,
            );
        }

        let wind_u = self.weights.apply(&values_u);
        let wind_v = self.weights.apply(&values_v);

        if let Ok(mut cfg) = self.source.write() {
            cfg.set_wind_field(&wind_u, &wind_v);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct UniformBoundaryProvider {
    times: Vec<f64>,
    values: Vec<ExternalForcing>,
}

impl UniformBoundaryProvider {
    fn new(series: Vec<(f64, ExternalForcing)>) -> Self {
        let mut series = series;
        series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let times = series.iter().map(|(t, _)| *t).collect();
        let values = series.iter().map(|(_, v)| *v).collect();
        Self { times, values }
    }

    fn interpolate(&self, time: f64) -> Option<ExternalForcing> {
        if self.times.is_empty() {
            return None;
        }
        if self.times.len() == 1 {
            return Some(self.values[0]);
        }
        if time <= self.times[0] {
            return Some(self.values[0]);
        }
        if time >= *self.times.last().unwrap() {
            return Some(*self.values.last().unwrap());
        }

        for i in 0..self.times.len() - 1 {
            let t0 = self.times[i];
            let t1 = self.times[i + 1];
            if t0 <= time && time <= t1 {
                let frac = if (t1 - t0).abs() < 1e-14 { 0.0 } else { (time - t0) / (t1 - t0) };
                let v0 = self.values[i];
                let v1 = self.values[i + 1];
                return Some(ExternalForcing::new(
                    v0.eta + (v1.eta - v0.eta) * frac,
                    v0.u() + (v1.u() - v0.u()) * frac,
                    v0.v() + (v1.v() - v0.v()) * frac,
                ));
            }
        }

        None
    }
}

impl BoundaryDataProvider for UniformBoundaryProvider {
    fn get_forcing(&self, _face_id: usize, time: f64) -> Option<ExternalForcing> {
        self.interpolate(time)
    }
}
