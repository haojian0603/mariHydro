// crates/mh_workflow/src/runner.rs

//! 任务运行器模块
//!
//! 提供任务执行的全生命周期管理，包括网格加载、求解器初始化、
//! 时间步进、检查点保存、VTU输出生成和进度跟踪。

use crate::events::WorkflowEvent;
use crate::job::{JobId, SimulationConfig, SimulationJob};
use crate::manager::{WorkflowError, WorkflowManager};
use crate::scheduler::{DeviceSelection, HybridScheduler};
use crate::storage::Storage;
use mh_physics::{
    engine::{ShallowWaterSolver, SolverStats, StabilityStatus,},
    state::ShallowWaterState,
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
use num_traits::ToPrimitive;

/// 运行器错误类型
#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("Workflow error: {0}")]
    Workflow(#[from] WorkflowError),

    #[error("Job {0} is already running")]
    AlreadyRunning(JobId),

    #[error("Job not found: {0}")]
    NotFound(JobId),

    #[error("Computation error: {0}")]
    Computation(String),

    #[error("Job was cancelled")]
    Cancelled,

    #[error("Job timed out after {0} seconds")]
    Timeout(u64),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Initialization error: {0}")]
    Initialization(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Numerical instability detected: {0}")]
    Instability(String),

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
    pub progress_interval: f64,
    pub checkpoint_interval: f64,
    pub timeout_secs: u64,
    pub num_threads: usize,
    pub output_interval: f64,
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
    pub solver: Arc<RwLock<ShallowWaterSolver<CpuBackend<f64>>>>,
    pub state: Arc<RwLock<ShallowWaterState<CpuBackend<f64>>>>,
    last_output_time: RwLock<f64>,
    output_counter: AtomicU64,
    last_checkpoint_time: RwLock<f64>,
}

impl RunContext {
    /// 创建并初始化运行上下文
    pub fn new(
        job: &SimulationJob,
        _runner_config: &RunnerConfig,
    ) -> Result<Self, RunnerError> {
        let mesh = load_mesh_from_project(&job.config.project_path)
            .map_err(|e| RunnerError::Initialization(format!("网格加载失败: {}", e)))?;
        let mesh = Arc::new(mesh);

        let layer4_config = load_layer4_config(&job.config.project_path)
            .map_err(|e| RunnerError::Config(format!("配置加载失败: {}", e)))?;
        
        let layer3_config: Layer3Config<f64> = Layer3Config::from_layer4(&layer4_config)
            .map_err(|e| RunnerError::Config(format!("配置转换失败: {}", e)))?;

        let backend = CpuBackend::<f64>::new();
        let solver = Arc::new(RwLock::new(
            ShallowWaterSolver::<CpuBackend<f64>>::new(mesh.clone(), layer3_config, backend)
        ));

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

    pub fn with_device(mut self, device: DeviceSelection) -> Self {
        self.device = Some(device);
        self
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
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
        now - *self.last_output_time.read() >= self.config.output_interval
    }

    pub fn should_checkpoint(&self, now: f64, config: &RunnerConfig) -> bool {
        config.checkpoint_interval > 0.0
            && now - *self.last_checkpoint_time.read() >= config.checkpoint_interval
    }

    pub fn is_finished(&self) -> bool {
        self.current_sim_time() >= self.config.end_time
    }

    pub fn solver_stats(&self) -> SolverStats {
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
            "Starting simulation {}: {:.2}s -> {:.2}s (dt_out={:.2}s, dt_chk={:.2}s)",
            context.job_id,
            context.config.start_time,
            context.config.end_time,
            context.config.output_interval,
            context.config.checkpoint_interval,
        );

        while !context.is_finished() && !context.is_cancelled() {
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

            if context.should_output(current_time) {
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
                        "dt={:.4}s, wave_speed={:.2}m/s, dry_cells={}, status={}",
                        stats.dt, stats.max_wave_speed, stats.dry_cells, stats.stability_status
                    )),
                )?;
                last_progress_time = Instant::now();
            }

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
    fn execute_timestep(&self, context: &RunContext) -> Result<(), RunnerError> {
        let mut solver = context.solver.write();
        let mut state = context.state.write();

        let dt_computed = solver.compute_dt(&state);
        let dt = dt_computed.max(1e-8).min(10.0);

        solver.step(&mut state, dt);

        let new_time = context.current_sim_time() + dt.to_f64().unwrap();
        context.set_current_sim_time(new_time);
        context.increment_steps(1);

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

        let snapshot = mh_io::snapshot::StateSnapshot::from_state_data(
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
            "Checkpoint saved for job {} at time {:.2}s to {:?}",
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
        );

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
            "Output written for job {}: step {}, time {:.2}s to {:?}",
            context.job_id,
            step,
            context.current_sim_time(),
            path
        );

        Ok(())
    }
}

/// 从项目目录加载网格文件
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
fn create_initial_state(
    mesh: &PhysicsMesh,
    config: &SimulationConfig,
) -> Result<ShallowWaterState<CpuBackend<f64>>, Box<dyn std::error::Error>> {
    let backend = CpuBackend::<f64>::new();
    
    let initial_file = config.project_path.join("initial_state.json");
    if initial_file.exists() {
        tracing::info!("加载初始状态文件: {:?}", initial_file);
        let content = std::fs::read_to_string(&initial_file)?;
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        let h = parse_f64_array(&data["h"])?;
        let n_cells = h.len();
        let hu = parse_f64_array(&data["hu"]).unwrap_or(vec![0.0; n_cells]);
        let hv = parse_f64_array(&data["hv"]).unwrap_or(vec![0.0; n_cells]);
        
        if h.len() != mesh.n_cells() {
            return Err(format!("初始状态h数组长度不匹配: 期望 {}, 实际 {}", mesh.n_cells(), h.len()).into());
        }
        
        let z_bed: Vec<f64> = (0..mesh.n_cells())
            .map(|i| mesh.cell_z_bed(mh_runtime::CellIndex::new(i)))
            .collect();
        
        Ok(ShallowWaterState::<CpuBackend<f64>>::from_data(
            backend,
            h,
            hu,
            hv,
            z_bed,
        ))
    } else {
        tracing::info!("未找到初始状态文件，使用默认静水条件 (h=1.0m)");
        let z_bed: Vec<f64> = (0..mesh.n_cells())
            .map(|i| mesh.cell_z_bed(mh_runtime::CellIndex::new(i)))
            .collect();
        Ok(ShallowWaterState::<CpuBackend<f64>>::cold_start(backend, 1.0, &z_bed))
    }
}

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

fn compute_config_hash(solver: &ShallowWaterSolver<CpuBackend<f64>>) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    let config = solver.config();
    config.params.cfl.to_bits().hash(&mut hasher);
    config.params.h_dry.to_bits().hash(&mut hasher);
    hasher.finish()
}

fn compute_mesh_hash(mesh: &PhysicsMesh) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    mesh.n_cells().hash(&mut hasher);
    mesh.n_nodes().hash(&mut hasher);
    
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