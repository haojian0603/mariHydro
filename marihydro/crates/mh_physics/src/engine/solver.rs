//! 浅水方程求解器（Backend泛型化版本）
//!
//! 基于有限体积法的非结构化网格求解器，支持f32/f64精度切换和GPU后端。
//! 本模块属于Layer 3(Engine层)，所有类型使用Backend泛型参数。

use crate::adapter::{CellIndex, FaceIndex, PhysicsMesh};
use crate::engine::timestep::TimeStepController;
use crate::schemes::{CentralSolver, HllcSolver, RoeSolver, RusanovSolver, RiemannFlux, RiemannSolver, SolverParams};
use crate::schemes::riemann::RiemannSolverAny;
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::numerics::{MusclConfig, MusclReconstructor, WenoConfig, WenoReconstructor};
use crate::numerics::reconstruction::ReconstructedState;
use crate::numerics::Reconstructor;
use crate::state::ShallowWaterState;
use crate::sources::traits::{SourceContextGeneric, SourceTermGeneric};
use crate::{BoundaryDataProvider, ExternalForcing};
use crate::types::{NumericalParams};
use crate::Layer3Config;

use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar, Vector2D};
use mh_config::solver_config::RiemannSolverType;
use num_traits::Float;
use rayon::prelude::*;
use std::sync::Arc;

/// 静水重构状态（Backend泛型化）
#[derive(Debug, Clone, Copy)]
pub struct HydrostaticFaceState<B: Backend> {
    pub h_left: B::Scalar,
    pub h_right: B::Scalar,
    pub vel_left: B::Vector2D,
    pub vel_right: B::Vector2D,
    pub z_face: B::Scalar,
}

/// 床坡源项修正（Backend泛型化）
#[derive(Debug, Clone, Copy)]
pub struct BedSlopeCorrection<B: Backend> {
    pub source_left_x: B::Scalar,
    pub source_left_y: B::Scalar,
    pub source_right_x: B::Scalar,
    pub source_right_y: B::Scalar,
}

/// 求解器统计信息（Backend 泛型版本）
/// 
/// 记录每个时间步的求解器性能指标和状态信息。
/// 所有数值类型使用 Backend 标量类型，确保精度一致性。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
#[derive(Debug, Clone)]
pub struct SolverStats<B: Backend> {
    /// 最大波速 [m/s]
    pub max_wave_speed: B::Scalar,
    /// 干单元数量
    pub dry_cells: usize,
    /// 被限制的面数量
    pub limited_faces: usize,
    /// 实际时间步长 [s]
    pub dt: B::Scalar,
    /// 回退次数
    pub fallback_count: u32,
    /// 当前数值格式
    pub current_scheme: NumericalScheme,
    /// 稳定性状态
    pub stability_status: StabilityStatus,
    /// 检测到的 NaN 数量
    pub nan_count: u32,
    /// 最后一个 NaN 位置
    pub last_nan_location: Option<usize>,
}

impl<B: Backend> Default for SolverStats<B> {
    fn default() -> Self {
        Self {
            max_wave_speed: B::Scalar::ZERO,
            dry_cells: 0,
            limited_faces: 0,
            dt: B::Scalar::ZERO,
            fallback_count: 0,
            current_scheme: NumericalScheme::default(),
            stability_status: StabilityStatus::default(),
            nan_count: 0,
            last_nan_location: None,
        }
    }
}

/// 稳定性状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StabilityStatus {
    #[default]
    Stable,
    Marginal,
    NeedsFallback,
    Unstable,
}

impl std::fmt::Display for StabilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Marginal => write!(f, "临界"),
            Self::Stable => write!(f, "稳定"),
            Self::NeedsFallback => write!(f, "需要回退"),
            Self::Unstable => write!(f, "不稳定"),
        }
    }
}

impl<B: Backend> SolverStats<B> {
    /// 检查是否需要回退
    pub fn needs_fallback(&self) -> bool {
        matches!(
            self.stability_status,
            StabilityStatus::NeedsFallback | StabilityStatus::Unstable
        )
    }

    /// 生成摘要字符串
    pub fn summary(&self) -> String {
        format!(
            "dt={:.4}s, 波速={:.2}m/s, 干单元={}, 限制面={}, 状态={}, 回退={}, NaN={}",
            self.dt.to_f64_lossy(),
            self.max_wave_speed.to_f64_lossy(),
            self.dry_cells,
            self.limited_faces,
            self.stability_status,
            self.fallback_count,
            self.nan_count,
        )
    }
    
    /// 转换为 f64 版本（用于日志输出和序列化）
    pub fn to_f64(&self) -> SolverStatsF64 {
        SolverStatsF64 {
            max_wave_speed: self.max_wave_speed.to_f64_lossy(),
            dry_cells: self.dry_cells,
            limited_faces: self.limited_faces,
            dt: self.dt.to_f64_lossy(),
            fallback_count: self.fallback_count,
            current_scheme: self.current_scheme,
            stability_status: self.stability_status,
            nan_count: self.nan_count,
            last_nan_location: self.last_nan_location,
        }
    }
}

/// 求解器统计信息（f64 版本，用于输出和序列化）
#[derive(Debug, Clone, Default)]
pub struct SolverStatsF64 {
    pub max_wave_speed: f64,
    pub dry_cells: usize,
    pub limited_faces: usize,
    pub dt: f64,
    pub fallback_count: u32,
    pub current_scheme: NumericalScheme,
    pub stability_status: StabilityStatus,
    pub nan_count: u32,
    pub last_nan_location: Option<usize>,
}

impl SolverStatsF64 {
    /// 生成摘要字符串
    pub fn summary(&self) -> String {
        format!(
            "dt={:.4}s, wave_speed={:.2}m/s, dry={}, limited={}, status={}, fallbacks={}, nan_detected={}",
            self.dt,
            self.max_wave_speed,
            self.dry_cells,
            self.limited_faces,
            self.stability_status,
            self.fallback_count,
            self.nan_count,
        )
    }
}

/// NaN检测结果
#[derive(Debug, Clone, Default)]
pub struct NanDetectionResult {
    pub found_nan: bool,
    pub affected_cells: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ParallelFluxAccumulation<B: Backend> {
    flux_h: Vec<B::Scalar>,
    flux_hu: Vec<B::Scalar>,
    flux_hv: Vec<B::Scalar>,
    source_hu: Vec<B::Scalar>,
    source_hv: Vec<B::Scalar>,
    max_wave_speed: B::Scalar,
}

impl<B: Backend> ParallelFluxAccumulation<B> {
    fn new(n_cells: usize) -> Self {
        let zero = B::Scalar::ZERO;
        Self {
            flux_h: vec![zero; n_cells],
            flux_hu: vec![zero; n_cells],
            flux_hv: vec![zero; n_cells],
            source_hu: vec![zero; n_cells],
            source_hv: vec![zero; n_cells],
            max_wave_speed: zero,
        }
    }

    #[inline]
    fn accumulate_face(
        &mut self,
        owner_idx: usize,
        neighbor_idx: Option<usize>,
        fh: B::Scalar,
        fhu: B::Scalar,
        fhv: B::Scalar,
        source_left_x: B::Scalar,
        source_left_y: B::Scalar,
        source_right_x: B::Scalar,
        source_right_y: B::Scalar,
        wave_speed: B::Scalar,
    ) {
        self.max_wave_speed = self.max_wave_speed.max(wave_speed);

        self.flux_h[owner_idx] -= fh;
        self.flux_hu[owner_idx] -= fhu;
        self.flux_hv[owner_idx] -= fhv;
        self.source_hu[owner_idx] += source_left_x;
        self.source_hv[owner_idx] += source_left_y;

        if let Some(neigh_idx) = neighbor_idx {
            self.flux_h[neigh_idx] += fh;
            self.flux_hu[neigh_idx] += fhu;
            self.flux_hv[neigh_idx] += fhv;
            self.source_hu[neigh_idx] += source_right_x;
            self.source_hv[neigh_idx] += source_right_y;
        }
    }

    fn merge(mut self, other: Self) -> Self {
        debug_assert_eq!(self.flux_h.len(), other.flux_h.len());

        for (dst, src) in self.flux_h.iter_mut().zip(other.flux_h) {
            *dst += src;
        }
        for (dst, src) in self.flux_hu.iter_mut().zip(other.flux_hu) {
            *dst += src;
        }
        for (dst, src) in self.flux_hv.iter_mut().zip(other.flux_hv) {
            *dst += src;
        }
        for (dst, src) in self.source_hu.iter_mut().zip(other.source_hu) {
            *dst += src;
        }
        for (dst, src) in self.source_hv.iter_mut().zip(other.source_hv) {
            *dst += src;
        }

        self.max_wave_speed = self.max_wave_speed.max(other.max_wave_speed);
        self
    }

    fn write_into(&self, workspace: &mut SolverWorkspaceGeneric<B>) {
        for i in 0..self.flux_h.len() {
            workspace.flux_h[i] = self.flux_h[i];
            workspace.flux_hu[i] = self.flux_hu[i];
            workspace.flux_hv[i] = self.flux_hv[i];
            workspace.source_hu[i] = self.source_hu[i];
            workspace.source_hv[i] = self.source_hv[i];
        }
    }
}

/// 求解器工作区（Backend泛型版本）
#[derive(Debug)]
pub struct SolverWorkspaceGeneric<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub flux_h: B::Buffer<B::Scalar>,
    pub flux_hu: B::Buffer<B::Scalar>,
    pub flux_hv: B::Buffer<B::Scalar>,
    pub source_h: B::Buffer<B::Scalar>,
    pub source_hu: B::Buffer<B::Scalar>,
    pub source_hv: B::Buffer<B::Scalar>,
    pub vel_u: B::Buffer<B::Scalar>,
    pub vel_v: B::Buffer<B::Scalar>,
    pub eta: B::Buffer<B::Scalar>,
}

impl<B: Backend> SolverWorkspaceGeneric<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(backend: &B, n_cells: usize) -> Self {
        Self {
            flux_h: backend.alloc(n_cells),
            flux_hu: backend.alloc(n_cells),
            flux_hv: backend.alloc(n_cells),
            source_h: backend.alloc(n_cells),
            source_hu: backend.alloc(n_cells),
            source_hv: backend.alloc(n_cells),
            vel_u: backend.alloc(n_cells),
            vel_v: backend.alloc(n_cells),
            eta: backend.alloc(n_cells),
        }
    }

    pub fn reset_fluxes(&mut self) {
        let zero = B::Scalar::ZERO;
        self.flux_h.fill(zero);
        self.flux_hu.fill(zero);
        self.flux_hv.fill(zero);
    }

    pub fn reset_sources(&mut self) {
        let zero = B::Scalar::ZERO;
        self.source_h.fill(zero);
        self.source_hu.fill(zero);
        self.source_hv.fill(zero);
    }

    pub fn reset(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
    }

    pub fn resize(&mut self, n_cells: usize) {
        use mh_runtime::DeviceBuffer;
        self.flux_h.resize(n_cells, B::Scalar::ZERO);
        self.flux_hu.resize(n_cells, B::Scalar::ZERO);
        self.flux_hv.resize(n_cells, B::Scalar::ZERO);
        self.source_h.resize(n_cells, B::Scalar::ZERO);
        self.source_hu.resize(n_cells, B::Scalar::ZERO);
        self.source_hv.resize(n_cells, B::Scalar::ZERO);
        self.vel_u.resize(n_cells, B::Scalar::ZERO);
        self.vel_v.resize(n_cells, B::Scalar::ZERO);
        self.eta.resize(n_cells, B::Scalar::ZERO);
    }
}

/// 数值格式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericalScheme {
    #[default]
    FirstOrder,
    SecondOrderMuscl,
    SecondOrderWeno,
}

/// 标量重构器封装（MUSCL/WENO）
enum ScalarReconstructor<B: Backend> {
    Muscl(MusclReconstructor<B>),
    Weno(WenoReconstructor<B>),
}

impl<B: Backend + Clone> ScalarReconstructor<B> {
    fn new(mesh: Arc<PhysicsMesh>, scheme: NumericalScheme, backend: B) -> Self {
        match scheme {
            NumericalScheme::SecondOrderWeno => {
                Self::Weno(WenoReconstructor::new(WenoConfig::default(), mesh, backend))
            }
            NumericalScheme::FirstOrder => {
                let mut recon = MusclReconstructor::new(MusclConfig::first_order(), mesh, backend);
                recon.set_config(MusclConfig::first_order());
                Self::Muscl(recon)
            }
            NumericalScheme::SecondOrderMuscl => {
                Self::Muscl(MusclReconstructor::new(MusclConfig::default(), mesh, backend))
            }
        }
    }

    fn configure_for_scheme(&mut self, scheme: NumericalScheme, mesh: Arc<PhysicsMesh>, backend: B) {
        match scheme {
            NumericalScheme::SecondOrderWeno => {
                if let ScalarReconstructor::Weno(recon) = self {
                    recon.set_config(WenoConfig::default());
                } else {
                    *self = Self::new(mesh, scheme, backend);
                }
            }
            NumericalScheme::SecondOrderMuscl => {
                if let ScalarReconstructor::Muscl(recon) = self {
                    recon.set_config(MusclConfig::default());
                } else {
                    *self = Self::new(mesh, scheme, backend);
                }
            }
            NumericalScheme::FirstOrder => {
                if let ScalarReconstructor::Muscl(recon) = self {
                    recon.set_config(MusclConfig::first_order());
                } else {
                    *self = Self::new(mesh, scheme, backend);
                }
            }
        }
    }

    fn compute_gradients(&mut self, values: &B::Buffer<B::Scalar>) {
        match self {
            ScalarReconstructor::Muscl(recon) => recon.compute_gradients(values),
            ScalarReconstructor::Weno(recon) => recon.compute_gradients(values),
        }
    }

    fn reconstruct_scalar(&self, face_id: usize, values: &B::Buffer<B::Scalar>) -> ReconstructedState<B> {
        match self {
            ScalarReconstructor::Muscl(recon) => recon.reconstruct_scalar(face_id, values),
            ScalarReconstructor::Weno(recon) => recon.reconstruct_scalar(face_id, values),
        }
    }
}

impl std::fmt::Display for NumericalScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstOrder => write!(f, "First Order"),
            Self::SecondOrderMuscl => write!(f, "MUSCL"),
            Self::SecondOrderWeno => write!(f, "WENO"),
        }
    }
}

impl NumericalScheme {
    /// 是否已实现
    pub fn is_implemented(&self) -> bool {
        true
    }

    /// 格式阶数
    pub fn order(&self) -> usize {
        match self {
            Self::FirstOrder => 1,
            Self::SecondOrderMuscl | Self::SecondOrderWeno => 2,
        }
    }

    /// 降级链
    pub fn fallback(&self) -> Option<Self> {
        match self {
            Self::SecondOrderWeno => Some(Self::SecondOrderMuscl),
            Self::SecondOrderMuscl => Some(Self::FirstOrder),
            Self::FirstOrder => None,
        }
    }
}

/// 回退策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackStrategy {
    #[default]
    NoFallback,
    FallbackToFirstOrder,
    ReduceTimestep,
    Progressive,
}

impl std::fmt::Display for FallbackStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFallback => write!(f, "无回退"),
            Self::FallbackToFirstOrder => write!(f, "回退一阶"),
            Self::ReduceTimestep => write!(f, "减小时间步"),
            Self::Progressive => write!(f, "渐进回退"),
        }
    }
}

/// 时间积分器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeIntegrator {
    #[default]
    Explicit,
    SemiImplicit,
}

impl std::fmt::Display for TimeIntegrator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Explicit => write!(f, "显式"),
            Self::SemiImplicit => write!(f, "半隐式"),
        }
    }
}

/// 稳定性检查选项
#[derive(Debug, Clone, Copy)]
pub struct StabilityOptions {
    pub check_nan: bool,
    pub check_negative_depth: bool,
    pub check_extreme_velocity: bool,
    pub velocity_limit: f64,
    pub depth_limit: f64,
}

impl Default for StabilityOptions {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: true,
            velocity_limit: 100.0,
            depth_limit: 1000.0,
        }
    }
}

/// 静水重构处理器（Backend泛型化）
#[derive(Debug, Clone)]
pub struct HydrostaticReconstruction<B: Backend> {
    _params: SolverParams<B::Scalar>,
    gravity: B::Scalar,
}

impl<B: Backend> HydrostaticReconstruction<B> {
    pub fn new(params: &SolverParams<B::Scalar>, g: B::Scalar) -> Self {
        Self {
            _params: params.clone(),
            gravity: g,
        }
    }

    /// 简单静水重构
    #[inline]
    pub fn reconstruct_face_simple(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        z_l: B::Scalar,
        z_r: B::Scalar,
        vel_l: B::Vector2D,
        vel_r: B::Vector2D,
    ) -> HydrostaticFaceState<B> {
        let z_face = if z_l > z_r { z_l } else { z_r };
        let eta_l = h_l + z_l;
        let eta_r = h_r + z_r;
        let zero = B::Scalar::ZERO;
        let h_left = if eta_l - z_face > zero { eta_l - z_face } else { zero };
        let h_right = if eta_r - z_face > zero { eta_r - z_face } else { zero };

        HydrostaticFaceState {
            h_left,
            h_right,
            vel_left: vel_l,
            vel_right: vel_r,
            z_face,
        }
    }

    /// 床坡源项修正
    #[inline]
    pub fn bed_slope_correction(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        h_l_star: B::Scalar,
        h_r_star: B::Scalar,
        normal: B::Vector2D,
        length: B::Scalar,
    ) -> BedSlopeCorrection<B> {
        let half = B::Scalar::HALF;
        let g = self.gravity;
        let pressure_diff_l = half * g * (h_l * h_l - h_l_star * h_l_star) * length;
        let pressure_diff_r = half * g * (h_r * h_r - h_r_star * h_r_star) * length;

        BedSlopeCorrection {
            source_left_x: -pressure_diff_l * normal.x(),
            source_left_y: -pressure_diff_l * normal.y(),
            source_right_x: pressure_diff_r * normal.x(),
            source_right_y: pressure_diff_r * normal.y(),
        }
    }
}

/// 浅水方程求解器（Backend泛型）
pub struct ShallowWaterSolver<B: Backend, S: SourceTermGeneric<B>>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    mesh: Arc<PhysicsMesh>,
    config: Layer3Config<B::Scalar>,
    params: NumericalParams<B::Scalar>,
    _gravity: B::Scalar,
    backend: B,
    workspace: SolverWorkspaceGeneric<B>,
    riemann: RiemannSolverAny<B>,
    riemann_fallback: RusanovSolver<B>,
    wetting_drying: WettingDryingHandler<B>,
    hydrostatic: HydrostaticReconstruction<B>,
    timestep_ctrl: TimeStepController<B>,
    stats: SolverStats<B>,
    recon_eta: ScalarReconstructor<B>,
    recon_u: ScalarReconstructor<B>,
    recon_v: ScalarReconstructor<B>,
    sources: Vec<S>,
    boundary_provider: Option<Arc<dyn BoundaryDataProvider>>,
}

impl<B: Backend, S: SourceTermGeneric<B>> ShallowWaterSolver<B, S>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    fn build_riemann_solver(
        config: &Layer3Config<B::Scalar>,
        params: &NumericalParams<B::Scalar>,
        solver_params: &SolverParams<B::Scalar>,
        gravity: B::Scalar,
    ) -> RiemannSolverAny<B> {
        match config.riemann_solver {
            RiemannSolverType::Hllc => RiemannSolverAny::Hllc(HllcSolver::<B>::new(solver_params, gravity)),
            RiemannSolverType::Roe => RiemannSolverAny::Roe(RoeSolver::<B>::new(solver_params, gravity)),
            RiemannSolverType::Rusanov => RiemannSolverAny::Rusanov(RusanovSolver::<B>::new(params, gravity)),
            RiemannSolverType::Central => RiemannSolverAny::Central(CentralSolver::<B>::new(params, gravity)),
        }
    }

    pub fn new(
        mesh: Arc<PhysicsMesh>, 
        config: Layer3Config<B::Scalar>, 
        backend: B
    ) -> Self {
        let n_cells = mesh.cell_count();
        let gravity = config.gravity;
        let params = config.params.clone();
        let timestep_ctrl = TimeStepController::<B>::new(gravity, &params);

        let workspace = SolverWorkspaceGeneric::new(&backend, n_cells);
        
        // 转换参数类型：Layer 4 NumericalParams → Layer 3 SolverParams
        let solver_params = crate::schemes::riemann::SolverParams::<B::Scalar>::from_numerical(&params, gravity);
        let riemann = Self::build_riemann_solver(&config, &params, &solver_params, gravity);
        let riemann_fallback = RusanovSolver::<B>::new(&params, gravity);
        let wetting_drying = WettingDryingHandler::<B>::from_params(&params)
            .expect("WettingDryingHandler 初始化失败");
        let hydrostatic = HydrostaticReconstruction::<B>::new(&solver_params, gravity);
        let recon_eta = ScalarReconstructor::new(mesh.clone(), config.scheme, backend.clone());
        let recon_u = ScalarReconstructor::new(mesh.clone(), config.scheme, backend.clone());
        let recon_v = ScalarReconstructor::new(mesh.clone(), config.scheme, backend.clone());

        Self {
            mesh,
            config,
            params,
            _gravity: gravity,
            backend,
            workspace,
            riemann,
            riemann_fallback,
            wetting_drying,
            hydrostatic,
            timestep_ctrl,
            stats: SolverStats::default(),
            recon_eta,
            recon_u,
            recon_v,
            sources: Vec::new(),
            boundary_provider: None,
        }
    }

    pub fn step(&mut self, state: &mut ShallowWaterState<B>, dt: B::Scalar) -> B::Scalar {
        self.step_with_sources(state, dt, 0.0)
    }

    pub fn step_with_sources(
        &mut self,
        state: &mut ShallowWaterState<B>,
        dt: B::Scalar,
        time: f64,
    ) -> B::Scalar {
        if self.config.stability.check_nan {
            let _ = self.detect_and_clean_nan(state);
        }

        self.workspace.reset();
        self.prepare_reconstruction(state);

        let max_wave_speed = if self.mesh.face_count() >= self.config.parallel_threshold as usize {
            self.compute_fluxes_parallel(state, time)
        } else {
            self.compute_fluxes_serial(state, time)
        };

        self.apply_sources(state, time, dt);
        self.update_state(state, dt);

        if self.config.stability.check_nan {
            let _ = self.detect_and_clean_nan(state);
        }

        let (dry_cells, limited_count) = self.enforce_positivity(state, dt);
        self.stats.max_wave_speed = max_wave_speed;
        self.stats.dry_cells = dry_cells;
        self.stats.limited_faces = limited_count;
        self.stats.dt = dt;

        dt
    }

    pub fn compute_dt(&mut self, state: &ShallowWaterState<B>) -> B::Scalar {
        self.timestep_ctrl.update(state, &self.mesh, &self.params)
    }

    #[inline]
    fn use_second_order(&self) -> bool {
        matches!(self.config.scheme, NumericalScheme::SecondOrderMuscl | NumericalScheme::SecondOrderWeno)
    }

    fn prepare_reconstruction(&mut self, state: &ShallowWaterState<B>) {
        let n = state.n_cells();
        if self.workspace.vel_u.len() != n {
            self.workspace.resize(n);
        }

        for cell in self.mesh.cell_indices() {
            let idx = cell.get();
            let (u, v) = self.params.safe_velocity_components(
                state.hu[idx], state.hv[idx], state.h[idx]
            );
            self.workspace.vel_u[idx] = u;
            self.workspace.vel_v[idx] = v;
            self.workspace.eta[idx] = state.h[idx] + state.z[idx];
        }

        let scheme = self.config.scheme;
        self.recon_eta
            .configure_for_scheme(scheme, self.mesh.clone(), self.backend.clone());
        self.recon_u
            .configure_for_scheme(scheme, self.mesh.clone(), self.backend.clone());
        self.recon_v
            .configure_for_scheme(scheme, self.mesh.clone(), self.backend.clone());

        if !self.use_second_order() {
            return;
        }

        self.recon_eta.compute_gradients(&self.workspace.eta);
        self.recon_u.compute_gradients(&self.workspace.vel_u);
        self.recon_v.compute_gradients(&self.workspace.vel_v);
    }

    #[inline]
    fn compute_boundary_pressure(
        &self,
        state: &ShallowWaterState<B>,
        face_idx: FaceIndex,
    ) -> (B::Scalar, B::Scalar) {
        let g = self.hydrostatic.gravity;
        let half = B::Scalar::HALF;
        let owner = self.mesh.face_owner(face_idx);
        let normal = self.mesh.face_normal_generic::<B>(face_idx).expect("边界面法向量转换失败");
        let length_f64 = self.mesh.face_length(face_idx);
        let length = self.backend.config_scalar(length_f64, "solver.boundary_pressure.length");
        let h = state.h[owner.get()].max(B::Scalar::ZERO);
        let pressure = half * g * h * h * length;
        let flux_hu = -pressure * normal.x();
        let flux_hv = -pressure * normal.y();
        (flux_hu, flux_hv)
    }

    fn apply_boundary_pressures(&mut self, state: &ShallowWaterState<B>) {
        for face in self.mesh.boundary_face_indices() {
            let (flux_hu, flux_hv) = self.compute_boundary_pressure(state, face);
            let owner = self.mesh.face_owner(face);
            self.workspace.source_hu[owner.get()] += flux_hu;
            self.workspace.source_hv[owner.get()] += flux_hv;
        }
    }

    #[inline]
    fn solve_riemann_with_fallback(
        &self,
        h_left: B::Scalar,
        h_right: B::Scalar,
        vel_left: B::Vector2D,
        vel_right: B::Vector2D,
        normal: B::Vector2D,
    ) -> RiemannFlux<B::Scalar> {
        self.riemann
            .solve(h_left, h_right, vel_left, vel_right, normal)
            .or_else(|_| {
                self.riemann_fallback
                    .solve(h_left, h_right, vel_left, vel_right, normal)
            })
            .unwrap_or_else(|_| RiemannFlux::zero())
    }

    fn apply_boundary_forcing(&mut self, state: &ShallowWaterState<B>, time: f64) {
        if let Some(provider) = &self.boundary_provider {
            for face in self.mesh.boundary_face_indices() {
                let face_idx = face.get();
                let forcing = provider
                    .get_forcing(face_idx, time)
                    .unwrap_or(ExternalForcing::ZERO);
                let (flux, length, owner) = self.compute_boundary_flux(state, face, &forcing);

                let fh = flux.mass * length;
                let fhu = flux.momentum_x * length;
                let fhv = flux.momentum_y * length;

                let owner_idx = owner.get();
                self.workspace.flux_h[owner_idx] -= fh;
                self.workspace.flux_hu[owner_idx] -= fhu;
                self.workspace.flux_hv[owner_idx] -= fhv;
            }
        } else {
            self.apply_boundary_pressures(state);
        }
    }

    fn compute_boundary_flux(
        &self,
        state: &ShallowWaterState<B>,
        face_idx: FaceIndex,
        forcing: &ExternalForcing,
    ) -> (RiemannFlux<B::Scalar>, B::Scalar, CellIndex) {
        let normal = self.mesh.face_normal_generic::<B>(face_idx)
            .expect("边界面法向量转换失败");
        let length_f64 = self.mesh.face_length(face_idx);
        let length = self.backend.config_scalar(length_f64, "solver.boundary_riemann.length");
        let owner = self.mesh.face_owner(face_idx);

        if !forcing.eta.is_finite() || !forcing.u().is_finite() || !forcing.v().is_finite() {
            return (RiemannFlux::zero(), length, owner);
        }

        let h_left = state.h[owner.get()];
        let (u_left, v_left) = self.params.safe_velocity_components(
            state.hu[owner.get()],
            state.hv[owner.get()],
            h_left,
        );
        let vel_left = B::vec2_new(u_left, v_left);

        let z = state.z[owner.get()];
        let z_f64 = z.to_f64_lossy();
        let h_right_f64 = (forcing.eta - z_f64).max(0.0);
        let h_right = self.backend.config_scalar(h_right_f64, "solver.boundary_riemann.h_right");
        let vel_right = B::vec2_new(
            self.backend.config_scalar(forcing.u(), "solver.boundary_riemann.u"),
            self.backend.config_scalar(forcing.v(), "solver.boundary_riemann.v"),
        );

        let flux = self.solve_riemann_with_fallback(h_left, h_right, vel_left, vel_right, normal);

        (flux, length, owner)
    }

    fn compute_fluxes_serial(&mut self, state: &ShallowWaterState<B>, time: f64) -> B::Scalar {
        let mut max_wave_speed = B::Scalar::ZERO;

        for face in self.mesh.interior_face_indices() {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face(state, face);

            if flux.max_wave_speed > max_wave_speed {
                max_wave_speed = flux.max_wave_speed;
            }

            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            let owner_idx = owner.get();
            self.workspace.flux_h[owner_idx] -= fh;
            self.workspace.flux_hu[owner_idx] -= fhu;
            self.workspace.flux_hv[owner_idx] -= fhv;
            self.workspace.source_hu[owner_idx] += bed_src.source_left_x;
            self.workspace.source_hv[owner_idx] += bed_src.source_left_y;

            if let Some(neigh) = neighbor {
                let neigh_idx = neigh.get();
                self.workspace.flux_h[neigh_idx] += fh;
                self.workspace.flux_hu[neigh_idx] += fhu;
                self.workspace.flux_hv[neigh_idx] += fhv;
                self.workspace.source_hu[neigh_idx] += bed_src.source_right_x;
                self.workspace.source_hv[neigh_idx] += bed_src.source_right_y;
            }
        }

        self.apply_boundary_forcing(state, time);
        max_wave_speed
    }

    fn compute_fluxes_parallel(&mut self, state: &ShallowWaterState<B>, time: f64) -> B::Scalar {
        let n_cells = self.mesh.cell_count();
        // Per-thread buffers avoid the old collect-then-serial-accumulate path.
        let accumulation = self
            .mesh
            .interior_face_indices()
            .collect::<Vec<_>>()
            .into_par_iter()
            .fold(
                || ParallelFluxAccumulation::<B>::new(n_cells),
                |mut local, face_idx| {
                    let (flux, bed_src, length, owner, neighbor) = self.compute_face(state, face_idx);
                    local.accumulate_face(
                        owner.get(),
                        neighbor.map(|cell| cell.get()),
                        flux.mass * length,
                        flux.momentum_x * length,
                        flux.momentum_y * length,
                        bed_src.source_left_x,
                        bed_src.source_left_y,
                        bed_src.source_right_x,
                        bed_src.source_right_y,
                        flux.max_wave_speed,
                    );
                    local
                },
            )
            .reduce(
                || ParallelFluxAccumulation::<B>::new(n_cells),
                |left, right| left.merge(right),
            );

        accumulation.write_into(&mut self.workspace);
        self.apply_boundary_forcing(state, time);
        accumulation.max_wave_speed
    }

    fn compute_face(
        &self,
        state: &ShallowWaterState<B>,
        face_idx: FaceIndex,
    ) -> (RiemannFlux<B::Scalar>, BedSlopeCorrection<B>, B::Scalar, CellIndex, Option<CellIndex>) {
        let normal = self.mesh.face_normal_generic::<B>(face_idx).expect("边界面法向量转换失败");
        let length_f64 = self.mesh.face_length(face_idx);
        let length = self.backend.config_scalar(length_f64, "solver.interior_riemann.length");
        let owner = self.mesh.face_owner(face_idx);
        let neighbor = self.mesh.face_neighbor(face_idx);

        let (h_l, vel_l, z_l, h_r, vel_r, z_r) = if self.use_second_order() {
            let eta_rec = self.recon_eta.reconstruct_scalar(face_idx.get(), &self.workspace.eta);
            let u_rec = self.recon_u.reconstruct_scalar(face_idx.get(), &self.workspace.vel_u);
            let v_rec = self.recon_v.reconstruct_scalar(face_idx.get(), &self.workspace.vel_v);

            if let Some(neigh) = neighbor {
                let z_owner = state.z[owner.get()];
                let z_neigh = state.z[neigh.get()];
                let z_face = if z_owner > z_neigh { z_owner } else { z_neigh };
                let h_left = (eta_rec.left - z_face).max(B::Scalar::ZERO);
                let h_right = (eta_rec.right - z_face).max(B::Scalar::ZERO);
                let u_l = u_rec.left;
                let v_l = v_rec.left;
                let u_r = u_rec.right;
                let v_r = v_rec.right;
                (
                    h_left,
                    B::vec2_new(u_l, v_l),
                    z_owner,
                    h_right,
                    B::vec2_new(u_r, v_r),
                    z_neigh,
                )
            } else {
                let z_owner = state.z[owner.get()];
                let h_left = (eta_rec.left - z_owner).max(B::Scalar::ZERO);
                let u_l = u_rec.left;
                let v_l = v_rec.left;
                let vel_left = B::vec2_new(u_l, v_l);
                let vn = B::vec2_dot(&vel_left, &normal);
                let two = B::Scalar::TWO;
                let vel_right = B::vec2_sub(&vel_left, &B::vec2_scale(&normal, vn * two));
                (h_left, vel_left, z_owner, h_left, vel_right, z_owner)
            }
        } else {
            let h_l = state.h[owner.get()];
            let z_l = state.z[owner.get()];
            let (u_l, v_l) = self.params.safe_velocity_components(
                state.hu[owner.get()], state.hv[owner.get()], h_l,
            );
            let vel_l = B::vec2_new(u_l, v_l);

            if let Some(neigh) = neighbor {
                let h_r = state.h[neigh.get()];
                let (u_r, v_r) = self.params.safe_velocity_components(
                    state.hu[neigh.get()], state.hv[neigh.get()], h_r,
                );
                (
                    h_l,
                    vel_l,
                    z_l,
                    h_r,
                    B::vec2_new(u_r, v_r),
                    state.z[neigh.get()],
                )
            } else {
                let vn = B::vec2_dot(&vel_l, &normal);
                let two = B::Scalar::TWO;
                let vel_r = B::vec2_sub(&vel_l, &B::vec2_scale(&normal, vn * two));
                (h_l, vel_l, z_l, h_l, vel_r, z_l)
            }
        };

        // 修复：reconstruct_face_simple返回结构体，不是元组
        let recon_state = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            let half = B::Scalar::HALF;
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: (z_l + z_r) * half,
            }
        };

        // 从结构体中提取字段
        let h_left = recon_state.h_left;
        let h_right = recon_state.h_right;
        let vel_left = recon_state.vel_left;
        let vel_right = recon_state.vel_right;
        let _z_face = recon_state.z_face;

        let wet_l = self.wetting_drying.get_state(h_left);
        let wet_r = self.wetting_drying.get_state(h_right);
        let flux_limiter: B::Scalar = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => B::Scalar::ZERO,
            (WetState::Dry, _) | (_, WetState::Dry) => B::Scalar::ONE,
            (WetState::PartiallyWet, WetState::PartiallyWet) => {
                let h_min = h_left.min(h_right);
                let fraction = (h_min - self.params.h_dry) / (self.params.h_wet - self.params.h_dry);
                let one = B::Scalar::ONE;
                let zero = B::Scalar::ZERO;
                if fraction > one { one } else if fraction < zero { zero } else { fraction }
            }
            _ => B::Scalar::ONE,
        };

        let flux = self.solve_riemann_with_fallback(h_left, h_right, vel_left, vel_right, normal);

        let limited_flux = flux.scaled(flux_limiter);
        let bed_src = self.hydrostatic.bed_slope_correction(h_l, h_r, recon_state.h_left, recon_state.h_right, normal, length);

        (limited_flux, bed_src, length, owner, neighbor)
    }

    fn update_state(&self, state: &mut ShallowWaterState<B>, dt: B::Scalar) {
        for cell in self.mesh.cell_indices() {
            let idx = cell.get();
            let Some(area_f64) = self.mesh.cell_area(cell) else {
                state.h[idx] = B::Scalar::ZERO;
                state.hu[idx] = B::Scalar::ZERO;
                state.hv[idx] = B::Scalar::ZERO;
                continue;
            };
            if !area_f64.is_finite() || area_f64 <= 0.0 {
                state.h[idx] = B::Scalar::ZERO;
                state.hu[idx] = B::Scalar::ZERO;
                state.hv[idx] = B::Scalar::ZERO;
                continue;
            }
            let inv_area = self.backend.config_scalar(1.0 / area_f64, "solver.update_state.inv_area");
            state.h[idx] = state.h[idx] + dt * inv_area * (self.workspace.flux_h[idx] + self.workspace.source_h[idx]);
            state.hu[idx] = state.hu[idx] + dt * inv_area * 
                (self.workspace.flux_hu[idx] + self.workspace.source_hu[idx]);
            state.hv[idx] = state.hv[idx] + dt * inv_area * 
                (self.workspace.flux_hv[idx] + self.workspace.source_hv[idx]);
        }
    }

    fn apply_sources(&mut self, state: &ShallowWaterState<B>, time: f64, dt: B::Scalar) {
        if self.sources.is_empty() {
            return;
        }

        let ctx = SourceContextGeneric::new(
            time,
            dt,
            self.config.gravity,
            self.params.h_dry,
            self.params.h_wet,
        );

        for source in &self.sources {
            if !source.is_enabled() {
                continue;
            }
            source.accumulate(
                state,
                &mut self.workspace.source_h,
                &mut self.workspace.source_hu,
                &mut self.workspace.source_hv,
                &ctx,
            );
        }
    }

    pub fn register_source(&mut self, source: S) {
        self.sources.push(source);
    }

    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    pub fn set_boundary_provider(&mut self, provider: Arc<dyn BoundaryDataProvider>) {
        self.boundary_provider = Some(provider);
    }

    pub fn clear_boundary_provider(&mut self) {
        self.boundary_provider = None;
    }

    pub fn has_boundary_provider(&self) -> bool {
        self.boundary_provider.is_some()
    }

    fn enforce_positivity(&mut self, state: &mut ShallowWaterState<B>, _dt: B::Scalar) -> (usize, usize) {
        let h_min = self.params.h_min;
        let h_dry = self.params.h_dry;
        let mut dry_count = 0;
        let mut limited_count = 0;

        for cell in self.mesh.cell_indices() {
            let idx = cell.get();
            if state.h[idx] < h_min {
                state.h[idx] = B::Scalar::ZERO;
                state.hu[idx] = B::Scalar::ZERO;
                state.hv[idx] = B::Scalar::ZERO;
                dry_count += 1;
            } else if state.h[idx] < h_dry {
                let factor = self.wetting_drying.wet_fraction_smooth(state.h[idx]);
                state.hu[idx] = state.hu[idx] * factor;
                state.hv[idx] = state.hv[idx] * factor;
                dry_count += 1;
                limited_count += 1;
            }
        }

        (dry_count, limited_count)
    }

    pub fn detect_and_clean_nan(&mut self, state: &mut ShallowWaterState<B>) -> NanDetectionResult {
        let mut result = NanDetectionResult::default();
        
        for cell in self.mesh.cell_indices() {
            let idx = cell.get();
            let mut has_nan = false;
            
            if !state.h[idx].is_finite() {
                state.h[idx] = B::Scalar::ZERO;
                has_nan = true;
            }
            
            if !state.hu[idx].is_finite() {
                state.hu[idx] = B::Scalar::ZERO;
                has_nan = true;
            }
            
            if !state.hv[idx].is_finite() {
                state.hv[idx] = B::Scalar::ZERO;
                has_nan = true;
            }

            if !state.z[idx].is_finite() {
                state.z[idx] = B::Scalar::ZERO;
                has_nan = true;
            }
            
            if has_nan {
                result.found_nan = true;
                result.affected_cells.push(idx);
                self.stats.nan_count += 1;
                self.stats.last_nan_location = Some(idx);
            }
        }
        
        result
    }

    pub fn mesh(&self) -> &PhysicsMesh {
        &self.mesh
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn stats(&self) -> &SolverStats<B> {
        &self.stats
    }

    pub fn max_wave_speed(&self) -> f64 {
        self.stats.max_wave_speed.to_f64_lossy()
    }

    pub fn dry_cell_count(&self) -> usize {
        self.stats.dry_cells
    }
    
    pub fn config(&self) -> &Layer3Config<B::Scalar> {
        &self.config
    }

    pub fn params(&self) -> &NumericalParams<B::Scalar> {
        &self.params
    }
}
