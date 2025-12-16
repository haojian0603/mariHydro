// crates/mh_physics/src/engine/solver.rs

//! 浅水方程求解器（Backend泛型化版本）
//!
//! 基于有限体积法的非结构化网格求解器，支持 f32/f64 精度切换和 GPU 后端扩展。
//!
//! # 架构改造
//!
//! **Phase 3 核心改造**：所有组件完全泛型化，支持 `CpuBackend<f32>` 和 `CpuBackend<f64>`。
//!
//! ## 关键改进
//!
//! 1. **`ShallowWaterSolver<B>`**：添加 `Backend` 泛型参数，统一状态、黎曼求解器、干湿处理器
//! 2. **`SolverWorkspaceGeneric<B>`**：工作区字段使用 `B::Buffer<B::Scalar>` 存储
//! 3. **索引类型安全**：所有几何查询强制使用 `FaceIndex/CellIndex`，杜绝 `usize` 泄露
//! 4. **Backend几何抽象**：完全移除 `glam::DVec2`，使用 `B::Vector2D` 和工厂方法
//! 5. **桥接层就绪**：实现 `DynSolver` trait，支持运行时多态分发
//!
//! # 使用示例
//!
//! ```rust
//! use mh_physics::engine::solver::{ShallowWaterSolver, SolverConfig};
//! use mh_runtime::{CpuBackend, Precision};
//!
//! // f64高精度模式
//! let backend_f64 = CpuBackend::<f64>::new();
//! let solver_f64 = ShallowWaterSolver::<CpuBackend<f64>>::new(mesh, config, backend_f64);
//!
//! // f32高性能模式
//! let backend_f32 = CpuBackend::<f32>::new();
//! let solver_f32 = ShallowWaterSolver::<CpuBackend<f32>>::new(mesh, config, backend_f32);
//! ```

use crate::adapter::{CellIndex, FaceIndex, PhysicsMesh};
use crate::engine::timestep::{TimeStepController, TimeStepControllerBuilder};
use crate::schemes::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::{ConservedState, ShallowWaterStateGeneric as ShallowWaterState};
use crate::types::{NumericalParams, NumericalParamsF64};

use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use num_traits::{Float, FromPrimitive, Zero};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================
// 求解器配置（Layer 4配置层）
// ============================================================

/// 数值格式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericalScheme {
    /// 一阶精度
    FirstOrder,
    /// 二阶 MUSCL
    #[default]
    SecondOrderMuscl,
    /// 二阶 WENO
    SecondOrderWeno,
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

/// 回退策略
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackStrategy {
    /// 不回退，失败时报错
    NoFallback,
    /// 回退到一阶格式
    #[default]
    FallbackToFirstOrder,
    /// 回退到较小时间步
    ReduceTimestep,
    /// 综合策略：先减小时间步，再降低格式精度
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

impl StabilityOptions {
    pub fn strict() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: true,
            velocity_limit: 50.0,
            depth_limit: 500.0,
        }
    }

    pub fn relaxed() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: false,
            velocity_limit: 200.0,
            depth_limit: 2000.0,
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

/// 求解器配置（Layer 4，保持f64）
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// 数值参数（f64配置）
    pub params: NumericalParamsF64,
    /// 重力加速度 [m/s²]
    pub gravity: f64,
    /// 是否启用静水重构
    pub use_hydrostatic_reconstruction: bool,
    /// 并行化阈值（面数）
    pub parallel_threshold: usize,
    /// 是否启用隐式摩擦（预留）
    pub implicit_friction: bool,
    /// 数值格式
    pub scheme: NumericalScheme,
    /// 回退策略
    pub fallback: FallbackStrategy,
    /// 稳定性检查选项
    pub stability: StabilityOptions,
    /// 最大回退次数
    pub max_fallback_attempts: u32,
    /// 时间步减小因子（回退时使用）
    pub timestep_reduction_factor: f64,
    /// 时间积分器类型
    pub integrator: TimeIntegrator,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            params: NumericalParamsF64::default(),
            gravity: 9.81,
            use_hydrostatic_reconstruction: true,
            parallel_threshold: 1000,
            implicit_friction: true,
            scheme: NumericalScheme::default(),
            fallback: FallbackStrategy::default(),
            stability: StabilityOptions::default(),
            max_fallback_attempts: 3,
            timestep_reduction_factor: 0.5,
            integrator: TimeIntegrator::default(),
        }
    }
}

impl SolverConfig {
    /// 创建构建器
    pub fn builder() -> SolverConfigBuilder {
        SolverConfigBuilder::default()
    }

    /// 快速配置：性能优先
    pub fn performance() -> Self {
        Self {
            scheme: NumericalScheme::FirstOrder,
            stability: StabilityOptions::relaxed(),
            parallel_threshold: 500,
            ..Default::default()
        }
    }

    /// 快速配置：精度优先
    pub fn accuracy() -> Self {
        Self {
            scheme: NumericalScheme::SecondOrderMuscl,
            stability: StabilityOptions::strict(),
            fallback: FallbackStrategy::Progressive,
            ..Default::default()
        }
    }

    /// 快速配置：稳健模式
    pub fn robust() -> Self {
        Self {
            scheme: NumericalScheme::FirstOrder,
            fallback: FallbackStrategy::Progressive,
            stability: StabilityOptions::strict(),
            max_fallback_attempts: 5,
            timestep_reduction_factor: 0.25,
            ..Default::default()
        }
    }
}

/// 配置构建器
#[derive(Default)]
pub struct SolverConfigBuilder {
    config: SolverConfig,
}

impl SolverConfigBuilder {
    pub fn params(mut self, params: NumericalParamsF64) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.gravity = g;
        self
    }

    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_threshold = threshold;
        self
    }

    pub fn implicit_friction(mut self, enable: bool) -> Self {
        self.config.implicit_friction = enable;
        self
    }

    /// 设置数值格式
    pub fn scheme(mut self, scheme: NumericalScheme) -> Self {
        self.config.scheme = scheme;
        self
    }

    /// 设置回退策略
    pub fn fallback(mut self, fallback: FallbackStrategy) -> Self {
        self.config.fallback = fallback;
        self
    }

    /// 设置稳定性选项
    pub fn stability(mut self, stability: StabilityOptions) -> Self {
        self.config.stability = stability;
        self
    }

    /// 设置最大回退次数
    pub fn max_fallback_attempts(mut self, attempts: u32) -> Self {
        self.config.max_fallback_attempts = attempts;
        self
    }

    /// 设置时间步减小因子
    pub fn timestep_reduction_factor(mut self, factor: f64) -> Self {
        self.config.timestep_reduction_factor = factor.clamp(0.1, 0.9);
        self
    }

    pub fn build(self) -> SolverConfig {
        self.config
    }
}

// ============================================================
// 求解器统计
// ============================================================

/// 求解器步进统计
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    /// 最大波速 [m/s]
    pub max_wave_speed: f64,
    /// 干单元数量
    pub dry_cells: usize,
    /// 被限制的面数量
    pub limited_faces: usize,
    /// 当前时间步长 [s]
    pub dt: f64,
    /// 回退次数
    pub fallback_count: u32,
    /// 当前使用的格式
    pub current_scheme: NumericalScheme,
    /// 稳定性状态
    pub stability_status: StabilityStatus,
}

/// 稳定性状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StabilityStatus {
    /// 稳定
    #[default]
    Stable,
    /// 接近不稳定
    Marginal,
    /// 需要回退
    NeedsFallback,
    /// 不稳定（计算失败）
    Unstable,
}

impl std::fmt::Display for StabilityStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "稳定"),
            Self::Marginal => write!(f, "临界"),
            Self::NeedsFallback => write!(f, "需回退"),
            Self::Unstable => write!(f, "不稳定"),
        }
    }
}

impl SolverStats {
    /// 检查是否需要回退
    pub fn needs_fallback(&self) -> bool {
        matches!(self.stability_status, StabilityStatus::NeedsFallback | StabilityStatus::Unstable)
    }

    /// 生成诊断摘要
    pub fn summary(&self) -> String {
        format!(
            "dt={:.4}s, wave_speed={:.2}m/s, dry={}, limited={}, status={}, fallbacks={}",
            self.dt,
            self.max_wave_speed,
            self.dry_cells,
            self.limited_faces,
            self.stability_status,
            self.fallback_count
        )
    }
}

// ============================================================
// 求解器工作区（Backend泛型化）
// ============================================================

/// 求解器工作区（泛型版本）
/// 
/// 存储中间计算结果，避免重复分配。所有字段使用 Backend 缓冲区。
#[derive(Debug)]
pub struct SolverWorkspaceGeneric<B: Backend> {
    /// 通量累加（质量）
    pub flux_h: B::Buffer<B::Scalar>,
    /// 通量累加（x动量）
    pub flux_hu: B::Buffer<B::Scalar>,
    /// 通量累加（y动量）
    pub flux_hv: B::Buffer<B::Scalar>,
    /// 源项累加（x动量）
    pub source_hu: B::Buffer<B::Scalar>,
    /// 源项累加（y动量）
    pub source_hv: B::Buffer<B::Scalar>,
    /// 单元速度 u 分量（用于重构）
    pub vel_u: B::Buffer<B::Scalar>,
    /// 单元速度 v 分量（用于重构）
    pub vel_v: B::Buffer<B::Scalar>,
    /// 水位 η = h + z（用于 well-balanced 重构）
    pub eta: B::Buffer<B::Scalar>,
}

impl<B: Backend> SolverWorkspaceGeneric<B> {
    /// 创建工作区
    pub fn new(backend: &B, n_cells: usize) -> Self {
        Self {
            flux_h: backend.alloc(n_cells),
            flux_hu: backend.alloc(n_cells),
            flux_hv: backend.alloc(n_cells),
            source_hu: backend.alloc(n_cells),
            source_hv: backend.alloc(n_cells),
            vel_u: backend.alloc(n_cells),
            vel_v: backend.alloc(n_cells),
            eta: backend.alloc(n_cells),
        }
    }

    /// 重置通量
    pub fn reset_fluxes(&mut self, backend: &B) {
        backend.fill(&mut self.flux_h, B::Scalar::zero());
        backend.fill(&mut self.flux_hu, B::Scalar::zero());
        backend.fill(&mut self.flux_hv, B::Scalar::zero());
    }

    /// 重置源项
    pub fn reset_sources(&mut self, backend: &B) {
        backend.fill(&mut self.source_hu, B::Scalar::zero());
        backend.fill(&mut self.source_hv, B::Scalar::zero());
    }

    /// 重置所有
    pub fn reset(&mut self, backend: &B) {
        self.reset_fluxes(backend);
        self.reset_sources(backend);
    }

    /// 调整大小
    pub fn resize(&mut self, backend: &B, n_cells: usize) {
        backend.resize(&mut self.flux_h, n_cells, B::Scalar::zero());
        backend.resize(&mut self.flux_hu, n_cells, B::Scalar::zero());
        backend.resize(&mut self.flux_hv, n_cells, B::Scalar::zero());
        backend.resize(&mut self.source_hu, n_cells, B::Scalar::zero());
        backend.resize(&mut self.source_hv, n_cells, B::Scalar::zero());
        backend.resize(&mut self.vel_u, n_cells, B::Scalar::zero());
        backend.resize(&mut self.vel_v, n_cells, B::Scalar::zero());
        backend.resize(&mut self.eta, n_cells, B::Scalar::zero());
    }
}

/// 向后兼容类型别名
pub type SolverWorkspace = SolverWorkspaceGeneric<CpuBackend<f64>>;

// ============================================================
// 静水重构（Backend几何化）
// ============================================================

/// 面上的静水重构状态（Backend泛型）
#[derive(Debug, Clone, Copy)]
pub struct HydrostaticFaceState<B: Backend> {
    /// 左侧有效水深
    pub h_left: B::Scalar,
    /// 右侧有效水深
    pub h_right: B::Scalar,
    /// 左侧速度
    pub vel_left: B::Vector2D,
    /// 右侧速度
    pub vel_right: B::Vector2D,
    /// 面处高程
    pub z_face: B::Scalar,
}

/// 床坡源项修正（Backend泛型）
#[derive(Debug, Clone, Copy)]
pub struct BedSlopeCorrection<B: Backend> {
    /// 左侧（owner）单元 x 方向源项
    pub source_left_x: B::Scalar,
    /// 左侧（owner）单元 y 方向源项
    pub source_left_y: B::Scalar,
    /// 右侧（neighbor）单元 x 方向源项
    pub source_right_x: B::Scalar,
    /// 右侧（neighbor）单元 y 方向源项
    pub source_right_y: B::Scalar,
}

impl<B: Backend> BedSlopeCorrection<B> {
    /// 零源项常量
    pub fn zero() -> Self {
        Self {
            source_left_x: B::Scalar::zero(),
            source_left_y: B::Scalar::zero(),
            source_right_x: B::Scalar::zero(),
            source_right_y: B::Scalar::zero(),
        }
    }
}

/// 静水重构处理器（Backend泛型化）
#[derive(Debug, Clone)]
pub struct HydrostaticReconstruction<B: Backend> {
    /// 数值参数（泛型）
    params: NumericalParams<B::Scalar>,
    /// 重力加速度（泛型）
    g: B::Scalar,
}

impl<B: Backend> HydrostaticReconstruction<B> {
    /// 创建静水重构处理器
    pub fn new(params: &NumericalParams<B::Scalar>, g: B::Scalar) -> Self {
        Self {
            params: params.clone(),
            g,
        }
    }

    /// 简单静水重构
    ///
    /// 对面两侧的水深进行修正，确保静水平衡
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
        // 面处高程取最大值（保守处理）
        let z_face = z_l.max(z_r);

        // 修正后的水深 = max(0, η - z_face)，其中 η = h + z 是水位
        let eta_l = h_l + z_l;
        let eta_r = h_r + z_r;
        let h_left = (eta_l - z_face).max(B::Scalar::zero());
        let h_right = (eta_r - z_face).max(B::Scalar::zero());

        HydrostaticFaceState {
            h_left,
            h_right,
            vel_left: vel_l,
            vel_right: vel_r,
            z_face,
        }
    }
}

// ============================================================
// 主求解器（Backend泛型化 - 核心改造）
// ============================================================

/// 浅水方程求解器（Backend泛型化）
///
/// 基于有限体积法的非结构化网格求解器，完全泛型化支持 f32/f64 精度切换。
pub struct ShallowWaterSolver<B: Backend> {
    /// 网格
    mesh: Arc<PhysicsMesh>,
    /// 配置（Layer 4）
    config: SolverConfig,
    /// Backend实例（计算策略层）
    backend: B,
    /// 工作区（Backend泛型）
    workspace: SolverWorkspaceGeneric<B>,
    /// 黎曼求解器（Backend泛型）
    riemann: HllcSolver<B>,
    /// 干湿处理器（Backend泛型）
    wetting_drying: WettingDryingHandler<B>,
    /// 静水重构（Backend泛型）
    hydrostatic: HydrostaticReconstruction<B>,
    /// 时间步控制器
    timestep_ctrl: TimeStepController,
    /// 统计信息
    stats: SolverStats,
    /// 水位重构器（用于 well-balanced 方法）
    muscl_eta: MusclReconstructor<B>,
    /// u 速度重构器
    muscl_u: MusclReconstructor<B>,
    /// v 速度重构器
    muscl_v: MusclReconstructor<B>,
}

impl<B: Backend> ShallowWaterSolver<B> {
    /// 创建求解器
    ///
    /// # 参数
    /// - `mesh`: 物理网格（数据层）
    /// - `config`: 求解器配置（Layer 4，f64）
    /// - `backend`: 计算后端（策略层，决定精度）
    pub fn new(mesh: Arc<PhysicsMesh>, config: SolverConfig, backend: B) -> Self {
        let n_cells = mesh.n_cells();

        // 创建时间步控制器
        let timestep_ctrl = TimeStepControllerBuilder::new(config.gravity)
            .with_cfl(config.params.cfl)
            .with_dt_limits(config.params.dt_min, config.params.dt_max)
            .build();

        // 根据配置选择重构器模式
        let muscl_config = if matches!(config.scheme, NumericalScheme::FirstOrder) {
            MusclConfig::first_order()
        } else {
            MusclConfig::default()
        };

        // 创建工作区（Backend分配）
        let workspace = SolverWorkspaceGeneric::new(&backend, n_cells);

        // 转换f64参数为Backend泛型
        let gravity_b = B::Scalar::from_f64(config.gravity)
            .unwrap_or_else(|| B::Scalar::from_f64(9.81).unwrap());
        let params_b = NumericalParams::<B::Scalar>::from_config(&config.params)
            .unwrap_or_else(|_| NumericalParams::<B::Scalar>::default());

        // 创建Backend泛型组件
        let riemann = HllcSolver::<B>::new(&params_b, gravity_b);
        let wetting_drying = WettingDryingHandler::<B>::from_params(&params_b);
        let hydrostatic = HydrostaticReconstruction::<B>::new(&params_b, gravity_b);
        let muscl_eta = MusclReconstructor::<B>::new(muscl_config.clone(), mesh.clone());
        let muscl_u = MusclReconstructor::<B>::new(muscl_config.clone(), mesh.clone());
        let muscl_v = MusclReconstructor::<B>::new(muscl_config.clone(), mesh.clone());

        Self {
            mesh,
            config,
            backend,
            workspace,
            riemann,
            wetting_drying,
            hydrostatic,
            timestep_ctrl,
            stats: SolverStats::default(),
            muscl_eta,
            muscl_u,
            muscl_v,
        }
    }

    /// 执行一个时间步
    ///
    /// 返回使用的时间步长
    pub fn step(&mut self, state: &mut ShallowWaterState<B>, dt: B::Scalar) -> B::Scalar {
        // 1. 重置工作区
        self.workspace.reset(&self.backend);

        // 2. 预计算速度并准备二阶重构
        self.prepare_reconstruction(state);

        // 3. 计算通量
        let max_wave_speed = if self.mesh.n_faces() >= self.config.parallel_threshold {
            self.compute_fluxes_parallel(state)
        } else {
            self.compute_fluxes_serial(state)
        };

        // 4. 更新状态
        self.update_state(state, dt);

        // 5. 强制正性
        let (dry_cells, _) = self.enforce_positivity(state, dt);

        // 6. 更新统计
        self.stats.max_wave_speed = max_wave_speed.to_f64().unwrap_or(0.0);
        self.stats.dry_cells = dry_cells;
        self.stats.dt = dt.to_f64().unwrap_or(0.0);

        dt
    }

    /// 计算自适应时间步长
    pub fn compute_dt(&mut self, state: &ShallowWaterState<B>) -> B::Scalar {
        self.timestep_ctrl.update(state, &self.mesh, &self.config.params)
    }

    /// 是否使用二阶格式
    #[inline]
    fn use_second_order(&self) -> bool {
        matches!(self.config.scheme, NumericalScheme::SecondOrderMuscl | NumericalScheme::SecondOrderWeno)
    }

    /// 根据配置同步重构器开关并计算梯度
    fn prepare_reconstruction(&mut self, state: &ShallowWaterState<B>) {
        let n = state.n_cells();
        if self.workspace.vel_u.len() != n {
            self.workspace.resize(&self.backend, n);
        }

        // 预计算安全速度和水位
        for cell_idx in self.mesh.cells() {
            let i = cell_idx.get();
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[i], state.hv[i], state.h[i]
            );
            self.workspace.vel_u[i] = u;
            self.workspace.vel_v[i] = v;
            // 计算水位 η = h + z（用于 well-balanced 重构）
            self.workspace.eta[i] = state.h[i] + state.z[i];
        }

        if !self.use_second_order() {
            let cfg = MusclConfig::first_order();
            self.muscl_eta.set_config(cfg.clone());
            self.muscl_u.set_config(cfg.clone());
            self.muscl_v.set_config(cfg);
            return;
        }

        // 二阶模式：使用默认配置
        let cfg = MusclConfig::default();
        self.muscl_eta.set_config(cfg.clone());
        self.muscl_u.set_config(cfg.clone());
        self.muscl_v.set_config(cfg);

        // 对水位 η 而非水深 h 计算梯度，保证 C-property
        self.muscl_eta.compute_gradients(&self.workspace.eta);
        self.muscl_u.compute_gradients(&self.workspace.vel_u);
        self.muscl_v.compute_gradients(&self.workspace.vel_v);
    }

    // =========================================================================
    // 通量计算（串行）
    // =========================================================================

    fn compute_fluxes_serial(&mut self, state: &ShallowWaterState<B>) -> B::Scalar {
        let mut max_wave_speed = B::Scalar::zero();

        for face_idx in self.mesh.interior_faces() {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face_flux(state, face_idx);

            max_wave_speed = max_wave_speed.max(flux.max_wave_speed);

            // 累加到 owner
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            let owner_idx = owner.get();
            self.workspace.flux_h[owner_idx] -= fh;
            self.workspace.flux_hu[owner_idx] -= fhu;
            self.workspace.flux_hv[owner_idx] -= fhv;
            self.workspace.source_hu[owner_idx] += bed_src.source_left_x;
            self.workspace.source_hv[owner_idx] += bed_src.source_left_y;

            // 累加到 neighbor（如果存在）
            if let Some(neigh) = neighbor {
                let neigh_idx = neigh.get();
                self.workspace.flux_h[neigh_idx] += fh;
                self.workspace.flux_hu[neigh_idx] += fhu;
                self.workspace.flux_hv[neigh_idx] += fhv;
                self.workspace.source_hu[neigh_idx] += bed_src.source_right_x;
                self.workspace.source_hv[neigh_idx] += bed_src.source_right_y;
            }
        }

        max_wave_speed
    }

    // =========================================================================
    // 通量计算（并行）
    // =========================================================================

    /// 使用"收集后累加"策略计算通量
    ///
    /// # 技术债务 (TD-5.3.2, TD-5.3.3)
    ///
    /// 当前实现是伪并行：累加阶段串行，未来需实现着色并行
    fn compute_fluxes_parallel(&mut self, state: &ShallowWaterState<B>) -> B::Scalar {
        let max_speed_atomic = AtomicU64::new(0u64);

        // 阶段1: 并行计算所有面的通量
        let face_results: Vec<_> = self.mesh.interior_faces()
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face_flux(state, face_idx);

                // 更新最大波速（原子操作）
                let bits = if std::mem::size_of::<B::Scalar>() == 4 {
                    // f32: 转换为f64再存储
                    (flux.max_wave_speed.to_f64().unwrap_or(0.0) as f32).to_bits() as u64
                } else {
                    // f64: 直接存储
                    flux.max_wave_speed.to_bits()
                };
                max_speed_atomic.fetch_max(bits, Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 阶段2: 串行累加到单元（TODO: 着色并行）
        for (flux, bed_src, length, owner, neighbor) in face_results {
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

        // 从原子值恢复标量
        let bits = max_speed_atomic.load(Ordering::Relaxed);
        if std::mem::size_of::<B::Scalar>() == 4 {
            B::Scalar::from_f64(f32::from_bits(bits as u32) as f64).unwrap_or(B::Scalar::zero())
        } else {
            B::Scalar::from_bits(bits)
        }
    }

    // =========================================================================
    // 单面通量计算（核心方法，Backend几何化）
    // =========================================================================

    fn compute_face_flux(
        &self,
        state: &ShallowWaterState<B>,
        face_idx: FaceIndex,
    ) -> (RiemannFlux<B::Scalar>, BedSlopeCorrection<B>, B::Scalar, CellIndex, Option<CellIndex>) {
        // 使用Backend几何接口
        let normal = self.mesh.face_normal_generic::<B>(face_idx);
        let length = B::Scalar::from_f64(self.mesh.face_length(face_idx)).unwrap_or(B::Scalar::one());
        let owner = self.mesh.face_owner(face_idx);
        let neighbor = self.mesh.face_neighbor(face_idx);

        // 重构后的左/右状态
        let (h_l, vel_l, z_l, h_r, vel_r, z_r) = if self.use_second_order() {
            // 重构水位 η 而非水深 h
            let eta_rec = self.muscl_eta.reconstruct_scalar(face_idx, &self.workspace.eta);
            let u_rec = self.muscl_u.reconstruct_scalar(face_idx, &self.workspace.vel_u);
            let v_rec = self.muscl_v.reconstruct_scalar(face_idx, &self.workspace.vel_v);

            if let Some(neigh) = neighbor {
                let z_owner = state.z[owner.get()];
                let z_neigh = state.z[neigh.get()];
                let h_left = (eta_rec.left - z_owner).max(B::Scalar::zero());
                let h_right = (eta_rec.right - z_neigh).max(B::Scalar::zero());
                (
                    h_left,
                    B::vec2_new(u_rec.left, v_rec.left),
                    z_owner,
                    h_right,
                    B::vec2_new(u_rec.right, v_rec.right),
                    z_neigh,
                )
            } else {
                // 边界：右侧使用反射条件
                let z_owner = state.z[owner.get()];
                let h_left = (eta_rec.left - z_owner).max(B::Scalar::zero());
                let vel_left = B::vec2_new(u_rec.left, v_rec.left);
                let vn = B::vec2_dot(&vel_left, &normal);
                let normal_2x = B::vec2_scale(&normal, B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::one()));
                let vel_right = B::vec2_sub(&vel_left, &B::vec2_scale(&normal_2x, vn));
                
                (
                    h_left,
                    vel_left,
                    z_owner,
                    h_left,
                    vel_right,
                    z_owner,
                )
            }
        } else {
            // 一阶：使用单元中心值
            let h_l = state.h[owner.get()];
            let z_l = state.z[owner.get()];
            let (u_l, v_l) = self.config.params.safe_velocity_components(
                state.hu[owner.get()], state.hv[owner.get()], h_l,
            );
            let vel_l = B::vec2_new(u_l, v_l);

            if let Some(neigh) = neighbor {
                let h_r = state.h[neigh.get()];
                let (u_r, v_r) = self.config.params.safe_velocity_components(
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
                // 边界反射
                let vn = B::vec2_dot(&vel_l, &normal);
                let normal_2x = B::vec2_scale(&normal, B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::one()));
                let vel_r = B::vec2_sub(&vel_l, &B::vec2_scale(&normal_2x, vn));
                (
                    h_l,
                    vel_l,
                    z_l,
                    h_l,
                    vel_r,
                    z_l,
                )
            }
        };

        // 静水重构
        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: (z_l + z_r) * B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::from_f64(0.5).unwrap()),
            }
        };

        // 干湿界面通量限制
        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => B::Scalar::zero(),
            (WetState::Dry, _) | (_, WetState::Dry) => B::Scalar::one(),
            (WetState::PartiallyWet, WetState::PartiallyWet) => {
                let h_min = recon.h_left.min(recon.h_right);
                let fraction = (h_min - self.config.params.h_dry) 
                    / (self.config.params.h_wet - self.config.params.h_dry);
                fraction.max(B::Scalar::zero()).min(B::Scalar::one())
            }
            _ => B::Scalar::one(),
        };

        // 求解黎曼问题
        let flux = self.riemann.solve(
            recon.h_left,
            recon.h_right,
            recon.vel_left,
            recon.vel_right,
            normal,
        ).unwrap_or_else(|_| RiemannFlux::zero());

        // 应用干湿限制
        let limited_flux = flux.scaled(flux_limiter);

        // 床坡源项（基于静水重构的水深差异）
        let bed_src = self.compute_hydrostatic_bed_slope(
            h_l, h_r, recon.h_left, recon.h_right, normal, length,
        );

        (limited_flux, bed_src, length, owner, neighbor)
    }

    /// 计算静水重构后的床坡源项
    ///
    /// 基于 Audusse (2004) 方法，使用 Backend 几何计算
    #[inline]
    fn compute_hydrostatic_bed_slope(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        h_l_star: B::Scalar,
        h_r_star: B::Scalar,
        normal: B::Vector2D,
        length: B::Scalar,
    ) -> BedSlopeCorrection<B> {
        let half = B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::from_f64(0.5).unwrap());
        
        // 左侧单元的压力补偿: 0.5 * g * (h_L² - h_L*²) * L
        let pressure_diff_l = half * self.hydrostatic.g * (h_l * h_l - h_l_star * h_l_star) * length;
        
        // 右侧单元的压力补偿: 0.5 * g * (h_R² - h_R*²) * L
        let pressure_diff_r = half * self.hydrostatic.g * (h_r * h_r - h_r_star * h_r_star) * length;

        // 左侧源项沿负法向，右侧源项沿正法向
        BedSlopeCorrection {
            source_left_x: -pressure_diff_l * normal.x(),
            source_left_y: -pressure_diff_l * normal.y(),
            source_right_x: pressure_diff_r * normal.x(),
            source_right_y: pressure_diff_r * normal.y(),
        }
    }

    /// 更新状态
    fn update_state(&self, state: &mut ShallowWaterState<B>, dt: B::Scalar) {
        let n = state.n_cells();

        for cell_idx in self.mesh.cells() {
            let i = cell_idx.get();
            let area = B::Scalar::from_f64(self.mesh.cell_area(cell_idx).unwrap_or(1.0)).unwrap_or(B::Scalar::one());
            let inv_area = B::Scalar::one() / area;

            state.h[i] = state.h[i] + dt * inv_area * self.workspace.flux_h[i];
            state.hu[i] = state.hu[i] + dt * inv_area * 
                (self.workspace.flux_hu[i] + self.workspace.source_hu[i]);
            state.hv[i] = state.hv[i] + dt * inv_area * 
                (self.workspace.flux_hv[i] + self.workspace.source_hv[i]);
        }
    }

    /// 强制正性约束
    fn enforce_positivity(&mut self, state: &mut ShallowWaterState<B>, _dt: B::Scalar) -> (usize, usize) {
        let h_min = B::Scalar::from_f64(self.config.params.h_min).unwrap_or(B::Scalar::zero());
        let h_dry = B::Scalar::from_f64(self.config.params.h_dry).unwrap_or(B::Scalar::zero());
        let mut dry_count = 0;
        let mut limited_count = 0;

        for cell_idx in self.mesh.cells() {
            let i = cell_idx.get();
            if state.h[i] < h_min {
                // 负水深修正
                state.h[i] = B::Scalar::zero();
                state.hu[i] = B::Scalar::zero();
                state.hv[i] = B::Scalar::zero();
                dry_count += 1;
            } else if state.h[i] < h_dry {
                // 干湿过渡区动量衰减
                let factor = self.wetting_drying.wet_fraction_smooth(state.h[i]);
                state.hu[i] = state.hu[i] * factor;
                state.hv[i] = state.hv[i] * factor;
                dry_count += 1;
                limited_count += 1;
            }
        }

        (dry_count, limited_count)
    }

    // =========================================================================
    // 访问器
    // =========================================================================

    /// 获取网格引用
    pub fn mesh(&self) -> &PhysicsMesh {
        &self.mesh
    }

    /// 获取配置引用
    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    /// 获取Backend引用
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// 获取统计信息
    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// 获取最大波速
    pub fn max_wave_speed(&self) -> f64 {
        self.stats.max_wave_speed
    }

    /// 获取干单元数量
    pub fn dry_cell_count(&self) -> usize {
        self.stats.dry_cells
    }
}

// ============================================================
// 求解器构建器（Backend感知）
// ============================================================

/// 求解器构建器
///
/// 提供链式API配置求解器，最终构建时需要指定Backend类型
#[derive(Debug, Default)]
pub struct SolverBuilder {
    /// 网格（必须）
    mesh: Option<Arc<PhysicsMesh>>,
    /// 配置（可选，有默认值）
    config: SolverConfig,
}

impl SolverBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            mesh: None,
            config: SolverConfig::default(),
        }
    }

    /// 设置网格（必须）
    pub fn mesh(mut self, mesh: Arc<PhysicsMesh>) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// 设置配置
    pub fn config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self
    }

    /// 设置数值参数
    pub fn params(mut self, params: NumericalParamsF64) -> Self {
        self.config.params = params;
        self
    }

    /// 设置重力加速度
    pub fn gravity(mut self, g: f64) -> Self {
        self.config.gravity = g;
        self
    }

    /// 设置是否使用静水重构
    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    /// 设置并行化阈值
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_threshold = threshold;
        self
    }

    /// 构建求解器
    ///
    /// # 参数
    /// - `backend`: 计算后端实例（决定精度）
    ///
    /// # 返回
    /// - `Ok(solver)`: 构建成功
    /// - `Err(e)`: 缺少必要配置（如网格）
    pub fn build<B: Backend>(self, backend: B) -> Result<ShallowWaterSolver<B>, BuildError> {
        let mesh = self.mesh.ok_or(BuildError::MissingMesh)?;
        Ok(ShallowWaterSolver::<B>::new(mesh, self.config, backend))
    }
}

impl Default for SolverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// 构建错误类型
#[derive(Debug, Clone)]
pub enum BuildError {
    /// 缺少网格
    MissingMesh,
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::MissingMesh => write!(f, "网格未设置（必须调用 .mesh()）"),
        }
    }
}

impl std::error::Error for BuildError {}

// 为了向后兼容，提供默认Backend的构建方法
impl SolverBuilder {
    /// 使用默认f64 Backend构建（向后兼容）
    pub fn build_f64(self) -> Result<ShallowWaterSolver<CpuBackend<f64>>, BuildError> {
        self.build(CpuBackend::<f64>::new())
    }

    /// 使用f32 Backend构建（高性能模式）
    pub fn build_f32(self) -> Result<ShallowWaterSolver<CpuBackend<f32>>, BuildError> {
        self.build(CpuBackend::<f32>::new())
    }
}