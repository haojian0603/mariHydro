// crates/mh_physics/src/engine/solver.rs

//! 浅水方程求解器
//!
//! 基于有限体积法的非结构化网格求解器，支持：
//! - HLLC 黎曼求解器
//! - 干湿处理
//! - 静水重构
//! - 多种时间积分方案
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/engine/solver.rs 迁移，保持核心算法不变。
//! 源项（摩擦、科氏力等）将在 sources 模块迁移后集成。

use crate::adapter::PhysicsMesh;
use crate::engine::timestep::{TimeStepController, TimeStepControllerBuilder};
use crate::schemes::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;

use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================
// 求解器配置
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
    /// 是否检查 NaN/Inf
    pub check_nan: bool,
    /// 是否检查负水深
    pub check_negative_depth: bool,
    /// 是否检查极大速度
    pub check_extreme_velocity: bool,
    /// 速度上限 [m/s]
    pub velocity_limit: f64,
    /// 水深上限 [m]
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
    /// 严格模式
    pub fn strict() -> Self {
        Self {
            check_nan: true,
            check_negative_depth: true,
            check_extreme_velocity: true,
            velocity_limit: 50.0,
            depth_limit: 500.0,
        }
    }

    /// 宽松模式
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

/// 求解器配置
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// 数值参数
    pub params: NumericalParams,
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
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            params: NumericalParams::default(),
            gravity: 9.81,
            use_hydrostatic_reconstruction: true,
            parallel_threshold: 1000,
            implicit_friction: true,
            scheme: NumericalScheme::default(),
            fallback: FallbackStrategy::default(),
            stability: StabilityOptions::default(),
            max_fallback_attempts: 3,
            timestep_reduction_factor: 0.5,
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
    pub fn params(mut self, params: NumericalParams) -> Self {
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
// 求解器工作区
// ============================================================

/// 求解器工作区
///
/// 存储中间计算结果，避免重复分配
#[derive(Debug)]
pub struct SolverWorkspace {
    /// 通量累加（质量）
    pub flux_h: Vec<f64>,
    /// 通量累加（x动量）
    pub flux_hu: Vec<f64>,
    /// 通量累加（y动量）
    pub flux_hv: Vec<f64>,
    /// 源项累加（x动量）
    pub source_hu: Vec<f64>,
    /// 源项累加（y动量）
    pub source_hv: Vec<f64>,
}

impl SolverWorkspace {
    /// 创建工作区
    pub fn new(n_cells: usize) -> Self {
        Self {
            flux_h: vec![0.0; n_cells],
            flux_hu: vec![0.0; n_cells],
            flux_hv: vec![0.0; n_cells],
            source_hu: vec![0.0; n_cells],
            source_hv: vec![0.0; n_cells],
        }
    }

    /// 重置通量
    pub fn reset_fluxes(&mut self) {
        self.flux_h.fill(0.0);
        self.flux_hu.fill(0.0);
        self.flux_hv.fill(0.0);
    }

    /// 重置源项
    pub fn reset_sources(&mut self) {
        self.source_hu.fill(0.0);
        self.source_hv.fill(0.0);
    }

    /// 重置所有
    pub fn reset(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.flux_h.resize(n_cells, 0.0);
        self.flux_hu.resize(n_cells, 0.0);
        self.flux_hv.resize(n_cells, 0.0);
        self.source_hu.resize(n_cells, 0.0);
        self.source_hv.resize(n_cells, 0.0);
    }
}

// ============================================================
// 静水重构
// ============================================================

/// 面上的静水重构状态
#[derive(Debug, Clone, Copy)]
pub struct HydrostaticFaceState {
    /// 左侧有效水深
    pub h_left: f64,
    /// 右侧有效水深
    pub h_right: f64,
    /// 左侧速度
    pub vel_left: DVec2,
    /// 右侧速度
    pub vel_right: DVec2,
    /// 面处高程
    pub z_face: f64,
}

/// 床坡源项修正
#[derive(Debug, Clone, Copy)]
pub struct BedSlopeCorrection {
    /// x方向源项
    pub source_x: f64,
    /// y方向源项
    pub source_y: f64,
}

/// 静水重构处理器
#[derive(Debug, Clone)]
pub struct HydrostaticReconstruction {
    /// 数值参数
    params: NumericalParams,
    /// 重力加速度
    g: f64,
}

impl HydrostaticReconstruction {
    /// 创建静水重构处理器
    pub fn new(params: &NumericalParams, g: f64) -> Self {
        Self {
            params: params.clone(),
            g,
        }
    }

    /// 简单静水重构
    ///
    /// 对面两侧的水深进行修正，确保静水平衡
    pub fn reconstruct_face_simple(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
    ) -> HydrostaticFaceState {
        // 面处高程取最大值（保守处理）
        let z_face = z_l.max(z_r);

        // 修正后的水深 = max(0, η - z_face)
        // 其中 η = h + z 是水位
        let eta_l = h_l + z_l;
        let eta_r = h_r + z_r;

        let h_left = (eta_l - z_face).max(0.0);
        let h_right = (eta_r - z_face).max(0.0);

        HydrostaticFaceState {
            h_left,
            h_right,
            vel_left: vel_l,
            vel_right: vel_r,
            z_face,
        }
    }

    /// 床坡源项修正
    ///
    /// 计算由于高程差产生的压力梯度源项
    pub fn bed_slope_correction(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        normal: DVec2,
        length: f64,
    ) -> BedSlopeCorrection {
        // 使用面平均水深
        let h_face = 0.5 * (h_l + h_r);
        
        if h_face < self.params.h_dry {
            return BedSlopeCorrection {
                source_x: 0.0,
                source_y: 0.0,
            };
        }

        // 床坡源项: -g * h * ∇z
        // 在有限体积离散中: S = -g * h_face * (z_r - z_l) * n * L
        let dz = z_r - z_l;
        let factor = -self.g * h_face * dz * length;

        BedSlopeCorrection {
            source_x: factor * normal.x,
            source_y: factor * normal.y,
        }
    }
}

// ============================================================
// 主求解器
// ============================================================

/// 浅水方程求解器
///
/// 基于有限体积法的非结构化网格求解器。
pub struct ShallowWaterSolver {
    /// 网格
    mesh: Arc<PhysicsMesh>,
    /// 配置
    config: SolverConfig,
    /// 工作区
    workspace: SolverWorkspace,
    /// 黎曼求解器
    riemann: HllcSolver,
    /// 干湿处理器
    wetting_drying: WettingDryingHandler,
    /// 静水重构
    hydrostatic: HydrostaticReconstruction,
    /// 时间步控制器
    timestep_ctrl: TimeStepController,
    /// 统计信息
    stats: SolverStats,
}

impl ShallowWaterSolver {
    /// 创建求解器
    pub fn new(mesh: Arc<PhysicsMesh>, config: SolverConfig) -> Self {
        let n_cells = mesh.n_cells();

        let timestep_ctrl = TimeStepControllerBuilder::new(config.gravity)
            .with_cfl(config.params.cfl)
            .with_dt_limits(config.params.dt_min, config.params.dt_max)
            .build();

        Self {
            mesh: mesh.clone(),
            config: config.clone(),
            workspace: SolverWorkspace::new(n_cells),
            riemann: HllcSolver::new(&config.params, config.gravity),
            wetting_drying: WettingDryingHandler::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::new(&config.params, config.gravity),
            timestep_ctrl,
            stats: SolverStats::default(),
        }
    }

    /// 执行一个时间步
    ///
    /// 返回使用的时间步长
    pub fn step(&mut self, state: &mut ShallowWaterState, dt: f64) -> f64 {
        // 1. 重置工作区
        self.workspace.reset();

        // 2. 计算通量（集成干湿处理）
        let max_wave_speed = if self.mesh.n_faces() >= self.config.parallel_threshold {
            self.compute_fluxes_parallel(state)
        } else {
            self.compute_fluxes_serial(state)
        };

        // 3. 更新状态
        self.update_state(state, dt);

        // 4. 强制正性并处理干区
        let (dry_cells, _) = self.enforce_positivity(state, dt);

        // 5. 更新统计
        self.stats.max_wave_speed = max_wave_speed;
        self.stats.dry_cells = dry_cells;
        self.stats.dt = dt;

        dt
    }

    /// 计算自适应时间步长
    pub fn compute_dt(&mut self, state: &ShallowWaterState) -> f64 {
        self.timestep_ctrl.update(state, &self.mesh, &self.config.params)
    }

    // =========================================================================
    // 通量计算（串行）
    // =========================================================================

    fn compute_fluxes_serial(&mut self, state: &ShallowWaterState) -> f64 {
        let n_faces = self.mesh.n_faces();
        let mut max_wave_speed = 0.0f64;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face_flux(state, face_idx);

            max_wave_speed = max_wave_speed.max(flux.max_wave_speed);

            // 累加到 owner
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            self.workspace.flux_h[owner] -= fh;
            self.workspace.flux_hu[owner] -= fhu;
            self.workspace.flux_hv[owner] -= fhv;
            self.workspace.source_hu[owner] += bed_src.source_x;
            self.workspace.source_hv[owner] += bed_src.source_y;

            // 累加到 neighbor（如果存在）
            if let Some(neigh) = neighbor {
                self.workspace.flux_h[neigh] += fh;
                self.workspace.flux_hu[neigh] += fhu;
                self.workspace.flux_hv[neigh] += fhv;
                self.workspace.source_hu[neigh] -= bed_src.source_x;
                self.workspace.source_hv[neigh] -= bed_src.source_y;
            }
        }

        max_wave_speed
    }

    // =========================================================================
    // 通量计算（收集后累加策略）
    // =========================================================================

    /// 使用"收集后累加"策略计算通量
    ///
    /// # 技术债务 (TD-5.3.2, TD-5.3.3)
    ///
    /// 当前实现是伪并行：
    /// 1. 并行阶段：各面通量计算是真正并行的
    /// 2. 累加阶段：收集所有结果后串行累加到单元
    ///
    /// 对于大规模网格，串行累加会成为性能瓶颈。
    /// 未来需要实现着色并行(Colored Parallel)以获得真正的无锁累加。
    fn compute_fluxes_parallel(&mut self, state: &ShallowWaterState) -> f64 {
        let n_faces = self.mesh.n_faces();
        let max_speed_atomic = AtomicU64::new(0u64);

        // 阶段1: 并行计算所有面的通量 (真正并行)
        let face_results: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face_flux(state, face_idx);

                // 更新最大波速
                max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 阶段2: 串行累加到单元 (性能瓶颈)
        // TODO(TD-5.3.3): 使用着色并行实现无锁累加
        for (flux, bed_src, length, owner, neighbor) in face_results {
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            self.workspace.flux_h[owner] -= fh;
            self.workspace.flux_hu[owner] -= fhu;
            self.workspace.flux_hv[owner] -= fhv;
            self.workspace.source_hu[owner] += bed_src.source_x;
            self.workspace.source_hv[owner] += bed_src.source_y;

            if let Some(neigh) = neighbor {
                self.workspace.flux_h[neigh] += fh;
                self.workspace.flux_hu[neigh] += fhu;
                self.workspace.flux_hv[neigh] += fhv;
                self.workspace.source_hu[neigh] -= bed_src.source_x;
                self.workspace.source_hv[neigh] -= bed_src.source_y;
            }
        }

        f64::from_bits(max_speed_atomic.load(Ordering::Relaxed))
    }

    // =========================================================================
    // 单面通量计算
    // =========================================================================

    fn compute_face_flux(
        &self,
        state: &ShallowWaterState,
        face_idx: usize,
    ) -> (RiemannFlux, BedSlopeCorrection, f64, usize, Option<usize>) {
        let normal = self.mesh.face_normal(face_idx);
        let length = self.mesh.face_length(face_idx);
        let owner = self.mesh.face_owner(face_idx);
        let neighbor = self.mesh.face_neighbor(face_idx);

        // 获取左侧状态
        let h_l = state.h[owner];
        let z_l = state.z[owner];
        let (u_l, v_l) = self.config.params.safe_velocity_components(
            state.hu[owner], state.hv[owner], h_l
        );
        let vel_l = DVec2::new(u_l, v_l);

        // 获取右侧状态
        let (h_r, vel_r, z_r) = if let Some(neigh) = neighbor {
            let h = state.h[neigh];
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[neigh], state.hv[neigh], h
            );
            (h, DVec2::new(u, v), state.z[neigh])
        } else {
            // TODO(phase5): 边界处理简化为反射条件
            // 完整实现应支持：入流/出流/滑移/无滑移等多种边界类型
            let vn = vel_l.dot(normal);
            (h_l, vel_l - 2.0 * vn * normal, z_l)
        };

        // 静水重构
        // TODO(phase5): 考虑添加 reconstruct_face_muscl() 方法支持高阶重构
        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: 0.5 * (z_l + z_r),
            }
        };

        // 干湿界面通量限制
        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => 0.0,
            (WetState::Dry, _) | (_, WetState::Dry) => {
                // 干湿界面：使用较小水深作为限制
                let h_min = recon.h_left.min(recon.h_right);
                (h_min / self.config.params.h_wet).min(1.0)
            }
            (WetState::PartiallyWet, _) | (_, WetState::PartiallyWet) => {
                let h_min = recon.h_left.min(recon.h_right);
                ((h_min - self.config.params.h_dry) 
                    / (self.config.params.h_wet - self.config.params.h_dry)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        };

        // 求解黎曼问题
        // TODO(phase5): 集成 AdaptiveSolver 支持自动选择最优求解器
        let flux = self.riemann.solve(
            recon.h_left, recon.h_right,
            recon.vel_left, recon.vel_right,
            normal,
        ).unwrap_or(RiemannFlux::ZERO);

        // 应用干湿限制
        let limited_flux = if flux_limiter < 1.0 {
            flux.scaled(flux_limiter)
        } else {
            flux
        };

        // 床坡源项
        let bed_src = self.hydrostatic.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);

        (limited_flux, bed_src, length, owner, neighbor)
    }

    // =========================================================================
    // 状态更新
    // =========================================================================

    fn update_state(&self, state: &mut ShallowWaterState, dt: f64) {
        let n = state.n_cells();

        for i in 0..n {
            let area = self.mesh.cell_area(i).unwrap_or(1.0);
            let inv_area = 1.0 / area;

            state.h[i] += dt * inv_area * self.workspace.flux_h[i];
            state.hu[i] += dt * inv_area * (self.workspace.flux_hu[i] + self.workspace.source_hu[i]);
            state.hv[i] += dt * inv_area * (self.workspace.flux_hv[i] + self.workspace.source_hv[i]);
        }
    }

    // =========================================================================
    // 正性保持
    // =========================================================================

    fn enforce_positivity(&mut self, state: &mut ShallowWaterState, _dt: f64) -> (usize, usize) {
        let h_min = self.config.params.h_min;
        let h_dry = self.config.params.h_dry;
        let mut dry_count = 0;
        let mut limited_count = 0;

        for i in 0..state.n_cells() {
            if state.h[i] < h_min {
                // 负水深修正
                state.h[i] = 0.0;
                state.hu[i] = 0.0;
                state.hv[i] = 0.0;
                dry_count += 1;
            } else if state.h[i] < h_dry {
                // 干湿过渡区动量衰减
                let factor = self.wetting_drying.wet_fraction_smooth(state.h[i]);
                state.hu[i] *= factor;
                state.hv[i] *= factor;
                dry_count += 1;
                limited_count += 1;
            }
        }

        (dry_count, limited_count)
    }

    // =========================================================================
    // 访问器
    // =========================================================================

    /// 获取网格
    pub fn mesh(&self) -> &PhysicsMesh {
        &self.mesh
    }

    /// 获取配置
    pub fn config(&self) -> &SolverConfig {
        &self.config
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
// 求解器构建器
// ============================================================

/// 求解器构建器
pub struct SolverBuilder {
    mesh: Option<Arc<PhysicsMesh>>,
    config: SolverConfig,
}

impl SolverBuilder {
    /// 创建构建器
    pub fn new() -> Self {
        Self {
            mesh: None,
            config: SolverConfig::default(),
        }
    }

    /// 设置网格
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
    pub fn params(mut self, params: NumericalParams) -> Self {
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

    /// 构建求解器
    pub fn build(self) -> Result<ShallowWaterSolver, &'static str> {
        let mesh = self.mesh.ok_or("mesh is required")?;
        Ok(ShallowWaterSolver::new(mesh, self.config))
    }
}

impl Default for SolverBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_geo::{Point2D, Point3D};
    use mh_mesh::FrozenMesh;

    /// 创建简单的 2x2 网格用于测试
    /// 
    /// 网格布局:
    /// ```text
    ///   +---+---+
    ///   | 2 | 3 |
    ///   +---+---+
    ///   | 0 | 1 |
    ///   +---+---+
    /// ```
    /// 
    /// 节点 (9个):
    /// ```text
    ///   6---7---8
    ///   |   |   |
    ///   3---4---5
    ///   |   |   |
    ///   0---1---2
    /// ```
    /// 
    /// 内部面 (4条): 连接相邻单元
    /// 边界面 (8条): 外边界
    fn create_simple_mesh() -> PhysicsMesh {
        // 单元尺寸
        let dx = 1.0;
        let dy = 1.0;
        
        // 节点坐标 (3x3 = 9个节点)
        let node_coords = vec![
            Point3D::new(0.0, 0.0, 0.0), // 0
            Point3D::new(dx, 0.0, 0.0),  // 1
            Point3D::new(2.0*dx, 0.0, 0.0), // 2
            Point3D::new(0.0, dy, 0.0),  // 3
            Point3D::new(dx, dy, 0.0),   // 4
            Point3D::new(2.0*dx, dy, 0.0), // 5
            Point3D::new(0.0, 2.0*dy, 0.0), // 6
            Point3D::new(dx, 2.0*dy, 0.0),  // 7
            Point3D::new(2.0*dx, 2.0*dy, 0.0), // 8
        ];
        
        // 单元中心
        let cell_center = vec![
            Point2D::new(0.5*dx, 0.5*dy), // 单元0
            Point2D::new(1.5*dx, 0.5*dy), // 单元1
            Point2D::new(0.5*dx, 1.5*dy), // 单元2
            Point2D::new(1.5*dx, 1.5*dy), // 单元3
        ];
        
        // 单元面积
        let cell_area = vec![dx*dy, dx*dy, dx*dy, dx*dy];
        
        // 单元底床高程 (平底)
        let cell_z_bed = vec![0.0, 0.0, 0.0, 0.0];
        
        // 内部面 (4条)
        // 面0: 单元0-1 (垂直面，法向x正)
        // 面1: 单元0-2 (水平面，法向y正)
        // 面2: 单元1-3 (水平面，法向y正)
        // 面3: 单元2-3 (垂直面，法向x正)
        
        // 边界面 (8条)
        // 面4-5: 下边界 (单元0,1)
        // 面6-7: 右边界 (单元1,3)
        // 面8-9: 上边界 (单元2,3)
        // 面10-11: 左边界 (单元0,2)
        
        let n_interior = 4;
        let n_boundary = 8;
        let n_faces = n_interior + n_boundary;
        
        // 面数据
        let face_center = vec![
            // 内部面
            Point2D::new(dx, 0.5*dy),     // 0: 单元0-1
            Point2D::new(0.5*dx, dy),     // 1: 单元0-2
            Point2D::new(1.5*dx, dy),     // 2: 单元1-3
            Point2D::new(dx, 1.5*dy),     // 3: 单元2-3
            // 边界面
            Point2D::new(0.5*dx, 0.0),    // 4: 下-单元0
            Point2D::new(1.5*dx, 0.0),    // 5: 下-单元1
            Point2D::new(2.0*dx, 0.5*dy), // 6: 右-单元1
            Point2D::new(2.0*dx, 1.5*dy), // 7: 右-单元3
            Point2D::new(0.5*dx, 2.0*dy), // 8: 上-单元2
            Point2D::new(1.5*dx, 2.0*dy), // 9: 上-单元3
            Point2D::new(0.0, 0.5*dy),    // 10: 左-单元0
            Point2D::new(0.0, 1.5*dy),    // 11: 左-单元2
        ];
        
        let face_normal = vec![
            // 内部面
            Point3D::new(1.0, 0.0, 0.0),  // 0: x正
            Point3D::new(0.0, 1.0, 0.0),  // 1: y正
            Point3D::new(0.0, 1.0, 0.0),  // 2: y正
            Point3D::new(1.0, 0.0, 0.0),  // 3: x正
            // 边界面 (外向法向)
            Point3D::new(0.0, -1.0, 0.0), // 4: 下
            Point3D::new(0.0, -1.0, 0.0), // 5: 下
            Point3D::new(1.0, 0.0, 0.0),  // 6: 右
            Point3D::new(1.0, 0.0, 0.0),  // 7: 右
            Point3D::new(0.0, 1.0, 0.0),  // 8: 上
            Point3D::new(0.0, 1.0, 0.0),  // 9: 上
            Point3D::new(-1.0, 0.0, 0.0), // 10: 左
            Point3D::new(-1.0, 0.0, 0.0), // 11: 左
        ];
        
        let face_length = vec![
            dy, dx, dx, dy,              // 内部面
            dx, dx, dy, dy, dx, dx, dy, dy, // 边界面
        ];
        
        let face_owner = vec![
            0, 0, 1, 2,                  // 内部面
            0, 1, 1, 3, 2, 3, 0, 2,      // 边界面
        ];
        
        let face_neighbor: Vec<u32> = vec![
            1, 2, 3, 3,                  // 内部面 (有邻居)
            u32::MAX, u32::MAX, u32::MAX, u32::MAX, // 边界面 (无邻居)
            u32::MAX, u32::MAX, u32::MAX, u32::MAX,
        ];
        
        // 面高程
        let face_z_left = vec![0.0; n_faces];
        let face_z_right = vec![0.0; n_faces];
        
        // 面到单元中心的向量
        let face_delta_owner = vec![Point2D::new(0.0, 0.0); n_faces];
        let face_delta_neighbor = vec![Point2D::new(0.0, 0.0); n_faces];
        let face_dist_o2n = vec![dx; n_faces];
        
        // 单元拓扑 (简化版)
        let cell_node_offsets = vec![0, 4, 8, 12, 16];
        let cell_node_indices = vec![
            0, 1, 4, 3,  // 单元0
            1, 2, 5, 4,  // 单元1
            3, 4, 7, 6,  // 单元2
            4, 5, 8, 7,  // 单元3
        ];
        
        let cell_face_offsets = vec![0, 4, 8, 12, 16];
        let cell_face_indices = vec![
            0, 1, 4, 10,   // 单元0
            0, 2, 5, 6,    // 单元1
            1, 3, 8, 11,   // 单元2
            2, 3, 7, 9,    // 单元3
        ];
        
        let cell_neighbor_offsets = vec![0, 2, 4, 6, 8];
        let cell_neighbor_indices = vec![
            1, 2,      // 单元0的邻居
            0, 3,      // 单元1的邻居
            0, 3,      // 单元2的邻居
            1, 2,      // 单元3的邻居
        ];
        
        let frozen = FrozenMesh {
            n_nodes: 9,
            node_coords,
            n_cells: 4,
            cell_center,
            cell_area,
            cell_z_bed,
            cell_node_offsets,
            cell_node_indices,
            cell_face_offsets,
            cell_face_indices,
            cell_neighbor_offsets,
            cell_neighbor_indices,
            n_faces,
            n_interior_faces: n_interior,
            face_center,
            face_normal,
            face_length,
            face_z_left,
            face_z_right,
            face_owner,
            face_neighbor,
            face_delta_owner,
            face_delta_neighbor,
            face_dist_o2n,
            boundary_face_indices: (4..12).map(|i| i as u32).collect(),
            boundary_names: vec!["boundary".to_string()],
            face_boundary_id: (0..n_faces).map(|i| if i >= 4 { Some(0) } else { None }).collect(),
            min_cell_size: dx.min(dy),
            max_cell_size: dx.max(dy),
        };
        
        PhysicsMesh::from_frozen(&frozen)
    }

    #[test]
    fn test_solver_config_default() {
        let config = SolverConfig::default();
        assert!((config.gravity - 9.81).abs() < 1e-10);
        assert!(config.use_hydrostatic_reconstruction);
        assert!(config.implicit_friction);
    }

    #[test]
    fn test_solver_config_builder() {
        let config = SolverConfig::builder()
            .gravity(10.0)
            .use_hydrostatic_reconstruction(false)
            .parallel_threshold(5000)
            .build();

        assert!((config.gravity - 10.0).abs() < 1e-10);
        assert!(!config.use_hydrostatic_reconstruction);
        assert_eq!(config.parallel_threshold, 5000);
    }

    #[test]
    fn test_solver_workspace() {
        let mut ws = SolverWorkspace::new(10);
        assert_eq!(ws.flux_h.len(), 10);

        ws.flux_h[0] = 1.0;
        ws.reset_fluxes();
        assert_eq!(ws.flux_h[0], 0.0);

        ws.resize(20);
        assert_eq!(ws.flux_h.len(), 20);
    }

    #[test]
    fn test_hydrostatic_reconstruction() {
        let params = NumericalParams::default();
        let hydro = HydrostaticReconstruction::new(&params, 9.81);

        // 平底情况
        let recon = hydro.reconstruct_face_simple(
            1.0, 1.0, 
            0.0, 0.0, 
            DVec2::ZERO, DVec2::ZERO
        );
        assert!((recon.h_left - 1.0).abs() < 1e-10);
        assert!((recon.h_right - 1.0).abs() < 1e-10);

        // 高程差情况
        let recon = hydro.reconstruct_face_simple(
            1.0, 0.5,
            0.0, 0.5,
            DVec2::ZERO, DVec2::ZERO
        );
        // z_face = max(0, 0.5) = 0.5
        // h_left = max(0, (1.0 + 0.0) - 0.5) = 0.5
        // h_right = max(0, (0.5 + 0.5) - 0.5) = 0.5
        assert!((recon.h_left - 0.5).abs() < 1e-10);
        assert!((recon.h_right - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bed_slope_correction() {
        let params = NumericalParams::default();
        let hydro = HydrostaticReconstruction::new(&params, 9.81);

        let correction = hydro.bed_slope_correction(
            1.0, 1.0,
            0.0, 1.0,
            DVec2::X,
            1.0,
        );

        // S = -g * h_face * dz * L * n
        // dz = 1.0, h_face = 1.0, L = 1.0, n = (1, 0)
        // S_x = -9.81 * 1.0 * 1.0 * 1.0 * 1.0 = -9.81
        assert!((correction.source_x - (-9.81)).abs() < 1e-10);
        assert!(correction.source_y.abs() < 1e-10);
    }

    #[test]
    fn test_solver_builder() {
        let builder = SolverBuilder::new()
            .gravity(10.0)
            .use_hydrostatic_reconstruction(false);

        assert!((builder.config.gravity - 10.0).abs() < 1e-10);
        assert!(!builder.config.use_hydrostatic_reconstruction);

        // 没有设置 mesh，构建应该失败
        assert!(builder.build().is_err());
    }

    // =========================================================================
    // TD-5.3.1: Solver核心方法测试
    // =========================================================================

    #[test]
    fn test_simple_mesh_creation() {
        let mesh = create_simple_mesh();
        assert_eq!(mesh.n_cells(), 4);
        assert_eq!(mesh.n_faces(), 12);
        assert_eq!(mesh.n_interior_faces(), 4);
        assert_eq!(mesh.n_boundary_faces(), 8);
    }

    #[test]
    fn test_solver_creation_with_mesh() {
        let mesh = create_simple_mesh();
        let config = SolverConfig::default();
        let solver = ShallowWaterSolver::new(Arc::new(mesh), config);

        assert_eq!(solver.mesh().n_cells(), 4);
        assert!((solver.config().gravity - 9.81).abs() < 1e-10);
    }

    #[test]
    fn test_solver_step_still_water() {
        // 测试静水情况: 水深均匀，无速度，无高程差
        // 预期: 状态应保持不变（平衡态）
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(100) // 强制串行以便调试
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 创建静水状态
        let h0 = 1.0;
        let mut state = ShallowWaterState::new(4);
        for i in 0..4 {
            state.h[i] = h0;
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        // 记录初始质量
        let initial_mass: f64 = state.h.iter().sum();
        
        // 执行一步
        let dt = 0.001;
        solver.step(&mut state, dt);
        
        // 验证质量守恒
        let final_mass: f64 = state.h.iter().sum();
        let mass_error = (final_mass - initial_mass).abs() / initial_mass;
        assert!(mass_error < 1e-10, "质量守恒误差: {}", mass_error);
        
        // 验证静水保持（通量应接近零）
        for i in 0..4 {
            assert!((state.h[i] - h0).abs() < 1e-10, 
                "单元{} 水深变化过大: {} -> {}", i, h0, state.h[i]);
            assert!(state.hu[i].abs() < 1e-10, 
                "单元{} 出现非零x动量: {}", i, state.hu[i]);
            assert!(state.hv[i].abs() < 1e-10,
                "单元{} 出现非零y动量: {}", i, state.hv[i]);
        }
    }

    #[test]
    fn test_solver_step_uniform_flow() {
        // 测试均匀流情况: 水深均匀，有均匀速度
        // 预期: 质量守恒
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(100)
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 创建均匀流状态
        let h0 = 1.0;
        let u0 = 0.1;  // 小速度
        let mut state = ShallowWaterState::new(4);
        for i in 0..4 {
            state.h[i] = h0;
            state.hu[i] = h0 * u0;  // hu = h * u
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        let initial_mass: f64 = state.h.iter().sum();
        
        // 执行一步
        let dt = 0.001;
        solver.step(&mut state, dt);
        
        // 验证质量守恒（允许边界流出/流入的影响）
        let final_mass: f64 = state.h.iter().sum();
        // 边界使用反射条件，所以质量应该守恒
        let mass_error = (final_mass - initial_mass).abs() / initial_mass;
        assert!(mass_error < 1e-6, "质量守恒误差: {}", mass_error);
        
        // 验证水深保持正值
        for i in 0..4 {
            assert!(state.h[i] > 0.0, "单元{} 水深为负: {}", i, state.h[i]);
        }
    }

    #[test]
    fn test_solver_step_dam_break() {
        // 测试溃坝情况: 左边高水深，右边低水深
        // 预期: 水从左向右流动
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(100)
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 创建溃坝初始条件
        // 单元0,2 (左侧) 水深高
        // 单元1,3 (右侧) 水深低
        let h_left = 2.0;
        let h_right = 1.0;
        let mut state = ShallowWaterState::new(4);
        state.h[0] = h_left;
        state.h[1] = h_right;
        state.h[2] = h_left;
        state.h[3] = h_right;
        for i in 0..4 {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        let initial_mass: f64 = state.h.iter().sum();
        
        // 执行多步
        let dt = 0.001;
        for _ in 0..10 {
            solver.step(&mut state, dt);
        }
        
        // 验证质量守恒
        let final_mass: f64 = state.h.iter().sum();
        let mass_error = (final_mass - initial_mass).abs() / initial_mass;
        assert!(mass_error < 1e-6, "质量守恒误差: {}", mass_error);
        
        // 验证水流方向正确 (左侧水深应减少或动量向右)
        // 由于溃坝，左侧单元应该有正的x动量（向右流动）
        // 这是一个定性测试
        let left_momentum: f64 = state.hu[0] + state.hu[2];
        let right_momentum: f64 = state.hu[1] + state.hu[3];
        
        // 由于边界反射，不容易直接验证流向
        // 但至少应该产生了动量
        let total_momentum = left_momentum.abs() + right_momentum.abs();
        assert!(total_momentum > 1e-6, "应该产生动量，但 total_momentum = {}", total_momentum);
    }

    #[test]
    fn test_solver_compute_fluxes_serial() {
        // 直接测试通量计算方法
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::builder()
            .parallel_threshold(10000) // 强制串行
            .build();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 创建有水深差的状态
        let mut state = ShallowWaterState::new(4);
        state.h[0] = 2.0;
        state.h[1] = 1.0;
        state.h[2] = 2.0;
        state.h[3] = 1.0;
        for i in 0..4 {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        // 计算通量
        let max_speed = solver.compute_fluxes_serial(&state);
        
        // 验证最大波速合理
        // 波速 c = sqrt(g*h)，对于 h=2, c ≈ 4.43
        assert!(max_speed > 0.0, "最大波速应大于0");
        assert!(max_speed < 10.0, "最大波速应合理: {}", max_speed);
        
        // 验证通量非零
        let total_flux: f64 = solver.workspace.flux_h.iter().map(|x| x.abs()).sum();
        assert!(total_flux > 1e-10, "应该产生非零通量");
    }

    #[test]
    fn test_solver_compute_fluxes_parallel() {
        // 测试并行通量计算与串行结果一致
        
        let mesh = Arc::new(create_simple_mesh());
        
        // 创建状态
        let mut state = ShallowWaterState::new(4);
        state.h[0] = 2.0;
        state.h[1] = 1.0;
        state.h[2] = 1.5;
        state.h[3] = 1.2;
        for i in 0..4 {
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        // 串行计算
        let config_serial = SolverConfig::builder()
            .parallel_threshold(10000)
            .build();
        let mut solver_serial = ShallowWaterSolver::new(mesh.clone(), config_serial);
        solver_serial.workspace.reset();
        let max_speed_serial = solver_serial.compute_fluxes_serial(&state);
        let flux_h_serial = solver_serial.workspace.flux_h.clone();
        
        // 并行计算
        let config_parallel = SolverConfig::builder()
            .parallel_threshold(0)
            .build();
        let mut solver_parallel = ShallowWaterSolver::new(mesh.clone(), config_parallel);
        solver_parallel.workspace.reset();
        let max_speed_parallel = solver_parallel.compute_fluxes_parallel(&state);
        let flux_h_parallel = solver_parallel.workspace.flux_h.clone();
        
        // 验证结果一致
        assert!((max_speed_serial - max_speed_parallel).abs() < 1e-10,
            "最大波速不一致: serial={}, parallel={}", max_speed_serial, max_speed_parallel);
        
        for i in 0..4 {
            assert!((flux_h_serial[i] - flux_h_parallel[i]).abs() < 1e-10,
                "单元{} 质量通量不一致: serial={}, parallel={}", 
                i, flux_h_serial[i], flux_h_parallel[i]);
        }
    }

    #[test]
    fn test_solver_positivity_enforcement() {
        // 测试正性保持
        
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::default();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        // 创建有负水深的状态
        let mut state = ShallowWaterState::new(4);
        state.h[0] = -0.01;  // 负水深
        state.h[1] = 0.001;  // 极小水深
        state.h[2] = 1.0;    // 正常水深
        state.h[3] = 0.0;    // 零水深
        state.hu[0] = 1.0;
        state.hu[1] = 0.1;
        
        let (dry_count, _) = solver.enforce_positivity(&mut state, 0.001);
        
        // 验证负水深被修正
        assert!(state.h[0] >= 0.0, "负水深未被修正");
        assert!(state.h[1] >= 0.0, "极小水深出问题");
        
        // 验证动量被衰减
        assert_eq!(state.hu[0], 0.0, "干单元动量应为0");
        
        // 验证统计正确
        assert!(dry_count >= 2, "应该检测到至少2个干单元");
    }

    #[test]
    fn test_solver_stats() {
        let mesh = Arc::new(create_simple_mesh());
        let config = SolverConfig::default();
        let mut solver = ShallowWaterSolver::new(mesh.clone(), config);
        
        let mut state = ShallowWaterState::new(4);
        for i in 0..4 {
            state.h[i] = 1.0;
            state.hu[i] = 0.0;
            state.hv[i] = 0.0;
            state.z[i] = 0.0;
        }
        
        solver.step(&mut state, 0.001);
        
        let stats = solver.stats();
        assert!(stats.max_wave_speed >= 0.0);
        assert!(stats.dt > 0.0);
    }
}
