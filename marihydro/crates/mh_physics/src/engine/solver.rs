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
//! 6. **边界条件修复**：恢复正确的固壁边界压力处理，确保静水平衡
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_physics::engine::solver::{ShallowWaterSolver};
//! use mh_physics::config_bridge::{Layer3Config, ConfigBridge};
//! use mh_config::SolverConfig;
//! use mh_runtime::{CpuBackend};
//!
//! // 从 Layer 4 配置转换
//! let layer4_config = SolverConfig::default();
//! let layer3_config: Layer3Config<f64> = ConfigBridge::convert(&layer4_config).unwrap();
//!
//! // f64高精度模式
//! let backend_f64 = CpuBackend::<f64>::new();
//! let solver_f64 = ShallowWaterSolver::<CpuBackend<f64>>::new(mesh, layer3_config, backend_f64);
//!
//! // f32高性能模式
//! let backend_f32 = CpuBackend::<f32>::new();
//! let solver_f32 = ShallowWaterSolver::<CpuBackend<f32>>::new(mesh, layer3_config, backend_f32);
//! ```

use crate::adapter::{CellIndex, FaceIndex, PhysicsMesh};
use crate::engine::timestep::TimeStepController;
use crate::schemes::{HllcSolver, RiemannFlux, RiemannSolver, SolverParams};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::numerics::{MusclConfig, MusclReconstructor};
use crate::numerics::reconstruction::Reconstructor;
use crate::state::ShallowWaterStateGeneric as ShallowWaterState;
use crate::types::{NumericalParams};
use crate::config_bridge::Layer3Config;

use mh_runtime::{Backend, CpuBackend, DeviceBuffer, RuntimeScalar, Vector2D};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ============================================================
// 求解器统计（保持不变）
// ============================================================

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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result{
        match self {
            Self::Marginal => write!(f, "临界"),
            Self::Stable => write!(f, "稳定"),
            Self::NeedsFallback => write!(f, "需要回退"),
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
    pub fn reset_fluxes(&mut self) {
        use mh_runtime::DeviceBuffer;
        self.flux_h.fill(B::Scalar::ZERO);
        self.flux_hu.fill(B::Scalar::ZERO);
        self.flux_hv.fill(B::Scalar::ZERO);
    }

    /// 重置源项
    pub fn reset_sources(&mut self) {
        use mh_runtime::DeviceBuffer;
        self.source_hu.fill(B::Scalar::ZERO);
        self.source_hv.fill(B::Scalar::ZERO);
    }

    /// 重置所有
    pub fn reset(&mut self) {
        self.reset_fluxes();
        self.reset_sources();
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        use mh_runtime::DeviceBuffer;
        self.flux_h.resize(n_cells, B::Scalar::ZERO);
        self.flux_hu.resize(n_cells, B::Scalar::ZERO);
        self.flux_hv.resize(n_cells, B::Scalar::ZERO);
        self.source_hu.resize(n_cells, B::Scalar::ZERO);
        self.source_hv.resize(n_cells, B::Scalar::ZERO);
        self.vel_u.resize(n_cells, B::Scalar::ZERO);
        self.vel_v.resize(n_cells, B::Scalar::ZERO);
        self.eta.resize(n_cells, B::Scalar::ZERO);
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
            source_left_x: B::Scalar::ZERO,
            source_left_y: B::Scalar::ZERO,
            source_right_x: B::Scalar::ZERO,
            source_right_y: B::Scalar::ZERO,
        }
    }
}

/// 类型别名
pub type BedSlopeCorrectionF64 = BedSlopeCorrection<CpuBackend<f64>>;

/// 静水重构处理器（Backend泛型化）
#[derive(Debug, Clone)]
pub struct HydrostaticReconstruction<B: Backend> {
    /// 数值参数（泛型）
    #[allow(dead_code)]
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

    /// 计算床坡源项
    #[inline]
    pub fn bed_slope_correction(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        z_l: B::Scalar,
        z_r: B::Scalar,
        normal: B::Vector2D,
        length: B::Scalar,
    ) -> BedSlopeCorrection<B> {
        let half = B::Scalar::from_f64(0.5).unwrap();
        let z_face = if z_l > z_r { z_l } else { z_r };
        let eta_l = h_l + z_l;
        let eta_r = h_r + z_r;
        let zero = B::Scalar::ZERO;
        let h_l_star = if eta_l - z_face > zero { eta_l - z_face } else { zero };
        let h_r_star = if eta_r - z_face > zero { eta_r - z_face } else { zero };
        
        let pressure_diff_l = half * self.g * (h_l * h_l - h_l_star * h_l_star) * length;
        let pressure_diff_r = half * self.g * (h_r * h_r - h_r_star * h_r_star) * length;

        BedSlopeCorrection {
            source_left_x: -pressure_diff_l * normal.x(),
            source_left_y: -pressure_diff_l * normal.y(),
            source_right_x: pressure_diff_r * normal.x(),
            source_right_y: pressure_diff_r * normal.y(),
        }
    }
}

// ============================================================
// 主求解器（Backend泛型化 - 最终版本）
// ============================================================

pub struct ShallowWaterSolver<B: Backend> {
    mesh: Arc<PhysicsMesh>,
    config: Layer3Config<B::Scalar>,
    params: NumericalParams<B::Scalar>,
    #[allow(dead_code)]
    gravity: B::Scalar,
    backend: B,
    workspace: SolverWorkspaceGeneric<B>,
    riemann: HllcSolver<B>,
    wetting_drying: WettingDryingHandler<B>,
    hydrostatic: HydrostaticReconstruction<B>,
    timestep_ctrl: TimeStepController<B>,
    stats: SolverStats,
    muscl_eta: MusclReconstructor,
    muscl_u: MusclReconstructor,
    muscl_v: MusclReconstructor,
}

impl<B: Backend> ShallowWaterSolver<B> {
    pub fn new(
        mesh: Arc<PhysicsMesh>, 
        config: Layer3Config<B::Scalar>, 
        backend: B
    ) -> Self {
        let n_cells = mesh.n_cells();
        let gravity = config.gravity;
        let params = config.params.clone();

        // 时间步控制器（已泛型化，接收 B::Scalar 参数）
        let timestep_ctrl = TimeStepController::<B>::new(gravity, &params);

        // 根据配置选择重构器模式
        let muscl_config = match config.scheme {
            NumericalScheme::SecondOrderMuscl | NumericalScheme::SecondOrderWeno => {
                MusclConfig::default()
            }
            NumericalScheme::FirstOrder => MusclConfig::first_order(),
        };

        let workspace = SolverWorkspaceGeneric::new(&backend, n_cells);
        let solver_params = SolverParams::<B::Scalar>::from_numerical(&params, gravity);
        
        let riemann = HllcSolver::<B>::new(&solver_params, gravity);
        let wetting_drying = WettingDryingHandler::<B>::from_params(&params);
        let hydrostatic = HydrostaticReconstruction::<B>::new(&params, gravity);
        let muscl_eta = MusclReconstructor::new(muscl_config.clone(), mesh.clone());
        let muscl_u = MusclReconstructor::new(muscl_config.clone(), mesh.clone());
        let muscl_v = MusclReconstructor::new(muscl_config, mesh.clone());

        Self {
            mesh,
            config,
            params,
            gravity,
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

    pub fn step(&mut self, state: &mut ShallowWaterState<B>, dt: B::Scalar) -> B::Scalar {
        self.workspace.reset();
        self.prepare_reconstruction(state);
        let max_wave_speed = if self.mesh.n_faces() >= self.config.parallel_threshold as usize {
            self.compute_fluxes_parallel(state)
        } else {
            self.compute_fluxes_serial(state)
        };
        self.update_state(state, dt);
        let (dry_cells, _) = self.enforce_positivity(state, dt);
        self.stats.max_wave_speed = max_wave_speed.to_f64().unwrap_or(0.0);
        self.stats.dry_cells = dry_cells;
        self.stats.dt = dt.to_f64().unwrap_or(0.0);
        dt
    }

    /// ✅ 修复类型不匹配：直接传递泛型参数
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

        for i in self.mesh.cells() {
            let (u, v) = self.params.safe_velocity_components(
                state.hu[i], state.hv[i], state.h[i]
            );
            self.workspace.vel_u[i] = u;
            self.workspace.vel_v[i] = v;
            self.workspace.eta[i] = state.h[i] + state.z[i];
        }

        // 一阶格式完全禁用梯度计算，直接返回
        if !self.use_second_order() {
            let cfg = MusclConfig::first_order();
            self.muscl_eta.set_config(cfg.clone());
            self.muscl_u.set_config(cfg.clone());
            self.muscl_v.set_config(cfg);
            // 关键：直接返回，不计算梯度
            return;
        }

        let cfg = MusclConfig::default();
        self.muscl_eta.set_config(cfg.clone());
        self.muscl_u.set_config(cfg.clone());
        self.muscl_v.set_config(cfg);

        // 注意：MusclReconstructor 使用 f64，需要转换
        let eta_f64: Vec<f64> = self.workspace.eta.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let vel_u_f64: Vec<f64> = self.workspace.vel_u.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        let vel_v_f64: Vec<f64> = self.workspace.vel_v.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
        self.muscl_eta.compute_gradients(&eta_f64);
        self.muscl_u.compute_gradients(&vel_u_f64);
        self.muscl_v.compute_gradients(&vel_v_f64);
    }

    /// 恢复并优化边界压力处理
    /// 固壁边界需要压力项保持静水平衡，但质量通量为零
    fn compute_fluxes_serial(&mut self, state: &ShallowWaterState<B>) -> f64 {
        let mut max_wave_speed = 0.0_f64;

        // 处理内部面
        for face_idx in self.mesh.interior_faces() {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face_flux(state, FaceIndex::new(face_idx));

            max_wave_speed = max_wave_speed.max(flux.max_wave_speed.to_f64().unwrap_or(0.0));

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

        // 恢复固壁边界压力处理
        // 静水平衡需要压力项抵消内部梯度，但质量通量保持为零
        let g = self.hydrostatic.g;
        let half = B::Scalar::from_f64(0.5).unwrap();
        
        for face_idx in self.mesh.boundary_faces() {
            let face = FaceIndex::new(face_idx);
            let owner = self.mesh.face_owner(face);
            let normal = self.mesh.face_normal_generic::<B>(face);
            let length_f64 = self.mesh.face_length(face);
            let length = B::Scalar::from_f64(length_f64).unwrap();
            
            let h = state.h[owner.get()];
            
            // 静水压力：F = 0.5 * g * h² * n * L
            // 仅作用于动量，质量通量为零（无穿透）
            let pressure = half * g * h * h * length;
            
            // 压力方向与法向相反（指向内部）
            self.workspace.flux_hu[owner.get()] -= pressure * normal.x();
            self.workspace.flux_hv[owner.get()] -= pressure * normal.y();
            
            // 静水状态下，此压力与内部床坡源项精确抵消
        }

        max_wave_speed
    }

    /// 并行版本同样恢复边界压力处理
    fn compute_fluxes_parallel(&mut self, state: &ShallowWaterState<B>) -> f64 {
        let max_speed_atomic = AtomicU64::new(0u64);

        let face_results: Vec<_> = self.mesh.interior_faces()
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face_flux(state, FaceIndex::new(face_idx));

                let speed_f64 = flux.max_wave_speed.to_f64().unwrap_or(0.0);
                let bits = speed_f64.to_bits();
                max_speed_atomic.fetch_max(bits, Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

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

        // 并行版本同样恢复边界压力
        let g = self.hydrostatic.g;
        let half = B::Scalar::from_f64(0.5).unwrap();
        
        for face_idx in self.mesh.boundary_faces() {
            let face = FaceIndex::new(face_idx);
            let owner = self.mesh.face_owner(face);
            let normal = self.mesh.face_normal_generic::<B>(face);
            let length_f64 = self.mesh.face_length(face);
            let length = B::Scalar::from_f64(length_f64).unwrap();
            
            let h = state.h[owner.get()];
            let pressure = half * g * h * h * length;
            
            self.workspace.flux_hu[owner.get()] -= pressure * normal.x();
            self.workspace.flux_hv[owner.get()] -= pressure * normal.y();
        }

        let bits = max_speed_atomic.load(Ordering::Relaxed);
        f64::from_bits(bits)
    }

    fn compute_face_flux(
        &self,
        state: &ShallowWaterState<B>,
        face_idx: FaceIndex,
    ) -> (RiemannFlux<B::Scalar>, BedSlopeCorrection<B>, B::Scalar, CellIndex, Option<CellIndex>) {
        let normal = self.mesh.face_normal_generic::<B>(face_idx);
        let length_f64 = self.mesh.face_length(face_idx);
        let length = B::Scalar::from_f64(length_f64).unwrap();
        let owner = self.mesh.face_owner(face_idx);
        let neighbor = self.mesh.face_neighbor(face_idx);

        let (h_l, vel_l, z_l, h_r, vel_r, z_r) = if self.use_second_order() {
            let eta_f64: Vec<f64> = self.workspace.eta.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
            let vel_u_f64: Vec<f64> = self.workspace.vel_u.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
            let vel_v_f64: Vec<f64> = self.workspace.vel_v.iter().map(|v| v.to_f64().unwrap_or(0.0)).collect();
            let eta_rec = self.muscl_eta.reconstruct_scalar(face_idx.get(), &eta_f64);
            let u_rec = self.muscl_u.reconstruct_scalar(face_idx.get(), &vel_u_f64);
            let v_rec = self.muscl_v.reconstruct_scalar(face_idx.get(), &vel_v_f64);

            if let Some(neigh) = neighbor {
                let z_owner = state.z[owner.get()];
                let z_neigh = state.z[neigh.get()];
                let z_owner_f64 = z_owner.to_f64().unwrap_or(0.0);
                let z_neigh_f64 = z_neigh.to_f64().unwrap_or(0.0);
                // Well-balanced: 使用统一的 z_face 来计算 h
                let z_face_f64 = z_owner_f64.max(z_neigh_f64);
                let h_left_f64 = (eta_rec.left - z_face_f64).max(0.0_f64);
                let h_right_f64 = (eta_rec.right - z_face_f64).max(0.0_f64);
                let h_left = B::Scalar::from_f64(h_left_f64).unwrap_or(B::Scalar::ZERO);
                let h_right = B::Scalar::from_f64(h_right_f64).unwrap_or(B::Scalar::ZERO);
                let u_l = B::Scalar::from_f64(u_rec.left).unwrap_or(B::Scalar::ZERO);
                let v_l = B::Scalar::from_f64(v_rec.left).unwrap_or(B::Scalar::ZERO);
                let u_r = B::Scalar::from_f64(u_rec.right).unwrap_or(B::Scalar::ZERO);
                let v_r = B::Scalar::from_f64(v_rec.right).unwrap_or(B::Scalar::ZERO);
                (
                    h_left,
                    B::vec2_new(u_l, v_l),
                    z_owner,
                    h_right,
                    B::vec2_new(u_r, v_r),
                    z_neigh,
                )
            } else {
                // 边界面：简化处理，使用单元自身值
                let z_owner = state.z[owner.get()];
                let z_owner_f64 = z_owner.to_f64().unwrap_or(0.0);
                let h_left_f64 = (eta_rec.left - z_owner_f64).max(0.0_f64);
                let h_left = B::Scalar::from_f64(h_left_f64).unwrap_or(B::Scalar::ZERO);
                let u_l = B::Scalar::from_f64(u_rec.left).unwrap_or(B::Scalar::ZERO);
                let v_l = B::Scalar::from_f64(v_rec.left).unwrap_or(B::Scalar::ZERO);
                let vel_left = B::vec2_new(u_l, v_l);
                // 速度反射：vn' = -vn，静水时vn=0
                let vn = B::vec2_dot(&vel_left, &normal);
                let two = B::Scalar::from_f64(2.0).unwrap();
                let vel_right = B::vec2_sub(&vel_left, &B::vec2_scale(&normal, vn * two));
                
                // 边界面两侧使用相同的状态（静水平衡）
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
            // 一阶格式：不使用重构
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
                // 边界面：速度反射
                let vn = B::vec2_dot(&vel_l, &normal);
                let two = B::Scalar::from_f64(2.0).unwrap();
                let vel_r = B::vec2_sub(&vel_l, &B::vec2_scale(&normal, vn * two));
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

        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            let half = B::Scalar::from_f64(0.5).unwrap();
            let z_face = (z_l + z_r) * half;
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face,
            }
        };

        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter: B::Scalar = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => B::Scalar::ZERO,
            (WetState::Dry, _) | (_, WetState::Dry) => B::Scalar::ONE,
            (WetState::PartiallyWet, WetState::PartiallyWet) => {
                let h_min = recon.h_left.min(recon.h_right);
                let fraction = (h_min - self.params.h_dry) 
                    / (self.params.h_wet - self.params.h_dry);
                let one = B::Scalar::ONE;
                let zero = B::Scalar::ZERO;
                if fraction > one { one } else if fraction < zero { zero } else { fraction }
            }
            _ => B::Scalar::ONE,
        };

        let flux = self.riemann.solve(
            recon.h_left,
            recon.h_right,
            recon.vel_left,
            recon.vel_right,
            normal,
        ).unwrap_or_else(|_| RiemannFlux::zero());

        let limited_flux = flux.scaled(flux_limiter);

        let bed_src = self.compute_hydrostatic_bed_slope(
            h_l, h_r, recon.h_left, recon.h_right, normal, length,
        );

        (limited_flux, bed_src, length, owner, neighbor)
    }

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
        let half = B::Scalar::from_f64(0.5).unwrap();
        let g = self.hydrostatic.g;
        
        // 使用精确的压力差计算，静水平衡时为零
        let pressure_diff_l = half * g * (h_l * h_l - h_l_star * h_l_star) * length;
        let pressure_diff_r = half * g * (h_r * h_r - h_r_star * h_r_star) * length;

        BedSlopeCorrection {
            source_left_x: -pressure_diff_l * normal.x(),
            source_left_y: -pressure_diff_l * normal.y(),
            source_right_x: pressure_diff_r * normal.x(),
            source_right_y: pressure_diff_r * normal.y(),
        }
    }

    fn update_state(&self, state: &mut ShallowWaterState<B>, dt: B::Scalar) {
        for i in self.mesh.cells() {
            let area_f64 = self.mesh.cell_area(CellIndex(i)).unwrap_or(1.0_f64);
            let inv_area = B::Scalar::from_f64(1.0 / area_f64).unwrap();

            state.h[i] = state.h[i] + dt * inv_area * self.workspace.flux_h[i];
            state.hu[i] = state.hu[i] + dt * inv_area * 
                (self.workspace.flux_hu[i] + self.workspace.source_hu[i]);
            state.hv[i] = state.hv[i] + dt * inv_area * 
                (self.workspace.flux_hv[i] + self.workspace.source_hv[i]);
        }
    }

    fn enforce_positivity(&mut self, state: &mut ShallowWaterState<B>, _dt: B::Scalar) -> (usize, usize) {
        let h_min = self.params.h_min;
        let h_dry = self.params.h_dry;
        let mut dry_count = 0;
        let mut limited_count = 0;

        for i in self.mesh.cells() {
            if state.h[i] < h_min {
                state.h[i] = B::Scalar::ZERO;
                state.hu[i] = B::Scalar::ZERO;
                state.hv[i] = B::Scalar::ZERO;
                dry_count += 1;
            } else if state.h[i] < h_dry {
                let factor = self.wetting_drying.wet_fraction_smooth(state.h[i]);
                state.hu[i] = state.hu[i] * factor;
                state.hv[i] = state.hv[i] * factor;
                dry_count += 1;
                limited_count += 1;
            }
        }

        (dry_count, limited_count)
    }

    // 访问器
    pub fn mesh(&self) -> &PhysicsMesh { &self.mesh }
    pub fn backend(&self) -> &B { &self.backend }
    pub fn stats(&self) -> &SolverStats { &self.stats }
    pub fn max_wave_speed(&self) -> f64 { self.stats.max_wave_speed }
    pub fn dry_cell_count(&self) -> usize { self.stats.dry_cells }
}

// ============================================================
// 辅助类型定义（引擎层内部使用）
// ============================================================

/// 数值格式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NumericalScheme {
    #[default]
    FirstOrder,
    SecondOrderMuscl,
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

// 类型别名
pub type ShallowWaterSolverF64 = ShallowWaterSolver<CpuBackend<f64>>;
pub type ShallowWaterSolverF32 = ShallowWaterSolver<CpuBackend<f32>>;