// src-tauri/src/marihydro/physics/engine/solver_v2.rs

//! 改进版求解器
//!
//! 集成新的数值子系统：
//! - 统一的干湿处理
//! - 多黎曼求解器支持
//! - 隐式源项处理
//!
//! # 关联问题
//!
//! - P1-001, P1-002: Solver集成干湿处理
//! - P1-022: 床坡源项处理

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContext, SourceTerm};
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use crate::marihydro::core::Workspace;
use crate::marihydro::domain::mesh::UnstructuredMesh;
use crate::marihydro::domain::state::ShallowWaterState;
use crate::marihydro::physics::schemes::{
    riemann::{HllcSolver, RiemannFlux, RiemannSolver, SolverParams},
    wetting_drying::{WetState, WettingDryingHandler},
    HydrostaticReconstruction,
};
use crate::marihydro::physics::sources::implicit::{ImplicitConfig, ImplicitMomentumDecay, ManningDamping};
use super::timestep::TimeStepController;
use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// 求解器配置
#[derive(Debug, Clone)]
pub struct SolverConfig {
    /// 数值参数
    pub params: NumericalParams,
    /// 重力加速度
    pub gravity: f64,
    /// 是否启用隐式摩擦
    pub implicit_friction: bool,
    /// 是否使用静水重构
    pub use_hydrostatic_reconstruction: bool,
    /// 并行化阈值（单元数）
    pub parallel_threshold: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            params: NumericalParams::default(),
            gravity: 9.81,
            implicit_friction: true,
            use_hydrostatic_reconstruction: true,
            parallel_threshold: 1000,
        }
    }
}

/// 改进版非结构化求解器
pub struct UnstructuredSolverV2 {
    mesh: Arc<UnstructuredMesh>,
    config: SolverConfig,

    // 工作区
    workspace: Workspace,

    // 子系统
    riemann: HllcSolver,
    dry_wet: WettingDryingHandler,
    hydrostatic: HydrostaticReconstruction,
    implicit_decay: ImplicitMomentumDecay,

    // 时间步控制
    timestep_ctrl: TimeStepController,

    // 源项
    source_terms: Vec<Box<dyn SourceTerm>>,

    // 统计
    max_wave_speed: f64,
    dry_cells: usize,
    limited_faces: usize,
}

impl UnstructuredSolverV2 {
    /// 创建新求解器
    pub fn new(mesh: Arc<UnstructuredMesh>, config: SolverConfig) -> Self {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();

        let solver_params = SolverParams::from_numerical(&config.params, config.gravity);

        Self {
            mesh: mesh.clone(),
            config: config.clone(),
            workspace: Workspace::new(n_cells, n_faces),
            riemann: HllcSolver::from_params(solver_params),
            dry_wet: WettingDryingHandler::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::new(&config.params, config.gravity),
            implicit_decay: ImplicitMomentumDecay::new(ImplicitConfig::default()),
            timestep_ctrl: TimeStepController::new(config.gravity, &config.params),
            source_terms: Vec::new(),
            max_wave_speed: 0.0,
            dry_cells: 0,
            limited_faces: 0,
        }
    }

    /// 添加源项
    pub fn add_source_term(&mut self, source: Box<dyn SourceTerm>) {
        self.source_terms.push(source);
    }

    /// 执行一个时间步
    pub fn step(&mut self, state: &mut ShallowWaterState, dt: f64) -> MhResult<f64> {
        if state.n_cells() != self.mesh.n_cells() {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "state",
                self.mesh.n_cells(),
                state.n_cells(),
            ));
        }

        // 1. 重置工作区
        self.workspace.reset_fluxes();
        self.workspace.reset_sources();

        // 2. 计算通量（集成干湿处理）
        self.max_wave_speed = self.compute_fluxes_with_dry_wet(state)?;

        // 3. 计算源项
        self.apply_source_terms(state, dt)?;

        // 4. 更新状态
        self.update_state(state, dt);

        // 5. 强制正性并处理干区动量
        self.enforce_positivity_and_dry_wet(state, dt);

        Ok(self.max_wave_speed)
    }

    /// 计算通量（带干湿处理）
    fn compute_fluxes_with_dry_wet(&mut self, state: &ShallowWaterState) -> MhResult<f64> {
        let n_faces = self.mesh.n_faces();
        let max_speed = AtomicU64::new(0u64);
        let limited_count = std::sync::atomic::AtomicUsize::new(0);

        // 并行计算每个面的通量
        let face_results: Vec<(RiemannFlux, DVec2, f64)> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let face = FaceIndex(face_idx);
                let owner = self.mesh.face_owner(face);
                let neighbor = self.mesh.face_neighbor(face);
                let normal = self.mesh.face_normal(face);
                let length = self.mesh.face_length(face);

                // 获取左侧状态
                let h_l = state.h[owner.0];
                let z_l = state.z[owner.0];
                let vel_l = self.dry_wet.safe_velocity(
                    state.hu[owner.0],
                    state.hv[owner.0],
                    h_l,
                );

                // 获取右侧状态（边界或邻居）
                let (h_r, vel_r, z_r) = if neighbor.is_valid() {
                    let h = state.h[neighbor.0];
                    let v = self.dry_wet.safe_velocity(
                        state.hu[neighbor.0],
                        state.hv[neighbor.0],
                        h,
                    );
                    (h, v, state.z[neighbor.0])
                } else {
                    // 反射边界
                    let vn = vel_l.dot(normal);
                    let vel_ghost = vel_l - 2.0 * vn * normal;
                    (h_l, vel_ghost, z_l)
                };

                // 静水重构
                let recon = if self.config.use_hydrostatic_reconstruction {
                    self.hydrostatic.reconstruct_face_simple(
                        h_l, h_r, z_l, z_r, vel_l, vel_r,
                    )
                } else {
                    crate::marihydro::physics::schemes::hydrostatic::HydrostaticFaceState {
                        h_left: h_l,
                        h_right: h_r,
                        vel_left: vel_l,
                        vel_right: vel_r,
                    }
                };

                // 干湿界面通量限制
                let flux_limiter = self.dry_wet.face_flux_limiter(recon.h_left, recon.h_right);
                if flux_limiter < 1.0 {
                    limited_count.fetch_add(1, Ordering::Relaxed);
                }

                // 求解黎曼问题
                let mut flux = self.riemann
                    .solve(
                        recon.h_left,
                        recon.h_right,
                        recon.vel_left,
                        recon.vel_right,
                        normal,
                    )
                    .unwrap_or(RiemannFlux::ZERO);

                // 应用干湿限制
                if flux_limiter < 1.0 {
                    flux = flux.scaled(flux_limiter);
                }

                // 床坡源项
                let bed_source = self.hydrostatic.bed_slope_correction(
                    h_l, h_r, z_l, z_r, normal, length,
                );

                // 更新最大波速
                let spd_bits = flux.max_wave_speed.to_bits();
                max_speed.fetch_max(spd_bits, Ordering::Relaxed);

                (flux, bed_source.as_vec(), length)
            })
            .collect();

        // 累加通量到单元
        self.workspace.flux_h.fill(0.0);
        self.workspace.flux_hu.fill(0.0);
        self.workspace.flux_hv.fill(0.0);

        for (face_idx, (flux, bed_src, length)) in face_results.iter().enumerate() {
            let face = FaceIndex(face_idx);
            let owner = self.mesh.face_owner(face);
            let neighbor = self.mesh.face_neighbor(face);

            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            // 所有者单元（通量流出为负）
            self.workspace.flux_h[owner.0] -= fh;
            self.workspace.flux_hu[owner.0] -= fhu;
            self.workspace.flux_hv[owner.0] -= fhv;

            // 床坡源项
            self.workspace.source_hu[owner.0] += bed_src.x;
            self.workspace.source_hv[owner.0] += bed_src.y;

            // 邻居单元（通量流入为正）
            if neighbor.is_valid() {
                self.workspace.flux_h[neighbor.0] += fh;
                self.workspace.flux_hu[neighbor.0] += fhu;
                self.workspace.flux_hv[neighbor.0] += fhv;

                self.workspace.source_hu[neighbor.0] -= bed_src.x;
                self.workspace.source_hv[neighbor.0] -= bed_src.y;
            }
        }

        self.limited_faces = limited_count.load(Ordering::Relaxed);

        Ok(f64::from_bits(max_speed.load(Ordering::Relaxed)))
    }

    /// 应用源项
    fn apply_source_terms(&mut self, state: &ShallowWaterState, dt: f64) -> MhResult<()> {
        let ctx = SourceContext {
            time: 0.0,
            dt,
            params: &self.config.params,
            workspace: &self.workspace,
        };

        for source in &self.source_terms {
            source.compute_all(
                self.mesh.as_ref(),
                state,
                &ctx,
                &mut self.workspace.source_h,
                &mut self.workspace.source_hu,
                &mut self.workspace.source_hv,
            )?;
        }

        Ok(())
    }

    /// 更新状态
    fn update_state(&self, state: &mut ShallowWaterState, dt: f64) {
        let n = state.n_cells();
        for i in 0..n {
            let area = self.mesh.cell_area(CellIndex(i));
            let inv_area = 1.0 / area;

            state.h[i] += dt * inv_area * self.workspace.flux_h[i];
            state.hu[i] += dt * inv_area * (self.workspace.flux_hu[i] + self.workspace.source_hu[i]);
            state.hv[i] += dt * inv_area * (self.workspace.flux_hv[i] + self.workspace.source_hv[i]);
        }
    }

    /// 强制正性并处理干区动量
    fn enforce_positivity_and_dry_wet(&mut self, state: &mut ShallowWaterState, dt: f64) {
        let h_min = self.config.params.h_min;
        let h_dry = self.config.params.h_dry;
        let mut dry_count = 0;

        for i in 0..state.n_cells() {
            // 修正负水深
            if state.h[i] < h_min {
                state.h[i] = 0.0;
                state.hu[i] = 0.0;
                state.hv[i] = 0.0;
                dry_count += 1;
            }
            // 干湿过渡动量处理
            else if state.h[i] < h_dry {
                let factor = self.dry_wet.smoothing_factor(state.h[i]);
                state.hu[i] *= factor;
                state.hv[i] *= factor;
                dry_count += 1;
            }
        }

        self.dry_cells = dry_count;
    }

    /// 计算时间步长
    pub fn compute_dt(&mut self, state: &ShallowWaterState) -> f64 {
        self.timestep_ctrl.update(state, self.mesh.as_ref(), &self.config.params)
    }

    // ================= 访问器 =================

    pub fn max_wave_speed(&self) -> f64 {
        self.max_wave_speed
    }

    pub fn mesh(&self) -> &UnstructuredMesh {
        &self.mesh
    }

    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    pub fn dry_cell_count(&self) -> usize {
        self.dry_cells
    }

    pub fn limited_face_count(&self) -> usize {
        self.limited_faces
    }
}

/// 求解器构建器
pub struct SolverBuilderV2 {
    mesh: Option<Arc<UnstructuredMesh>>,
    config: SolverConfig,
    sources: Vec<Box<dyn SourceTerm>>,
}

impl SolverBuilderV2 {
    pub fn new() -> Self {
        Self {
            mesh: None,
            config: SolverConfig::default(),
            sources: Vec::new(),
        }
    }

    pub fn mesh(mut self, mesh: Arc<UnstructuredMesh>) -> Self {
        self.mesh = Some(mesh);
        self
    }

    pub fn config(mut self, config: SolverConfig) -> Self {
        self.config = config;
        self
    }

    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.gravity = g;
        self
    }

    pub fn implicit_friction(mut self, enable: bool) -> Self {
        self.config.implicit_friction = enable;
        self
    }

    pub fn add_source(mut self, source: Box<dyn SourceTerm>) -> Self {
        self.sources.push(source);
        self
    }

    pub fn build(self) -> MhResult<UnstructuredSolverV2> {
        let mesh = self
            .mesh
            .ok_or_else(|| crate::marihydro::core::MhError::missing_config("mesh"))?;

        let mut solver = UnstructuredSolverV2::new(mesh, self.config);
        for source in self.sources {
            solver.add_source_term(source);
        }

        Ok(solver)
    }
}

impl Default for SolverBuilderV2 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 测试需要模拟网格，这里只测试配置
    #[test]
    fn test_solver_config() {
        let config = SolverConfig::default();
        assert_eq!(config.gravity, 9.81);
        assert!(config.implicit_friction);
    }

    #[test]
    fn test_builder() {
        let builder = SolverBuilderV2::new()
            .gravity(10.0)
            .implicit_friction(false);

        assert!((builder.config.gravity - 10.0).abs() < 1e-10);
        assert!(!builder.config.implicit_friction);
    }
}
