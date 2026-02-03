// crates/mh_physics/src/sediment/morphology.rs

//! 河床演变求解器
//!
//! 基于 Exner 方程的河床变形计算，包括：
//! - 推移质通量散度计算
//! - 河床与水深的强耦合更新
//! - 崩塌处理（超过安息角时的重分布）
//!
//! # 基本方程
//!
//! Exner 方程：
//! $$\frac{\partial z_b}{\partial t} + \frac{1}{1-p} \nabla \cdot \vec{q}_b = 0$$
//!
//! 其中：
//! - $z_b$: 床面高程
//! - $p$: 床面孔隙率
//! - $\vec{q}_b$: 推移质输沙率向量
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::sediment::morphology::{MorphodynamicsSolver, MorphologyConfig};
//!
//! let config = MorphologyConfig::default();
//! let mut solver = MorphodynamicsSolver::new(mesh.cell_count(), config);
//!
//! // 在每个时间步更新河床
//! solver.step(&mut state, &mesh, &qb_x, &qb_y, dt);
//! ```

use crate::adapter::{CellIndex, FaceIndex, PhysicsMesh};
use crate::core::Backend;
use crate::state::ShallowWaterState;
use mh_runtime::{DeviceBuffer, RuntimeScalar, Vector2D};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};

/// 河床演变配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyConfig {
    /// 床面孔隙率（默认 0.4）
    pub porosity: f64,
    /// 干单元水深阈值 [m]
    pub h_dry: f64,
    /// 是否启用崩塌处理
    pub avalanche_enabled: bool,
    /// 湿润时的安息角 [rad]
    pub angle_repose_wet: f64,
    /// 干燥时的安息角 [rad]
    pub angle_repose_dry: f64,
    /// 最大崩塌迭代次数
    pub max_avalanche_iter: usize,
    /// 崩塌松弛因子（0-1）
    pub avalanche_relaxation: f64,
    /// 最大允许床面变化率 [m/s]
    pub max_dz_rate: f64,
}

impl Default for MorphologyConfig {
    fn default() -> Self {
        Self {
            porosity: 0.4,
            h_dry: 1e-4,
            avalanche_enabled: true,
            angle_repose_wet: 30.0_f64.to_radians(),
            angle_repose_dry: 34.0_f64.to_radians(),
            max_avalanche_iter: 5,
            avalanche_relaxation: 0.5,
            max_dz_rate: 1.0,
        }
    }
}

impl MorphologyConfig {
    /// 创建禁用崩塌的配置
    pub fn no_avalanche() -> Self {
        Self {
            avalanche_enabled: false,
            ..Default::default()
        }
    }

    /// 创建高孔隙率配置（粗砂/砾石）
    pub fn coarse_sediment() -> Self {
        Self {
            porosity: 0.35,
            angle_repose_wet: 35.0_f64.to_radians(),
            angle_repose_dry: 40.0_f64.to_radians(),
            ..Default::default()
        }
    }

    /// 创建低孔隙率配置（细砂/粉砂）
    pub fn fine_sediment() -> Self {
        Self {
            porosity: 0.45,
            angle_repose_wet: 25.0_f64.to_radians(),
            angle_repose_dry: 30.0_f64.to_radians(),
            ..Default::default()
        }
    }
}

/// 河床演变统计
#[derive(Debug, Clone)]
pub struct MorphologyStats<S> {
    /// 最大侵蚀深度 [m]
    pub max_erosion: S,
    /// 最大淤积深度 [m]
    pub max_deposition: S,
    /// 总侵蚀量 [m³]
    pub total_erosion: S,
    /// 总淤积量 [m³]
    pub total_deposition: S,
    /// 崩塌迭代次数
    pub avalanche_iterations: usize,
    /// 发生崩塌的面数
    pub avalanche_faces: usize,
}

impl<S: RuntimeScalar> Default for MorphologyStats<S> {
    fn default() -> Self {
        Self {
            max_erosion: S::ZERO,
            max_deposition: S::ZERO,
            total_erosion: S::ZERO,
            total_deposition: S::ZERO,
            avalanche_iterations: 0,
            avalanche_faces: 0,
        }
    }
}

/// 河床演变求解器
pub struct MorphodynamicsSolver<B: Backend> {
    /// 配置
    config: MorphologyConfig,
    /// 河床变化率 dz/dt [m/s]
    dz_dt: B::Buffer<B::Scalar>,
    /// 临时存储：通量散度
    flux_divergence: B::Buffer<B::Scalar>,
    /// 最新统计
    stats: MorphologyStats<B::Scalar>,
    /// 后端（保留以支持后续缓冲区分配）
    backend: B,
}

impl<B> MorphodynamicsSolver<B>
where
    B: Backend + Clone,
    B::Scalar: RuntimeScalar + Float + FromPrimitive,
{
    /// 创建新的河床演变求解器
    ///
    /// # 参数
    ///
    /// - `n_cells`: 单元数量
    /// - `config`: 配置
    pub fn new_with_backend(backend: B, n_cells: usize, config: MorphologyConfig) -> Self {
        Self {
            config,
            dz_dt: backend.alloc_init(n_cells, B::Scalar::ZERO),
            flux_divergence: backend.alloc_init(n_cells, B::Scalar::ZERO),
            stats: MorphologyStats::default(),
            backend,
        }
    }

    /// 便捷构造（使用默认后端）
    pub fn new(n_cells: usize, config: MorphologyConfig) -> Self
    where
        B: Default,
    {
        Self::new_with_backend(B::default(), n_cells, config)
    }

    /// 获取配置引用
    pub fn config(&self) -> &MorphologyConfig {
        &self.config
    }

    /// 获取可变配置引用
    pub fn config_mut(&mut self) -> &mut MorphologyConfig {
        &mut self.config
    }

    /// 获取最新统计
    pub fn stats(&self) -> &MorphologyStats<B::Scalar> {
        &self.stats
    }

    /// 执行河床更新（强耦合）
    ///
    /// # 参数
    ///
    /// - `state`: 浅水状态（将被修改）
    /// - `mesh`: 物理网格
    /// - `qb_x`: 推移质输沙率 x 分量 [m²/s]
    /// - `qb_y`: 推移质输沙率 y 分量 [m²/s]
    /// - `dt`: 时间步长 [s]
    pub fn step(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        qb_x: &B::Buffer<B::Scalar>,
        qb_y: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) {
        // 重置统计
        self.stats = MorphologyStats::default();

        // 1. 计算推移质通量散度（迎风格式）
        self.compute_divergence_upwind(mesh, state, qb_x, qb_y);

        // 2. 强耦合更新河床和水深
        self.update_bed_coupled(state, mesh, dt);

        // 3. 崩塌处理
        if self.config.avalanche_enabled {
            self.apply_avalanche(state, mesh);
        }
    }

    /// 半隐式河床更新（Picard 迭代）
    ///
    /// 使用 Picard 迭代求解非线性床变方程，适用于大时间步长。
    ///
    /// # 参数
    ///
    /// - `state`: 浅水状态（将被修改）
    /// - `mesh`: 物理网格
    /// - `compute_transport`: 计算输沙率的闭包
    /// - `dt`: 时间步长 [s]
    /// - `max_iter`: 最大迭代次数
    /// - `tol`: 收敛容差
    pub fn step_semi_implicit<F>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        compute_transport: F,
        dt: B::Scalar,
        max_iter: usize,
        tol: B::Scalar,
    ) -> usize
    where
        F: Fn(&ShallowWaterState<B>) -> (B::Buffer<B::Scalar>, B::Buffer<B::Scalar>),
    {
        // 重置统计
        self.stats = MorphologyStats::default();

        // 保存初始河床
        let z_old: Vec<B::Scalar> = state.z.as_slice().to_vec();
        let mut z_prev = z_old.clone();
        let mut iterations = 0;

        for iter in 0..max_iter {
            // 计算当前状态下的输沙率
            let (qb_x, qb_y) = compute_transport(state);

            // 计算通量散度
            self.compute_divergence_upwind(mesh, state, &qb_x, &qb_y);

            // 临时更新河床
            let factor = self.backend.scalar_from_f64(1.0 / (1.0 - self.config.porosity));
            let max_dz = self.backend.scalar_from_f64(self.config.max_dz_rate) * dt;

            let mut max_change = B::Scalar::ZERO;

            for i in 0..state.n_cells() {
                // Picard: z^{n+1} = z^n - dt * div(q^{k})
                let mut dz = -self.flux_divergence[i] * dt * factor;
                dz = dz.clamp(-max_dz, max_dz);

                let z_new = z_old[i] + dz;

                // 干湿边界约束
                let eta = z_old[i] + state.h[i];
                let h_dry = self.backend.scalar_from_f64(self.config.h_dry);
                if z_new > eta && state.h[i] > h_dry {
                    // 不能高于水面
                    state.z[i] = eta;
                    state.h[i] = B::Scalar::ZERO;
                } else {
                    state.z[i] = z_new;
                    state.h[i] = (eta - z_new).max(B::Scalar::ZERO);
                }

                // 检查收敛
                let change = (state.z[i] - z_prev[i]).abs();
                max_change = max_change.max(change);
                z_prev[i] = state.z[i];
            }

            iterations = iter + 1;

            // 绝对收敛准则
            if max_change < tol {
                break;
            }
        }

        // 崩塌处理
        if self.config.avalanche_enabled {
            self.apply_avalanche(state, mesh);
        }

        iterations
    }

    /// 计算输沙率雅可比矩阵对角元（用于隐式求解）
    pub fn compute_jacobian_diagonal(
        &self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState<B>,
        qb_x: &B::Buffer<B::Scalar>,
        qb_y: &B::Buffer<B::Scalar>,
    ) -> B::Buffer<B::Scalar> {
        let n = state.n_cells();
        let mut jacobian = self.backend.alloc_init(n, B::Scalar::ZERO);
        let _eps = self.backend.scalar_from_f64(1e-6);

        for i in 0..n {
            let q_mag = (qb_x[i] * qb_x[i] + qb_y[i] * qb_y[i]).sqrt();
            if q_mag < _eps {
                continue;
            }

            let h = state.h[i];
            let h_dry = self.backend.scalar_from_f64(self.config.h_dry);
            if h > h_dry {
                let coeff = self.backend.scalar_from_f64(-1.5);
                jacobian[i] = coeff * q_mag / h;
            }
        }

        let _ = mesh.cell_count();
        jacobian
    }

    /// 仅计算通量散度（不更新状态）
    pub fn compute_divergence(
        &mut self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState<B>,
        qb_x: &[B::Scalar],
        qb_y: &[B::Scalar],
    ) -> &[B::Scalar] {
        self.compute_divergence_upwind(mesh, state, qb_x, qb_y);
        self.dz_dt.as_slice()
    }

    /// 使用迎风格式计算通量散度
    fn compute_divergence_upwind(
        &mut self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState<B>,
        qb_x: &[B::Scalar],
        qb_y: &[B::Scalar],
    ) {
        let factor = self.backend.scalar_from_f64(1.0 / (1.0 - self.config.porosity));

        // 清零
        for v in self.dz_dt.as_slice_mut() {
            *v = B::Scalar::ZERO;
        }
        for v in self.flux_divergence.as_slice_mut() {
            *v = B::Scalar::ZERO;
        }

        for face_idx in 0..mesh.face_count() {
            let fi = FaceIndex::new(face_idx);
            let owner_ci = mesh.face_owner(fi);
            let neighbor_ci = mesh.face_neighbor(fi);

            let owner: usize = owner_ci.get();
            let neighbor = neighbor_ci.map(|c| c.get());

            let normal = mesh
                .face_normal_generic::<B>(fi)
                .expect("face_normal out of range");
            let nx = normal.x();
            let ny = normal.y();
            let length = self.backend.scalar_from_f64(mesh.face_length(fi));

            // Owner 的法向通量
            let q_n_owner = qb_x[owner] * nx + qb_y[owner] * ny;

            // 迎风选择通量
            let q_n = if let Some(neigh) = neighbor {
                let q_n_neigh = qb_x[neigh] * nx + qb_y[neigh] * ny;
                if q_n_owner + q_n_neigh >= B::Scalar::ZERO {
                    q_n_owner
                } else {
                    q_n_neigh
                }
            } else {
                q_n_owner
            };

            let flux = q_n * length * factor;

            // 累加到通量散度
            let area_o = self.backend.scalar_from_f64(mesh.cell_area_unchecked(owner_ci));
            self.flux_divergence[owner] += flux / area_o;

            if let Some(neigh) = neighbor_ci {
                let neigh_idx: usize = neigh.get();
                let area_n = self.backend.scalar_from_f64(mesh.cell_area_unchecked(neigh));
                self.flux_divergence[neigh_idx] -= flux / area_n;
            }
        }

        // dz/dt = -div(q) / (1-p)
        for i in 0..state.n_cells() {
            self.dz_dt[i] = -self.flux_divergence[i];
        }
    }

    /// 强耦合更新河床和水深
    fn update_bed_coupled(&mut self, state: &mut ShallowWaterState<B>, mesh: &PhysicsMesh, dt: B::Scalar) {
        let max_dz = self.backend.scalar_from_f64(self.config.max_dz_rate) * dt;

        for i in 0..state.n_cells() {
            let mut dz = self.dz_dt[i] * dt;

            // 限制变化率
            dz = dz.clamp(-max_dz, max_dz);

            let eps = self.backend.scalar_from_f64(1e-14);
            if dz.abs() < eps {
                continue;
            }

            let z_old = state.z[i];
            let h_old = state.h[i];
            let eta = z_old + h_old;

            let z_new = z_old + dz;
            let h_new = (eta - z_new).max(B::Scalar::ZERO);

            let h_dry = self.backend.scalar_from_f64(self.config.h_dry);
            if dz > h_old && h_old > h_dry {
                state.z[i] = eta;
                state.h[i] = B::Scalar::ZERO;
            } else {
                state.z[i] = z_new;
                state.h[i] = h_new;
            }

            // 更新统计
            let area = self.backend.scalar_from_f64(mesh.cell_area_unchecked(CellIndex::new(i)));
            if dz < B::Scalar::ZERO {
                self.stats.max_erosion = self.stats.max_erosion.max(-dz);
                self.stats.total_erosion += -dz * area;
            } else {
                self.stats.max_deposition = self.stats.max_deposition.max(dz);
                self.stats.total_deposition += dz * area;
            }
        }
    }

    /// 应用崩塌处理
    fn apply_avalanche(&mut self, state: &mut ShallowWaterState<B>, mesh: &PhysicsMesh) {
        let mut total_faces = 0;

        for iter in 0..self.config.max_avalanche_iter {
            let mut changed = false;

            for fi in mesh.interior_face_indices() {
                let owner_ci = mesh.face_owner(fi);
                let neigh_ci = mesh
                    .face_neighbor(fi)
                    .expect("interior face must have neighbor");

                let owner = owner_ci.get();
                let neigh = neigh_ci.get();

                let dist = mesh.face_dist_o2n(fi);
                let dist_s = self.backend.scalar_from_f64(dist);
                if dist_s <= self.backend.scalar_from_f64(1e-10) {
                    continue;
                }

                let dz = state.z[neigh] - state.z[owner];
                let slope = dz.abs() / dist_s;

                let h_dry = self.backend.scalar_from_f64(self.config.h_dry);
                let is_wet = state.h[owner] > h_dry || state.h[neigh] > h_dry;
                let max_slope = if is_wet {
                    self.backend.scalar_from_f64(self.config.angle_repose_wet.tan())
                } else {
                    self.backend.scalar_from_f64(self.config.angle_repose_dry.tan())
                };

                if slope > max_slope {
                    let target_dz = max_slope * dist_s * dz.signum();
                    let correction = (dz - target_dz) * self.backend.scalar_from_f64(self.config.avalanche_relaxation);

                    let area_owner = self.backend.scalar_from_f64(mesh.cell_area_unchecked(owner_ci));
                    let area_neigh = self.backend.scalar_from_f64(mesh.cell_area_unchecked(neigh_ci));
                    let total_area = area_owner + area_neigh;

                    let dz_owner = correction * area_neigh / total_area;
                    let dz_neigh = -correction * area_owner / total_area;

                    state.z[owner] += dz_owner;
                    state.z[neigh] += dz_neigh;

                    state.h[owner] = (state.h[owner] - dz_owner).max(B::Scalar::ZERO);
                    state.h[neigh] = (state.h[neigh] - dz_neigh).max(B::Scalar::ZERO);

                    changed = true;
                    total_faces += 1;
                }
            }

            self.stats.avalanche_iterations = iter + 1;

            if !changed {
                break;
            }
        }

        self.stats.avalanche_faces = total_faces;
    }

    /// 获取河床变化率场引用
    pub fn dz_dt(&self) -> &[B::Scalar] {
        self.dz_dt.as_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 注意：完整测试需要 PhysicsMesh，这里只测试配置
    #[test]
    fn test_config_defaults() {
        let config = MorphologyConfig::default();
        assert!((config.porosity - 0.4).abs() < 1e-10);
        assert!(config.avalanche_enabled);
    }

    #[test]
    fn test_config_variants() {
        let coarse = MorphologyConfig::coarse_sediment();
        let fine = MorphologyConfig::fine_sediment();

        assert!(coarse.porosity < fine.porosity);
        assert!(coarse.angle_repose_dry > fine.angle_repose_dry);
    }

    #[test]
    fn test_no_avalanche_config() {
        let config = MorphologyConfig::no_avalanche();
        assert!(!config.avalanche_enabled);
    }
}
