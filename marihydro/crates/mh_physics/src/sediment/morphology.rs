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
//! let mut solver = MorphodynamicsSolver::new(mesh.n_cells(), config);
//!
//! // 在每个时间步更新河床
//! solver.step(&mut state, &mesh, &qb_x, &qb_y, dt);
//! ```

use crate::adapter::PhysicsMesh;
use crate::state::ShallowWaterState;
use mh_foundation::{AlignedVec, Scalar};
use serde::{Deserialize, Serialize};

/// 河床演变配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyConfig {
    /// 床面孔隙率（默认 0.4）
    pub porosity: Scalar,
    /// 干单元水深阈值 [m]
    pub h_dry: Scalar,
    /// 是否启用崩塌处理
    pub avalanche_enabled: bool,
    /// 湿润时的安息角 [rad]
    pub angle_repose_wet: Scalar,
    /// 干燥时的安息角 [rad]
    pub angle_repose_dry: Scalar,
    /// 最大崩塌迭代次数
    pub max_avalanche_iter: usize,
    /// 崩塌松弛因子（0-1）
    pub avalanche_relaxation: Scalar,
    /// 最大允许床面变化率 [m/s]
    pub max_dz_rate: Scalar,
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
#[derive(Debug, Clone, Default)]
pub struct MorphologyStats {
    /// 最大侵蚀深度 [m]
    pub max_erosion: Scalar,
    /// 最大淤积深度 [m]
    pub max_deposition: Scalar,
    /// 总侵蚀量 [m³]
    pub total_erosion: Scalar,
    /// 总淤积量 [m³]
    pub total_deposition: Scalar,
    /// 崩塌迭代次数
    pub avalanche_iterations: usize,
    /// 发生崩塌的面数
    pub avalanche_faces: usize,
}

/// 河床演变求解器
pub struct MorphodynamicsSolver {
    /// 配置
    config: MorphologyConfig,
    /// 河床变化率 dz/dt [m/s]
    dz_dt: AlignedVec<Scalar>,
    /// 临时存储：通量散度
    flux_divergence: AlignedVec<Scalar>,
    /// 最新统计
    stats: MorphologyStats,
}

impl MorphodynamicsSolver {
    /// 创建新的河床演变求解器
    ///
    /// # 参数
    ///
    /// - `n_cells`: 单元数量
    /// - `config`: 配置
    pub fn new(n_cells: usize, config: MorphologyConfig) -> Self {
        Self {
            config,
            dz_dt: AlignedVec::zeros(n_cells),
            flux_divergence: AlignedVec::zeros(n_cells),
            stats: MorphologyStats::default(),
        }
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
    pub fn stats(&self) -> &MorphologyStats {
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
        state: &mut ShallowWaterState,
        mesh: &PhysicsMesh,
        qb_x: &[Scalar],
        qb_y: &[Scalar],
        dt: Scalar,
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

    /// 仅计算通量散度（不更新状态）
    pub fn compute_divergence(
        &mut self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState,
        qb_x: &[Scalar],
        qb_y: &[Scalar],
    ) -> &[Scalar] {
        self.compute_divergence_upwind(mesh, state, qb_x, qb_y);
        self.dz_dt.as_slice()
    }

    /// 使用迎风格式计算通量散度
    fn compute_divergence_upwind(
        &mut self,
        mesh: &PhysicsMesh,
        state: &ShallowWaterState,
        qb_x: &[Scalar],
        qb_y: &[Scalar],
    ) {
        let factor = 1.0 / (1.0 - self.config.porosity);

        // 清零
        self.dz_dt.as_mut_slice().fill(0.0);
        self.flux_divergence.as_mut_slice().fill(0.0);

        for face_idx in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);

            let normal = mesh.face_normal(face_idx);
            let length = mesh.face_length(face_idx);

            // Owner 的法向通量
            let q_n_owner = qb_x[owner] * normal.x + qb_y[owner] * normal.y;

            // 迎风选择通量
            let q_n = if let Some(neigh) = neighbor {
                let q_n_neigh = qb_x[neigh] * normal.x + qb_y[neigh] * normal.y;
                // 选择上游值
                if q_n_owner + q_n_neigh >= 0.0 {
                    q_n_owner
                } else {
                    q_n_neigh
                }
            } else {
                // 边界面：使用内部值（假设零梯度）
                q_n_owner
            };

            let flux = q_n * length * factor;

            // 累加到通量散度
            let area_o = mesh.cell_area_unchecked(owner);
            self.flux_divergence[owner] += flux / area_o;

            if let Some(neigh) = neighbor {
                let area_n = mesh.cell_area_unchecked(neigh);
                self.flux_divergence[neigh] -= flux / area_n;
            }
        }

        // dz/dt = -div(q) / (1-p)
        for i in 0..state.n_cells() {
            self.dz_dt[i] = -self.flux_divergence[i];
        }
    }

    /// 强耦合更新河床和水深
    fn update_bed_coupled(&mut self, state: &mut ShallowWaterState, mesh: &PhysicsMesh, dt: Scalar) {
        let max_dz = self.config.max_dz_rate * dt;

        for i in 0..state.n_cells() {
            let mut dz = self.dz_dt[i] * dt;

            // 限制变化率
            dz = dz.clamp(-max_dz, max_dz);

            if dz.abs() < 1e-14 {
                continue;
            }

            let z_old = state.z[i];
            let h_old = state.h[i];
            let eta = z_old + h_old; // 水位

            // 更新河床
            let z_new = z_old + dz;

            // 保持水位不变，调整水深
            let h_new = (eta - z_new).max(0.0);

            // 淤积超过水深时的限制处理
            if dz > h_old && h_old > self.config.h_dry {
                // 河床升至水面
                state.z[i] = eta;
                state.h[i] = 0.0;
                log::trace!(
                    "Cell {} deposition limited: dz={:.4}, h={:.4}",
                    i,
                    dz,
                    h_old
                );
            } else {
                state.z[i] = z_new;
                state.h[i] = h_new;
            }

            // 更新统计
            let area = mesh.cell_area_unchecked(i);
            if dz < 0.0 {
                self.stats.max_erosion = self.stats.max_erosion.max(-dz);
                self.stats.total_erosion += -dz * area;
            } else {
                self.stats.max_deposition = self.stats.max_deposition.max(dz);
                self.stats.total_deposition += dz * area;
            }
        }
    }

    /// 应用崩塌处理
    ///
    /// 当相邻单元间坡度超过安息角时，进行泥沙重分布
    fn apply_avalanche(&mut self, state: &mut ShallowWaterState, mesh: &PhysicsMesh) {
        let mut total_faces = 0;

        for iter in 0..self.config.max_avalanche_iter {
            let mut changed = false;

            for face_idx in mesh.interior_faces() {
                let owner = mesh.face_owner(face_idx);
                // SAFETY: interior_faces() guarantees neighbor exists
                let neigh = mesh
                    .face_neighbor(face_idx)
                    .expect("interior face must have neighbor");

                let dist = mesh.face_dist_o2n(face_idx);
                if dist < 1e-10 {
                    continue;
                }

                let dz = state.z[neigh] - state.z[owner];
                let slope = dz.abs() / dist;

                // 根据干湿选择安息角
                let is_wet =
                    state.h[owner] > self.config.h_dry || state.h[neigh] > self.config.h_dry;
                let max_slope = if is_wet {
                    self.config.angle_repose_wet.tan()
                } else {
                    self.config.angle_repose_dry.tan()
                };

                if slope > max_slope {
                    let target_dz = max_slope * dist * dz.signum();
                    let correction = (dz - target_dz) * self.config.avalanche_relaxation;

                    // 质量守恒的重分布
                    let area_owner = mesh.cell_area_unchecked(owner);
                    let area_neigh = mesh.cell_area_unchecked(neigh);
                    let total_area = area_owner + area_neigh;

                    // 按面积加权分配
                    let dz_owner = correction * area_neigh / total_area;
                    let dz_neigh = -correction * area_owner / total_area;

                    state.z[owner] += dz_owner;
                    state.z[neigh] += dz_neigh;

                    // 保持水位，调整水深
                    state.h[owner] = (state.h[owner] - dz_owner).max(0.0);
                    state.h[neigh] = (state.h[neigh] - dz_neigh).max(0.0);

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
    pub fn dz_dt(&self) -> &[Scalar] {
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
