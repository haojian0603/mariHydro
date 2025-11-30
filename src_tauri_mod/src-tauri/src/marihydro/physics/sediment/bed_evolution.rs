// src-tauri/src/marihydro/physics/sediment/bed_evolution.rs
//! 床面演变 (Exner 方程)
//! 
//! ∂z_b/∂t = -1/(1-p) × ∇·q_b

use rayon::prelude::*;

use crate::marihydro::core::error::MhResult;
use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;

/// Exner 方程配置
#[derive(Debug, Clone)]
pub struct ExnerConfig {
    /// 床面孔隙率
    pub porosity: f64,
    /// 最小床面变化率 [m/s]
    pub min_rate: f64,
    /// 最大床面变化率 [m/s]
    pub max_rate: f64,
    /// 使用形态加速因子
    pub morphological_factor: f64,
    /// 是否启用崩塌检查
    pub enable_avalanche: bool,
    /// 静止摩擦角 [度]
    pub angle_of_repose: f64,
}

impl Default for ExnerConfig {
    fn default() -> Self {
        Self {
            porosity: 0.4,
            min_rate: -0.01,
            max_rate: 0.01,
            morphological_factor: 1.0,
            enable_avalanche: true,
            angle_of_repose: 32.0,
        }
    }
}

impl ExnerConfig {
    /// 设置形态加速因子
    pub fn with_morphological_factor(mut self, mf: f64) -> Self {
        self.morphological_factor = mf.max(1.0);
        self
    }

    /// 设置孔隙率
    pub fn with_porosity(mut self, p: f64) -> Self {
        self.porosity = p.clamp(0.2, 0.6);
        self
    }
}

/// 床面演变求解器
pub struct BedEvolutionSolver {
    /// 配置
    config: ExnerConfig,
    /// 床面高程 [m]
    bed_level: Vec<f64>,
    /// 床面变化率 [m/s]
    dz_dt: Vec<f64>,
    /// 输沙通量散度
    div_qb: Vec<f64>,
}

impl BedEvolutionSolver {
    /// 创建求解器
    pub fn new(n_cells: usize, initial_bed: &[f64]) -> Self {
        let bed_level = if initial_bed.len() == n_cells {
            initial_bed.to_vec()
        } else {
            vec![0.0; n_cells]
        };

        Self {
            config: ExnerConfig::default(),
            bed_level,
            dz_dt: vec![0.0; n_cells],
            div_qb: vec![0.0; n_cells],
        }
    }

    /// 设置配置
    pub fn with_config(mut self, config: ExnerConfig) -> Self {
        self.config = config;
        self
    }

    /// 计算输沙通量散度
    /// 
    /// div(q_b) ≈ Σ(q_b · n) × L / A
    pub fn compute_flux_divergence(
        &mut self,
        mesh: &UnstructuredMesh,
        qbx: &[f64],
        qby: &[f64],
    ) {
        self.div_qb.fill(0.0);

        // 遍历内部面
        for face_idx in mesh.interior_faces() {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];

            let nx = mesh.face_normal_x[face_idx];
            let ny = mesh.face_normal_y[face_idx];
            let length = mesh.face_length[face_idx];

            // 面心插值
            let qb_face_x = 0.5 * (qbx[owner] + qbx[neighbor]);
            let qb_face_y = 0.5 * (qby[owner] + qby[neighbor]);

            // 通量
            let flux = (qb_face_x * nx + qb_face_y * ny) * length;

            self.div_qb[owner] += flux;
            self.div_qb[neighbor] -= flux;
        }

        // 除以单元面积得到散度
        for i in 0..mesh.n_cells {
            let area = mesh.cell_area[i];
            if area > 1e-14 {
                self.div_qb[i] /= area;
            }
        }
    }

    /// 演进床面一步
    pub fn advance(&mut self, dt: f64) {
        let factor = -1.0 / (1.0 - self.config.porosity);
        let mf = self.config.morphological_factor;
        let min_rate = self.config.min_rate;
        let max_rate = self.config.max_rate;

        self.dz_dt
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, dz)| {
                // Exner 方程: dz/dt = -1/(1-p) × div(q_b)
                *dz = factor * self.div_qb[i];
                
                // 限制变化率
                *dz = dz.clamp(min_rate, max_rate);
            });

        // 更新床面高程
        self.bed_level
            .par_iter_mut()
            .zip(self.dz_dt.par_iter())
            .for_each(|(z, &dz)| {
                *z += dz * dt * mf;
            });
    }

    /// 求解一个完整步
    pub fn solve_step(
        &mut self,
        mesh: &UnstructuredMesh,
        qbx: &[f64],
        qby: &[f64],
        dt: f64,
    ) -> MhResult<()> {
        self.compute_flux_divergence(mesh, qbx, qby);
        self.advance(dt);

        if self.config.enable_avalanche {
            self.apply_avalanche(mesh);
        }

        Ok(())
    }

    /// 应用崩塌（简化的坡度限制）
    fn apply_avalanche(&mut self, mesh: &UnstructuredMesh) {
        let tan_phi = (self.config.angle_of_repose * std::f64::consts::PI / 180.0).tan();
        let max_iterations = 10;

        for _ in 0..max_iterations {
            let mut changed = false;

            // 遍历所有内部面
            for face_idx in mesh.interior_faces() {
                let owner = mesh.face_owner[face_idx];
                let neighbor = mesh.face_neighbor[face_idx];

                let dz = self.bed_level[neighbor] - self.bed_level[owner];
                let dist = mesh.face_dist_o2n[face_idx];

                if dist < 1e-10 {
                    continue;
                }

                let slope = dz.abs() / dist;
                
                if slope > tan_phi {
                    // 需要崩塌
                    let excess = (slope - tan_phi) * dist;
                    let transfer = 0.5 * excess;

                    if dz > 0.0 {
                        self.bed_level[neighbor] -= transfer;
                        self.bed_level[owner] += transfer;
                    } else {
                        self.bed_level[neighbor] += transfer;
                        self.bed_level[owner] -= transfer;
                    }
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// 获取床面高程
    pub fn bed_level(&self) -> &[f64] {
        &self.bed_level
    }

    /// 获取床面变化率
    pub fn bed_change_rate(&self) -> &[f64] {
        &self.dz_dt
    }

    /// 计算总床面变化量
    pub fn total_bed_change(&self, mesh: &UnstructuredMesh) -> f64 {
        self.dz_dt
            .iter()
            .enumerate()
            .map(|(i, &dz)| dz * mesh.cell_area[i])
            .sum()
    }

    /// 更新初始床面
    pub fn set_bed_level(&mut self, levels: &[f64]) {
        let n = self.bed_level.len().min(levels.len());
        self.bed_level[..n].copy_from_slice(&levels[..n]);
    }
}

/// 快捷函数：单步求解
pub fn solve_exner_step(
    bed_level: &mut [f64],
    mesh: &UnstructuredMesh,
    qbx: &[f64],
    qby: &[f64],
    porosity: f64,
    dt: f64,
) -> MhResult<()> {
    let mut solver = BedEvolutionSolver::new(mesh.n_cells, bed_level);
    solver.config.porosity = porosity;
    solver.solve_step(mesh, qbx, qby, dt)?;
    
    let n = bed_level.len();
    bed_level.copy_from_slice(&solver.bed_level()[..n]);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exner_config() {
        let config = ExnerConfig::default()
            .with_morphological_factor(10.0)
            .with_porosity(0.35);
        
        assert!((config.morphological_factor - 10.0).abs() < 1e-10);
        assert!((config.porosity - 0.35).abs() < 1e-10);
    }

    #[test]
    fn test_bed_evolution_solver() {
        let initial_bed = vec![0.0; 100];
        let solver = BedEvolutionSolver::new(100, &initial_bed);
        
        assert_eq!(solver.bed_level().len(), 100);
    }
}
