// crates/mh_physics/src/sediment/bed_evolution.rs

//! 床面演变计算
//! 基于 Exner 方程的床面高程变化求解器

use super::properties::SedimentProperties;
use serde::{Deserialize, Serialize};

/// Exner 方程求解器配置
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExnerConfig {
    /// 沉降孔隙率（默认0.4）
    pub porosity: f64,
    /// 最小水深限制（低于此值不计算床面变化）
    pub min_depth: f64,
    /// 最大床面变化率限制（用于数值稳定性）
    pub max_bed_change_rate: f64,
    /// 是否启用边坡崩塌
    pub enable_avalanche: bool,
    /// 边坡临界角度（弧度）
    pub critical_slope: f64,
    /// 松弛系数（用于稳定性）
    pub relaxation_factor: f64,
}

impl Default for ExnerConfig {
    fn default() -> Self {
        Self {
            porosity: 0.4,
            min_depth: 0.01,
            max_bed_change_rate: 0.01, // 每秒最大1cm变化
            enable_avalanche: true,
            critical_slope: 0.58, // 约33度
            relaxation_factor: 0.8,
        }
    }
}

impl ExnerConfig {
    /// 创建默认配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置孔隙率
    pub fn with_porosity(mut self, porosity: f64) -> Self {
        self.porosity = porosity.clamp(0.2, 0.6);
        self
    }

    /// 设置边坡临界角度（度）
    pub fn with_critical_slope_degrees(mut self, degrees: f64) -> Self {
        self.critical_slope = degrees.to_radians().tan();
        self
    }

    /// 启用边坡崩塌
    pub fn with_avalanche(mut self, enable: bool) -> Self {
        self.enable_avalanche = enable;
        self
    }
}

/// 床面演变求解器
pub struct BedEvolutionSolver {
    /// 配置
    config: ExnerConfig,
    /// 床面高程
    zb: Vec<f64>,
    /// 床面变化速率
    dzb_dt: Vec<f64>,
    /// 泥沙通量 x 分量
    flux_x: Vec<f64>,
    /// 泥沙通量 y 分量
    flux_y: Vec<f64>,
}

impl BedEvolutionSolver {
    /// 创建新的求解器
    pub fn new(n_cells: usize, config: ExnerConfig) -> Self {
        Self {
            config,
            zb: vec![0.0; n_cells],
            dzb_dt: vec![0.0; n_cells],
            flux_x: vec![0.0; n_cells],
            flux_y: vec![0.0; n_cells],
        }
    }

    /// 设置初始床面高程
    pub fn set_bed_elevation(&mut self, zb: &[f64]) {
        let n = self.zb.len().min(zb.len());
        self.zb[..n].copy_from_slice(&zb[..n]);
    }

    /// 获取床面高程
    pub fn bed_elevation(&self) -> &[f64] {
        &self.zb
    }

    /// 获取可变床面高程
    pub fn bed_elevation_mut(&mut self) -> &mut [f64] {
        &mut self.zb
    }

    /// 获取床面变化速率
    pub fn bed_change_rate(&self) -> &[f64] {
        &self.dzb_dt
    }

    /// 获取配置
    pub fn config(&self) -> &ExnerConfig {
        &self.config
    }

    /// 设置泥沙通量
    pub fn set_sediment_flux(&mut self, qx: &[f64], qy: &[f64]) {
        let n = self.flux_x.len().min(qx.len()).min(qy.len());
        self.flux_x[..n].copy_from_slice(&qx[..n]);
        self.flux_y[..n].copy_from_slice(&qy[..n]);
    }

    /// 计算通量散度（简化版本，需要网格信息进行完整计算）
    /// 
    /// Exner 方程：(1-p) × ∂zb/∂t + ∇·q = 0
    /// 
    /// 返回床面变化速率
    pub fn compute_flux_divergence_simplified(
        &mut self,
        cell_areas: &[f64],
        neighbor_indices: &[Vec<usize>],
        edge_lengths: &[Vec<f64>],
        edge_normals_x: &[Vec<f64>],
        edge_normals_y: &[Vec<f64>],
    ) {
        let n_cells = self.zb.len().min(cell_areas.len());
        let one_minus_p = 1.0 - self.config.porosity;

        for i in 0..n_cells {
            if cell_areas[i] < 1e-14 {
                self.dzb_dt[i] = 0.0;
                continue;
            }

            let mut div_q = 0.0;
            let neighbors = &neighbor_indices[i];
            let edges = &edge_lengths[i];
            let nx_list = &edge_normals_x[i];
            let ny_list = &edge_normals_y[i];

            for j in 0..neighbors.len().min(edges.len()) {
                let ni = neighbors[j];
                let edge_len = edges[j];
                let nx = nx_list[j];
                let ny = ny_list[j];

                // 边界上的通量（上风格式）
                let qx_face = 0.5 * (self.flux_x[i] + self.flux_x[ni]);
                let qy_face = 0.5 * (self.flux_y[i] + self.flux_y[ni]);

                // 通量穿过边界
                let flux_normal = qx_face * nx + qy_face * ny;
                div_q += flux_normal * edge_len;
            }

            // 散度除以面积
            div_q /= cell_areas[i];

            // Exner 方程
            self.dzb_dt[i] = -div_q / one_minus_p;

            // 限制最大变化率
            self.dzb_dt[i] = self.dzb_dt[i].clamp(
                -self.config.max_bed_change_rate,
                self.config.max_bed_change_rate,
            );
        }
    }

    /// 简化的通量散度计算（结构化网格）
    pub fn compute_flux_divergence_structured(
        &mut self,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) {
        let one_minus_p = 1.0 - self.config.porosity;
        let inv_dx = 1.0 / dx;
        let inv_dy = 1.0 / dy;

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                
                // x 方向散度
                let dqx_dx = if i == 0 {
                    (self.flux_x[idx + 1] - self.flux_x[idx]) * inv_dx
                } else if i == nx - 1 {
                    (self.flux_x[idx] - self.flux_x[idx - 1]) * inv_dx
                } else {
                    (self.flux_x[idx + 1] - self.flux_x[idx - 1]) * 0.5 * inv_dx
                };

                // y 方向散度
                let dqy_dy = if j == 0 {
                    (self.flux_y[idx + nx] - self.flux_y[idx]) * inv_dy
                } else if j == ny - 1 {
                    (self.flux_y[idx] - self.flux_y[idx - nx]) * inv_dy
                } else {
                    (self.flux_y[idx + nx] - self.flux_y[idx - nx]) * 0.5 * inv_dy
                };

                // Exner 方程
                self.dzb_dt[idx] = -(dqx_dx + dqy_dy) / one_minus_p;

                // 限制
                self.dzb_dt[idx] = self.dzb_dt[idx].clamp(
                    -self.config.max_bed_change_rate,
                    self.config.max_bed_change_rate,
                );
            }
        }
    }

    /// 时间步进（显式欧拉）
    pub fn advance(&mut self, dt: f64) {
        for i in 0..self.zb.len() {
            self.zb[i] += self.dzb_dt[i] * dt * self.config.relaxation_factor;
        }
    }

    /// 边坡崩塌修正（结构化网格）
    pub fn apply_avalanche_structured(
        &mut self,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
    ) {
        if !self.config.enable_avalanche {
            return;
        }

        let tan_crit = self.config.critical_slope;
        let max_height_diff_x = dx * tan_crit;
        let max_height_diff_y = dy * tan_crit;

        // 多次迭代以确保收敛
        for _ in 0..3 {
            for j in 0..ny {
                for i in 0..nx {
                    let idx = j * nx + i;

                    // 检查 x 方向邻居
                    if i < nx - 1 {
                        let idx_right = idx + 1;
                        let diff = self.zb[idx] - self.zb[idx_right];
                        if diff.abs() > max_height_diff_x {
                            let correction = (diff.abs() - max_height_diff_x) * 0.5;
                            if diff > 0.0 {
                                self.zb[idx] -= correction;
                                self.zb[idx_right] += correction;
                            } else {
                                self.zb[idx] += correction;
                                self.zb[idx_right] -= correction;
                            }
                        }
                    }

                    // 检查 y 方向邻居
                    if j < ny - 1 {
                        let idx_up = idx + nx;
                        let diff = self.zb[idx] - self.zb[idx_up];
                        if diff.abs() > max_height_diff_y {
                            let correction = (diff.abs() - max_height_diff_y) * 0.5;
                            if diff > 0.0 {
                                self.zb[idx] -= correction;
                                self.zb[idx_up] += correction;
                            } else {
                                self.zb[idx] += correction;
                                self.zb[idx_up] -= correction;
                            }
                        }
                    }
                }
            }
        }
    }

    /// 计算床面变化统计
    pub fn compute_statistics(&self) -> BedEvolutionStats {
        if self.zb.is_empty() {
            return BedEvolutionStats::default();
        }

        let n = self.zb.len() as f64;
        let mean = self.zb.iter().sum::<f64>() / n;
        let min = self.zb.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.zb.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        let variance = self.zb.iter().map(|&z| (z - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let total_change_rate: f64 = self.dzb_dt.iter().sum();

        BedEvolutionStats {
            min_elevation: min,
            max_elevation: max,
            mean_elevation: mean,
            std_deviation: std_dev,
            total_change_rate,
        }
    }
}

/// 床面演变统计信息
#[derive(Debug, Clone, Copy, Default)]
pub struct BedEvolutionStats {
    /// 最小高程
    pub min_elevation: f64,
    /// 最大高程
    pub max_elevation: f64,
    /// 平均高程
    pub mean_elevation: f64,
    /// 高程标准差
    pub std_deviation: f64,
    /// 总变化率
    pub total_change_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exner_config_default() {
        let config = ExnerConfig::default();
        assert!((config.porosity - 0.4).abs() < 1e-10);
        assert!(config.enable_avalanche);
    }

    #[test]
    fn test_exner_config_builder() {
        let config = ExnerConfig::new()
            .with_porosity(0.35)
            .with_critical_slope_degrees(30.0)
            .with_avalanche(false);

        assert!((config.porosity - 0.35).abs() < 1e-10);
        assert!(!config.enable_avalanche);
    }

    #[test]
    fn test_bed_evolution_solver_new() {
        let config = ExnerConfig::default();
        let solver = BedEvolutionSolver::new(100, config);
        
        assert_eq!(solver.bed_elevation().len(), 100);
        assert_eq!(solver.bed_change_rate().len(), 100);
    }

    #[test]
    fn test_set_bed_elevation() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(10, config);
        
        let zb: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        solver.set_bed_elevation(&zb);
        
        assert!((solver.bed_elevation()[5] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_flux_divergence_structured() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(25, config);
        
        // 设置均匀通量（无散度）
        solver.flux_x = vec![1.0; 25];
        solver.flux_y = vec![0.0; 25];
        
        solver.compute_flux_divergence_structured(5, 5, 1.0, 1.0);
        
        // 内部单元应该接近零散度
        let dzb_dt = solver.bed_change_rate();
        for j in 1..4 {
            for i in 1..4 {
                let idx = j * 5 + i;
                assert!(dzb_dt[idx].abs() < 0.1);
            }
        }
    }

    #[test]
    fn test_advance() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(10, config);
        
        // 设置初始床面
        let zb: Vec<f64> = vec![0.0; 10];
        solver.set_bed_elevation(&zb);
        
        // 手动设置变化率
        solver.dzb_dt = vec![0.001; 10];
        
        solver.advance(100.0);
        
        // 验证床面变化
        assert!(solver.bed_elevation()[0] > 0.0);
    }

    #[test]
    fn test_avalanche_structured() {
        let config = ExnerConfig::new().with_critical_slope_degrees(45.0);
        let mut solver = BedEvolutionSolver::new(9, config);
        
        // 创建一个陡峭的床面
        let mut zb = vec![0.0; 9];
        zb[4] = 10.0;  // 中心高点
        solver.set_bed_elevation(&zb);
        
        solver.apply_avalanche_structured(3, 3, 1.0, 1.0);
        
        // 中心应该降低
        assert!(solver.bed_elevation()[4] < 10.0);
    }

    #[test]
    fn test_compute_statistics() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(10, config);
        
        let zb: Vec<f64> = (0..10).map(|i| i as f64).collect();
        solver.set_bed_elevation(&zb);
        
        let stats = solver.compute_statistics();
        
        assert!((stats.min_elevation - 0.0).abs() < 1e-10);
        assert!((stats.max_elevation - 9.0).abs() < 1e-10);
        assert!((stats.mean_elevation - 4.5).abs() < 1e-10);
    }
}
