// crates/mh_physics/src/sediment/bed_evolution.rs

//! 床面演变计算
//! 基于 Exner 方程的床面高程变化求解器

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

    /// 计算通量散度（上风格式 - 推荐用于非结构化网格）
    /// 
    /// Exner 方程：(1-p) × ∂zb/∂t + ∇·q = 0
    /// 
    /// 使用上风格式计算界面通量：
    /// - 如果流动从 i 到 j (v_n > 0)，使用 q_i
    /// - 如果流动从 j 到 i (v_n < 0)，使用 q_j
    /// 
    /// # 参数
    /// - `cell_areas`: 单元面积
    /// - `neighbor_indices`: 每个单元的邻居索引
    /// - `edge_lengths`: 每条边的长度
    /// - `edge_normals_x`: 边法向量x分量（指向外部）
    /// - `edge_normals_y`: 边法向量y分量
    /// - `cell_velocities_x`: 单元速度x分量（用于确定上风方向）
    /// - `cell_velocities_y`: 单元速度y分量
    pub fn compute_flux_divergence_upwind(
        &mut self,
        cell_areas: &[f64],
        neighbor_indices: &[Vec<usize>],
        edge_lengths: &[Vec<f64>],
        edge_normals_x: &[Vec<f64>],
        edge_normals_y: &[Vec<f64>],
        cell_velocities_x: &[f64],
        cell_velocities_y: &[f64],
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

                // 计算界面处的速度（平均）
                let u_face = 0.5 * (cell_velocities_x[i] + cell_velocities_x[ni]);
                let v_face = 0.5 * (cell_velocities_y[i] + cell_velocities_y[ni]);
                
                // 法向速度（正值表示从 i 流向外部）
                let v_n = u_face * nx + v_face * ny;
                
                // 上风格式选择泥沙通量
                let (qx_face, qy_face) = if v_n >= 0.0 {
                    // 流出单元 i，使用 i 的通量
                    (self.flux_x[i], self.flux_y[i])
                } else {
                    // 流入单元 i，使用邻居 ni 的通量
                    (self.flux_x[ni], self.flux_y[ni])
                };

                // 通量穿过边界
                let flux_normal = qx_face * nx + qy_face * ny;
                div_q += flux_normal * edge_len;
            }

            // 散度除以面积
            div_q /= cell_areas[i];

            // Exner 方程：(1-p) * dzb/dt = -div(q)
            self.dzb_dt[i] = -div_q / one_minus_p;

            // 限制最大变化率
            self.dzb_dt[i] = self.dzb_dt[i].clamp(
                -self.config.max_bed_change_rate,
                self.config.max_bed_change_rate,
            );
        }
    }

    /// 计算通量散度（中心差分 - 仅用于平滑流场）
    /// 
    /// **警告**：中心差分在非平滑流场中可能产生振荡
    /// 建议使用 `compute_flux_divergence_upwind` 替代
    #[deprecated(since = "0.2.0", note = "使用 compute_flux_divergence_upwind 替代，中心差分易产生振荡")]
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

                // 边界上的通量（中心差分 - 不推荐）
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

    /// 上风通量散度计算（结构化网格）
    /// 
    /// 使用上风格式计算界面通量，根据流动方向选择上游单元的通量值
    /// 
    /// # 参数
    /// - `nx, ny`: 网格维度
    /// - `dx, dy`: 网格间距
    /// - `u, v`: 速度场（用于确定上风方向）
    pub fn compute_flux_divergence_structured_upwind(
        &mut self,
        nx: usize,
        ny: usize,
        dx: f64,
        dy: f64,
        u: &[f64],
        v: &[f64],
    ) {
        let one_minus_p = 1.0 - self.config.porosity;
        let inv_dx = 1.0 / dx;
        let inv_dy = 1.0 / dy;

        for j in 0..ny {
            for i in 0..nx {
                let idx = j * nx + i;
                
                // x 方向上风格式
                let dqx_dx = if i == 0 {
                    // 左边界：只能用右边
                    (self.flux_x[idx + 1] - self.flux_x[idx]) * inv_dx
                } else if i == nx - 1 {
                    // 右边界：只能用左边
                    (self.flux_x[idx] - self.flux_x[idx - 1]) * inv_dx
                } else {
                    // 内部单元：上风格式
                    let u_local = u[idx];
                    if u_local >= 0.0 {
                        // 向右流动，使用左侧差分
                        (self.flux_x[idx] - self.flux_x[idx - 1]) * inv_dx
                    } else {
                        // 向左流动，使用右侧差分
                        (self.flux_x[idx + 1] - self.flux_x[idx]) * inv_dx
                    }
                };

                // y 方向上风格式
                let dqy_dy = if j == 0 {
                    // 下边界
                    (self.flux_y[idx + nx] - self.flux_y[idx]) * inv_dy
                } else if j == ny - 1 {
                    // 上边界
                    (self.flux_y[idx] - self.flux_y[idx - nx]) * inv_dy
                } else {
                    // 内部单元：上风格式
                    let v_local = v[idx];
                    if v_local >= 0.0 {
                        // 向上流动，使用下侧差分
                        (self.flux_y[idx] - self.flux_y[idx - nx]) * inv_dy
                    } else {
                        // 向下流动，使用上侧差分
                        (self.flux_y[idx + nx] - self.flux_y[idx]) * inv_dy
                    }
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

    /// 简化的通量散度计算（结构化网格，中心差分）
    /// 
    /// **警告**：中心差分在非平滑流场中易产生振荡
    /// 建议使用 `compute_flux_divergence_structured_upwind` 替代
    #[deprecated(since = "0.2.0", note = "使用 compute_flux_divergence_structured_upwind 替代")]
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
        
        #[allow(deprecated)]
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
    fn test_flux_divergence_structured_upwind() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(25, config);
        
        // 设置均匀通量（无散度）
        solver.flux_x = vec![1.0; 25];
        solver.flux_y = vec![0.0; 25];
        
        // 均匀向右流动
        let u = vec![1.0; 25];
        let v = vec![0.0; 25];
        
        solver.compute_flux_divergence_structured_upwind(5, 5, 1.0, 1.0, &u, &v);
        
        // 均匀通量，内部应接近零散度
        let dzb_dt = solver.bed_change_rate();
        for j in 1..4 {
            for i in 1..4 {
                let idx = j * 5 + i;
                assert!(dzb_dt[idx].abs() < 0.1, "dzb_dt[{}] = {}", idx, dzb_dt[idx]);
            }
        }
    }

    #[test]
    fn test_upwind_captures_gradient() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(9, config);
        
        // 设置通量梯度：从左到右通量增加
        // qx = [0, 1, 2] 每行
        for j in 0..3 {
            for i in 0..3 {
                let idx = j * 3 + i;
                solver.flux_x[idx] = i as f64;
                solver.flux_y[idx] = 0.0;
            }
        }
        
        // 向右流动
        let u = vec![1.0; 9];
        let v = vec![0.0; 9];
        
        solver.compute_flux_divergence_structured_upwind(3, 3, 1.0, 1.0, &u, &v);
        
        // dqx/dx > 0，所以 dzb/dt < 0（侵蚀）
        let dzb_dt = solver.bed_change_rate();
        let idx_center = 4; // 中心单元
        assert!(dzb_dt[idx_center] < 0.0, "应该是侵蚀，但 dzb_dt = {}", dzb_dt[idx_center]);
    }

    #[test]
    fn test_upwind_unstructured() {
        let config = ExnerConfig::default();
        let mut solver = BedEvolutionSolver::new(4, config);
        
        // 简单的4单元网格（2x2）
        // 0 - 1
        // |   |
        // 2 - 3
        
        // 设置通量
        solver.flux_x = vec![0.0, 1.0, 0.0, 1.0];
        solver.flux_y = vec![0.0, 0.0, 0.0, 0.0];
        
        let cell_areas = vec![1.0, 1.0, 1.0, 1.0];
        let neighbor_indices = vec![
            vec![1, 2],    // 单元0的邻居
            vec![0, 3],    // 单元1的邻居
            vec![0, 3],    // 单元2的邻居
            vec![1, 2],    // 单元3的邻居
        ];
        let edge_lengths = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
            vec![1.0, 1.0],
        ];
        // 法向量：指向邻居
        let edge_normals_x = vec![
            vec![1.0, 0.0],   // 单元0：右、下
            vec![-1.0, 0.0],  // 单元1：左、下
            vec![0.0, 1.0],   // 单元2：上、右
            vec![0.0, -1.0],  // 单元3：上、左
        ];
        let edge_normals_y = vec![
            vec![0.0, -1.0],
            vec![0.0, -1.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        
        // 向右流动
        let u = vec![1.0; 4];
        let v = vec![0.0; 4];
        
        solver.compute_flux_divergence_upwind(
            &cell_areas,
            &neighbor_indices,
            &edge_lengths,
            &edge_normals_x,
            &edge_normals_y,
            &u,
            &v,
        );
        
        // 单元0：出口通量 = 0（自己的qx），入口通量 = 0
        // 单元1：入口通量 = 0（从0），出口通量 = 1
        // 所以单元0和1的散度应该不同
        let dzb_dt = solver.bed_change_rate();
        // 通量从0流向1，单元0应该侵蚀（dzb < 0 如果有出流）
        println!("dzb_dt: {:?}", dzb_dt);
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
