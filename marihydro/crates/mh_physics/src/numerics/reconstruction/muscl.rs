//! MUSCL 重构器实现

use glam::DVec2;
use std::sync::Arc;

use super::config::{GradientType, MusclConfig};
use super::traits::{ReconstructedState, Reconstructor};
use crate::adapter::PhysicsMesh;
use crate::numerics::gradient::{
    GradientMethod, GreenGaussGradient, LeastSquaresGradient, ScalarGradientStorage,
};
use crate::numerics::limiter::{create_limiter, LimiterContext, SlopeLimiter};

/// MUSCL 重构器
///
/// 实现完整的二阶 MUSCL 重构流程：
/// 1. 使用 Green-Gauss 或 Least-Squares 计算梯度
/// 2. 使用选定的限制器进行梯度限制
/// 3. 线性外推到面中心
pub struct MusclReconstructor {
    /// 配置
    config: MusclConfig,
    
    /// 网格引用
    mesh: Arc<PhysicsMesh>,
    
    /// 梯度存储
    gradients: ScalarGradientStorage,
    
    /// 限制因子存储
    limiters: Vec<f64>,
    
    /// 梯度计算器
    gradient_computer: GradientComputer,
    
    /// 限制器
    limiter: Box<dyn SlopeLimiter>,
    
    /// 网格特征尺度（用于 Venkatakrishnan）
    mesh_scale: f64,
}

/// 梯度计算器枚举
enum GradientComputer {
    GreenGauss(GreenGaussGradient),
    LeastSquares(LeastSquaresGradient),
}

impl MusclReconstructor {
    /// 创建新的 MUSCL 重构器
    pub fn new(config: MusclConfig, mesh: Arc<PhysicsMesh>) -> Self {
        let n_cells = mesh.n_cells();
        
        // 计算网格特征尺度
        let mesh_scale = compute_mesh_scale(&mesh);
        
        // 创建梯度计算器
        let gradient_computer = match config.gradient_type {
            GradientType::GreenGauss => {
                GradientComputer::GreenGauss(GreenGaussGradient::new())
            }
            GradientType::LeastSquares => {
                GradientComputer::LeastSquares(LeastSquaresGradient::new())
            }
        };
        
        // 创建限制器
        let limiter = create_limiter(config.limiter_type, config.venkat_k, mesh_scale);
        
        Self {
            config,
            mesh,
            gradients: ScalarGradientStorage::new(n_cells),
            limiters: vec![1.0; n_cells],
            gradient_computer,
            limiter,
            mesh_scale,
        }
    }
    
    /// 更新配置
    pub fn set_config(&mut self, config: MusclConfig) {
        // 如果限制器类型改变，重新创建
        if config.limiter_type != self.config.limiter_type 
           || config.venkat_k != self.config.venkat_k {
            self.limiter = create_limiter(config.limiter_type, config.venkat_k, self.mesh_scale);
        }
        
        // 如果梯度类型改变，重新创建
        if config.gradient_type != self.config.gradient_type {
            self.gradient_computer = match config.gradient_type {
                GradientType::GreenGauss => {
                    GradientComputer::GreenGauss(GreenGaussGradient::new())
                }
                GradientType::LeastSquares => {
                    GradientComputer::LeastSquares(LeastSquaresGradient::new())
                }
            };
        }
        
        self.config = config;
    }
    
    /// 获取配置
    pub fn config(&self) -> &MusclConfig {
        &self.config
    }
    
    /// 计算并限制梯度
    fn compute_and_limit_gradients(&mut self, values: &[f64]) {
        let n_cells = self.mesh.n_cells();
        
        if !self.config.second_order {
            // 一阶精度：梯度为零
            self.gradients.resize(n_cells);
            self.limiters.fill(1.0);
            return;
        }
        
        // 步骤1：计算原始梯度
        match &self.gradient_computer {
            GradientComputer::GreenGauss(gg) => {
                gg.compute_scalar_gradient(values, &self.mesh, &mut self.gradients);
            }
            GradientComputer::LeastSquares(ls) => {
                ls.compute_scalar_gradient(values, &self.mesh, &mut self.gradients);
            }
        }
        
        // 步骤2：计算限制因子
        self.compute_limiters(values);
        
        // 步骤3：应用限制
        self.gradients.apply_limiter(&self.limiters);
    }
    
    /// 计算限制因子
    fn compute_limiters(&mut self, values: &[f64]) {
        let n_cells = self.mesh.n_cells();
        
        for cell_id in 0..n_cells {
            let cell_value = values[cell_id];
            
            // 检查干单元
            if cell_value < self.config.dry_tolerance {
                self.limiters[cell_id] = 0.0;
                continue;
            }
            
            // 查找邻居的最小/最大值
            let (min_neighbor, max_neighbor) = self.find_neighbor_extrema(cell_id, values);
            
            // 计算最大梯度投影
            let (grad_projection, max_distance) = self.compute_max_gradient_projection(cell_id);
            
            // 创建限制器上下文
            let ctx = LimiterContext::new(
                cell_value,
                grad_projection,
                min_neighbor,
                max_neighbor,
                max_distance,
            );
            
            self.limiters[cell_id] = self.limiter.compute_limiter(&ctx);
            
            // 正定保持：确保重构后水深非负
            if self.config.positivity_preserving {
                self.apply_positivity_constraint(cell_id, cell_value);
            }
        }
    }
    
    /// 查找邻居单元的极值
    /// 
    /// 使用 cell_neighbors 进行 O(邻居数量) 查找
    fn find_neighbor_extrema(&self, cell_id: usize, values: &[f64]) -> (f64, f64) {
        let cell_value = values[cell_id];
        let mut min_val = cell_value;
        let mut max_val = cell_value;
        
        // 直接遍历邻居单元，复杂度 O(邻居数)
        for neighbor_id in self.mesh.cell_neighbors(cell_id) {
            let neighbor_value = values[neighbor_id];
            min_val = min_val.min(neighbor_value);
            max_val = max_val.max(neighbor_value);
        }
        
        (min_val, max_val)
    }
    
    /// 计算最大梯度投影和距离
    /// 
    /// 使用 cell_faces 进行 O(面数量) 查找
    fn compute_max_gradient_projection(&self, cell_id: usize) -> (f64, f64) {
        let grad_x = self.gradients.grad_x[cell_id];
        let grad_y = self.gradients.grad_y[cell_id];
        let cell_center = self.mesh.cell_center(cell_id);
        
        let mut max_projection = 0.0f64;
        let mut max_distance = 0.0f64;
        
        // 直接遍历单元的面，复杂度 O(单元面数)
        for face_id in self.mesh.cell_faces(cell_id) {
            let face_center = self.mesh.face_center(face_id);
            let dx = face_center.x - cell_center.x;
            let dy = face_center.y - cell_center.y;
            
            let distance = (dx * dx + dy * dy).sqrt();
            let projection = (grad_x * dx + grad_y * dy).abs();
            
            if projection > max_projection {
                max_projection = projection;
                max_distance = distance;
            }
        }
        
        (max_projection, max_distance)
    }
    
    /// 应用正定约束
    /// 
    /// 使用 cell_faces 进行 O(单元面数) 查找
    fn apply_positivity_constraint(&mut self, cell_id: usize, cell_value: f64) {
        if cell_value <= 0.0 {
            self.limiters[cell_id] = 0.0;
            return;
        }
        
        let grad_x = self.gradients.grad_x[cell_id];
        let grad_y = self.gradients.grad_y[cell_id];
        let cell_center = self.mesh.cell_center(cell_id);
        
        // 收集面列表避免借用冲突
        let faces: Vec<usize> = self.mesh.cell_faces(cell_id).collect();
        
        for face_id in faces {
            let face_center = self.mesh.face_center(face_id);
            let dx = face_center.x - cell_center.x;
            let dy = face_center.y - cell_center.y;
            
            let reconstructed = cell_value + self.limiters[cell_id] * (grad_x * dx + grad_y * dy);
            
            if reconstructed < 0.0 {
                // 计算保持正定的最大限制因子
                let denominator = grad_x * dx + grad_y * dy;
                if denominator.abs() > 1e-12 {
                    let alpha_safe = (-cell_value / denominator).abs().min(1.0);
                    self.limiters[cell_id] = self.limiters[cell_id].min(alpha_safe * 0.9);
                } else {
                    self.limiters[cell_id] = 0.0;
                }
            }
        }
    }
    
    /// 重构面值（内部方法）
    fn reconstruct_at_face(&self, face_id: usize, values: &[f64]) -> ReconstructedState {
        let left_cell = self.mesh.face_owner(face_id);
        let right_cell = self.mesh.face_neighbor(face_id);
        let face_center = self.mesh.face_center(face_id);
        
        // 左侧重构
        let left_value = self.reconstruct_at_point(left_cell, face_center, values);
        
        // 右侧重构
        let right_value = if let Some(right_id) = right_cell {
            self.reconstruct_at_point(right_id, face_center, values)
        } else {
            // 边界面：使用左侧值
            left_value
        };
        
        ReconstructedState::new(left_value, right_value)
    }
    
    /// 从单元中心重构到指定点
    fn reconstruct_at_point(&self, cell_id: usize, point: DVec2, values: &[f64]) -> f64 {
        if !self.config.second_order {
            return values[cell_id];
        }
        
        let cell_center = self.mesh.cell_center(cell_id);
        let dx = point.x - cell_center.x;
        let dy = point.y - cell_center.y;
        
        // 注意：梯度已经被限制
        let grad_x = self.gradients.grad_x[cell_id];
        let grad_y = self.gradients.grad_y[cell_id];
        
        values[cell_id] + grad_x * dx + grad_y * dy
    }
}

impl Reconstructor for MusclReconstructor {
    fn compute_gradients(&mut self, values: &[f64]) {
        self.compute_and_limit_gradients(values);
    }
    
    fn reconstruct_scalar(&self, face_id: usize, values: &[f64]) -> ReconstructedState {
        self.reconstruct_at_face(face_id, values)
    }
    
    fn get_limited_gradient(&self, cell_id: usize) -> DVec2 {
        DVec2::new(
            self.gradients.grad_x[cell_id],
            self.gradients.grad_y[cell_id],
        )
    }
    
    fn is_second_order(&self) -> bool {
        self.config.second_order
    }
    
    fn name(&self) -> &'static str {
        "MUSCL"
    }
}

/// 计算网格特征尺度
fn compute_mesh_scale(mesh: &PhysicsMesh) -> f64 {
    if mesh.n_cells() == 0 {
        return 1.0;
    }
    
    // 使用平均单元面积的平方根作为特征尺度
    let total_area: f64 = (0..mesh.n_cells())
        .filter_map(|i| mesh.cell_area(i))
        .sum();
    
    (total_area / mesh.n_cells() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // 注意：完整测试需要 PhysicsMesh 实例，这里只测试辅助函数
    
    #[test]
    fn test_config_creation() {
        let config = MusclConfig::default();
        assert!(config.second_order);
        assert!(config.positivity_preserving);
    }
    
    #[test]
    fn test_first_order_config() {
        let config = MusclConfig::first_order();
        assert!(!config.second_order);
    }
    
    #[test]
    fn test_reconstructed_state_basic() {
        let state = ReconstructedState::new(1.5, 2.0);
        assert_eq!(state.average(), 1.75);
        assert_eq!(state.jump(), 0.5);
    }
}
