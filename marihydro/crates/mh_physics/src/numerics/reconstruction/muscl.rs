//! MUSCL 重构器实现 - Backend 泛型版本
//!
//! **层级**: Layer 3 - Engine Layer
//!
//! 实现完整的二阶 MUSCL 重构流程：
//! 1. 使用 Green-Gauss 或 Least-Squares 计算梯度
//! 2. 使用选定的限制器进行梯度限制
//! 3. 线性外推到面中心
//!
//! # 设计原则
//!
//! 1. **全泛型**: 实现 `Reconstructor<B>` 支持任意 Backend
//! 2. **无 DVec2**: 所有几何操作使用 `Vector2D` 接口
//! 3. **泛型标量**: 几何计算在 `S` 上完成

use std::marker::PhantomData;
use std::sync::Arc;

use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar, Vector2D};
use num_traits::Float;

use super::config::{GradientType, MusclConfig};
use super::traits::{ReconstructedState, Reconstructor};
use crate::adapter::PhysicsMesh;
use crate::types::{CellIndex, FaceIndex};
use crate::numerics::gradient::{GradientMethod, GreenGaussGradient, LeastSquaresGradient, ScalarGradientStorage};
use crate::numerics::limiter::{create_limiter, LimiterAny, LimiterContext, SlopeLimiter};

// ============================================================================
// MUSCL 重构器 - 泛型版本
// ============================================================================

/// MUSCL 重构器 - Backend 泛型版本
///
/// 实现完整的二阶 MUSCL 重构流程，支持任意 Backend 精度。
pub struct MusclReconstructor<B: Backend> {
    /// 配置
    config: MusclConfig,
    
    /// 网格引用
    mesh: Arc<PhysicsMesh>,

    /// 计算后端
    backend: B,
    
    /// 梯度存储 (S 空间)
    gradients: ScalarGradientStorage<B>,
    
    /// 限制因子存储 (S 空间)
    limiters: B::Buffer<B::Scalar>,
    
    /// 梯度计算器
    gradient_computer: GradientComputer,
    
    /// 限制器
    limiter: LimiterAny<B>,
    
    /// 网格特征尺度 (f64, 几何量)
    mesh_scale: f64,
    
    _marker: PhantomData<B>,
}

/// 梯度计算器枚举
enum GradientComputer {
    GreenGauss(GreenGaussGradient),
    LeastSquares(LeastSquaresGradient),
}

impl<B: Backend> MusclReconstructor<B> {
    /// 创建新的 MUSCL 重构器
    pub fn new(config: MusclConfig, mesh: Arc<PhysicsMesh>, backend: B) -> Self {
        let n_cells = mesh.cell_count();
        
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
        
        // 创建限制器 (泛型版本)
        let limiter = create_limiter(&backend, config.limiter_type, config.venkat_k, mesh_scale);

        let mut limiters = backend.alloc(n_cells);
        limiters.fill(B::Scalar::ONE);
        let gradients = ScalarGradientStorage::with_backend(&backend, n_cells);

        Self {
            config,
            mesh,
            backend,
            gradients,
            limiters,
            gradient_computer,
            limiter,
            mesh_scale,
            _marker: PhantomData,
        }
    }
    
    /// 更新配置
    pub fn set_config(&mut self, config: MusclConfig) {
        // 如果限制器类型改变，重新创建
        if config.limiter_type != self.config.limiter_type 
           || config.venkat_k != self.config.venkat_k {
            self.limiter = create_limiter(&self.backend, config.limiter_type, config.venkat_k, self.mesh_scale);
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
    fn compute_and_limit_gradients(&mut self, values: &B::Buffer<B::Scalar>) {
        let n_cells = self.mesh.cell_count();
        
        if !self.config.second_order {
            // 一阶精度：梯度为零
            self.gradients.resize(n_cells);
            self.limiters.fill(B::Scalar::ONE);
            return;
        }
        
        // 步骤1：计算原始梯度
        match &self.gradient_computer {
            GradientComputer::GreenGauss(gg) => {
                gg.compute_scalar_gradient(&self.backend, values, &self.mesh, &mut self.gradients);
            }
            GradientComputer::LeastSquares(ls) => {
                ls.compute_scalar_gradient(&self.backend, values, &self.mesh, &mut self.gradients);
            }
        }
        
        // 步骤2：计算限制因子
        self.compute_limiters(values);
        
        // 步骤3：应用限制
        self.gradients.apply_limiter(&self.limiters);
    }
    
    /// 计算限制因子
    fn compute_limiters(&mut self, values: &B::Buffer<B::Scalar>) {
        let n_cells = self.mesh.cell_count();
        let dry_tol = B::Scalar::from_config(self.config.dry_tolerance)
            .unwrap_or(B::Scalar::ZERO);
        let values_slice = values.as_slice();
        let mut temp_limiters = vec![B::Scalar::ONE; n_cells];
        
        for cell_id in 0..n_cells {
            let cell_value = values_slice[cell_id];
            
            // 检查干单元
            if cell_value < dry_tol {
                temp_limiters[cell_id] = B::Scalar::ZERO;
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
            
            temp_limiters[cell_id] = self.limiter.compute_limiter(&ctx);
            
            // 正定保持：确保重构后水深非负
            if self.config.positivity_preserving {
                self.apply_positivity_constraint(cell_id, cell_value, &mut temp_limiters);
            }
        }

        let limiters_slice = self.limiters.as_slice_mut();
        for (dst, src) in limiters_slice.iter_mut().zip(temp_limiters.iter()) {
            *dst = *src;
        }
    }
    
    /// 查找邻居单元的极值
    fn find_neighbor_extrema(&self, cell_id: usize, values: &B::Buffer<B::Scalar>) -> (B::Scalar, B::Scalar) {
        let values_slice = values.as_slice();
        let cell_value = values_slice[cell_id];
        let mut min_val = cell_value;
        let mut max_val = cell_value;
        
        for neighbor_id in self.mesh.cell_neighbors(CellIndex::new(cell_id)) {
            let neighbor_value = values_slice[neighbor_id.0];
            if neighbor_value < min_val {
                min_val = neighbor_value;
            }
            if neighbor_value > max_val {
                max_val = neighbor_value;
            }
        }
        
        (min_val, max_val)
    }
    
    /// 计算最大梯度投影和距离
    fn compute_max_gradient_projection(&self, cell_id: usize) -> (B::Scalar, B::Scalar) {
        let (grad_x, grad_y) = self.gradients.get_tuple(cell_id);
        let cell_center = self
            .mesh
            .cell_center_generic::<B>(CellIndex::new(cell_id))
            .expect("cell_center out of range");
        let cell_center_x = cell_center.x();
        let cell_center_y = cell_center.y();
        
        let mut max_projection = B::Scalar::ZERO;
        let mut max_distance = B::Scalar::ZERO;
        
        for face_id in self.mesh.cell_faces(CellIndex::new(cell_id)) {
            let face_center = self
                .mesh
                .face_center_generic::<B>(face_id)
                .expect("face_center out of range");

            let dx = face_center.x() - cell_center_x;
            let dy = face_center.y() - cell_center_y;
            
            let distance = (dx * dx + dy * dy).sqrt();
            let projection = (grad_x * dx + grad_y * dy).abs();
            let distance_s = distance;
            
            if projection > max_projection {
                max_projection = projection;
                max_distance = distance_s;
            }
        }
        
        (max_projection, max_distance)
    }
    
    /// 应用正定约束
    fn apply_positivity_constraint(
        &self,
        cell_id: usize,
        cell_value: B::Scalar,
        limiters: &mut [B::Scalar],
    ) {
        if cell_value <= B::Scalar::ZERO {
            limiters[cell_id] = B::Scalar::ZERO;
            return;
        }
        
        let (grad_x, grad_y) = self.gradients.get_tuple(cell_id);
        let cell_center = self
            .mesh
            .cell_center_generic::<B>(CellIndex::new(cell_id))
            .expect("cell_center out of range");
        let cell_center_x = cell_center.x();
        let cell_center_y = cell_center.y();
        
        let faces: Vec<usize> = self
            .mesh
            .cell_faces(CellIndex::new(cell_id))
            .map(|f| f.into())
            .collect();
        
        for face_id in faces {
            let face_center = self
                .mesh
                .face_center_generic::<B>(FaceIndex::new(face_id))
                .expect("face_center out of range");

            let dx = face_center.x() - cell_center_x;
            let dy = face_center.y() - cell_center_y;
            
            let reconstructed = cell_value + limiters[cell_id] * (grad_x * dx + grad_y * dy);
            
            if reconstructed < B::Scalar::ZERO {
                let denominator = grad_x * dx + grad_y * dy;
                let eps = self.backend.config_scalar(1e-12, "muscl.positivity_epsilon");
                if denominator.abs() > eps {
                    let alpha_safe = (-cell_value / denominator).abs();
                    let alpha_safe = if alpha_safe < B::Scalar::ONE { alpha_safe } else { B::Scalar::ONE };
                    let shrink = self.backend.config_scalar(0.9, "muscl.positivity_shrink");
                    let new_limiter = limiters[cell_id] * shrink;
                    limiters[cell_id] = if alpha_safe < new_limiter { alpha_safe } else { new_limiter };
                } else {
                    limiters[cell_id] = B::Scalar::ZERO;
                }
            }
        }
    }
    
    /// 重构面值
    fn reconstruct_at_face(&self, face_id: usize, values: &B::Buffer<B::Scalar>) -> ReconstructedState<B> {
        let fi = FaceIndex::new(face_id);
        let left_cell: usize = self.mesh.face_owner(fi).into();
        let right_cell: Option<usize> = self.mesh.face_neighbor(fi).map(|c| c.into());
        let face_center = self
            .mesh
            .face_center_generic::<B>(fi)
            .expect("face_center out of range");
        
        // 左侧重构
        let left_value = self.reconstruct_at_point(left_cell, &face_center, values);
        
        // 右侧重构
        let right_value = if let Some(right_id) = right_cell {
            self.reconstruct_at_point(right_id, &face_center, values)
        } else {
            left_value
        };
        
        ReconstructedState::new(left_value, right_value)
    }
    
    /// 从单元中心重构到指定点
    fn reconstruct_at_point(&self, cell_id: usize, point: &impl Vector2D<Scalar = B::Scalar>, values: &B::Buffer<B::Scalar>) -> B::Scalar {
        if !self.config.second_order {
            return values[cell_id];
        }
        
        let cell_center = self
            .mesh
            .cell_center_generic::<B>(CellIndex::new(cell_id))
            .expect("cell_center out of range");
        
        let dx = point.x() - cell_center.x();
        let dy = point.y() - cell_center.y();
        
        let (grad_x, grad_y) = self.gradients.get_tuple(cell_id);
        
        values[cell_id] + grad_x * dx + grad_y * dy
    }
}

// ============================================================================
// Trait 实现
// ============================================================================

impl<B: Backend> Reconstructor<B> for MusclReconstructor<B> {
    fn compute_gradients(&mut self, values: &B::Buffer<B::Scalar>) {
        self.compute_and_limit_gradients(values);
    }
    
    fn reconstruct_scalar(&self, face_id: usize, values: &B::Buffer<B::Scalar>) -> ReconstructedState<B> {
        self.reconstruct_at_face(face_id, values)
    }
    
    fn get_limited_gradient_tuple(&self, cell_id: usize) -> (B::Scalar, B::Scalar) {
        self.gradients.get_tuple(cell_id)
    }
    
    fn is_second_order(&self) -> bool {
        self.config.second_order
    }
    
    fn name(&self) -> &'static str {
        "MUSCL"
    }
}

// ============================================================================
// 工具函数
// ============================================================================

/// 计算网格特征尺度
fn compute_mesh_scale(mesh: &PhysicsMesh) -> f64 {
    if mesh.cell_count() == 0 {
        return 1.0;
    }
    
    // 使用平均单元面积的平方根作为特征尺度
    let total_area: f64 = (0..mesh.cell_count())
        .filter_map(|i| mesh.cell_area(CellIndex::new(i)))
        .sum();
    
    (total_area / mesh.cell_count() as f64).sqrt()
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
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
    fn test_reconstructed_state_basic_f64() {
        let state = ReconstructedState::<CpuBackend<f64>>::new(1.5, 2.0);
        assert_eq!(state.average(), 1.75);
        assert_eq!(state.jump(), 0.5);
    }
    
    #[test]
    fn test_reconstructed_state_basic_f32() {
        let state = ReconstructedState::<CpuBackend<f32>>::new(1.5, 2.0);
        assert!((state.average() - 1.75).abs() < 1e-5);
        assert!((state.jump() - 0.5).abs() < 1e-5);
    }
}
