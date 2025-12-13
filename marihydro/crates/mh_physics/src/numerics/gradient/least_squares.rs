// crates/mh_physics/src/numerics/gradient/least_squares.rs

//! 最小二乘梯度计算
//!
//! 通过最小化加权最小二乘误差来计算梯度:
//! min Σ w_j * (φ_j - φ_i - ∇φ_i · r_ij)²
//!
//! 对于 2D 情况，求解 2x2 法方程：
//! [a11 a12] [∂φ/∂x]   [b1]
//! [a12 a22] [∂φ/∂y] = [b2]
//!
//! 当法方程奇异时，回退到 Green-Gauss 方法。

use super::traits::{GradientMethodGeneric, ScalarGradientStorage, VectorGradientStorage};
use super::green_gauss::GreenGaussGradient;
use crate::adapter::PhysicsMesh;
use crate::types::NumericalParams;

use glam::DVec2;

// ============================================================
// 配置 (Layer 4 - 配置参数允许 f64)
// ============================================================

/// 最小二乘梯度配置
#[derive(Debug, Clone)]
pub struct LeastSquaresConfig {
    /// 行列式最小值（判断奇异性）
    pub det_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// 是否启用边界贡献（虚拟点策略）
    pub use_boundary_contributions: bool,
    /// 是否启用并行
    pub parallel: bool,
    /// 并行阈值
    pub parallel_threshold: usize,
}

impl Default for LeastSquaresConfig {
    fn default() -> Self {
        Self {
            det_min: 1e-12,
            use_boundary_contributions: true,
            parallel: true,
            parallel_threshold: 1000,
        }
    }
}

// ============================================================
// 最小二乘梯度计算器
// ============================================================

/// 最小二乘梯度计算器
#[derive(Debug, Clone)]
pub struct LeastSquaresGradient {
    config: LeastSquaresConfig,
    /// 回退方法
    fallback: GreenGaussGradient,
}

impl Default for LeastSquaresGradient {
    fn default() -> Self {
        Self {
            config: LeastSquaresConfig::default(),
            fallback: GreenGaussGradient::new(),
        }
    }
}

impl LeastSquaresGradient {
    /// 创建新实例
    pub fn new() -> Self {
        Self::default()
    }

    /// 从数值参数创建
    pub fn from_params(params: &NumericalParams) -> Self {
        Self {
            config: LeastSquaresConfig {
                det_min: params.det_min,
                ..Default::default()
            },
            fallback: GreenGaussGradient::new(),
        }
    }

    /// 设置行列式最小值
    // ALLOW_F64: Layer 4 配置参数设置方法
    pub fn with_det_min(mut self, det_min: f64) -> Self {
        self.config.det_min = det_min;
        self
    }

    /// 设置边界贡献开关
    pub fn with_boundary_contributions(mut self, enable: bool) -> Self {
        self.config.use_boundary_contributions = enable;
        self
    }

    /// 设置并行开关
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// 求解 2x2 对称正定系统
    ///
    /// [a11 a12] [x1]   [b1]
    /// [a12 a22] [x2] = [b2]
    // ALLOW_F64: PhysicsMesh 返回 DVec2 (f64)，此辅助方法配合 DVec2 使用
    #[inline]
    fn solve_2x2(
        a11: f64, // ALLOW_F64: DVec2 坐标计算
        a12: f64, // ALLOW_F64: DVec2 坐标计算
        a22: f64, // ALLOW_F64: DVec2 坐标计算
        b1: f64, // ALLOW_F64: DVec2 坐标计算
        b2: f64, // ALLOW_F64: DVec2 坐标计算
        det_min: f64, // ALLOW_F64: DVec2 坐标计算
    ) -> Option<(f64, f64)> { // ALLOW_F64: DVec2 坐标计算
        let det = a11 * a22 - a12 * a12;
        if det.abs() < det_min {
            return None;
        }
        let inv = 1.0 / det;
        let x1 = (a22 * b1 - a12 * b2) * inv;
        let x2 = (a11 * b2 - a12 * b1) * inv;
        if x1.is_finite() && x2.is_finite() {
            Some((x1, x2))
        } else {
            None
        }
    }

    /// 计算单个单元的梯度
    fn compute_cell_gradient(
        &self,
        cell: usize,
        field: &[f64],
        mesh: &PhysicsMesh,
    ) -> Option<DVec2> {
        let cell_center = mesh.cell_center(cell);
        let phi_c = field[cell];

        let mut a11 = 0.0;
        let mut a12 = 0.0;
        let mut a22 = 0.0;
        let mut b1 = 0.0;
        let mut b2 = 0.0;
        let mut neighbor_count = 0;

        // 收集邻居贡献
        for face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor_opt = mesh.face_neighbor(face);

            let is_owner = owner == cell;
            let is_neighbor = neighbor_opt == Some(cell);

            if !is_owner && !is_neighbor {
                continue;
            }

            if let Some(neighbor) = neighbor_opt {
                // 内部面：使用邻居单元
                let other = if is_owner { neighbor } else { owner };
                let other_center = mesh.cell_center(other);

                let dx = other_center.x - cell_center.x;
                let dy = other_center.y - cell_center.y;
                let dphi = field[other] - phi_c;

                let dist_sq = dx * dx + dy * dy;
                if dist_sq < 1e-20 {
                    continue;
                }

                // 距离平方反比加权
                let w = 1.0 / dist_sq;
                a11 += w * dx * dx;
                a12 += w * dx * dy;
                a22 += w * dy * dy;
                b1 += w * dx * dphi;
                b2 += w * dy * dphi;
                neighbor_count += 1;
            } else if self.config.use_boundary_contributions {
                // 边界面：使用镜像点策略
                let face_center = mesh.face_center(face);
                let normal = mesh.face_normal(face);

                // 单元中心到面的距离
                let to_face = face_center - cell_center;
                let dist_to_face = to_face.dot(normal);

                if dist_to_face.abs() < 1e-14 {
                    continue;
                }

                // 虚拟点：面的另一侧镜像
                let ghost_point = face_center + normal * dist_to_face.abs();
                let dx = ghost_point.x - cell_center.x;
                let dy = ghost_point.y - cell_center.y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1e-20 {
                    continue;
                }

                // 零梯度边界条件：dphi = 0
                let w = 1.0 / dist_sq;
                a11 += w * dx * dx;
                a12 += w * dx * dy;
                a22 += w * dy * dy;
                // b1, b2 不变（dphi = 0）
                neighbor_count += 1;
            }
        }

        // 邻居不足时返回零梯度
        if neighbor_count < 2 {
            return Some(DVec2::ZERO);
        }

        Self::solve_2x2(a11, a12, a22, b1, b2, self.config.det_min)
            .map(|(x, y)| DVec2::new(x, y))
    }
}

impl GradientMethodGeneric<f64> for LeastSquaresGradient {
    fn compute_scalar_gradient(
        &self,
        field: &[f64],
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage,
    ) {
        if output.len() != mesh.n_cells() {
            output.resize(mesh.n_cells());
        }
        output.reset();

        let mut fallback_cells = Vec::new();

        // 计算所有单元梯度
        for cell in 0..mesh.n_cells() {
            match self.compute_cell_gradient(cell, field, mesh) {
                Some(g) => output.set(cell, g),
                None => fallback_cells.push(cell),
            }
        }

        // 回退处理奇异单元
        if !fallback_cells.is_empty() {
            let mut fb = ScalarGradientStorage::new(mesh.n_cells());
            self.fallback.compute_scalar_gradient(field, mesh, &mut fb);
            for cell in fallback_cells {
                output.set(cell, fb.get(cell));
            }
        }
    }

    fn compute_vector_gradient(
        &self,
        field_u: &[f64],
        field_v: &[f64],
        mesh: &PhysicsMesh,
        output: &mut VectorGradientStorage,
    ) {
        if output.len() != mesh.n_cells() {
            output.resize(mesh.n_cells());
        }

        // 分别计算 u 和 v 的梯度
        let mut grad_u = ScalarGradientStorage::new(mesh.n_cells());
        let mut grad_v = ScalarGradientStorage::new(mesh.n_cells());

        self.compute_scalar_gradient(field_u, mesh, &mut grad_u);
        self.compute_scalar_gradient(field_v, mesh, &mut grad_v);

        output.du_dx = grad_u.grad_x;
        output.du_dy = grad_u.grad_y;
        output.dv_dx = grad_v.grad_x;
        output.dv_dy = grad_v.grad_y;
    }

    fn name(&self) -> &'static str {
        "Least-Squares"
    }

    fn supports_parallel(&self) -> bool {
        self.config.parallel
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

    fn create_test_mesh() -> PhysicsMesh {
        let frozen = FrozenMesh {
            n_nodes: 6,
            node_coords: vec![
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(2.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
                Point3D::new(1.0, 1.0, 0.0),
                Point3D::new(2.0, 1.0, 0.0),
            ],
            n_cells: 2,
            cell_center: vec![
                Point2D::new(0.5, 0.5),
                Point2D::new(1.5, 0.5),
            ],
            cell_area: vec![1.0, 1.0],
            cell_z_bed: vec![0.0, 0.0],
            cell_node_offsets: vec![0, 4, 8],
            cell_node_indices: vec![0, 1, 4, 3, 1, 2, 5, 4],
            cell_face_offsets: vec![0, 4, 8],
            cell_face_indices: vec![0, 1, 2, 3, 0, 4, 5, 6],
            cell_neighbor_offsets: vec![0, 1, 2],
            cell_neighbor_indices: vec![1, 0],
            n_faces: 7,
            n_interior_faces: 1,
            face_center: vec![
                Point2D::new(1.0, 0.5),
                Point2D::new(0.5, 0.0),
                Point2D::new(0.0, 0.5),
                Point2D::new(0.5, 1.0),
                Point2D::new(1.5, 0.0),
                Point2D::new(2.0, 0.5),
                Point2D::new(1.5, 1.0),
            ],
            face_normal: vec![
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, -1.0, 0.0),
                Point3D::new(-1.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
                Point3D::new(0.0, -1.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
            ],
            face_length: vec![1.0; 7],
            face_z_left: vec![0.0; 7],
            face_z_right: vec![0.0; 7],
            face_owner: vec![0, 0, 0, 0, 1, 1, 1],
            face_neighbor: vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            face_delta_owner: vec![Point2D::new(0.0, 0.0); 7],
            face_delta_neighbor: vec![Point2D::new(0.0, 0.0); 7],
            face_dist_o2n: vec![1.0; 7],
            boundary_face_indices: (1..7).map(|i| i as u32).collect(),
            boundary_names: vec!["boundary".to_string()],
            face_boundary_id: vec![None, Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)],
            min_cell_size: 1.0,
            max_cell_size: 1.0,
            // AMR 预分配字段
            cell_refinement_level: vec![0; 2],
            cell_parent: vec![0, 1],
            ghost_capacity: 0,
            // ID 映射与排列字段
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
        };

        PhysicsMesh::from_frozen(&frozen)
    }

    #[test]
    fn test_least_squares_uniform_field() {
        let mesh = create_test_mesh();
        let ls = LeastSquaresGradient::new();

        let field = vec![1.0, 1.0];
        let mut output = ScalarGradientStorage::new(2);

        ls.compute_scalar_gradient(&field, &mesh, &mut output);

        for i in 0..2 {
            let grad = output.get(i);
            assert!(grad.length() < 1e-6, "单元{} 梯度应接近零: {:?}", i, grad);
        }
    }

    #[test]
    fn test_least_squares_linear_field() {
        let mesh = create_test_mesh();
        let ls = LeastSquaresGradient::new();

        // 线性场 φ = x
        let field = vec![0.5, 1.5];
        let mut output = ScalarGradientStorage::new(2);

        ls.compute_scalar_gradient(&field, &mesh, &mut output);

        // x 方向梯度应为正
        let grad0 = output.get(0);
        let grad1 = output.get(1);
        
        assert!(grad0.x > 0.0 || grad1.x > 0.0, 
            "x方向应有正梯度: grad0={:?}, grad1={:?}", grad0, grad1);
    }

    #[test]
    fn test_solve_2x2() {
        // 测试简单情况: [2 0][x] = [4]
        //               [0 2][y]   [6]
        let result = LeastSquaresGradient::solve_2x2(2.0, 0.0, 2.0, 4.0, 6.0, 1e-12);
        assert!(result.is_some());
        let (x, y) = result.unwrap();
        assert!((x - 2.0).abs() < 1e-10);
        assert!((y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_2x2_singular() {
        // 奇异矩阵
        let result = LeastSquaresGradient::solve_2x2(1.0, 1.0, 1.0, 1.0, 1.0, 1e-12);
        assert!(result.is_none());
    }

    #[test]
    fn test_least_squares_config() {
        let ls = LeastSquaresGradient::new()
            .with_det_min(1e-10)
            .with_boundary_contributions(false)
            .with_parallel(false);

        assert!((ls.config.det_min - 1e-10).abs() < 1e-15);
        assert!(!ls.config.use_boundary_contributions);
        assert!(!ls.supports_parallel());
    }
}
