// crates/mh_physics/src/numerics/gradient/least_squares.rs

//! 最小二乘梯度计算 - 泛型版本
//!
//! 通过最小化加权最小二乘误差来计算梯度:
//! min Σ w_j * (φ_j - φ_i - ∇φ_i · r_ij)²
//!
//! 对于 2D 情况，求解 2x2 法方程：
//! [a11 a12] [∂φ/∂x]   [b1]
//! [a12 a22] [∂φ/∂y] = [b2]
//!
//! # 设计原则
//!
//! 1. **全泛型**: 实现 `GradientMethodGeneric<S>` 支持任意 RuntimeScalar
//! 2. **无 DVec2**: 所有几何操作使用元组 `(f64, f64)` 或 `(S, S)`
//! 3. **几何数据 f64**: PhysicsMesh 几何数据保持 f64，在计算时转换为 S

use mh_runtime::RuntimeScalar;

use super::traits::{GradientMethodGeneric, ScalarGradientStorageGeneric, VectorGradientStorageGeneric};
use super::green_gauss::GreenGaussGradient;
use crate::adapter::PhysicsMesh;
use crate::types::NumericalParams;

// ============================================================
// 配置
// ============================================================

/// 最小二乘梯度配置
#[derive(Debug, Clone)]
pub struct LeastSquaresConfig {
    /// 行列式最小值（判断奇异性）
    pub det_min: f64,
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
    pub fn from_params(params: &NumericalParams<f64>) -> Self {
        Self {
            config: LeastSquaresConfig {
                det_min: params.det_min,
                ..Default::default()
            },
            fallback: GreenGaussGradient::new(),
        }
    }

    /// 设置行列式最小值
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
    #[inline]
    fn solve_2x2<S: RuntimeScalar>(
        a11: S,
        a12: S,
        a22: S,
        b1: S,
        b2: S,
        det_min: S,
    ) -> Option<(S, S)> {
        let det = a11 * a22 - a12 * a12;
        if det.abs() < det_min {
            return None;
        }
        let inv = S::ONE / det;
        let x1 = (a22 * b1 - a12 * b2) * inv;
        let x2 = (a11 * b2 - a12 * b1) * inv;
        if x1.is_finite() && x2.is_finite() {
            Some((x1, x2))
        } else {
            None
        }
    }

    /// 计算单个单元的梯度 - 泛型版本
    fn compute_cell_gradient<S: RuntimeScalar>(
        &self,
        cell: usize,
        field: &[S],
        mesh: &PhysicsMesh,
    ) -> Option<(S, S)> {
        let cell_idx = mh_runtime::CellIndex(cell);
        let cell_center = mesh.cell_center_tuple(cell);
        let phi_c = field[cell];

        let mut a11 = S::ZERO;
        let mut a12 = S::ZERO;
        let mut a22 = S::ZERO;
        let mut b1 = S::ZERO;
        let mut b2 = S::ZERO;
        let mut neighbor_count = 0;

        // 收集邻居贡献
        for face in mesh.cell_faces(cell_idx) {
            let owner = mesh.face_owner(face);
            let neighbor_opt = mesh.face_neighbor(face);

            let is_owner = owner == cell_idx;
            let is_neighbor = neighbor_opt == Some(cell_idx);

            if !is_owner && !is_neighbor {
                continue;
            }

            if let Some(neighbor) = neighbor_opt {
                // 内部面：使用邻居单元
                let other = if is_owner { neighbor } else { owner };
                let other_center = mesh.cell_center_tuple(other.get());

                let dx_f64 = other_center.0 - cell_center.0;
                let dy_f64 = other_center.1 - cell_center.1;
                let dphi = field[other.get()] - phi_c;

                let dist_sq_f64 = dx_f64 * dx_f64 + dy_f64 * dy_f64;
                if dist_sq_f64 < 1e-20 {
                    continue;
                }

                // 转换为 S
                let dx = S::from_f64(dx_f64).unwrap_or(S::ZERO);
                let dy = S::from_f64(dy_f64).unwrap_or(S::ZERO);
                let w = S::from_f64(1.0 / dist_sq_f64).unwrap_or(S::ONE);

                a11 = a11 + w * dx * dx;
                a12 = a12 + w * dx * dy;
                a22 = a22 + w * dy * dy;
                b1 = b1 + w * dx * dphi;
                b2 = b2 + w * dy * dphi;
                neighbor_count += 1;
            } else if self.config.use_boundary_contributions {
                // 边界面：使用镜像点策略
                let face_center = mesh.face_center_tuple(face.get());
                let (nx, ny) = mesh.face_normal_2d_tuple(face.get());

                // 单元中心到面的距离
                let to_face_x = face_center.0 - cell_center.0;
                let to_face_y = face_center.1 - cell_center.1;
                let dist_to_face = to_face_x * nx + to_face_y * ny;

                if dist_to_face.abs() < 1e-14 {
                    continue;
                }

                // 虚拟点：面的另一侧镜像
                let ghost_x = face_center.0 + nx * dist_to_face.abs();
                let ghost_y = face_center.1 + ny * dist_to_face.abs();
                let dx_f64 = ghost_x - cell_center.0;
                let dy_f64 = ghost_y - cell_center.1;
                let dist_sq_f64 = dx_f64 * dx_f64 + dy_f64 * dy_f64;

                if dist_sq_f64 < 1e-20 {
                    continue;
                }

                // 转换为 S
                let dx = S::from_f64(dx_f64).unwrap_or(S::ZERO);
                let dy = S::from_f64(dy_f64).unwrap_or(S::ZERO);
                let w = S::from_f64(1.0 / dist_sq_f64).unwrap_or(S::ONE);

                a11 = a11 + w * dx * dx;
                a12 = a12 + w * dx * dy;
                a22 = a22 + w * dy * dy;
                // b1, b2 不变（dphi = 0 对于零梯度边界条件）
                neighbor_count += 1;
            }
        }

        // 邻居不足时返回零梯度
        if neighbor_count < 2 {
            return Some((S::ZERO, S::ZERO));
        }

        let det_min = S::from_f64(self.config.det_min).unwrap_or(S::MIN_POSITIVE);
        Self::solve_2x2(a11, a12, a22, b1, b2, det_min)
    }
}

// ============================================================
// 泛型 trait 实现
// ============================================================

impl<S: RuntimeScalar> GradientMethodGeneric<S> for LeastSquaresGradient {
    fn compute_scalar_gradient(
        &self,
        field: &[S],
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorageGeneric<S>,
    ) {
        if output.len() != mesh.n_cells() {
            output.resize(mesh.n_cells());
        }
        output.reset();

        let mut fallback_cells = Vec::new();

        // 计算所有单元梯度
        for cell in 0..mesh.n_cells() {
            match self.compute_cell_gradient(cell, field, mesh) {
                Some(g) => output.set_tuple(cell, g),
                None => fallback_cells.push(cell),
            }
        }

        // 回退处理奇异单元
        if !fallback_cells.is_empty() {
            let mut fb = ScalarGradientStorageGeneric::<S>::new(mesh.n_cells());
            self.fallback.compute_scalar_gradient(field, mesh, &mut fb);
            for cell in fallback_cells {
                output.set_tuple(cell, fb.get_tuple(cell));
            }
        }
    }

    fn compute_vector_gradient(
        &self,
        field_u: &[S],
        field_v: &[S],
        mesh: &PhysicsMesh,
        output: &mut VectorGradientStorageGeneric<S>,
    ) {
        if output.len() != mesh.n_cells() {
            output.resize(mesh.n_cells());
        }

        // 分别计算 u 和 v 的梯度
        let mut grad_u = ScalarGradientStorageGeneric::<S>::new(mesh.n_cells());
        let mut grad_v = ScalarGradientStorageGeneric::<S>::new(mesh.n_cells());

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
            cell_refinement_level: vec![0; 2],
            cell_parent: vec![0, 1],
            ghost_capacity: 0,
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
        };

        PhysicsMesh::from_frozen(&frozen)
    }

    #[test]
    fn test_least_squares_uniform_field_f64() {
        let mesh = create_test_mesh();
        let ls = LeastSquaresGradient::new();

        let field: Vec<f64> = vec![1.0, 1.0];
        let mut output = ScalarGradientStorageGeneric::<f64>::new(2);

        ls.compute_scalar_gradient(&field, &mesh, &mut output);

        for i in 0..2 {
            let (gx, gy) = output.get_tuple(i);
            let len = (gx * gx + gy * gy).sqrt();
            assert!(len < 1e-6, "单元{} 梯度应接近零: ({}, {})", i, gx, gy);
        }
    }

    #[test]
    fn test_least_squares_uniform_field_f32() {
        let mesh = create_test_mesh();
        let ls = LeastSquaresGradient::new();

        let field: Vec<f32> = vec![1.0f32, 1.0f32];
        let mut output = ScalarGradientStorageGeneric::<f32>::new(2);

        ls.compute_scalar_gradient(&field, &mesh, &mut output);

        for i in 0..2 {
            let (gx, gy) = output.get_tuple(i);
            let len = (gx * gx + gy * gy).sqrt();
            assert!(len < 1e-4, "单元{} 梯度应接近零: ({}, {})", i, gx, gy);
        }
    }

    #[test]
    fn test_least_squares_linear_field() {
        let mesh = create_test_mesh();
        let ls = LeastSquaresGradient::new();

        let field: Vec<f64> = vec![0.5, 1.5];
        let mut output = ScalarGradientStorageGeneric::<f64>::new(2);

        ls.compute_scalar_gradient(&field, &mesh, &mut output);

        let (grad0x, _) = output.get_tuple(0);
        let (grad1x, _) = output.get_tuple(1);
        
        assert!(grad0x > 0.0 || grad1x > 0.0, 
            "x方向应有正梯度: grad0x={}, grad1x={}", grad0x, grad1x);
    }

    #[test]
    fn test_solve_2x2() {
        // 测试简单情况: [2 0][x] = [4]
        //               [0 2][y]   [6]
        let result = LeastSquaresGradient::solve_2x2(2.0f64, 0.0, 2.0, 4.0, 6.0, 1e-12);
        assert!(result.is_some());
        let (x, y) = result.unwrap();
        assert!((x - 2.0).abs() < 1e-10);
        assert!((y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_2x2_singular() {
        // 奇异矩阵
        let result = LeastSquaresGradient::solve_2x2(1.0f64, 1.0, 1.0, 1.0, 1.0, 1e-12);
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
