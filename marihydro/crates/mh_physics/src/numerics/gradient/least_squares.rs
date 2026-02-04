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
//! 1. **全泛型**: 实现 `GradientMethod<B>` 支持任意 Backend
//! 2. **无 DVec2**: 所有几何操作使用 `Vector2D` 接口
//! 3. **泛型标量**: 几何与梯度计算在 `S` 上完成

use mh_runtime::{Backend, Vector2D};
use rayon::prelude::*;

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use super::green_gauss::GreenGaussGradient;
use crate::adapter::PhysicsMesh;
use crate::types::CellIndex;
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

    /// 是否支持并行
    pub fn supports_parallel(&self) -> bool {
        self.config.parallel
    }

    /// 求解 2x2 对称正定系统
    ///
    /// [a11 a12] [x1]   [b1]
    /// [a12 a22] [x2] = [b2]
    #[inline]
    fn solve_2x2<B: Backend>(
        a11: B::Scalar,
        a12: B::Scalar,
        a22: B::Scalar,
        b1: B::Scalar,
        b2: B::Scalar,
        det_min: B::Scalar,
    ) -> Option<(B::Scalar, B::Scalar)> {
        let det = a11 * a22 - a12 * a12;
        if det.abs() < det_min {
            return None;
        }
        let inv = B::Scalar::ONE / det;
        let x1 = (a22 * b1 - a12 * b2) * inv;
        let x2 = (a11 * b2 - a12 * b1) * inv;
        if x1.is_finite() && x2.is_finite() {
            Some((x1, x2))
        } else {
            None
        }
    }

    /// 计算单个单元的梯度 - 泛型版本
    fn compute_cell_gradient<B: Backend>(
        &self,
        backend: &B,
        cell: usize,
        field: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
    ) -> Option<(B::Scalar, B::Scalar)> {
        let cell_idx = mh_runtime::CellIndex(cell);
        let cell_center = mesh
            .cell_center_generic::<B>(CellIndex::new(cell))
            .expect("cell_center out of range");
        let cell_center_x = cell_center.x();
        let cell_center_y = cell_center.y();
        let phi_c = field[cell];

        let mut a11 = B::Scalar::ZERO;
        let mut a12 = B::Scalar::ZERO;
        let mut a22 = B::Scalar::ZERO;
        let mut b1 = B::Scalar::ZERO;
        let mut b2 = B::Scalar::ZERO;
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
                let other_center = mesh
                    .cell_center_generic::<B>(other)
                    .expect("cell_center out of range");

                let dx = other_center.x() - cell_center_x;
                let dy = other_center.y() - cell_center_y;
                let dphi = field[other.get()] - phi_c;

                let dist_sq = dx * dx + dy * dy;
                if dist_sq < 1e-20 {
                    continue;
                }

                let w = backend.scalar_from_f64(1.0 / dist_sq);
                let dx_s = backend.scalar_from_f64(dx);
                let dy_s = backend.scalar_from_f64(dy);

                a11 = a11 + w * dx_s * dx_s;
                a12 = a12 + w * dx_s * dy_s;
                a22 = a22 + w * dy_s * dy_s;
                b1 = b1 + w * dx_s * dphi;
                b2 = b2 + w * dy_s * dphi;
                neighbor_count += 1;
            } else if self.config.use_boundary_contributions {
                // 边界面：使用镜像点策略
                let face_center = mesh
                    .face_center_generic::<B>(face)
                    .expect("face_center out of range");
                let normal = mesh
                    .face_normal_generic::<B>(face)
                    .expect("face_normal out of range");

                // 单元中心到面的距离
                let to_face_x = face_center.x() - cell_center_x;
                let to_face_y = face_center.y() - cell_center_y;
                let dist_to_face = to_face_x * normal.x() + to_face_y * normal.y();

                if dist_to_face.abs() < 1e-14 {
                    continue;
                }

                // 虚拟点：面的另一侧镜像
                let abs_dist = dist_to_face.abs();
                let ghost_x = face_center.x() + normal.x() * abs_dist;
                let ghost_y = face_center.y() + normal.y() * abs_dist;
                let dx = ghost_x - cell_center_x;
                let dy = ghost_y - cell_center_y;
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1e-20 {
                    continue;
                }

                let w = backend.scalar_from_f64(1.0 / dist_sq);
                let dx_s = backend.scalar_from_f64(dx);
                let dy_s = backend.scalar_from_f64(dy);

                a11 = a11 + w * dx_s * dx_s;
                a12 = a12 + w * dx_s * dy_s;
                a22 = a22 + w * dy_s * dy_s;
                // b1, b2 不变（dphi = 0 对于零梯度边界条件）
                neighbor_count += 1;
            }
        }

        // 邻居不足时返回零梯度
        if neighbor_count < 2 {
            return Some((B::Scalar::ZERO, B::Scalar::ZERO));
        }

        let det_min = backend.scalar_from_f64(self.config.det_min);
        Self::solve_2x2::<B>(a11, a12, a22, b1, b2, det_min)
    }
}

// ============================================================
// 泛型 trait 实现
// ============================================================

impl<B: Backend> GradientMethod<B> for LeastSquaresGradient {
    fn compute_scalar_gradient(
        &self,
        backend: &B,
        field: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage<B>,
    ) {
        if output.len() != mesh.cell_count() {
            output.resize(mesh.cell_count());
        }
        output.reset();

        let use_parallel = self.config.parallel && mesh.cell_count() >= self.config.parallel_threshold;

        let mut fallback_cells = Vec::new();

        if use_parallel {
            let grads: Vec<Option<(B::Scalar, B::Scalar)>> = (0..mesh.cell_count())
                .into_par_iter()
                .map(|cell| self.compute_cell_gradient(backend, cell, field, mesh))
                .collect();

            for (cell, grad) in grads.into_iter().enumerate() {
                match grad {
                    Some(g) => output.set_tuple(cell, g),
                    None => fallback_cells.push(cell),
                }
            }
        } else {
            for cell in 0..mesh.cell_count() {
                match self.compute_cell_gradient(backend, cell, field, mesh) {
                    Some(g) => output.set_tuple(cell, g),
                    None => fallback_cells.push(cell),
                }
            }
        }

        if !fallback_cells.is_empty() {
            let mut fb = ScalarGradientStorage::with_backend(backend, mesh.cell_count());
            self.fallback.compute_scalar_gradient(backend, field, mesh, &mut fb);
            for cell in fallback_cells {
                output.set_tuple(cell, fb.get_tuple(cell));
            }
        }
    }

    fn compute_vector_gradient(
        &self,
        backend: &B,
        field_u: &B::Buffer<B::Scalar>,
        field_v: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
        output: &mut VectorGradientStorage<B>,
    ) {
        if output.len() != mesh.cell_count() {
            output.resize(mesh.cell_count());
        }

        // 分别计算 u 和 v 的梯度
        let mut grad_u = ScalarGradientStorage::with_backend(backend, mesh.cell_count());
        let mut grad_v = ScalarGradientStorage::with_backend(backend, mesh.cell_count());

        self.compute_scalar_gradient(backend, field_u, mesh, &mut grad_u);
        self.compute_scalar_gradient(backend, field_v, mesh, &mut grad_v);

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
    use mh_runtime::CpuBackend;

    fn create_test_mesh() -> PhysicsMesh {
        let mut frozen = FrozenMesh::empty_with_cells(2);
        frozen.n_nodes = 6;
        frozen.node_coords = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(2.0, 1.0, 0.0),
        ];
        frozen.n_cells = 2;
        frozen.cell_center = vec![Point2D::new(0.5, 0.5), Point2D::new(1.5, 0.5)];
        frozen.cell_area = vec![1.0, 1.0];
        frozen.cell_z_bed = vec![0.0, 0.0];
        frozen.cell_node_offsets = vec![0, 4, 8];
        frozen.cell_node_indices = vec![0, 1, 4, 3, 1, 2, 5, 4];
        frozen.cell_face_offsets = vec![0, 4, 8];
        frozen.cell_face_indices = vec![0, 1, 2, 3, 0, 4, 5, 6];
        frozen.cell_neighbor_offsets = vec![0, 1, 2];
        frozen.cell_neighbor_indices = vec![1, 0];
        frozen.n_faces = 7;
        frozen.n_interior_faces = 1;
        frozen.face_center = vec![
            Point2D::new(1.0, 0.5),
            Point2D::new(0.5, 0.0),
            Point2D::new(0.0, 0.5),
            Point2D::new(0.5, 1.0),
            Point2D::new(1.5, 0.0),
            Point2D::new(2.0, 0.5),
            Point2D::new(1.5, 1.0),
        ];
        frozen.face_normal = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, -1.0, 0.0),
            Point3D::new(-1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, -1.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        frozen.face_length = vec![1.0; 7];
        frozen.face_z_left = vec![0.0; 7];
        frozen.face_z_right = vec![0.0; 7];
        frozen.face_owner = vec![0, 0, 0, 0, 1, 1, 1];
        frozen.face_neighbor = vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        frozen.face_delta_owner = vec![Point2D::new(0.0, 0.0); 7];
        frozen.face_delta_neighbor = vec![Point2D::new(0.0, 0.0); 7];
        frozen.face_dist_o2n = vec![1.0; 7];
        frozen.boundary_face_indices = (1..7).map(|i| i as u32).collect();
        frozen.boundary_names = vec!["boundary".to_string()];
        frozen.face_boundary_id = vec![None, Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)];
        frozen.min_cell_size = 1.0;
        frozen.max_cell_size = 1.0;
        frozen.cell_refinement_level = vec![0; 2];
        frozen.cell_parent = vec![0, 1];
        frozen.ghost_capacity = 0;
        frozen.cell_original_id = Vec::new();
        frozen.face_original_id = Vec::new();
        frozen.cell_permutation = Vec::new();
        frozen.cell_inv_permutation = Vec::new();

        PhysicsMesh::from_frozen(&frozen)
    }

    #[test]
    fn test_least_squares_uniform_field_f64() {
        let mesh = create_test_mesh();
        let ls = LeastSquaresGradient::new();
        let backend = CpuBackend::<f64>::new();

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[1.0, 1.0]);
        let mut output = ScalarGradientStorage::with_backend(&backend, 2);

        ls.compute_scalar_gradient(&backend, &field, &mesh, &mut output);

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
        let backend = CpuBackend::<f32>::new();

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[1.0f32, 1.0f32]);
        let mut output = ScalarGradientStorage::with_backend(&backend, 2);

        ls.compute_scalar_gradient(&backend, &field, &mesh, &mut output);

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
        let backend = CpuBackend::<f64>::new();

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[0.5, 1.5]);
        let mut output = ScalarGradientStorage::with_backend(&backend, 2);

        ls.compute_scalar_gradient(&backend, &field, &mesh, &mut output);

        let (grad0x, _) = output.get_tuple(0);
        let (grad1x, _) = output.get_tuple(1);
        
        assert!(grad0x > 0.0 || grad1x > 0.0, 
            "x方向应有正梯度: grad0x={}, grad1x={}", grad0x, grad1x);
    }

    #[test]
    fn test_solve_2x2() {
        // 测试简单情况: [2 0][x] = [4]
        //               [0 2][y]   [6]
        let result = LeastSquaresGradient::solve_2x2::<CpuBackend<f64>>(2.0f64, 0.0, 2.0, 4.0, 6.0, 1e-12);
        assert!(result.is_some());
        let (x, y) = result.unwrap();
        assert!((x - 2.0).abs() < 1e-10);
        assert!((y - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_solve_2x2_singular() {
        // 奇异矩阵
        let result = LeastSquaresGradient::solve_2x2::<CpuBackend<f64>>(1.0f64, 1.0, 1.0, 1.0, 1.0, 1e-12);
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
