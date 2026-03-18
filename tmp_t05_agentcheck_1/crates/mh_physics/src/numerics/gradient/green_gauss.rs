// crates/mh_physics/src/numerics/gradient/green_gauss.rs

//! Green-Gauss 梯度计算 - 泛型版本
//!
//! 使用 Green 定理将体积分转化为面积分:
//! ∇φ ≈ (1/V) ∮ φ·n dS
//!
//! 对于离散网格:
//! ∇φ_i ≈ (1/A_i) Σ_f φ_f · n_f · L_f
//!
//! # 设计原则
//!
//! 1. **全泛型**: 实现 `GradientMethod<B>` 支持任意 Backend
//! 2. **无 DVec2**: 所有几何操作使用元组 `(f64, f64)` 或 `(S, S)`
//! 3. **几何数据 f64**: PhysicsMesh 几何数据保持 f64，在计算时转换为 S

use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use num_traits::Float;
use rayon::prelude::*;
use log::debug;

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use crate::adapter::PhysicsMesh;
use crate::types::CellIndex;

// ============================================================
// 配置
// ============================================================

/// 面插值方法
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FaceInterpolation {
    /// 简单算术平均
    #[default]
    Arithmetic,
    /// 距离加权插值 (更精确，适用于非均匀网格)
    DistanceWeighted,
}

/// Green-Gauss 梯度配置
#[derive(Debug, Clone)]
pub struct GreenGaussConfig {
    /// 是否启用并行
    pub parallel: bool,
    /// 并行阈值（单元数）
    pub parallel_threshold: usize,
    /// 面插值方法
    pub face_interpolation: FaceInterpolation,
    /// 是否在边界单元强制零梯度（保持静水平衡）
    pub force_zero_gradient_at_boundary: bool,
}

impl Default for GreenGaussConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            parallel_threshold: 1000,
            face_interpolation: FaceInterpolation::Arithmetic,
            force_zero_gradient_at_boundary: true,
        }
    }
}

// ============================================================
// Green-Gauss 梯度计算器
// ============================================================

/// Green-Gauss 梯度计算器
///
/// # 泛型支持
///
/// 实现 `GradientMethod<B>` 支持 f32/f64 精度切换。
/// 内部几何数据使用 f64（来自 PhysicsMesh），在计算时转换为目标精度 S。
#[derive(Debug, Clone)]
pub struct GreenGaussGradient {
    config: GreenGaussConfig,
    /// 边界单元索引缓存（可选性能优化）
    boundary_cells: Option<Vec<bool>>,
}

impl GreenGaussGradient {
    /// 创建新实例
    pub fn new() -> Self {
        Self::with_config(GreenGaussConfig::default())
    }

    /// 使用配置创建
    pub fn with_config(config: GreenGaussConfig) -> Self {
        Self { 
            config,
            boundary_cells: None,
        }
    }

    /// 设置并行开关
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// 设置并行阈值
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.config.parallel_threshold = threshold;
        self
    }

    /// 使用距离加权插值
    pub fn with_distance_weighted(mut self) -> Self {
        self.config.face_interpolation = FaceInterpolation::DistanceWeighted;
        self
    }

    /// 禁用边界零梯度强制
    pub fn without_boundary_zero(mut self) -> Self {
        self.config.force_zero_gradient_at_boundary = false;
        self
    }

    /// 预计算并缓存边界单元索引
    ///
    /// 预计算后每次梯度计算可节省O(N×F)的检测时间。
    pub fn with_boundary_cache(mut self, mesh: &PhysicsMesh) -> Self {
        let mut boundary_cells = vec![false; mesh.cell_count()];
        for cell in 0..mesh.cell_count() {
            let cell_idx = CellIndex::new(cell);
            boundary_cells[cell] = mesh
                .cell_faces(cell_idx)
                .any(|face| mesh.face_neighbor(face).is_none());
        }

        debug!("Cached {} boundary cells out of {} total cells", 
            boundary_cells.iter().filter(|&&is_boundary| is_boundary).count(), mesh.cell_count());

        self.boundary_cells = Some(boundary_cells);
        self
    }

    /// 检查单元是否是边界单元
    #[inline]
    fn is_boundary_cell(&self, cell: usize, mesh: &PhysicsMesh) -> bool {
        if let Some(ref boundary_cells) = self.boundary_cells {
            boundary_cells.get(cell).copied().unwrap_or(false)
        } else {
            let cell_idx = CellIndex::new(cell);
            mesh.cell_faces(cell_idx)
                .any(|face| mesh.face_neighbor(face).is_none())
        }
    }

    /// 计算单个单元的梯度 - 泛型版本
    ///
    /// 几何数据从 PhysicsMesh 获取 (f64)，转换为 S 进行计算
    fn compute_cell_gradient<B: Backend>(
        &self,
        backend: &B,
        cell: usize,
        field: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
    ) -> (B::Scalar, B::Scalar) {
        let cell_idx = CellIndex::new(cell);
        let area_f64 = mesh.cell_area_unchecked(cell_idx);
        
        if area_f64 < 1e-14 {
            return (B::Scalar::ZERO, B::Scalar::ZERO);
        }

        // 边界单元强制零梯度
        if self.config.force_zero_gradient_at_boundary && self.is_boundary_cell(cell, mesh) {
            return (B::Scalar::ZERO, B::Scalar::ZERO);
        }

        let cell_center = mesh
            .cell_center_generic::<B>(cell_idx)
            .expect("cell_center out of range");
        let cell_center_x = cell_center.x();
        let cell_center_y = cell_center.y();
        let phi_c = field[cell];
        let mut grad_x = B::Scalar::ZERO;
        let mut grad_y = B::Scalar::ZERO;

        for face in mesh.cell_faces(cell_idx) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let is_owner = owner == cell_idx;
            let is_neighbor = neighbor == Some(cell_idx);

            if !is_owner && !is_neighbor {
                continue;
            }

            // 几何数据 (f64)
            let normal = mesh
                .face_normal_generic::<B>(face)
                .expect("face_normal out of range");
            let length = backend.scalar_from_f64(mesh.face_length(face));
            let sign = if is_owner { B::Scalar::ONE } else { -B::Scalar::ONE };
            let ds_x = normal.x() * length * sign;
            let ds_y = normal.y() * length * sign;

            // 计算面值
            let phi_face = if let Some(neigh) = neighbor {
                let other = if is_owner { neigh } else { owner };

                match self.config.face_interpolation {
                    FaceInterpolation::Arithmetic => {
                        B::Scalar::HALF * (phi_c + field[other.get()])
                    }
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh
                            .face_center_generic::<B>(face)
                            .expect("face_center out of range");
                        let other_center = mesh
                            .cell_center_generic::<B>(other)
                            .expect("cell_center out of range");

                        let dx_self = face_center.x() - cell_center_x;
                        let dy_self = face_center.y() - cell_center_y;
                        let d_self = (dx_self * dx_self + dy_self * dy_self).sqrt();

                        let dx_other = face_center.x() - other_center.x();
                        let dy_other = face_center.y() - other_center.y();
                        let d_other = (dx_other * dx_other + dy_other * dy_other).sqrt();

                        let eps = B::Scalar::from_config(1e-14).unwrap_or(B::Scalar::MIN_POSITIVE);
                        Self::distance_weighted_interpolate::<B>(
                            phi_c,
                            field[other.get()],
                            d_self,
                            d_other,
                            eps,
                        )
                    }
                }
            } else {
                phi_c
            };

            grad_x = grad_x + ds_x * phi_face;
            grad_y = grad_y + ds_y * phi_face;
        }

        let area = B::Scalar::from_config(area_f64).unwrap_or(B::Scalar::ZERO);
        (grad_x / area, grad_y / area)
    }

    /// 距离加权插值 - 泛型版本
    #[inline]
    fn distance_weighted_interpolate<B: Backend>(
        phi_o: B::Scalar,
        phi_n: B::Scalar,
        d_o: B::Scalar,
        d_n: B::Scalar,
        eps: B::Scalar,
    ) -> B::Scalar {
        let d_total = d_o + d_n;
        if d_total < eps {
            B::Scalar::HALF * (phi_o + phi_n)
        } else {
            (phi_n * d_o + phi_o * d_n) / d_total
        }
    }

    /// 串行计算标量梯度 - 泛型版本
    fn compute_scalar_serial<B: Backend>(
        &self,
        backend: &B,
        field: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage<B>,
    ) {
        output.reset();
        for cell in 0..mesh.cell_count() {
            let grad = self.compute_cell_gradient(backend, cell, field, mesh);
            output.set_tuple(cell, grad);
        }
    }

    /// 并行计算标量梯度 - 泛型版本
    fn compute_scalar_parallel<B: Backend>(
        &self,
        backend: &B,
        field: &B::Buffer<B::Scalar>,
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage<B>,
    ) {
        let grads: Vec<(B::Scalar, B::Scalar)> = (0..mesh.cell_count())
            .into_par_iter()
            .map(|cell| self.compute_cell_gradient(backend, cell, field, mesh))
            .collect();

        for (i, g) in grads.into_iter().enumerate() {
            output.set_tuple(i, g);
        }
    }
}

impl Default for GreenGaussGradient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 泛型 trait 实现
// ============================================================

impl<B: Backend> GradientMethod<B> for GreenGaussGradient {
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

        if self.config.parallel && mesh.cell_count() >= self.config.parallel_threshold {
            self.compute_scalar_parallel(backend, field, mesh, output);
        } else {
            self.compute_scalar_serial(backend, field, mesh, output);
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

        // 复制到输出
        output.du_dx = grad_u.grad_x;
        output.du_dy = grad_u.grad_y;
        output.dv_dx = grad_v.grad_x;
        output.dv_dy = grad_v.grad_y;
    }

    fn name(&self) -> &'static str {
        "Green-Gauss"
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

    /// 创建简单的 2x1 网格
    fn create_test_mesh() -> PhysicsMesh {
        let backend = CpuBackend::<f64>::new();
        let mut frozen = FrozenMesh::empty_with_cells_backend(backend.clone(), 2);
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
        let mut cell_area = backend.alloc(2);
        cell_area.copy_from_slice(&[1.0, 1.0]);
        frozen.cell_area = cell_area;
        let mut cell_z_bed = backend.alloc(2);
        cell_z_bed.copy_from_slice(&[0.0, 0.0]);
        frozen.cell_z_bed = cell_z_bed;
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
        let mut face_length = backend.alloc(7);
        face_length.copy_from_slice(&[1.0; 7]);
        frozen.face_length = face_length;
        let mut face_z_left = backend.alloc(7);
        face_z_left.copy_from_slice(&[0.0; 7]);
        frozen.face_z_left = face_z_left;
        let mut face_z_right = backend.alloc(7);
        face_z_right.copy_from_slice(&[0.0; 7]);
        frozen.face_z_right = face_z_right;
        frozen.face_owner = vec![0, 0, 0, 0, 1, 1, 1];
        frozen.face_neighbor = vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        frozen.face_delta_owner = vec![Point2D::new(0.0, 0.0); 7];
        frozen.face_delta_neighbor = vec![Point2D::new(0.0, 0.0); 7];
        let mut face_dist_o2n = backend.alloc(7);
        face_dist_o2n.copy_from_slice(&[1.0; 7]);
        frozen.face_dist_o2n = face_dist_o2n;
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
    fn test_green_gauss_uniform_field_f64() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new();
        let backend = CpuBackend::<f64>::new();

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[1.0, 1.0]);
        let mut output = ScalarGradientStorage::with_backend(&backend, 2);

        gg.compute_scalar_gradient(&backend, &field, &mesh, &mut output);

        for i in 0..2 {
            let (gx, gy) = output.get_tuple(i);
            let len = (gx * gx + gy * gy).sqrt();
            assert!(len < 1e-6, "单元{} 梯度应接近零: ({}, {})", i, gx, gy);
        }
    }

    #[test]
    fn test_green_gauss_uniform_field_f32() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new();
        let backend = CpuBackend::<f32>::new();

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[1.0f32, 1.0f32]);
        let mut output = ScalarGradientStorage::with_backend(&backend, 2);

        gg.compute_scalar_gradient(&backend, &field, &mesh, &mut output);

        for i in 0..2 {
            let (gx, gy) = output.get_tuple(i);
            let len = (gx * gx + gy * gy).sqrt();
            assert!(len < 1e-4, "单元{} 梯度应接近零: ({}, {})", i, gx, gy);
        }
    }

    #[test]
    fn test_green_gauss_linear_field() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new()
            .with_parallel(false)
            .without_boundary_zero();
        let backend = CpuBackend::<f64>::new();

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[0.5, 1.5]);
        let mut output = ScalarGradientStorage::with_backend(&backend, 2);

        gg.compute_scalar_gradient(&backend, &field, &mesh, &mut output);

        let (grad0x, _) = output.get_tuple(0);
        let (grad1x, _) = output.get_tuple(1);
        
        assert!(grad0x > 0.0, "单元0 x方向梯度应为正: {}", grad0x);
        assert!(grad1x > 0.0, "单元1 x方向梯度应为正: {}", grad1x);
    }

    #[test]
    fn test_green_gauss_config() {
        let gg = GreenGaussGradient::new()
            .with_parallel(false)
            .with_threshold(500)
            .with_distance_weighted();

        assert!(!<GreenGaussGradient as GradientMethod<CpuBackend<f64>>>::supports_parallel(&gg));
        assert_eq!(gg.config.parallel_threshold, 500);
        assert_eq!(gg.config.face_interpolation, FaceInterpolation::DistanceWeighted);
    }

    #[test]
    fn test_vector_gradient() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new()
            .with_parallel(false)
            .without_boundary_zero();
        let backend = CpuBackend::<f64>::new();

        let mut u = backend.alloc(2);
        let mut v = backend.alloc(2);
        u.copy_from_slice(&[0.5, 1.5]);
        v.copy_from_slice(&[0.0, 0.0]);
        let mut output = VectorGradientStorage::with_backend(&backend, 2);

        gg.compute_vector_gradient(&backend, &u, &v, &mesh, &mut output);

        assert!(output.du_dx[0] > 0.0 || output.du_dx[1] > 0.0);
        assert!(output.dv_dy[0].abs() < 1e-6);
        assert!(output.dv_dy[1].abs() < 1e-6);
    }

    #[test]
    fn test_boundary_caching() {
        let mesh = create_test_mesh();
        let backend = CpuBackend::<f64>::new();
        
        let gg_no_cache = GreenGaussGradient::new();
        let gg_with_cache = GreenGaussGradient::new().with_boundary_cache(&mesh);

        let mut field = backend.alloc(2);
        field.copy_from_slice(&[1.0, 1.0]);
        let mut output1 = ScalarGradientStorage::with_backend(&backend, 2);
        let mut output2 = ScalarGradientStorage::with_backend(&backend, 2);

        gg_no_cache.compute_scalar_gradient(&backend, &field, &mesh, &mut output1);
        gg_with_cache.compute_scalar_gradient(&backend, &field, &mesh, &mut output2);

        for i in 0..2 {
            let (g1x, g1y) = output1.get_tuple(i);
            let (g2x, g2y) = output2.get_tuple(i);
            let diff = ((g1x - g2x).powi(2) + (g1y - g2y).powi(2)).sqrt();
            assert!(diff < 1e-10, "缓存与非缓存结果不一致 at cell {}", i);
        }
    }
}
