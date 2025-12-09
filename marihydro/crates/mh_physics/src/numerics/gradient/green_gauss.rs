// crates/mh_physics/src/numerics/gradient/green_gauss.rs

//! Green-Gauss 梯度计算
//!
//! 使用 Green 定理将体积分转化为面积分:
//! ∇φ ≈ (1/V) ∮ φ·n dS
//!
//! 对于离散网格:
//! ∇φ_i ≈ (1/A_i) Σ_f φ_f · n_f · L_f

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use crate::adapter::PhysicsMesh;

use glam::DVec2;
use rayon::prelude::*;

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
}

impl Default for GreenGaussConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            parallel_threshold: 1000,
            face_interpolation: FaceInterpolation::Arithmetic,
        }
    }
}

// ============================================================
// Green-Gauss 梯度计算器
// ============================================================

/// Green-Gauss 梯度计算器
#[derive(Debug, Clone)]
pub struct GreenGaussGradient {
    config: GreenGaussConfig,
}

impl Default for GreenGaussGradient {
    fn default() -> Self {
        Self {
            config: GreenGaussConfig::default(),
        }
    }
}

impl GreenGaussGradient {
    /// 创建新实例
    pub fn new() -> Self {
        Self::default()
    }

    /// 使用配置创建
    pub fn with_config(config: GreenGaussConfig) -> Self {
        Self { config }
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

    /// 设置面插值方法
    pub fn with_face_interpolation(mut self, method: FaceInterpolation) -> Self {
        self.config.face_interpolation = method;
        self
    }

    /// 使用距离加权插值（推荐用于非均匀网格）
    pub fn with_distance_weighted(self) -> Self {
        self.with_face_interpolation(FaceInterpolation::DistanceWeighted)
    }

    /// 计算单个单元的标量梯度
    fn compute_cell_gradient(
        &self,
        cell: usize,
        field: &[f64],
        mesh: &PhysicsMesh,
    ) -> DVec2 {
        let area = mesh.cell_area_unchecked(cell);
        if area < 1e-14 {
            return DVec2::ZERO;
        }

        let cell_center = mesh.cell_center(cell);
        let phi_c = field[cell];
        let mut grad = DVec2::ZERO;

        // 仅遍历该单元关联的面，避免 O(N^2)
        for face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let is_owner = owner == cell;
            let is_neighbor = neighbor == Some(cell);

            if !is_owner && !is_neighbor {
                continue;
            }

            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);

            // owner 侧法向指向外侧，neighbor 取相反号
            let sign = if is_owner { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let phi_face = if let Some(neigh) = neighbor {
                let other = if is_owner { neigh } else { owner };

                match self.config.face_interpolation {
                    FaceInterpolation::Arithmetic => 0.5 * (phi_c + field[other]),
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh.face_center(face);
                        let other_center = mesh.cell_center(other);
                        let d_self = (face_center - cell_center).length();
                        let d_other = (face_center - other_center).length();
                        Self::distance_weighted_interpolate(phi_c, field[other], d_self, d_other)
                    }
                }
            } else {
                // 边界面：使用单元中心值
                phi_c
            };

            grad += ds * phi_face;
        }

        grad / area
    }

    /// 计算单个单元的水面高度梯度（C-property 保持）
    ///
    /// 对于静水，梯度应精确为零。使用水位 η = h + z_b 作为梯度变量。
    pub fn compute_water_level_gradient(
        &self,
        cell: usize,
        h: &[f64],
        z_bed: &[f64],
        mesh: &PhysicsMesh,
    ) -> DVec2 {
        let area = mesh.cell_area_unchecked(cell);
        if area < 1e-14 {
            return DVec2::ZERO;
        }

        let cell_center = mesh.cell_center(cell);
        let eta_c = h[cell] + z_bed[cell];
        let mut grad = DVec2::ZERO;

        for face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let is_owner = owner == cell;
            let is_neighbor = neighbor == Some(cell);

            if !is_owner && !is_neighbor {
                continue;
            }

            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            let sign = if is_owner { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let eta_face = if let Some(neigh) = neighbor {
                let other = if is_owner { neigh } else { owner };
                let eta_other = h[other] + z_bed[other];

                match self.config.face_interpolation {
                    FaceInterpolation::Arithmetic => 0.5 * (eta_c + eta_other),
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh.face_center(face);
                        let other_center = mesh.cell_center(other);
                        let d_self = (face_center - cell_center).length();
                        let d_other = (face_center - other_center).length();
                        Self::distance_weighted_interpolate(eta_c, eta_other, d_self, d_other)
                    }
                }
            } else {
                eta_c
            };

            grad += ds * eta_face;
        }

        grad / area
    }

    /// 距离加权插值
    ///
    /// phi_face = (phi_n * d_o + phi_o * d_n) / (d_o + d_n)
    #[inline]
    fn distance_weighted_interpolate(phi_o: f64, phi_n: f64, d_o: f64, d_n: f64) -> f64 {
        let d_total = d_o + d_n;
        if d_total < 1e-14 {
            0.5 * (phi_o + phi_n)
        } else {
            (phi_n * d_o + phi_o * d_n) / d_total
        }
    }

    /// 串行计算标量梯度
    fn compute_scalar_serial(
        &self,
        field: &[f64],
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage,
    ) {
        output.reset();
        for cell in 0..mesh.n_cells() {
            let grad = self.compute_cell_gradient(cell, field, mesh);
            output.set(cell, grad);
        }
    }

    /// 并行计算标量梯度
    fn compute_scalar_parallel(
        &self,
        field: &[f64],
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage,
    ) {
        let grads: Vec<DVec2> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|cell| self.compute_cell_gradient(cell, field, mesh))
            .collect();

        for (i, g) in grads.into_iter().enumerate() {
            output.set(i, g);
        }
    }

    /// 并行计算所有单元梯度（返回 (grad_x, grad_y) 向量）
    pub fn compute_all_parallel(
        &self,
        field: &[f64],
        mesh: &PhysicsMesh,
    ) -> (Vec<f64>, Vec<f64>) {
        let grads: Vec<DVec2> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|cell| self.compute_cell_gradient(cell, field, mesh))
            .collect();

        let n = grads.len();
        let mut grad_x = Vec::with_capacity(n);
        let mut grad_y = Vec::with_capacity(n);

        for g in grads {
            grad_x.push(g.x);
            grad_y.push(g.y);
        }

        (grad_x, grad_y)
    }

    /// 并行计算水面梯度（C-property 保持）
    pub fn compute_water_level_parallel(
        &self,
        h: &[f64],
        z_bed: &[f64],
        mesh: &PhysicsMesh,
    ) -> (Vec<f64>, Vec<f64>) {
        let grads: Vec<DVec2> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|cell| self.compute_water_level_gradient(cell, h, z_bed, mesh))
            .collect();

        let n = grads.len();
        let mut grad_x = Vec::with_capacity(n);
        let mut grad_y = Vec::with_capacity(n);

        for g in grads {
            grad_x.push(g.x);
            grad_y.push(g.y);
        }

        (grad_x, grad_y)
    }
}

impl GradientMethod for GreenGaussGradient {
    fn compute_scalar_gradient(
        &self,
        field: &[f64],
        mesh: &PhysicsMesh,
        output: &mut ScalarGradientStorage,
    ) {
        if output.len() != mesh.n_cells() {
            output.resize(mesh.n_cells());
        }

        if self.config.parallel && mesh.n_cells() >= self.config.parallel_threshold {
            self.compute_scalar_parallel(field, mesh, output);
        } else {
            self.compute_scalar_serial(field, mesh, output);
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

    /// 创建简单的 2x1 网格
    fn create_test_mesh() -> PhysicsMesh {
        // 两个单元并排
        // +---+---+
        // | 0 | 1 |
        // +---+---+
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
                Point2D::new(1.0, 0.5), // 内部面
                Point2D::new(0.5, 0.0), // 下边界
                Point2D::new(0.0, 0.5), // 左边界
                Point2D::new(0.5, 1.0), // 上边界
                Point2D::new(1.5, 0.0), // 下边界
                Point2D::new(2.0, 0.5), // 右边界
                Point2D::new(1.5, 1.0), // 上边界
            ],
            face_normal: vec![
                Point3D::new(1.0, 0.0, 0.0),  // 内部: 指向右
                Point3D::new(0.0, -1.0, 0.0), // 下
                Point3D::new(-1.0, 0.0, 0.0), // 左
                Point3D::new(0.0, 1.0, 0.0),  // 上
                Point3D::new(0.0, -1.0, 0.0), // 下
                Point3D::new(1.0, 0.0, 0.0),  // 右
                Point3D::new(0.0, 1.0, 0.0),  // 上
            ],
            face_length: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
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
    fn test_green_gauss_uniform_field() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new();

        // 均匀场，梯度应为零
        let field = vec![1.0, 1.0];
        let mut output = ScalarGradientStorage::new(2);

        gg.compute_scalar_gradient(&field, &mesh, &mut output);

        for i in 0..2 {
            let grad = output.get(i);
            assert!(grad.length() < 1e-6, "单元{} 梯度应接近零: {:?}", i, grad);
        }
    }

    #[test]
    fn test_green_gauss_linear_field() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new().with_parallel(false);

        // 线性场 φ = x，梯度应为 (1, 0)
        // 单元0中心 (0.5, 0.5)，单元1中心 (1.5, 0.5)
        let field = vec![0.5, 1.5];
        let mut output = ScalarGradientStorage::new(2);

        gg.compute_scalar_gradient(&field, &mesh, &mut output);

        // 由于边界条件简化，梯度可能不是精确的 (1, 0)
        // 但应该在 x 方向有正梯度
        let grad0 = output.get(0);
        let grad1 = output.get(1);
        
        assert!(grad0.x > 0.0, "单元0 x方向梯度应为正: {}", grad0.x);
        assert!(grad1.x > 0.0, "单元1 x方向梯度应为正: {}", grad1.x);
    }

    #[test]
    fn test_green_gauss_config() {
        let gg = GreenGaussGradient::new()
            .with_parallel(false)
            .with_threshold(500)
            .with_distance_weighted();

        assert!(!gg.supports_parallel());
        assert_eq!(gg.config.parallel_threshold, 500);
        assert_eq!(gg.config.face_interpolation, FaceInterpolation::DistanceWeighted);
    }

    #[test]
    fn test_vector_gradient() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new().with_parallel(false);

        let u = vec![0.5, 1.5];  // u 随 x 增加
        let v = vec![0.0, 0.0];  // v 均匀
        let mut output = VectorGradientStorage::new(2);

        gg.compute_vector_gradient(&u, &v, &mesh, &mut output);

        // du/dx 应该为正
        assert!(output.du_dx[0] > 0.0 || output.du_dx[1] > 0.0);
        
        // dv/dy 应该接近零
        assert!(output.dv_dy[0].abs() < 1e-6);
        assert!(output.dv_dy[1].abs() < 1e-6);
    }
}
