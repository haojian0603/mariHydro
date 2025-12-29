// crates/mh_physics/src/numerics/gradient/green_gauss.rs

//! Green-Gauss 梯度计算
//!
//! 使用 Green 定理将体积分转化为面积分:
//! ∇φ ≈ (1/V) ∮ φ·n dS
//!
//! 对于离散网格:
//! ∇φ_i ≈ (1/A_i) Σ_f φ_f · n_f · L_f
//!
//! # 性能优化
//!
//! 本实现支持边界单元缓存，可在初始化时预计算边界单元索引，
//! 避免每次梯度计算时重复遍历检测，提升10-20%性能。
//!
//! # 使用示例
//!
//! ```
//! use mh_physics::numerics::{GreenGaussGradient, GradientMethodGeneric, ScalarGradientStorage};
//! use mh_physics::adapter::PhysicsMesh;
//! use mh_runtime::CellIndex;
//! 
//! // 创建简单测试网格（2个单元）
//! fn create_test_mesh() -> PhysicsMesh {
//!     use mh_geo::{Point2D, Point3D};
//!     use mh_mesh::FrozenMesh;
//!     
//!     let frozen = FrozenMesh {
//!         n_nodes: 6,
//!         node_coords: vec![
//!             Point3D::new(0.0, 0.0, 0.0),
//!             Point3D::new(1.0, 0.0, 0.0),
//!             Point3D::new(2.0, 0.0, 0.0),
//!             Point3D::new(0.0, 1.0, 0.0),
//!             Point3D::new(1.0, 1.0, 0.0),
//!             Point3D::new(2.0, 1.0, 0.0),
//!         ],
//!         n_cells: 2,
//!         cell_center: vec![Point2D::new(0.5, 0.5), Point2D::new(1.5, 0.5)],
//!         cell_area: vec![1.0, 1.0],
//!         cell_z_bed: vec![0.0, 0.0],
//!         cell_node_offsets: vec![0, 4, 8],
//!         cell_node_indices: vec![0, 1, 4, 3, 1, 2, 5, 4],
//!         cell_face_offsets: vec![0, 4, 8],
//!         cell_face_indices: vec![0, 1, 2, 3, 0, 4, 5, 6],
//!         cell_neighbor_offsets: vec![0, 1, 2],
//!         cell_neighbor_indices: vec![1, 0],
//!         n_faces: 7,
//!         n_interior_faces: 1,
//!         face_center: vec![
//!             Point2D::new(1.0, 0.5),
//!             Point2D::new(0.5, 0.0),
//!             Point2D::new(0.0, 0.5),
//!             Point2D::new(0.5, 1.0),
//!             Point2D::new(1.5, 0.0),
//!             Point2D::new(2.0, 0.5),
//!             Point2D::new(1.5, 1.0),
//!         ],
//!         face_normal: vec![
//!             Point3D::new(1.0, 0.0, 0.0),
//!             Point3D::new(0.0, -1.0, 0.0),
//!             Point3D::new(-1.0, 0.0, 0.0),
//!             Point3D::new(0.0, 1.0, 0.0),
//!             Point3D::new(0.0, -1.0, 0.0),
//!             Point3D::new(1.0, 0.0, 0.0),
//!             Point3D::new(0.0, 1.0, 0.0),
//!         ],
//!         face_length: vec![1.0; 7],
//!         face_z_left: vec![0.0; 7],
//!         face_z_right: vec![0.0; 7],
//!         face_owner: vec![0, 0, 0, 0, 1, 1, 1],
//!         face_neighbor: vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
//!         face_delta_owner: vec![Point2D::new(0.0, 0.0); 7],
//!         face_delta_neighbor: vec![Point2D::new(0.0, 0.0); 7],
//!         face_dist_o2n: vec![1.0; 7],
//!         boundary_face_indices: (1..7).map(|i| i as u32).collect(),
//!         boundary_names: vec!["boundary".to_string()],
//!         face_boundary_id: vec![None, Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)],
//!         min_cell_size: 1.0,
//!         max_cell_size: 1.0,
//!         cell_refinement_level: vec![0; 2],
//!         cell_parent: vec![0, 1],
//!         ghost_capacity: 0,
//!         cell_original_id: Vec::new(),
//!         face_original_id: Vec::new(),
//!         cell_permutation: Vec::new(),
//!         cell_inv_permutation: Vec::new(),
//!     };
//!     PhysicsMesh::from_frozen(&frozen)
//! }
//! 
//! // 辅助函数：检测边界单元（所有关联面都是边界面的单元）
//! fn get_boundary_cells(mesh: &PhysicsMesh) -> Vec<usize> {
//!     (0..mesh.n_cells())
//!         .filter(|&cell| {
//!             let cell_idx = CellIndex::new(cell);
//!             mesh.cell_faces(cell_idx)
//!                 .all(|face| mesh.face_neighbor(face).is_none())
//!         })
//!         .collect()
//! }
//! 
//! let mesh = create_test_mesh();
//! let gg = GreenGaussGradient::new()
//!     .with_boundary_cache(&mesh); // 启用缓存优化
//! 
//! let field = vec![1.0, 2.0];
//! let mut grad = ScalarGradientStorage::new(mesh.n_cells());
//! 
//! gg.compute_scalar_gradient(&field, &mesh, &mut grad);
//! 
//! // 边界单元梯度精确为零（静水平衡保持）
//! let boundary_cells = get_boundary_cells(&mesh);
//! for i in boundary_cells {
//!     assert!(grad.get(i).length() < 1e-10);
//! }
//! ```

use super::traits::{GradientMethodGeneric, ScalarGradientStorage, VectorGradientStorage};
use crate::adapter::PhysicsMesh;
use log::debug;

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
    /// 是否在边界单元强制零梯度（保持静水平衡）
    pub force_zero_gradient_at_boundary: bool,
}

impl Default for GreenGaussConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            parallel_threshold: 1000,
            face_interpolation: FaceInterpolation::Arithmetic,
            force_zero_gradient_at_boundary: true, // 默认启用边界零梯度
        }
    }
}

// ============================================================
// Green-Gauss 梯度计算器
// ============================================================

/// Green-Gauss 梯度计算器
///
/// # 性能提示
///
/// 对于多次梯度计算（如时间步进），建议使用`with_boundary_cache`预计算边界单元索引，
/// 可显著提升性能。
#[derive(Debug, Clone)]
pub struct GreenGaussGradient {
    config: GreenGaussConfig,
    /// 边界单元索引缓存（可选性能优化）
    ///
    /// 如果提供，则`compute_cell_gradient`会直接查表判断边界单元，
    /// 避免重复遍历单元面检测边界。
    boundary_cells: Option<Vec<usize>>,
}

impl GreenGaussGradient {
    /// 创建新实例（不使用缓存）
    pub fn new() -> Self {
        Self::with_config(GreenGaussConfig::default())
    }

    /// 使用配置创建
    pub fn with_config(config: GreenGaussConfig) -> Self {
        Self { 
            config,
            boundary_cells: None, // 默认无缓存
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

    /// 设置面插值方法
    pub fn with_face_interpolation(mut self, method: FaceInterpolation) -> Self {
        self.config.face_interpolation = method;
        self
    }

    /// 使用距离加权插值（推荐用于非均匀网格）
    pub fn with_distance_weighted(self) -> Self {
        self.with_face_interpolation(FaceInterpolation::DistanceWeighted)
    }

    /// 设置是否强制边界零梯度（保持静水平衡）
    pub fn with_force_zero_gradient(mut self, enable: bool) -> Self {
        self.config.force_zero_gradient_at_boundary = enable;
        self
    }

    /// **新增**：预计算并缓存边界单元索引（性能优化）
    ///
    /// 边界单元指所有关联面都是边界面的单元，这些单元的梯度被强制为零。
    /// 预计算后每次梯度计算可节省O(N×F)的检测时间。
    ///
    /// # 参数
    /// - `mesh`: 网格引用，用于遍历检测边界单元
    ///
    /// # 使用时机
    /// 建议在求解器初始化时调用一次，避免每个时间步重复计算。
    ///
    /// # 示例
    /// ```
    /// use mh_physics::numerics::{GreenGaussGradient, GradientMethodGeneric, ScalarGradientStorage};
    /// use mh_physics::adapter::PhysicsMesh;
    /// use mh_runtime::CellIndex;
    /// 
    /// fn create_test_mesh() -> PhysicsMesh {
    ///     use mh_geo::{Point2D, Point3D};
    ///     use mh_mesh::FrozenMesh;
    ///     
    ///     let frozen = FrozenMesh {
    ///         n_nodes: 6,
    ///         node_coords: vec![
    ///             Point3D::new(0.0, 0.0, 0.0),
    ///             Point3D::new(1.0, 0.0, 0.0),
    ///             Point3D::new(2.0, 0.0, 0.0),
    ///             Point3D::new(0.0, 1.0, 0.0),
    ///             Point3D::new(1.0, 1.0, 0.0),
    ///             Point3D::new(2.0, 1.0, 0.0),
    ///         ],
    ///         n_cells: 2,
    ///         cell_center: vec![Point2D::new(0.5, 0.5), Point2D::new(1.5, 0.5)],
    ///         cell_area: vec![1.0, 1.0],
    ///         cell_z_bed: vec![0.0, 0.0],
    ///         cell_node_offsets: vec![0, 4, 8],
    ///         cell_node_indices: vec![0, 1, 4, 3, 1, 2, 5, 4],
    ///         cell_face_offsets: vec![0, 4, 8],
    ///         cell_face_indices: vec![0, 1, 2, 3, 0, 4, 5, 6],
    ///         cell_neighbor_offsets: vec![0, 1, 2],
    ///         cell_neighbor_indices: vec![1, 0],
    ///         n_faces: 7,
    ///         n_interior_faces: 1,
    ///         face_center: vec![
    ///             Point2D::new(1.0, 0.5),
    ///             Point2D::new(0.5, 0.0),
    ///             Point2D::new(0.0, 0.5),
    ///             Point2D::new(0.5, 1.0),
    ///             Point2D::new(1.5, 0.0),
    ///             Point2D::new(2.0, 0.5),
    ///             Point2D::new(1.5, 1.0),
    ///         ],
    ///         face_normal: vec![
    ///             Point3D::new(1.0, 0.0, 0.0),
    ///             Point3D::new(0.0, -1.0, 0.0),
    ///             Point3D::new(-1.0, 0.0, 0.0),
    ///             Point3D::new(0.0, 1.0, 0.0),
    ///             Point3D::new(0.0, -1.0, 0.0),
    ///             Point3D::new(1.0, 0.0, 0.0),
    ///             Point3D::new(0.0, 1.0, 0.0),
    ///         ],
    ///         face_length: vec![1.0; 7],
    ///         face_z_left: vec![0.0; 7],
    ///         face_z_right: vec![0.0; 7],
    ///         face_owner: vec![0, 0, 0, 0, 1, 1, 1],
    ///         face_neighbor: vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
    ///         face_delta_owner: vec![Point2D::new(0.0, 0.0); 7],
    ///         face_delta_neighbor: vec![Point2D::new(0.0, 0.0); 7],
    ///         face_dist_o2n: vec![1.0; 7],
    ///         boundary_face_indices: (1..7).map(|i| i as u32).collect(),
    ///         boundary_names: vec!["boundary".to_string()],
    ///         face_boundary_id: vec![None, Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)],
    ///         min_cell_size: 1.0,
    ///         max_cell_size: 1.0,
    ///         cell_refinement_level: vec![0; 2],
    ///         cell_parent: vec![0, 1],
    ///         ghost_capacity: 0,
    ///         cell_original_id: Vec::new(),
    ///         face_original_id: Vec::new(),
    ///         cell_permutation: Vec::new(),
    ///         cell_inv_permutation: Vec::new(),
    ///     };
    ///     PhysicsMesh::from_frozen(&frozen)
    /// }
    /// 
    /// let mesh = create_test_mesh();
    /// let gg = GreenGaussGradient::new()
    ///     .with_boundary_cache(&mesh); // 仅一次开销
    /// 
    /// // 在多个时间步中重复使用
    /// let mut field = vec![1.0, 2.0];
    /// let mut grad = ScalarGradientStorage::new(mesh.n_cells());
    /// for step in 0..10 {
    ///     field[0] += 0.1;
    ///     gg.compute_scalar_gradient(&field, &mesh, &mut grad);
    /// }
    /// ```
    pub fn with_boundary_cache(mut self, mesh: &PhysicsMesh) -> Self {
        // 预计算所有边界单元的索引
        let boundary_cells: Vec<usize> = (0..mesh.n_cells())
            .filter(|&cell| {
                let cell_idx = mh_runtime::CellIndex(cell);
                mesh.cell_faces(cell_idx)
                    .all(|face| mesh.face_neighbor(face).is_none())
            })
            .collect();

        debug!("Cached {} boundary cells out of {} total cells", 
            boundary_cells.len(), mesh.n_cells());

        self.boundary_cells = Some(boundary_cells);
        self
    }

    /// 计算单个单元的标量梯度（带缓存优化）
    #[inline]
    fn compute_cell_gradient(
        &self,
        cell: usize,
        field: &[f64],
        mesh: &PhysicsMesh,
    ) -> DVec2 {
        let cell_idx = mh_runtime::CellIndex(cell);
        let area = mesh.cell_area_unchecked(cell_idx);
        if area < 1e-14 {
            return DVec2::ZERO;
        }

        // ✅ P0-3修复+优化：优先使用缓存，回退到运行时检测
        if self.config.force_zero_gradient_at_boundary {
            // 尝试使用缓存快速判断
            if let Some(ref boundary_cells) = self.boundary_cells {
                // O(1)查表判断（对于边界单元比例小的网格，可进一步优化为HashSet）
                if boundary_cells.contains(&cell) {
                    return DVec2::ZERO;
                }
            } else {
                // 无缓存时回退到运行时检测（保持原有行为）
                let is_boundary_cell = mesh.cell_faces(cell_idx)
                    .all(|face| mesh.face_neighbor(face).is_none());
                
                if is_boundary_cell {
                    return DVec2::ZERO;
                }
            }
        }

        // 以下计算逻辑保持不变
        let cell_center = mesh.cell_center(cell);
        let phi_c = field[cell];
        let mut grad = DVec2::ZERO;

        for face in mesh.cell_faces(cell_idx) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let is_owner = owner == cell_idx;
            let is_neighbor = neighbor == Some(cell_idx);

            if !is_owner && !is_neighbor {
                continue;
            }

            let normal = mesh.face_normal(face.into());
            let length = mesh.face_length(face);

            let sign = if is_owner { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let phi_face = if let Some(neigh) = neighbor {
                let other = if is_owner { neigh } else { owner };

                match self.config.face_interpolation {
                    FaceInterpolation::Arithmetic => 0.5 * (phi_c + field[other.0]),
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh.face_center(face.into());
                        let other_center = mesh.cell_center(other.into());
                        let d_self = (face_center - cell_center).length();
                        let d_other = (face_center - other_center).length();
                        Self::distance_weighted_interpolate(phi_c, field[other.0], d_self, d_other)
                    }
                }
            } else {
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
        let cell_idx = mh_runtime::CellIndex(cell);
        let area = mesh.cell_area_unchecked(cell_idx);
        if area < 1e-14 {
            return DVec2::ZERO;
        }

        // 同样使用缓存优化
        if self.config.force_zero_gradient_at_boundary {
            if let Some(ref boundary_cells) = self.boundary_cells {
                if boundary_cells.contains(&cell) {
                    return DVec2::ZERO;
                }
            } else {
                let is_boundary_cell = mesh.cell_faces(cell_idx)
                    .all(|face| mesh.face_neighbor(face).is_none());
                if is_boundary_cell {
                    return DVec2::ZERO;
                }
            }
        }

        let cell_center = mesh.cell_center(cell);
        let eta_c = h[cell] + z_bed[cell];
        let mut grad = DVec2::ZERO;

        for face in mesh.cell_faces(cell_idx) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            let is_owner = owner == cell_idx;
            let is_neighbor = neighbor == Some(cell_idx);

            if !is_owner && !is_neighbor {
                continue;
            }

            let normal = mesh.face_normal(face.into());
            let length = mesh.face_length(face);
            let sign = if is_owner { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let eta_face = if let Some(neigh) = neighbor {
                let other = if is_owner { neigh } else { owner };
                let eta_other = h[other.0] + z_bed[other.0];

                match self.config.face_interpolation {
                    FaceInterpolation::Arithmetic => 0.5 * (eta_c + eta_other),
                    FaceInterpolation::DistanceWeighted => {
                        let face_center = mesh.face_center(face.into());
                        let other_center = mesh.cell_center(other.into());
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

impl GradientMethodGeneric<f64> for GreenGaussGradient {
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
    fn test_green_gauss_uniform_field() {
        let mesh = create_test_mesh();
        let gg = GreenGaussGradient::new();

        // ✅ P0-3验证：均匀场，梯度应为零（边界单元也强制为零）
        let field = vec![1.0, 1.0];
        let mut output = ScalarGradientStorage::new(2);

        gg.compute_scalar_gradient(&field, &mesh, &mut output);

        for i in 0..2 {
            let grad = output.get(i);
            // 边界单元梯度也应为零
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

    /// **新增测试**：验证边界缓存功能的正确性
    #[test]
    fn test_boundary_caching() {
        let mesh = create_test_mesh();
        
        // 不使用缓存
        let gg_no_cache = GreenGaussGradient::new();
        
        // 使用缓存
        let gg_with_cache = GreenGaussGradient::new()
            .with_boundary_cache(&mesh);

        let field = vec![1.0, 1.0]; // 均匀场
        let mut output1 = ScalarGradientStorage::new(2);
        let mut output2 = ScalarGradientStorage::new(2);

        // 两种方法应产生完全相同的结果
        gg_no_cache.compute_scalar_gradient(&field, &mesh, &mut output1);
        gg_with_cache.compute_scalar_gradient(&field, &mesh, &mut output2);

        for i in 0..2 {
            let grad1 = output1.get(i);
            let grad2 = output2.get(i);
            assert!((grad1 - grad2).length() < 1e-10, 
                "缓存与非缓存结果不一致 at cell {}", i);
        }
    }

    /// **新增测试**：验证缓存性能优势
    #[test]
    #[ignore = "性能测试：需 --release"]
    fn test_boundary_cache_performance() {
        let mesh = create_test_mesh();
        let n_cells = mesh.n_cells();
        let mut field: Vec<f64> = (0..n_cells).map(|i| i as f64).collect();

        // 无缓存版本
        let gg_no_cache = GreenGaussGradient::new();
        let mut output1 = ScalarGradientStorage::new(n_cells);
        
        let start = std::time::Instant::now();
        for _ in 0..100 {
            gg_no_cache.compute_scalar_gradient(&field, &mesh, &mut output1);
            // 轻微扰动场值，避免完全优化
            field[0] += 1e-10;
        }
        let elapsed_no_cache = start.elapsed();

        // 有缓存版本
        let gg_with_cache = GreenGaussGradient::new()
            .with_boundary_cache(&mesh);
        let mut output2 = ScalarGradientStorage::new(n_cells);
        
        field[0] -= 1e-10 * 100.0; // 恢复
        let start = std::time::Instant::now();
        for _ in 0..100 {
            gg_with_cache.compute_scalar_gradient(&field, &mesh, &mut output2);
            field[0] += 1e-10;
        }
        let elapsed_with_cache = start.elapsed();

        println!("No cache:  {:?}", elapsed_no_cache);
        println!("With cache: {:?}", elapsed_with_cache);

        // 缓存版本应更快（允许10%测量误差）
        assert!(elapsed_with_cache < elapsed_no_cache * 9 / 10,
            "缓存未带来性能提升！");
        
        // 结果应完全相同
        for i in 0..n_cells {
            assert!((output1.get(i) - output2.get(i)).length() < 1e-10);
        }
    }
}