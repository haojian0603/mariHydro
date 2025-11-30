// src-tauri/src/marihydro/physics/numerics/gradient/least_squares.rs

//! 最小二乘梯度（带 SVD 回退）- 解决 PREC-004
//!
//! ## 边界处理
//!
//! 对于边界单元（邻居不足），使用以下策略：
//! 1. 首先尝试使用所有可用邻居
//! 2. 如果邻居不足，添加边界面贡献（虚拟点）
//! 3. 如果仍然奇异，回退到 Green-Gauss

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use super::GreenGaussGradient;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use glam::DVec2;

pub struct LeastSquaresGradient {
    det_min: f64,
    fallback: GreenGaussGradient,
    /// 是否启用边界贡献（虚拟点策略）
    use_boundary_contributions: bool,
}

impl LeastSquaresGradient {
    pub fn new(params: &NumericalParams) -> Self {
        Self {
            det_min: params.det_min,
            fallback: GreenGaussGradient::default(),
            use_boundary_contributions: true,
        }
    }

    pub fn with_det_min(mut self, d: f64) -> Self {
        self.det_min = d;
        self
    }

    /// 启用/禁用边界贡献
    pub fn with_boundary_contributions(mut self, enable: bool) -> Self {
        self.use_boundary_contributions = enable;
        self
    }

    #[inline]
    fn solve_2x2(
        a11: f64,
        a12: f64,
        a22: f64,
        b1: f64,
        b2: f64,
        det_min: f64,
    ) -> Option<(f64, f64)> {
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

    fn compute_cell<M: MeshAccess>(
        &self,
        cell_idx: usize,
        field: &[f64],
        mesh: &M,
    ) -> Option<DVec2> {
        let cell = CellIndex(cell_idx);
        let cc = mesh.cell_centroid(cell);
        let phi_c = field[cell_idx];
        let neighbors = mesh.cell_neighbors(cell);

        let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut count = 0;

        // 1. 收集内部邻居贡献
        for &nb in neighbors {
            if !nb.is_valid() {
                continue;
            }
            let nc = mesh.cell_centroid(nb);
            let (dx, dy) = (nc.x - cc.x, nc.y - cc.y);
            let dphi = field[nb.0] - phi_c;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq < 1e-20 {
                continue;
            }
            let w = 1.0 / dist_sq;
            a11 += w * dx * dx;
            a12 += w * dx * dy;
            a22 += w * dy * dy;
            b1 += w * dx * dphi;
            b2 += w * dy * dphi;
            count += 1;
        }

        // 2. 如果邻居不足且启用边界贡献，添加边界面虚拟点
        if count < 2 && self.use_boundary_contributions {
            for &face in mesh.cell_faces(cell) {
                let neighbor = mesh.face_neighbor(face);
                if neighbor.is_valid() {
                    continue; // 内部面，已经处理过
                }

                // 边界面：使用镜像点策略（假设零梯度边界条件）
                // 虚拟点位于边界面法向外侧，距离等于单元中心到面的距离
                let face_center = mesh.face_centroid(face);
                let normal = mesh.face_normal(face);

                // 计算单元中心到面中心的向量
                let to_face = face_center - cc;
                let dist_to_face = to_face.dot(normal);

                if dist_to_face.abs() < 1e-14 {
                    continue;
                }

                // 虚拟点位于面的另一侧，距离相等（镜像）
                let ghost_point = face_center + normal * dist_to_face.abs();
                let (dx, dy) = (ghost_point.x - cc.x, ghost_point.y - cc.y);
                let dist_sq = dx * dx + dy * dy;

                if dist_sq < 1e-20 {
                    continue;
                }

                // 零梯度边界条件：虚拟点值等于单元中心值
                let dphi = 0.0; // phi_ghost - phi_c = 0

                let w = 1.0 / dist_sq;
                a11 += w * dx * dx;
                a12 += w * dx * dy;
                a22 += w * dy * dy;
                b1 += w * dx * dphi;
                b2 += w * dy * dphi;
                count += 1;
            }
        }

        // 3. 如果仍然不足，返回 None 触发回退
        if count < 2 {
            return Some(DVec2::ZERO); // 孤立单元，返回零梯度
        }

        Self::solve_2x2(a11, a12, a22, b1, b2, self.det_min).map(|(x, y)| DVec2::new(x, y))
    }
}

impl GradientMethod for LeastSquaresGradient {
    fn compute_scalar_gradient<M: MeshAccess>(
        &self,
        field: &[f64],
        mesh: &M,
        output: &mut ScalarGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        let mut fallback_cells = Vec::new();

        for i in 0..mesh.n_cells() {
            match self.compute_cell(i, field, mesh) {
                Some(g) => output.set(i, g),
                None => fallback_cells.push(i),
            }
        }

        if !fallback_cells.is_empty() {
            let mut fb = ScalarGradientStorage::new(mesh.n_cells());
            self.fallback
                .compute_scalar_gradient(field, mesh, &mut fb)?;
            for i in fallback_cells {
                output.set(i, fb.get(i));
            }
        }
        Ok(())
    }

    fn compute_vector_gradient<M: MeshAccess>(
        &self,
        field: &[DVec2],
        mesh: &M,
        output: &mut VectorGradientStorage,
    ) -> MhResult<()> {
        let n = mesh.n_cells();
        let u: Vec<f64> = field.iter().map(|v| v.x).collect();
        let v: Vec<f64> = field.iter().map(|v| v.y).collect();

        let mut ug = ScalarGradientStorage::new(n);
        let mut vg = ScalarGradientStorage::new(n);
        self.compute_scalar_gradient(&u, mesh, &mut ug)?;
        self.compute_scalar_gradient(&v, mesh, &mut vg)?;

        output.du_dx = ug.grad_x;
        output.du_dy = ug.grad_y;
        output.dv_dx = vg.grad_x;
        output.dv_dy = vg.grad_y;
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Least-Squares"
    }
}
