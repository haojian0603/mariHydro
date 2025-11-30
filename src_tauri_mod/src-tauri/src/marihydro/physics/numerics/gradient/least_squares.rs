// src-tauri/src/marihydro/physics/numerics/gradient/least_squares.rs

//! 最小二乘梯度（带 SVD 回退）- 解决 PREC-004

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use super::GreenGaussGradient;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use glam::DVec2;

pub struct LeastSquaresGradient {
    det_min: f64,
    fallback: GreenGaussGradient,
}

impl LeastSquaresGradient {
    pub fn new(params: &NumericalParams) -> Self {
        Self {
            det_min: params.det_min,
            fallback: GreenGaussGradient::default(),
        }
    }
    pub fn with_det_min(mut self, d: f64) -> Self {
        self.det_min = d;
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
        if neighbors.is_empty() {
            return Some(DVec2::ZERO);
        }

        let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let mut count = 0;

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
        if count < 2 {
            return Some(DVec2::ZERO);
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
