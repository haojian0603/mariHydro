// src-tauri/src/marihydro/physics/numerics/gradient/parallel.rs
use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex};
use glam::DVec2;
use rayon::prelude::*;

pub struct ParallelGreenGauss { threshold: usize }

impl Default for ParallelGreenGauss { fn default() -> Self { Self { threshold: 1000 } } }
impl ParallelGreenGauss {
    pub fn new(threshold: usize) -> Self { Self { threshold } }
}

impl GradientMethod for ParallelGreenGauss {
    fn compute_scalar_gradient<M: MeshAccess>(&self, field: &[f64], mesh: &M, output: &mut ScalarGradientStorage) -> MhResult<()> {
        output.reset();
        let n = mesh.n_cells();
        if n < self.threshold {
            for i in 0..n { output.set(i, compute_cell_gradient(i, field, mesh)); }
        } else {
            let grads: Vec<DVec2> = (0..n).into_par_iter().map(|i| compute_cell_gradient(i, field, mesh)).collect();
            for (i, g) in grads.into_iter().enumerate() { output.set(i, g); }
        }
        Ok(())
    }

    fn compute_vector_gradient<M: MeshAccess>(&self, field: &[DVec2], mesh: &M, output: &mut VectorGradientStorage) -> MhResult<()> {
        output.reset();
        let n = mesh.n_cells();
        for i in 0..n {
            let (gu, gv) = compute_cell_vector_gradient(i, field, mesh);
            output.set_grad_u(i, gu);
            output.set_grad_v(i, gv);
        }
        Ok(())
    }

    fn name(&self) -> &'static str { "ParallelGreenGauss" }
}

fn compute_cell_gradient<M: MeshAccess>(cell_idx: usize, field: &[f64], mesh: &M) -> DVec2 {
    let cell = CellIndex(cell_idx);
    let area = mesh.cell_area(cell);
    if area < 1e-14 { return DVec2::ZERO; }
    let mut grad = DVec2::ZERO;
    for &face in mesh.cell_faces(cell) {
        let owner = mesh.face_owner(face);
        let neighbor = mesh.face_neighbor(face);
        let normal = mesh.face_normal(face);
        let length = mesh.face_length(face);
        let sign = if cell_idx == owner.0 { 1.0 } else { -1.0 };
        let ds = normal * length * sign;
        let phi_face = if !neighbor.is_valid() { field[cell_idx] }
        else { 0.5 * (field[owner.0] + field[neighbor.0]) };
        grad += ds * phi_face;
    }
    grad / area
}

fn compute_cell_vector_gradient<M: MeshAccess>(cell_idx: usize, field: &[DVec2], mesh: &M) -> (DVec2, DVec2) {
    let cell = CellIndex(cell_idx);
    let area = mesh.cell_area(cell);
    if area < 1e-14 { return (DVec2::ZERO, DVec2::ZERO); }
    let mut gu = DVec2::ZERO; let mut gv = DVec2::ZERO;
    for &face in mesh.cell_faces(cell) {
        let owner = mesh.face_owner(face);
        let neighbor = mesh.face_neighbor(face);
        let normal = mesh.face_normal(face);
        let length = mesh.face_length(face);
        let sign = if cell_idx == owner.0 { 1.0 } else { -1.0 };
        let ds = normal * length * sign;
        let v = if !neighbor.is_valid() { field[cell_idx] } else { (field[owner.0] + field[neighbor.0]) * 0.5 };
        gu += ds * v.x; gv += ds * v.y;
    }
    (gu / area, gv / area)
}
