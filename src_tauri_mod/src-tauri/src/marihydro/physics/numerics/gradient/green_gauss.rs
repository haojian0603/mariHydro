//! Green-Gauss 梯度 - 消除 PERF-004

use super::traits::{GradientMethod, ScalarGradientStorage, VectorGradientStorage};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex};
use glam::DVec2;
use rayon::prelude::*;

pub struct GreenGaussGradient {
    parallel: bool,
    parallel_threshold: usize,
}

impl Default for GreenGaussGradient {
    fn default() -> Self {
        Self {
            parallel: true,
            parallel_threshold: 1000,
        }
    }
}

impl GreenGaussGradient {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }
    pub fn with_threshold(mut self, t: usize) -> Self {
        self.parallel_threshold = t;
        self
    }

    fn compute_cell_scalar<M: MeshAccess>(
        &self,
        cell_idx: usize,
        field: &[f64],
        mesh: &M,
    ) -> DVec2 {
        let cell = CellIndex(cell_idx);
        let area = mesh.cell_area(cell);
        if area < 1e-14 {
            return DVec2::ZERO;
        }

        let mut grad = DVec2::ZERO;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);

            let sign = if cell_idx == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let phi_face = if !neighbor.is_valid() {
                field[cell_idx]
            } else {
                let other = if cell_idx == owner.0 {
                    neighbor.0
                } else {
                    owner.0
                };
                0.5 * (field[cell_idx] + field[other])
            };
            grad += ds * phi_face;
        }
        grad / area
    }

    fn compute_cell_vector<M: MeshAccess>(
        &self,
        cell_idx: usize,
        field: &[DVec2],
        mesh: &M,
    ) -> (DVec2, DVec2) {
        let cell = CellIndex(cell_idx);
        let area = mesh.cell_area(cell);
        if area < 1e-14 {
            return (DVec2::ZERO, DVec2::ZERO);
        }

        let mut grad_u = DVec2::ZERO;
        let mut grad_v = DVec2::ZERO;

        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);

            let sign = if cell_idx == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let vel_face = if !neighbor.is_valid() {
                field[cell_idx]
            } else {
                let other = if cell_idx == owner.0 {
                    neighbor.0
                } else {
                    owner.0
                };
                (field[cell_idx] + field[other]) * 0.5
            };
            grad_u += ds * vel_face.x;
            grad_v += ds * vel_face.y;
        }
        (grad_u / area, grad_v / area)
    }
}

impl GradientMethod for GreenGaussGradient {
    fn compute_scalar_gradient<M: MeshAccess>(
        &self,
        field: &[f64],
        mesh: &M,
        output: &mut ScalarGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        for i in 0..mesh.n_cells() {
            output.set(i, self.compute_cell_scalar(i, field, mesh));
        }
        Ok(())
    }

    fn compute_vector_gradient<M: MeshAccess>(
        &self,
        field: &[DVec2],
        mesh: &M,
        output: &mut VectorGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        for i in 0..mesh.n_cells() {
            let (gu, gv) = self.compute_cell_vector(i, field, mesh);
            output.set_grad_u(i, gu);
            output.set_grad_v(i, gv);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Green-Gauss"
    }
}

/// 专用并行版本
pub mod optimized {
    use super::*;
    use crate::marihydro::domain::mesh::UnstructuredMesh;

    pub fn compute_scalar_gradient_parallel(
        field: &[f64],
        mesh: &UnstructuredMesh,
        output: &mut ScalarGradientStorage,
    ) -> MhResult<()> {
        output.reset();
        let grads: Vec<DVec2> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|i| {
                let cell = CellIndex(i);
                let area = mesh.cell_area(cell);
                if area < 1e-14 {
                    return DVec2::ZERO;
                }
                let mut g = DVec2::ZERO;
                for &face in mesh.cell_faces(cell) {
                    let owner = mesh.face_owner(face);
                    let neighbor = mesh.face_neighbor(face);
                    let normal = mesh.face_normal(face);
                    let length = mesh.face_length(face);
                    let sign = if i == owner.0 { 1.0 } else { -1.0 };
                    let ds = normal * length * sign;
                    let phi = if !neighbor.is_valid() {
                        field[i]
                    } else {
                        let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                        0.5 * (field[i] + field[o])
                    };
                    g += ds * phi;
                }
                g / area
            })
            .collect();
        for (i, g) in grads.into_iter().enumerate() {
            output.set(i, g);
        }
        Ok(())
    }
}
