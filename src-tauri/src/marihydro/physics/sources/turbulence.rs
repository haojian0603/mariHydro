// src-tauri/src/marihydro/physics/sources/turbulence.rs

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::domain::state::ConservedState;
use crate::marihydro::infra::error::MhResult;

#[derive(Debug, Clone, Copy)]
pub struct SmagorinskyModel {
    pub cs: f64,
}

impl Default for SmagorinskyModel {
    fn default() -> Self {
        Self { cs: 0.15 }
    }
}

impl SmagorinskyModel {
    pub fn new(cs: f64) -> Self {
        Self { cs }
    }

    pub fn compute_eddy_viscosity(
        &self,
        state: &ConservedState,
        mesh: &UnstructuredMesh,
        h_min: f64,
    ) -> MhResult<Vec<f64>> {
        let n_cells = mesh.n_cells;

        let velocities: Vec<DVec2> = (0..n_cells).map(|i| state.velocity(i, h_min)).collect();

        let (grad_u, grad_v) = self.compute_velocity_gradients(mesh, &velocities);

        let nu_t: Vec<f64> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let area = mesh.cell_area[i];
                let delta = area.sqrt();
                let cs_delta_sq = (self.cs * delta).powi(2);

                let dudx = grad_u[i].x;
                let dudy = grad_u[i].y;
                let dvdx = grad_v[i].x;
                let dvdy = grad_v[i].y;

                let s_xx = dudx;
                let s_yy = dvdy;
                let s_xy = 0.5 * (dudy + dvdx);

                let s_mag_sq = s_xx.powi(2) + s_yy.powi(2) + 2.0 * s_xy.powi(2);
                let s_mag = (2.0 * s_mag_sq).sqrt();

                cs_delta_sq * s_mag
            })
            .collect();

        Ok(nu_t)
    }

    fn compute_velocity_gradients(
        &self,
        mesh: &UnstructuredMesh,
        velocities: &[DVec2],
    ) -> (Vec<DVec2>, Vec<DVec2>) {
        let n_cells = mesh.n_cells;
        let mut grad_u = vec![DVec2::ZERO; n_cells];
        let mut grad_v = vec![DVec2::ZERO; n_cells];

        for face_idx in 0..mesh.n_faces {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];

            let normal = mesh.face_normal[face_idx];
            let length = mesh.face_length[face_idx];
            let ds = normal * length;

            let vel_face = if neighbor != usize::MAX {
                (velocities[owner] + velocities[neighbor]) * 0.5
            } else {
                velocities[owner]
            };

            grad_u[owner] += ds * vel_face.x;
            grad_v[owner] += ds * vel_face.y;

            if neighbor != usize::MAX {
                grad_u[neighbor] -= ds * vel_face.x;
                grad_v[neighbor] -= ds * vel_face.y;
            }
        }

        for i in 0..n_cells {
            let inv_area = 1.0 / mesh.cell_area[i];
            grad_u[i] *= inv_area;
            grad_v[i] *= inv_area;
        }

        (grad_u, grad_v)
    }
}

pub fn compute_gradient_field(field: &[f64], mesh: &UnstructuredMesh) -> Vec<DVec2> {
    let n_cells = mesh.n_cells;
    let mut grad = vec![DVec2::ZERO; n_cells];

    for face_idx in 0..mesh.n_faces {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let normal = mesh.face_normal[face_idx];
        let length = mesh.face_length[face_idx];
        let ds = normal * length;

        let phi_face = if neighbor != usize::MAX {
            0.5 * (field[owner] + field[neighbor])
        } else {
            field[owner]
        };

        grad[owner] += ds * phi_face;

        if neighbor != usize::MAX {
            grad[neighbor] -= ds * phi_face;
        }
    }

    for i in 0..n_cells {
        grad[i] /= mesh.cell_area[i];
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cs_default_range() {
        let model = SmagorinskyModel::default();
        assert!(model.cs > 0.0 && model.cs < 0.5);
    }

    #[test]
    fn test_custom_cs() {
        let model = SmagorinskyModel::new(0.12);
        assert_eq!(model.cs, 0.12);
    }
}
