use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

use glam::DVec2;

use crate::marihydro::domain::mesh::indices::INVALID_CELL;
use crate::marihydro::domain::mesh::unstructured::{BoundaryKind, UnstructuredMesh};
use crate::marihydro::domain::state::{ConservedState, Flux, GradientState};
use crate::marihydro::infra::error::{MhError, MhResult};

use super::schemes::hllc::solve_hllc;
use super::schemes::hydrostatic::{
    compute_bed_slope_source, hydrostatic_reconstruction, muscl_reconstruct, SlopeLimiter,
};

struct WorkSpace {
    fluxes: Vec<Flux>,
    bed_sources: Vec<DVec2>,
}

impl WorkSpace {
    fn new(n_faces: usize) -> Self {
        Self {
            fluxes: vec![Flux::default(); n_faces],
            bed_sources: vec![DVec2::ZERO; n_faces],
        }
    }

    fn reset(&mut self) {
        self.fluxes.iter_mut().for_each(|f| *f = Flux::default());
        self.bed_sources.iter_mut().for_each(|s| *s = DVec2::ZERO);
    }
}

pub struct UnstructuredSolver {
    pub mesh: UnstructuredMesh,
    pub gravity: f64,
    pub h_min: f64,
    pub gradients: Option<GradientState>,
    pub limiter: SlopeLimiter,
    pub max_wave_speed: f64,
    pub total_flux_calls: AtomicU64,
    pub manning_coef: f64,
    workspace: WorkSpace,
}

impl UnstructuredSolver {
    pub fn new(mesh: UnstructuredMesh, gravity: f64, h_min: f64) -> Self {
        let n_faces = mesh.n_faces;
        Self {
            mesh,
            gravity,
            h_min,
            gradients: None,
            limiter: SlopeLimiter::VanLeer,
            max_wave_speed: 0.0,
            total_flux_calls: AtomicU64::new(0),
            manning_coef: 0.025,
            workspace: WorkSpace::new(n_faces),
        }
    }

    pub fn enable_muscl(&mut self) {
        self.gradients = Some(GradientState::new(self.mesh.n_cells));
    }

    pub fn step(
        &mut self,
        state: &ConservedState,
        next_state: &mut ConservedState,
        dt: f64,
    ) -> MhResult<f64> {
        next_state.h.copy_from_slice(&state.h);
        next_state.hu.copy_from_slice(&state.hu);
        next_state.hv.copy_from_slice(&state.hv);

        if let Some(ref mut grad) = self.gradients {
            self.compute_gradients(state, grad);
        }

        self.workspace.reset();
        self.compute_face_fluxes_with_source(state);

        self.update_cells_with_source(next_state, dt);

        self.apply_friction(next_state, state, dt);

        next_state.validate(0.0)?;

        Ok(self.max_wave_speed)
    }

    fn compute_gradients(&self, state: &ConservedState, grad: &mut GradientState) {
        grad.reset();

        for face_idx in 0..self.mesh.n_faces {
            let owner = self.mesh.face_owner[face_idx];
            let neighbor = self.mesh.face_neighbor[face_idx];

            let normal = self.mesh.face_normal[face_idx];
            let length = self.mesh.face_length[face_idx];
            let ds = normal * length;

            let (h_face, hu_face, hv_face) = if neighbor != INVALID_CELL {
                let (h_ghost, vel_ghost) = self.compute_ghost_state_for_gradient(
                    state.h[neighbor],
                    state.velocity(neighbor, self.h_min),
                    face_idx,
                );
                let h = 0.5 * (state.h[owner] + h_ghost);
                let vel = 0.5 * (state.velocity(owner, self.h_min) + vel_ghost);
                (h, h * vel.x, h * vel.y)
            } else {
                let bc_idx = face_idx - self.mesh.n_interior_faces;
                let bc_kind = self.mesh.bc_kind[bc_idx];
                let (h_ghost, vel_ghost) = self.compute_ghost_state(
                    state.h[owner],
                    state.velocity(owner, self.h_min),
                    normal,
                    bc_kind,
                    bc_idx,
                );
                let h = 0.5 * (state.h[owner] + h_ghost);
                let vel = 0.5 * (state.velocity(owner, self.h_min) + vel_ghost);
                (h, h * vel.x, h * vel.y)
            };

            grad.grad_h[owner] += ds * h_face;
            grad.grad_hu[owner] += ds * hu_face;
            grad.grad_hv[owner] += ds * hv_face;

            if neighbor != INVALID_CELL {
                grad.grad_h[neighbor] -= ds * h_face;
                grad.grad_hu[neighbor] -= ds * hu_face;
                grad.grad_hv[neighbor] -= ds * hv_face;
            }
        }

        for i in 0..self.mesh.n_cells {
            let inv_area = 1.0 / self.mesh.cell_area[i];
            grad.grad_h[i] *= inv_area;
            grad.grad_hu[i] *= inv_area;
            grad.grad_hv[i] *= inv_area;
        }
    }

    fn compute_ghost_state_for_gradient(
        &self,
        h_neighbor: f64,
        vel_neighbor: DVec2,
        _face_idx: usize,
    ) -> (f64, DVec2) {
        (h_neighbor, vel_neighbor)
    }

    fn compute_face_fluxes_with_source(&mut self, state: &ConservedState) {
        let n_interior = self.mesh.n_interior_faces;
        let max_speed = AtomicU64::new(0);

        let (interior_fluxes, boundary_fluxes) = self.workspace.fluxes.split_at_mut(n_interior);
        let (interior_sources, boundary_sources) =
            self.workspace.bed_sources.split_at_mut(n_interior);

        interior_fluxes
            .par_iter_mut()
            .zip(interior_sources.par_iter_mut())
            .enumerate()
            .for_each(|(face_idx, (flux, bed_source))| {
                let owner = self.mesh.face_owner[face_idx];
                let neighbor = self.mesh.face_neighbor[face_idx];

                let normal = self.mesh.face_normal[face_idx];
                let length = self.mesh.face_length[face_idx];

                let (h_l, vel_l) = if let Some(ref grad) = self.gradients {
                    let recon = muscl_reconstruct(
                        state.h[owner],
                        state.hu[owner],
                        state.hv[owner],
                        grad.grad_h[owner],
                        grad.grad_hu[owner],
                        grad.grad_hv[owner],
                        self.mesh.face_delta_owner[face_idx],
                        self.limiter,
                        self.h_min,
                    );
                    (recon.h, recon.vel)
                } else {
                    (state.h[owner], state.velocity(owner, self.h_min))
                };

                let (h_r, vel_r) = if let Some(ref grad) = self.gradients {
                    let recon = muscl_reconstruct(
                        state.h[neighbor],
                        state.hu[neighbor],
                        state.hv[neighbor],
                        grad.grad_h[neighbor],
                        grad.grad_hu[neighbor],
                        grad.grad_hv[neighbor],
                        -self.mesh.face_delta_neighbor[face_idx],
                        self.limiter,
                        self.h_min,
                    );
                    (recon.h, recon.vel)
                } else {
                    (state.h[neighbor], state.velocity(neighbor, self.h_min))
                };

                let z_owner = self.mesh.cell_z_bed[owner];
                let z_neighbor = self.mesh.cell_z_bed[neighbor];

                let recon = hydrostatic_reconstruction(
                    h_l,
                    z_owner,
                    vel_l,
                    h_r,
                    z_neighbor,
                    vel_r,
                    self.mesh.face_z_left[face_idx],
                    self.mesh.face_z_right[face_idx],
                    self.h_min,
                    self.gravity,
                );

                let result = solve_hllc(
                    recon.h_left,
                    recon.vel_left,
                    recon.h_right,
                    recon.vel_right,
                    normal,
                    self.gravity,
                    self.h_min,
                );

                *flux = result.flux;

                let source = compute_bed_slope_source(
                    recon.h_left,
                    recon.h_right,
                    z_owner,
                    z_neighbor,
                    normal,
                    length,
                    self.gravity,
                );

                *bed_source = DVec2::new(source.source_x, source.source_y);

                let speed_bits = result.max_wave_speed.to_bits();
                max_speed.fetch_max(speed_bits, Ordering::Relaxed);
            });

        boundary_fluxes
            .par_iter_mut()
            .zip(boundary_sources.par_iter_mut())
            .enumerate()
            .for_each(|(local_idx, (flux, bed_source))| {
                let face_idx = self.mesh.n_interior_faces + local_idx;
                let owner = self.mesh.face_owner[face_idx];
                let bc_idx = local_idx;

                let normal = self.mesh.face_normal[face_idx];
                let length = self.mesh.face_length[face_idx];
                let bc_kind = self.mesh.bc_kind[bc_idx];

                let (h_l, vel_l) = if let Some(ref grad) = self.gradients {
                    let recon = muscl_reconstruct(
                        state.h[owner],
                        state.hu[owner],
                        state.hv[owner],
                        grad.grad_h[owner],
                        grad.grad_hu[owner],
                        grad.grad_hv[owner],
                        self.mesh.face_delta_owner[face_idx],
                        self.limiter,
                        self.h_min,
                    );
                    (recon.h, recon.vel)
                } else {
                    (state.h[owner], state.velocity(owner, self.h_min))
                };

                let (h_ghost, vel_ghost) =
                    self.compute_ghost_state(h_l, vel_l, normal, bc_kind, bc_idx);

                let z_owner = self.mesh.cell_z_bed[owner];

                let recon = hydrostatic_reconstruction(
                    h_l,
                    z_owner,
                    vel_l,
                    h_ghost,
                    z_owner,
                    vel_ghost,
                    self.mesh.face_z_left[face_idx],
                    self.mesh.face_z_right[face_idx],
                    self.h_min,
                    self.gravity,
                );

                let result = solve_hllc(
                    recon.h_left,
                    recon.vel_left,
                    recon.h_right,
                    recon.vel_right,
                    normal,
                    self.gravity,
                    self.h_min,
                );

                *flux = result.flux;

                let source = compute_bed_slope_source(
                    recon.h_left,
                    recon.h_right,
                    z_owner,
                    z_owner,
                    normal,
                    length,
                    self.gravity,
                );

                *bed_source = DVec2::new(source.source_x, source.source_y);

                let speed_bits = result.max_wave_speed.to_bits();
                max_speed.fetch_max(speed_bits, Ordering::Relaxed);
            });

        self.max_wave_speed = f64::from_bits(max_speed.load(Ordering::Relaxed));
        self.total_flux_calls
            .fetch_add(self.mesh.n_faces as u64, Ordering::Relaxed);
    }

    #[inline]
    fn compute_ghost_state(
        &self,
        h_owner: f64,
        vel_owner: DVec2,
        normal: DVec2,
        bc_kind: BoundaryKind,
        bc_idx: usize,
    ) -> (f64, DVec2) {
        match bc_kind {
            BoundaryKind::Wall | BoundaryKind::Symmetry => {
                let vn = vel_owner.dot(normal);
                let vel_ghost = vel_owner - 2.0 * vn * normal;
                (h_owner, vel_ghost)
            }

            BoundaryKind::OpenSea => {
                let eta_bc = self.mesh.bc_value_h[bc_idx];
                let face_idx = self.mesh.n_interior_faces + bc_idx;
                let owner_idx = self.mesh.face_owner[face_idx];
                let z_bed = self.mesh.cell_z_bed[owner_idx];
                let h_ghost = (eta_bc - z_bed).max(0.0);

                (h_ghost, vel_owner)
            }

            BoundaryKind::RiverInflow => {
                let q_bc = self.mesh.bc_value_q[bc_idx];

                let h_ghost = h_owner.max(self.h_min);
                let vel_mag = q_bc / h_ghost.max(self.h_min);

                let vel_ghost = -normal * vel_mag;

                (h_ghost, vel_ghost)
            }

            BoundaryKind::Outflow => (h_owner, vel_owner),
        }
    }

    fn update_cells_with_source(&self, next_state: &mut ConservedState, dt: f64) {
        next_state
            .h
            .par_iter_mut()
            .zip(&mut next_state.hu)
            .zip(&mut next_state.hv)
            .enumerate()
            .for_each(|(cell_idx, ((h_next, hu_next), hv_next))| {
                let cell_faces = &self.mesh.cell_faces[cell_idx];
                let area = self.mesh.cell_area[cell_idx];

                let mut delta_h = 0.0;
                let mut delta_hu = 0.0;
                let mut delta_hv = 0.0;

                let mut source_hu = 0.0;
                let mut source_hv = 0.0;

                for (local_idx, &face_id) in cell_faces.faces.iter().enumerate() {
                    let face_idx = face_id.idx();
                    let flux = self.workspace.fluxes[face_idx];
                    let bed_source = self.workspace.bed_sources[face_idx];
                    let length = self.mesh.face_length[face_idx];

                    let sign = if cell_faces.is_owner(local_idx) {
                        -1.0
                    } else {
                        1.0
                    };

                    delta_h += sign * flux.mass * length;
                    delta_hu += sign * flux.mom_x * length;
                    delta_hv += sign * flux.mom_y * length;

                    source_hu += sign * bed_source.x;
                    source_hv += sign * bed_source.y;
                }

                *h_next += (dt / area) * delta_h;
                *hu_next += (dt / area) * (delta_hu + source_hu);
                *hv_next += (dt / area) * (delta_hv + source_hv);
            });
    }

    fn apply_friction(
        &self,
        next_state: &mut ConservedState,
        prev_state: &ConservedState,
        dt: f64,
    ) {
        next_state
            .h
            .par_iter()
            .zip(&mut next_state.hu)
            .zip(&mut next_state.hv)
            .zip(&prev_state.h)
            .for_each(|(((h_new, hu_new), hv_new), &h_old)| {
                if *h_new > self.h_min {
                    let h_avg = 0.5 * (h_old + *h_new);
                    let u = *hu_new / *h_new;
                    let v = *hv_new / *h_new;
                    let vel_mag = (u * u + v * v).sqrt();

                    let n = self.manning_coef;
                    let h_pow = h_avg.powf(4.0 / 3.0);
                    let cf = self.gravity * n * n * vel_mag / h_pow.max(self.h_min);

                    let denom = 1.0 + dt * cf;
                    *hu_new /= denom;
                    *hv_new /= denom;
                } else {
                    *hu_new = 0.0;
                    *hv_new = 0.0;
                }
            });
    }

    pub fn compute_cfl_dt(&self, state: &ConservedState, cfl_factor: f64) -> f64 {
        let min_char_length = self
            .mesh
            .cell_area
            .par_iter()
            .zip(&self.mesh.cell_faces)
            .map(|(&area, faces)| {
                let perimeter: f64 = faces
                    .faces
                    .iter()
                    .map(|&fid| self.mesh.face_length[fid.idx()])
                    .sum();
                2.0 * area / perimeter.max(1e-12)
            })
            .reduce(|| f64::MAX, f64::min);

        let max_speed = state
            .h
            .par_iter()
            .zip(&state.hu)
            .zip(&state.hv)
            .map(|((&h, &hu), &hv)| {
                if h > self.h_min {
                    let u = hu / h;
                    let v = hv / h;
                    let vel_mag = (u * u + v * v).sqrt();
                    let c = (self.gravity * h).sqrt();
                    vel_mag + c
                } else {
                    0.0
                }
            })
            .reduce(|| 0.0, f64::max);

        if max_speed < 1e-6 {
            return 10.0;
        }

        cfl_factor * min_char_length / max_speed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;

    fn create_simple_mesh() -> UnstructuredMesh {
        UnstructuredMesh::new()
    }

    #[test]
    fn test_solver_creation() {
        let mesh = create_simple_mesh();
        let solver = UnstructuredSolver::new(mesh, 9.81, 1e-3);

        assert_eq!(solver.gravity, 9.81);
    }

    #[test]
    fn test_ghost_state_wall() {
        let mesh = create_simple_mesh();
        let solver = UnstructuredSolver::new(mesh, 9.81, 1e-3);

        let h = 1.0;
        let vel = DVec2::new(1.0, 0.5);
        let normal = DVec2::X;

        let (h_ghost, vel_ghost) =
            solver.compute_ghost_state(h, vel, normal, BoundaryKind::Wall, 0);

        assert_eq!(h_ghost, h);
        assert_eq!(vel_ghost.x, -1.0);
        assert_eq!(vel_ghost.y, 0.5);
    }
}
