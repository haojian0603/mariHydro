// src-tauri/src/marihydro/physics/sources/turbulence.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use crate::marihydro::physics::numerics::gradient::{GradientMethod, GreenGaussGradient, VectorGradientStorage};
use glam::DVec2;
use rayon::prelude::*;

pub struct SmagorinskyTurbulence {
    cs: f64,
    nu_min: f64,
    nu_max: f64,
}

impl SmagorinskyTurbulence {
    pub fn new(cs: f64) -> Self {
        Self { cs, nu_min: 1e-6, nu_max: 1000.0 }
    }

    pub fn with_limits(mut self, nu_min: f64, nu_max: f64) -> Self {
        self.nu_min = nu_min;
        self.nu_max = nu_max;
        self
    }

    fn compute_eddy_viscosity<M: MeshAccess, S: StateAccess>(&self, mesh: &M, state: &S, params: &NumericalParams) -> Vec<f64> {
        let n = mesh.n_cells();
        let velocities: Vec<DVec2> = (0..n).map(|i| {
            let h = state.h(i);
            if params.is_dry(h) { DVec2::ZERO }
            else { DVec2::new(state.hu(i) / h, state.hv(i) / h) }
        }).collect();
        let grad_method = GreenGaussGradient::new();
        let mut grad_storage = VectorGradientStorage::new(n);
        let _ = grad_method.compute_vector_gradient(&velocities, mesh, &mut grad_storage);
        (0..n).map(|i| {
            let area = mesh.cell_area(CellIndex(i));
            if area < 1e-14 { return self.nu_min; }
            let delta = area.sqrt();
            let du_dx = grad_storage.du_dx[i];
            let du_dy = grad_storage.du_dy[i];
            let dv_dx = grad_storage.dv_dx[i];
            let dv_dy = grad_storage.dv_dy[i];
            let s11 = du_dx;
            let s22 = dv_dy;
            let s12 = 0.5 * (du_dy + dv_dx);
            let s_mag = (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt();
            let nu = (self.cs * delta).powi(2) * s_mag;
            nu.clamp(self.nu_min, self.nu_max)
        }).collect()
    }

    fn compute_diffusion_flux<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, nu_t: &[f64], params: &NumericalParams,
        acc_hu: &mut [f64], acc_hv: &mut [f64],
    ) {
        let n_faces = mesh.n_faces();
        for face_idx in 0..n_faces {
            let face = FaceIndex(face_idx);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            if !neighbor.is_valid() { continue; }
            let h_o = state.h(owner.0);
            let h_n = state.h(neighbor.0);
            if params.is_dry(h_o) && params.is_dry(h_n) { continue; }
            let u_o = if params.is_dry(h_o) { 0.0 } else { state.hu(owner.0) / h_o };
            let v_o = if params.is_dry(h_o) { 0.0 } else { state.hv(owner.0) / h_o };
            let u_n = if params.is_dry(h_n) { 0.0 } else { state.hu(neighbor.0) / h_n };
            let v_n = if params.is_dry(h_n) { 0.0 } else { state.hv(neighbor.0) / h_n };
            let dist = (mesh.cell_centroid(neighbor) - mesh.cell_centroid(owner)).length();
            if dist < 1e-14 { continue; }
            let nu_face = 0.5 * (nu_t[owner.0] + nu_t[neighbor.0]);
            let h_face = 0.5 * (h_o + h_n);
            let length = mesh.face_length(face);
            let flux_u = nu_face * h_face * (u_n - u_o) / dist * length;
            let flux_v = nu_face * h_face * (v_n - v_o) / dist * length;
            acc_hu[owner.0] += flux_u;
            acc_hv[owner.0] += flux_v;
            acc_hu[neighbor.0] -= flux_u;
            acc_hv[neighbor.0] -= flux_v;
        }
    }
}

impl SourceTerm for SmagorinskyTurbulence {
    fn name(&self) -> &'static str { "SmagorinskyTurbulence" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, _cell_idx: usize, _mesh: &M, _state: &S, _ctx: &SourceContext,
    ) -> SourceContribution {
        SourceContribution::ZERO
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        _output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        let nu_t = self.compute_eddy_viscosity(mesh, state, ctx.params);
        self.compute_diffusion_flux(mesh, state, &nu_t, ctx.params, output_hu, output_hv);
        Ok(())
    }
}

pub fn compute_vorticity<M: MeshAccess, S: StateAccess>(mesh: &M, state: &S, params: &NumericalParams) -> Vec<f64> {
    let n = mesh.n_cells();
    let velocities: Vec<DVec2> = (0..n).map(|i| {
        let h = state.h(i);
        if params.is_dry(h) { DVec2::ZERO }
        else { DVec2::new(state.hu(i) / h, state.hv(i) / h) }
    }).collect();
    let grad = GreenGaussGradient::new();
    let mut storage = VectorGradientStorage::new(n);
    let _ = grad.compute_vector_gradient(&velocities, mesh, &mut storage);
    (0..n).map(|i| storage.dv_dx[i] - storage.du_dy[i]).collect()
}
