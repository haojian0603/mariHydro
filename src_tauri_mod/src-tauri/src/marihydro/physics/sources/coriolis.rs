// src-tauri/src/marihydro/physics/sources/coriolis.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use std::f64::consts::PI;

pub struct CoriolisSource {
    f: f64,
    use_exact_rotation: bool,
}

impl CoriolisSource {
    pub fn new(f: f64) -> Self {
        Self { f, use_exact_rotation: true }
    }

    pub fn from_latitude(lat_deg: f64) -> Self {
        let omega = 7.2921e-5;
        let f = 2.0 * omega * (lat_deg * PI / 180.0).sin();
        Self::new(f)
    }

    pub fn with_linear_approximation(mut self) -> Self {
        self.use_exact_rotation = false;
        self
    }

    #[inline]
    fn apply_exact(&self, hu: f64, hv: f64, dt: f64) -> (f64, f64) {
        let theta = self.f * dt;
        let (sin_t, cos_t) = if theta.abs() < 1e-3 {
            let t2 = theta * theta;
            (theta * (1.0 - t2 / 6.0), 1.0 - t2 * 0.5)
        } else {
            theta.sin_cos()
        };
        (hu * cos_t + hv * sin_t, -hu * sin_t + hv * cos_t)
    }

    #[inline]
    fn apply_linear(&self, hu: f64, hv: f64, dt: f64) -> (f64, f64) {
        let dhu = self.f * hv * dt;
        let dhv = -self.f * hu * dt;
        (hu + dhu, hv + dhv)
    }

    pub fn coriolis_parameter(&self) -> f64 { self.f }

    pub fn is_stable(&self, dt: f64) -> bool {
        (self.f * dt).abs() < 0.1
    }

    pub fn max_stable_dt(&self, safety: f64) -> f64 {
        if self.f.abs() < 1e-14 { f64::INFINITY }
        else { safety * 0.1 / self.f.abs() }
    }
}

impl SourceTerm for CoriolisSource {
    fn name(&self) -> &'static str { "CoriolisSource" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, cell_idx: usize, _mesh: &M, state: &S, ctx: &SourceContext,
    ) -> SourceContribution {
        let hu = state.hu(cell_idx);
        let hv = state.hv(cell_idx);
        if ctx.params.is_dry(state.h(cell_idx)) {
            return SourceContribution::ZERO;
        }
        let (hu_new, hv_new) = if self.use_exact_rotation {
            self.apply_exact(hu, hv, ctx.dt)
        } else {
            self.apply_linear(hu, hv, ctx.dt)
        };
        SourceContribution {
            s_h: 0.0,
            s_hu: (hu_new - hu) / ctx.dt,
            s_hv: (hv_new - hv) / ctx.dt,
        }
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        _output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        let n = mesh.n_cells();
        for i in 0..n {
            if ctx.params.is_dry(state.h(i)) { continue; }
            let hu = state.hu(i);
            let hv = state.hv(i);
            let (hu_new, hv_new) = if self.use_exact_rotation {
                self.apply_exact(hu, hv, ctx.dt)
            } else {
                self.apply_linear(hu, hv, ctx.dt)
            };
            output_hu[i] += (hu_new - hu) / ctx.dt;
            output_hv[i] += (hv_new - hv) / ctx.dt;
        }
        Ok(())
    }
}
