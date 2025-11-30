// src-tauri/src/marihydro/physics/sources/atmosphere.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::{CellIndex, FaceIndex};
use crate::marihydro::physics::numerics::gradient::{GradientMethod, GreenGaussGradient, ScalarGradientStorage};
use glam::DVec2;

const MAX_WIND_SPEED: f64 = 100.0;

#[inline]
pub fn wind_drag_coefficient_lp81(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 11.0 { 1.2e-3 }
    else if w < 25.0 { (0.49 + 0.065 * w) * 1e-3 }
    else { 2.11e-3 }
}

#[inline]
pub fn wind_drag_coefficient_wu82(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    (0.8 + 0.065 * w) * 1e-3
}

pub struct WindStressSource {
    wind_u: Vec<f64>,
    wind_v: Vec<f64>,
    rho_air: f64,
    rho_water: f64,
}

impl WindStressSource {
    pub fn new(n_cells: usize, rho_air: f64, rho_water: f64) -> Self {
        Self { wind_u: vec![0.0; n_cells], wind_v: vec![0.0; n_cells], rho_air, rho_water }
    }

    pub fn set_uniform_wind(&mut self, u: f64, v: f64) {
        self.wind_u.fill(u);
        self.wind_v.fill(v);
    }

    pub fn set_wind_field(&mut self, u: Vec<f64>, v: Vec<f64>) {
        self.wind_u = u;
        self.wind_v = v;
    }

    pub fn update_wind(&mut self, idx: usize, u: f64, v: f64) {
        if idx < self.wind_u.len() {
            self.wind_u[idx] = u;
            self.wind_v[idx] = v;
        }
    }

    fn compute_stress(&self, u: f64, v: f64) -> (f64, f64) {
        let mag = (u * u + v * v).sqrt();
        if mag < 1e-8 { return (0.0, 0.0); }
        let cd = wind_drag_coefficient_lp81(mag);
        let tau = self.rho_air * cd * mag;
        (tau * u, tau * v)
    }

    fn compute_acceleration(&self, u: f64, v: f64, h: f64) -> (f64, f64) {
        if h < 1e-6 { return (0.0, 0.0); }
        let (tau_x, tau_y) = self.compute_stress(u, v);
        let factor = 1.0 / (self.rho_water * h);
        (tau_x * factor, tau_y * factor)
    }
}

impl SourceTerm for WindStressSource {
    fn name(&self) -> &'static str { "WindStress" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, cell_idx: usize, _mesh: &M, state: &S, ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h(cell_idx);
        if ctx.params.is_dry(h) { return SourceContribution::ZERO; }
        let u = self.wind_u.get(cell_idx).copied().unwrap_or(0.0);
        let v = self.wind_v.get(cell_idx).copied().unwrap_or(0.0);
        let (ax, ay) = self.compute_acceleration(u, v, h);
        SourceContribution { s_h: 0.0, s_hu: h * ax, s_hv: h * ay }
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        _output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        for i in 0..mesh.n_cells() {
            let h = state.h(i);
            if ctx.params.is_dry(h) { continue; }
            let u = self.wind_u.get(i).copied().unwrap_or(0.0);
            let v = self.wind_v.get(i).copied().unwrap_or(0.0);
            let (ax, ay) = self.compute_acceleration(u, v, h);
            output_hu[i] += h * ax;
            output_hv[i] += h * ay;
        }
        Ok(())
    }
}

pub struct PressureGradientSource {
    pressure: Vec<f64>,
    rho_water: f64,
}

impl PressureGradientSource {
    pub fn new(n_cells: usize, rho_water: f64) -> Self {
        Self { pressure: vec![101325.0; n_cells], rho_water }
    }

    pub fn set_pressure_field(&mut self, p: Vec<f64>) {
        self.pressure = p;
    }

    pub fn update_pressure(&mut self, idx: usize, p: f64) {
        if idx < self.pressure.len() {
            self.pressure[idx] = p;
        }
    }
}

impl SourceTerm for PressureGradientSource {
    fn name(&self) -> &'static str { "PressureGradient" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, _cell_idx: usize, _mesh: &M, _state: &S, _ctx: &SourceContext,
    ) -> SourceContribution {
        SourceContribution::ZERO
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        _output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        let n = mesh.n_cells();
        let grad = GreenGaussGradient::new();
        let mut storage = ScalarGradientStorage::new(n);
        grad.compute_scalar_gradient(&self.pressure, mesh, &mut storage)?;
        let factor = -1.0 / self.rho_water;
        for i in 0..n {
            let h = state.h(i);
            if ctx.params.is_dry(h) { continue; }
            let gp = storage.get(i);
            output_hu[i] += h * gp.x * factor;
            output_hv[i] += h * gp.y * factor;
        }
        Ok(())
    }
}
