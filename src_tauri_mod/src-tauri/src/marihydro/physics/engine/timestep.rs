// src-tauri/src/marihydro/physics/engine/timestep.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use crate::marihydro::domain::state::ShallowWaterState;
use rayon::prelude::*;

pub struct CflCalculator {
    g: f64,
    cfl: f64,
    dt_min: f64,
    dt_max: f64,
}

impl CflCalculator {
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self { g, cfl: params.cfl, dt_min: params.dt_min, dt_max: params.dt_max }
    }

    pub fn compute_dt<M: MeshAccess + Sync>(&self, state: &ShallowWaterState, mesh: &M, params: &NumericalParams) -> f64 {
        let n_cells = mesh.n_cells();
        if n_cells == 0 { return self.dt_max; }
        let max_speed = self.compute_max_wave_speed(state, mesh, params);
        let min_length = self.compute_min_char_length(mesh);
        if max_speed < params.min_wave_speed { return self.dt_max; }
        let dt = self.cfl * min_length / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }

    fn compute_max_wave_speed<M: MeshAccess + Sync>(&self, state: &ShallowWaterState, mesh: &M, params: &NumericalParams) -> f64 {
        let n = mesh.n_cells();
        (0..n).into_par_iter().map(|i| {
            let h = state.h[i];
            if params.is_dry(h) { return 0.0; }
            let vel = params.safe_velocity(state.hu[i], state.hv[i], h);
            let speed = (vel.u * vel.u + vel.v * vel.v).sqrt();
            let c = (self.g * h).sqrt();
            speed + c
        }).reduce(|| 0.0, f64::max)
    }

    fn compute_min_char_length<M: MeshAccess + Sync>(&self, mesh: &M) -> f64 {
        let n = mesh.n_cells();
        (0..n).into_par_iter().map(|i| {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            let faces = mesh.cell_faces(cell);
            let perimeter: f64 = faces.iter().map(|&f| mesh.face_length(f)).sum();
            if perimeter < 1e-14 { return f64::MAX; }
            2.0 * area / perimeter
        }).reduce(|| f64::MAX, f64::min)
    }

    pub fn compute_from_max_speed(&self, max_speed: f64, min_length: f64) -> f64 {
        if max_speed < 1e-12 { return self.dt_max; }
        let dt = self.cfl * min_length / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }
}

pub struct TimeStepController {
    calculator: CflCalculator,
    current_dt: f64,
    growth_factor: f64,
    shrink_factor: f64,
}

impl TimeStepController {
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self {
            calculator: CflCalculator::new(g, params),
            current_dt: params.dt_max,
            growth_factor: 1.1,
            shrink_factor: 0.5,
        }
    }

    pub fn update<M: MeshAccess + Sync>(&mut self, state: &ShallowWaterState, mesh: &M, params: &NumericalParams) -> f64 {
        let suggested = self.calculator.compute_dt(state, mesh, params);
        let grown = self.current_dt * self.growth_factor;
        self.current_dt = suggested.min(grown);
        self.current_dt
    }

    pub fn shrink(&mut self) {
        self.current_dt *= self.shrink_factor;
        self.current_dt = self.current_dt.max(self.calculator.dt_min);
    }

    pub fn current_dt(&self) -> f64 { self.current_dt }
}
