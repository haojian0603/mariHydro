// src-tauri/src/marihydro/physics/sources/friction.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::NumericalParams;
use super::base::FrictionDecayCalculator;
use rayon::prelude::*;

pub struct ManningFriction {
    g: f64,
    manning_n: Vec<f64>,
    uniform_n: Option<f64>,
}

impl ManningFriction {
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> Self {
        Self { g, manning_n: vec![default_n; n_cells], uniform_n: Some(default_n) }
    }

    pub fn with_field(g: f64, manning_n: Vec<f64>) -> Self {
        Self { g, manning_n, uniform_n: None }
    }

    pub fn set_coefficient(&mut self, cell_idx: usize, n: f64) {
        if cell_idx < self.manning_n.len() {
            self.manning_n[cell_idx] = n;
            self.uniform_n = None;
        }
    }

    fn get_n(&self, idx: usize) -> f64 {
        self.uniform_n.unwrap_or_else(|| self.manning_n.get(idx).copied().unwrap_or(0.025))
    }
}

impl SourceTerm for ManningFriction {
    fn name(&self) -> &'static str { "ManningFriction" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, cell_idx: usize, _mesh: &M, state: &S, ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h(cell_idx);
        let hu = state.hu(cell_idx);
        let hv = state.hv(cell_idx);
        if ctx.params.is_dry(h) {
            return SourceContribution { s_h: 0.0, s_hu: -hu / ctx.dt, s_hv: -hv / ctx.dt };
        }
        let n = self.get_n(cell_idx);
        let calc = FrictionDecayCalculator::new(self.g, ctx.params);
        let cf = calc.compute_manning_cf(h, n);
        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 {
            return SourceContribution::ZERO;
        }
        let speed = speed_sq.sqrt();
        let decay = calc.compute_decay(cf, speed, ctx.dt);
        let factor = (decay - 1.0) / ctx.dt;
        SourceContribution { s_h: 0.0, s_hu: hu * factor, s_hv: hv * factor }
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        let n_cells = mesh.n_cells();
        let calc = FrictionDecayCalculator::new(self.g, ctx.params);
        for i in 0..n_cells {
            let h = state.h(i);
            let hu = state.hu(i);
            let hv = state.hv(i);
            if ctx.params.is_dry(h) {
                output_hu[i] -= hu / ctx.dt;
                output_hv[i] -= hv / ctx.dt;
                continue;
            }
            let n = self.get_n(i);
            let cf = calc.compute_manning_cf(h, n);
            let speed_sq = (hu * hu + hv * hv) / (h * h);
            if speed_sq < 1e-20 { continue; }
            let speed = speed_sq.sqrt();
            let decay = calc.compute_decay(cf, speed, ctx.dt);
            let factor = (decay - 1.0) / ctx.dt;
            output_hu[i] += hu * factor;
            output_hv[i] += hv * factor;
        }
        Ok(())
    }
}

pub struct ChezyFriction {
    g: f64,
    chezy_c: f64,
}

impl ChezyFriction {
    pub fn new(g: f64, chezy_c: f64) -> Self { Self { g, chezy_c } }
}

impl SourceTerm for ChezyFriction {
    fn name(&self) -> &'static str { "ChezyFriction" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, cell_idx: usize, _mesh: &M, state: &S, ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h(cell_idx);
        let hu = state.hu(cell_idx);
        let hv = state.hv(cell_idx);
        if ctx.params.is_dry(h) {
            return SourceContribution { s_h: 0.0, s_hu: -hu / ctx.dt, s_hv: -hv / ctx.dt };
        }
        let calc = FrictionDecayCalculator::new(self.g, ctx.params);
        let cf = calc.compute_chezy_cf(self.chezy_c);
        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 { return SourceContribution::ZERO; }
        let speed = speed_sq.sqrt();
        let decay = calc.compute_decay(cf, speed, ctx.dt);
        let factor = (decay - 1.0) / ctx.dt;
        SourceContribution { s_h: 0.0, s_hu: hu * factor, s_hv: hv * factor }
    }
}
