// src-tauri/src/marihydro/physics/sources/friction.rs
// Optimized: 2025-11-30
// Performance: 预计算gn²，内联函数，消除循环内重复乘法

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::CellIndex;
use rayon::prelude::*;

pub struct ManningFriction {
    g: f64,
    manning_n: Vec<f64>,
    uniform_n: Option<f64>,
    /// 预计算 g * n^2，避免循环内重复计算
    precomputed_gn2: Option<f64>,
}

impl ManningFriction {
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> Self {
        let gn2 = g * default_n * default_n;
        Self { 
            g, 
            manning_n: vec![default_n; n_cells], 
            uniform_n: Some(default_n),
            precomputed_gn2: Some(gn2),
        }
    }

    pub fn with_field(g: f64, manning_n: Vec<f64>) -> Self {
        Self { g, manning_n, uniform_n: None, precomputed_gn2: None }
    }

    pub fn set_coefficient(&mut self, cell_idx: usize, n: f64) {
        if cell_idx < self.manning_n.len() {
            self.manning_n[cell_idx] = n;
            self.uniform_n = None;
            self.precomputed_gn2 = None;
        }
    }

    /// 计算摩擦系数 cf = g*n^2 / h^(1/3)
    #[inline(always)]
    fn compute_cf(&self, h: f64, idx: usize) -> f64 {
        let h_safe = h.max(1e-4);
        if let Some(gn2) = self.precomputed_gn2 {
            gn2 / h_safe.cbrt()
        } else {
            let n = self.manning_n.get(idx).copied().unwrap_or(0.025);
            self.g * n * n / h_safe.cbrt()
        }
    }

    /// 计算衰减因子
    #[inline(always)]
    fn compute_decay(&self, cf: f64, speed: f64, dt: f64) -> f64 {
        1.0 / (1.0 + dt * cf * speed)
    }
}

impl SourceTerm for ManningFriction {
    fn name(&self) -> &'static str { "ManningFriction" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, cell_idx: usize, _mesh: &M, state: &S, ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h(CellIndex(cell_idx));
        let hu = state.hu(CellIndex(cell_idx));
        let hv = state.hv(CellIndex(cell_idx));
        if ctx.params.is_dry(h) {
            return SourceContribution { s_h: 0.0, s_hu: -hu / ctx.dt, s_hv: -hv / ctx.dt };
        }
        // P1-001 修复：使用 safe_velocity 避免小水深时的数值不稳定
        let vel = ctx.params.safe_velocity(hu, hv, h);
        let speed_sq = vel.speed_squared();
        if speed_sq < 1e-20 { return SourceContribution::ZERO; }
        let speed = speed_sq.sqrt();
        let cf = self.compute_cf(h, cell_idx);
        let decay = self.compute_decay(cf, speed, ctx.dt);
        let factor = (decay - 1.0) / ctx.dt;
        SourceContribution { s_h: 0.0, s_hu: hu * factor, s_hv: hv * factor }
    }

    fn compute_all<M: MeshAccess, S: StateAccess>(
        &self, mesh: &M, state: &S, ctx: &SourceContext,
        _output_h: &mut [f64], output_hu: &mut [f64], output_hv: &mut [f64],
    ) -> MhResult<()> {
        let n_cells = mesh.n_cells();
        let dt = ctx.dt;
        let h_dry = ctx.params.h_dry;
        let h_min = ctx.params.h_min;
        let use_uniform = self.precomputed_gn2.is_some();
        let gn2 = self.precomputed_gn2.unwrap_or(0.0);
        
        for i in 0..n_cells {
            let h = state.h(CellIndex(i));
            let hu = state.hu(CellIndex(i));
            let hv = state.hv(CellIndex(i));
            if h < h_dry {
                output_hu[i] -= hu / dt;
                output_hv[i] -= hv / dt;
                continue;
            }
            // P1-001 修复：使用安全水深计算速度，避免小水深时数值不稳定
            let h_safe = h.max(h_min);
            let u = hu / h_safe;
            let v = hv / h_safe;
            let speed_sq = u * u + v * v;
            if speed_sq < 1e-20 { continue; }
            let cf = if use_uniform { 
                gn2 / h.max(1e-4).cbrt() 
            } else { 
                let n = self.manning_n[i]; 
                self.g * n * n / h.max(1e-4).cbrt() 
            };
            let speed = speed_sq.sqrt();
            let decay = 1.0 / (1.0 + dt * cf * speed);
            let factor = (decay - 1.0) / dt;
            output_hu[i] += hu * factor;
            output_hv[i] += hv * factor;
        }
        Ok(())
    }
}

pub struct ChezyFriction {
    g: f64,
    chezy_c: f64,
    /// 预计算 cf = g / C^2
    cf: f64,
}

impl ChezyFriction {
    pub fn new(g: f64, chezy_c: f64) -> Self { 
        let cf = g / (chezy_c * chezy_c);
        Self { g, chezy_c, cf } 
    }
}

impl SourceTerm for ChezyFriction {
    fn name(&self) -> &'static str { "ChezyFriction" }

    fn compute_cell<M: MeshAccess, S: StateAccess>(
        &self, cell_idx: usize, _mesh: &M, state: &S, ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h(CellIndex(cell_idx));
        let hu = state.hu(CellIndex(cell_idx));
        let hv = state.hv(CellIndex(cell_idx));
        if ctx.params.is_dry(h) {
            return SourceContribution { s_h: 0.0, s_hu: -hu / ctx.dt, s_hv: -hv / ctx.dt };
        }
        // P1-001 修复：使用 safe_velocity 避免小水深时的数值不稳定
        let vel = ctx.params.safe_velocity(hu, hv, h);
        let speed_sq = vel.speed_squared();
        if speed_sq < 1e-20 { return SourceContribution::ZERO; }
        let speed = speed_sq.sqrt();
        let decay = 1.0 / (1.0 + ctx.dt * self.cf * speed);
        let factor = (decay - 1.0) / ctx.dt;
        SourceContribution { s_h: 0.0, s_hu: hu * factor, s_hv: hv * factor }
    }
}
