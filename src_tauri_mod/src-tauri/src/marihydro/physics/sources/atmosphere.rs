// src-tauri/src/marihydro/physics/sources/atmosphere.rs
// 优化版本：预计算因子，并行化梯度计算
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::source::{SourceContribution, SourceContext, SourceTerm};
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::CellIndex;
use crate::marihydro::physics::numerics::gradient::ScalarGradientStorage;
use glam::DVec2;
use rayon::prelude::*;

const MAX_WIND_SPEED: f64 = 100.0;

/// Large and Pond (1981) 风阻系数
#[inline(always)]
pub fn wind_drag_coefficient_lp81(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 11.0 { 1.2e-3 }
    else if w < 25.0 { (0.49 + 0.065 * w) * 1e-3 }
    else { 2.11e-3 }
}

/// Wu (1982) 风阻系数
#[inline(always)]
pub fn wind_drag_coefficient_wu82(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    (0.8 + 0.065 * w) * 1e-3
}

pub struct WindStressSource {
    wind_u: Vec<f64>,
    wind_v: Vec<f64>,
    rho_air: f64,
    rho_water: f64,
    /// 预计算 rho_air / rho_water
    stress_factor: f64,
}

impl WindStressSource {
    pub fn new(n_cells: usize, rho_air: f64, rho_water: f64) -> Self {
        Self { 
            wind_u: vec![0.0; n_cells], 
            wind_v: vec![0.0; n_cells], 
            rho_air, 
            rho_water, 
            stress_factor: rho_air / rho_water 
        }
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

    /// 优化后的加速度计算，使用预计算的 stress_factor
    #[inline(always)]
    fn compute_acceleration(&self, wu: f64, wv: f64, h: f64) -> (f64, f64) {
        if h < 1e-6 { return (0.0, 0.0); }
        let mag = (wu * wu + wv * wv).sqrt();
        if mag < 1e-8 { return (0.0, 0.0); }
        let cd = wind_drag_coefficient_lp81(mag);
        let factor = self.stress_factor * cd * mag / h;
        (factor * wu, factor * wv)
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
        let n = mesh.n_cells();
        let h_dry = ctx.params.h_dry;
        for i in 0..n {
            let h = state.h(i);
            if h < h_dry { continue; }
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
    /// 预计算 -1.0 / rho_water
    inv_rho: f64,
    /// 缓存梯度存储，避免重复分配
    grad_storage: ScalarGradientStorage,
}

impl PressureGradientSource {
    pub fn new(n_cells: usize, rho_water: f64) -> Self {
        Self { 
            pressure: vec![101325.0; n_cells], 
            rho_water, 
            inv_rho: -1.0 / rho_water,
            grad_storage: ScalarGradientStorage::new(n_cells),
        }
    }

    pub fn set_pressure_field(&mut self, p: Vec<f64>) {
        if p.len() != self.pressure.len() {
            self.grad_storage = ScalarGradientStorage::new(p.len());
        }
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
        // 使用 unsafe 获取可变引用（grad_storage 在此调用期间是独占的）
        let storage = unsafe { &mut *(&self.grad_storage as *const _ as *mut ScalarGradientStorage) };
        compute_scalar_gradient_parallel(&self.pressure, mesh, storage);
        let factor = self.inv_rho;
        let h_dry = ctx.params.h_dry;
        for i in 0..n {
            let h = state.h(i);
            if h < h_dry { continue; }
            let gp = storage.get(i);
            output_hu[i] += h * gp.x * factor;
            output_hv[i] += h * gp.y * factor;
        }
        Ok(())
    }
}

/// 并行计算标量场梯度 (Green-Gauss)
fn compute_scalar_gradient_parallel<M: MeshAccess>(
    field: &[f64], mesh: &M, output: &mut ScalarGradientStorage,
) {
    output.reset();
    let n = mesh.n_cells();
    let grads: Vec<DVec2> = (0..n).into_par_iter().map(|i| {
        let cell = CellIndex(i);
        let area = mesh.cell_area(cell);
        if area < 1e-14 { return DVec2::ZERO; }
        let mut g = DVec2::ZERO;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            let sign = if i == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;
            let phi = if !neighbor.is_valid() { field[i] }
            else { 
                let o = if i == owner.0 { neighbor.0 } else { owner.0 }; 
                0.5 * (field[i] + field[o]) 
            };
            g += ds * phi;
        }
        g / area
    }).collect();
    for (i, g) in grads.into_iter().enumerate() { 
        output.set(i, g); 
    }
}
