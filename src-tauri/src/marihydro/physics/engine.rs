// src-tauri/src/marihydro/physics/engine.rs

use ndarray::{Array2, Axis};
use rayon::prelude::*;
use std::mem;
use std::time::Instant;

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::{PhysicsParameters, ProjectManifest};
use crate::marihydro::infra::time::{TimeManager, TimezoneConfig};

use crate::marihydro::domain::boundary::BoundaryManager;
use crate::marihydro::domain::mesh::{IndexStorage, Mesh};
use crate::marihydro::domain::state::State;
use crate::marihydro::forcing::manager::ForcingManager;

use super::schemes::{
    hllc::{solve_hllc, ComputeAxis},
    muscl::{reconstruct_interface, SlopeLimiterType},
    ConservedVars, FluxVars, PrimitiveVars,
};
use super::sources::atmosphere::wind_drag_coefficient_lp81;
use super::sources::friction::apply_friction_implicit;

#[cfg(feature = "turbulence")]
use super::sources::turbulence::SmagorinskyModel;

#[derive(Debug, Clone, Copy)]
struct NumericalConstants {
    min_speed_threshold: f64,
    default_large_dt: f64,
    wind_threshold: f64,
    wind_mag_min: f64,
    tile_size_min: usize,
    tile_size_max: usize,
    tile_adjust_threshold_high: f64,
    tile_adjust_threshold_low: f64,
    machine_eps_dt: f64,
}

impl NumericalConstants {
    fn from_physics(mesh: &Mesh, params: &PhysicsParameters) -> Self {
        let (dx, dy) = mesh.transform.resolution();
        let min_dx = dx.min(dy);

        let max_depth = mesh
            .zb
            .iter()
            .map(|&z| if z < 0.0 { -z } else { 0.0 })
            .fold(0.0, f64::max)
            .max(10.0);

        let machine_eps_dt = (min_dx / (params.gravity * max_depth).sqrt()) * 1e-6;

        log::debug!(
            "数值常量: 最大水深={:.1}m, 机器精度dt={:.3e}s",
            max_depth,
            machine_eps_dt
        );

        Self {
            min_speed_threshold: 1e-6,
            default_large_dt: 10.0,
            wind_threshold: 1e-12,
            wind_mag_min: 1e-8,
            tile_size_min: 16,
            tile_size_max: 64,
            tile_adjust_threshold_high: 0.5,
            tile_adjust_threshold_low: 0.3,
            machine_eps_dt: machine_eps_dt.clamp(1e-10, 1e-3),
        }
    }
}

struct StepContext<'a> {
    h_curr: &'a [f64],
    u_curr: &'a [f64],
    v_curr: &'a [f64],
    c_curr: &'a [f64],

    h_next_ptr: *mut f64,
    u_next_ptr: *mut f64,
    v_next_ptr: *mut f64,
    c_next_ptr: *mut f64,

    zb: &'a [f64],
    roughness: &'a [f64],

    wind_u: &'a [f64],
    wind_v: &'a [f64],
    wind_stride: usize,

    stride: usize,
    ng: usize,

    #[cfg(debug_assertions)]
    array_len: usize,
}

impl<'a> StepContext<'a> {
    unsafe fn new(engine: &'a mut SimulationEngine) -> Result<Self, &'static str> {
        let slices = engine.state.as_slices().map_err(|_| "State内存非连续")?;

        let zb = engine
            .state
            .zb
            .as_slice_memory_order()
            .ok_or("zb内存非连续")?;
        let roughness = engine
            .mesh
            .roughness
            .as_slice_memory_order()
            .ok_or("roughness内存非连续")?;

        let forcing = engine.forcing.get_context();
        let wind_u = forcing
            .wind_u
            .as_slice_memory_order()
            .ok_or("wind_u内存非连续")?;
        let wind_v = forcing
            .wind_v
            .as_slice_memory_order()
            .ok_or("wind_v内存非连续")?;

        let h_next_ptr = engine.next_state.h.as_mut_ptr();
        let u_next_ptr = engine.next_state.u.as_mut_ptr();
        let v_next_ptr = engine.next_state.v.as_mut_ptr();
        let c_next_ptr = engine.next_state.c.as_mut_ptr();

        Ok(Self {
            h_curr: slices.h,
            u_curr: slices.u,
            v_curr: slices.v,
            c_curr: slices.c,
            h_next_ptr,
            u_next_ptr,
            v_next_ptr,
            c_next_ptr,
            zb,
            roughness,
            wind_u,
            wind_v,
            wind_stride: forcing.wind_u.strides()[0] as usize,
            stride: engine.state.h.strides()[0] as usize,
            ng: engine.mesh.ng,

            #[cfg(debug_assertions)]
            array_len: slices.h.len(),
        })
    }

    #[inline(always)]
    fn get_primitive(&self, j: usize, i: usize) -> PrimitiveVars {
        let idx = j * self.stride + i;

        #[cfg(debug_assertions)]
        debug_assert!(idx < self.array_len, "索引越界: {}", idx);

        unsafe {
            let h = *self.h_curr.get_unchecked(idx);
            let z = *self.zb.get_unchecked(idx);

            PrimitiveVars {
                h,
                u: *self.u_curr.get_unchecked(idx),
                v: *self.v_curr.get_unchecked(idx),
                c: *self.c_curr.get_unchecked(idx),
                z,
                eta: h + z,
            }
        }
    }

    #[inline(always)]
    unsafe fn read_h_next(&self, idx: usize) -> f64 {
        #[cfg(debug_assertions)]
        debug_assert!(idx < self.array_len);

        *self.h_next_ptr.add(idx)
    }

    #[inline(always)]
    fn read_roughness(&self, idx: usize) -> f64 {
        unsafe { *self.roughness.get_unchecked(idx) }
    }

    #[inline(always)]
    fn read_wind(&self, j: usize, i: usize) -> (f64, f64) {
        let idx = (j - self.ng) * self.wind_stride + (i - self.ng);
        unsafe {
            (
                *self.wind_u.get_unchecked(idx),
                *self.wind_v.get_unchecked(idx),
            )
        }
    }

    #[inline(always)]
    unsafe fn write_next(&self, idx: usize, h: f64, u: f64, v: f64, c: f64) {
        #[cfg(debug_assertions)]
        debug_assert!(idx < self.array_len);

        *self.h_next_ptr.add(idx) = h;
        *self.u_next_ptr.add(idx) = u;
        *self.v_next_ptr.add(idx) = v;
        *self.c_next_ptr.add(idx) = c;
    }

    #[inline(always)]
    unsafe fn write_velocity(&self, idx: usize, u: f64, v: f64) {
        #[cfg(debug_assertions)]
        debug_assert!(idx < self.array_len);

        *self.u_next_ptr.add(idx) = u;
        *self.v_next_ptr.add(idx) = v;
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimulationMetrics {
    pub avg_cfl_dt: f64,
    pub flux_compute_ms: u64,
    pub cell_update_ms: u64,
    pub source_apply_ms: u64,
    pub total_step_ms: u64,
    pub active_cells: usize,
    pub max_wave_speed: f64,
    pub steps_completed: u64,
    pub numerical_errors: u64,
}

impl SimulationMetrics {
    pub fn reset_timers(&mut self) {
        self.flux_compute_ms = 0;
        self.cell_update_ms = 0;
        self.source_apply_ms = 0;
        self.total_step_ms = 0;
    }

    pub fn report(&self) -> String {
        format!(
            "步数: {}, CFL: {:.4}s, 波速: {:.2}m/s, 耗时: {}ms (通量{}ms, 更新{}ms, 源项{}ms), 单元: {}, 异常: {}",
            self.steps_completed, self.avg_cfl_dt, self.max_wave_speed,
            self.total_step_ms, self.flux_compute_ms, self.cell_update_ms, self.source_apply_ms,
            self.active_cells, self.numerical_errors
        )
    }

    pub fn cells_per_second(&self) -> f64 {
        if self.total_step_ms == 0 {
            return 0.0;
        }
        (self.active_cells as f64 * 1000.0) / self.total_step_ms as f64
    }
}

pub struct SimulationEngine {
    pub state: State,
    next_state: State,
    pub mesh: Mesh,
    pub forcing: ForcingManager,
    pub boundary: BoundaryManager,
    pub time: TimeManager,

    flux_x: Array2<FluxVars>,
    flux_y: Array2<FluxVars>,

    params: PhysicsParameters,
    pub metrics: SimulationMetrics,
    constants: NumericalConstants,

    rho_ratio: f64,
    sqrt_g: f64,
    cfl_factor: f64,

    tile_size: usize,
    ghost_cells_valid: bool,

    #[cfg(feature = "turbulence")]
    turbulence_model: Option<SmagorinskyModel>,
}

impl SimulationEngine {
    pub fn init(manifest: &ProjectManifest) -> MhResult<Self> {
        log::info!("初始化模拟引擎");

        let mesh = Mesh::init(manifest)?;
        let ng = mesh.ng;

        if ng < 2 {
            return Err(MhError::ConfigError {
                field: "mesh.ng".into(),
                message: format!("MUSCL需要至少2层Ghost单元，当前: {}", ng),
            });
        }

        let state = State::cold_start(mesh.nx, mesh.ny, ng, 0.0, mesh.zb_arc())?;
        let next_state = state.clone_structure();

        if !state.is_standard_layout() || !next_state.is_standard_layout() {
            return Err(MhError::InternalError(
                "State数组必须是C-contiguous布局".into(),
            ));
        }
        log::debug!("内存布局验证通过");

        Self::validate_active_indices(&mesh.active_indices, state.h.dim())?;
        log::debug!("活跃索引验证通过: {}个单元", mesh.active_indices.len());

        let mut boundary = BoundaryManager::from_manifest(manifest);
        boundary.register_structured_mesh(&mesh, manifest);
        boundary.validate()?;

        let forcing = ForcingManager::init(manifest, &mesh)?;
        let time = TimeManager::new(&manifest.start_time.to_rfc3339(), TimezoneConfig::Utc)?;

        let flux_x = Array2::from_elem((mesh.ny, mesh.nx + 1), FluxVars::default());
        let flux_y = Array2::from_elem((mesh.ny + 1, mesh.nx), FluxVars::default());

        let params = manifest.physics.clone();
        let constants = NumericalConstants::from_physics(&mesh, &params);

        let mut engine = Self {
            state,
            next_state,
            mesh,
            forcing,
            boundary,
            time,
            flux_x,
            flux_y,
            params,
            metrics: SimulationMetrics::default(),
            constants,
            rho_ratio: 1.225 / 1025.0,
            sqrt_g: params.gravity.sqrt(),
            cfl_factor: 0.9,
            tile_size: 32,
            ghost_cells_valid: false,

            #[cfg(feature = "turbulence")]
            turbulence_model: None,
        };

        engine.metrics.active_cells = engine.mesh.active_indices.len();
        log::info!("模拟引擎初始化完成");

        Ok(engine)
    }

    fn validate_active_indices(indices: &IndexStorage, shape: (usize, usize)) -> MhResult<()> {
        let (max_j, max_i) = shape;

        for (j, i) in indices.iter() {
            if j >= max_j || i >= max_i {
                return Err(MhError::ConfigError {
                    field: "active_indices".into(),
                    message: format!("索引越界: ({}, {}), 尺寸: ({}, {})", j, i, max_j, max_i),
                });
            }
        }
        Ok(())
    }

    pub fn step(&mut self, target_dt: f64) -> MhResult<f64> {
        let step_start = Instant::now();

        self.ghost_cells_valid = false;
        log::trace!("开始时间步进 (目标dt: {:.4}s)", target_dt);

        let current_time = self.time.current_utc();
        self.forcing.update(current_time, &self.mesh)?;

        let tide_data = self.forcing.get_boundary_forcing();
        self.boundary
            .update_ghost_cells(&mut self.state, &self.mesh.zb, tide_data);
        self.ghost_cells_valid = true;
        log::trace!("Ghost单元已更新");

        let dt = {
            let ctx =
                unsafe { StepContext::new(self).map_err(|e| MhError::InternalError(e.into()))? };

            let dt_cfl = self.compute_cfl_dt(&ctx);
            let dt = target_dt.min(dt_cfl).max(self.constants.machine_eps_dt);
            self.metrics.avg_cfl_dt = dt_cfl;
            log::trace!("CFL步长: {:.4}s, 实际: {:.4}s", dt_cfl, dt);

            let t0 = Instant::now();
            self.compute_fluxes(&ctx);
            self.metrics.flux_compute_ms = t0.elapsed().as_millis() as u64;

            let t1 = Instant::now();
            self.update_cells(&ctx, dt);
            self.metrics.cell_update_ms = t1.elapsed().as_millis() as u64;

            let t2 = Instant::now();
            self.apply_sources(&ctx, dt);
            self.metrics.source_apply_ms = t2.elapsed().as_millis() as u64;

            dt
        };

        if let Err(_) = self.next_state.validate(self.time.elapsed_seconds()) {
            self.metrics.numerical_errors += 1;
            self.ghost_cells_valid = false;
            log::error!(
                "数值不稳定 (步数: {}, 时间: {:.2}s)",
                self.metrics.steps_completed,
                self.time.elapsed_seconds()
            );
            return Err(MhError::NumericalInstability {
                message: "检测到NaN/Inf".into(),
                time: self.time.elapsed_seconds(),
                location: None,
            });
        }

        mem::swap(&mut self.state, &mut self.next_state);
        self.ghost_cells_valid = false;

        self.time.advance(dt);
        self.metrics.total_step_ms = step_start.elapsed().as_millis() as u64;
        self.metrics.steps_completed += 1;

        log::trace!("时间步完成 (耗时: {}ms)", self.metrics.total_step_ms);

        Ok(dt)
    }

    pub fn run_until(&mut self, end_time: f64, target_dt: f64, log_interval: u64) -> MhResult<()> {
        log::info!("开始批量模拟: 目标时间={:.2}s", end_time);

        while self.time.elapsed_seconds() < end_time {
            let _dt = self.step(target_dt)?;

            if self.metrics.steps_completed % log_interval == 0 {
                log::info!("{}", self.metrics.report());
            }

            if self.metrics.steps_completed % 100 == 0 {
                self.adjust_tile_size();
            }
        }

        log::info!("批量模拟完成");
        Ok(())
    }

    pub fn adjust_tile_size(&mut self) {
        let flux_ratio =
            self.metrics.flux_compute_ms as f64 / (self.metrics.total_step_ms as f64 + 1e-6);

        let old_size = self.tile_size;

        if flux_ratio > self.constants.tile_adjust_threshold_high
            && self.tile_size < self.constants.tile_size_max
        {
            self.tile_size *= 2;
        } else if flux_ratio < self.constants.tile_adjust_threshold_low
            && self.tile_size > self.constants.tile_size_min
        {
            self.tile_size /= 2;
        }

        if self.tile_size != old_size {
            log::debug!(
                "Tile大小调整: {} -> {} (通量占比: {:.1}%)",
                old_size,
                self.tile_size,
                flux_ratio * 100.0
            );
        }
    }

    fn compute_cfl_dt(&mut self, ctx: &StepContext) -> f64 {
        let h_min = self.params.h_min;
        let (dx, dy) = self.mesh.transform.resolution();
        let min_d = dx.min(dy);

        let max_speed = self
            .mesh
            .active_indices
            .par_iter()
            .map(|&(j, i)| {
                let idx = j * ctx.stride + i;

                unsafe {
                    let h = *ctx.h_curr.get_unchecked(idx);
                    if h <= h_min * (1.0 + 1e-12) {
                        return 0.0;
                    }

                    let u = *ctx.u_curr.get_unchecked(idx);
                    let v = *ctx.v_curr.get_unchecked(idx);

                    let vel_mag = (u * u + v * v).sqrt();
                    let c = self.sqrt_g * h.sqrt();
                    vel_mag + c
                }
            })
            .reduce(|| 0.0, f64::max);

        self.metrics.max_wave_speed = max_speed;

        if max_speed < self.constants.min_speed_threshold {
            return self.constants.default_large_dt;
        }

        self.cfl_factor * min_d / max_speed
    }

    fn compute_fluxes(&mut self, ctx: &StepContext) {
        debug_assert!(self.ghost_cells_valid);

        let ng = ctx.ng;
        let g = self.params.gravity;
        let h_min = self.params.h_min;
        let limiter = SlopeLimiterType::VanLeer;

        self.flux_x
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j_phys, mut row)| {
                let j = j_phys + ng;
                let mut w = [PrimitiveVars::default(); 4];

                for k in 0..4 {
                    w[k] = ctx.get_primitive(j, ng - 1 + k);
                }

                for (k, flux) in row.iter_mut().enumerate() {
                    if k > 0 {
                        w[0] = w[1];
                        w[1] = w[2];
                        w[2] = w[3];
                        w[3] = ctx.get_primitive(j, k + ng + 2);
                    }

                    let (u_l, u_r) =
                        reconstruct_interface(&w[0], &w[1], &w[2], &w[3], limiter, h_min);
                    *flux = solve_hllc(&u_l, &u_r, ComputeAxis::X, g, h_min);
                }
            });

        self.flux_y
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(j_phys, mut row)| {
                let j = j_phys + ng - 1;

                for (i_phys, flux) in row.iter_mut().enumerate() {
                    let i = i_phys + ng;

                    let w0 = ctx.get_primitive(j - 1, i);
                    let w1 = ctx.get_primitive(j, i);
                    let w2 = ctx.get_primitive(j + 1, i);
                    let w3 = ctx.get_primitive(j + 2, i);

                    let (u_b, u_t) = reconstruct_interface(&w0, &w1, &w2, &w3, limiter, h_min);
                    *flux = solve_hllc(&u_b, &u_t, ComputeAxis::Y, g, h_min);
                }
            });
    }

    fn update_cells(&self, ctx: &StepContext, dt: f64) {
        let (dx, dy) = self.mesh.transform.resolution();
        let dt_dx = dt / dx;
        let dt_dy = dt / dy;
        let ng = ctx.ng;
        let h_min = self.params.h_min;

        self.mesh.active_indices.par_iter().for_each(|&(j, i)| {
            let idx = j * ctx.stride + i;

            let (h_old, u_old, v_old, c_old, z) = unsafe {
                (
                    *ctx.h_curr.get_unchecked(idx),
                    *ctx.u_curr.get_unchecked(idx),
                    *ctx.v_curr.get_unchecked(idx),
                    *ctx.c_curr.get_unchecked(idx),
                    *ctx.zb.get_unchecked(idx),
                )
            };

            let p_old = PrimitiveVars {
                h: h_old,
                u: u_old,
                v: v_old,
                c: c_old,
                z,
                eta: h_old + z,
            };
            let mut u_curr = ConservedVars::from_primitive(&p_old);

            let fj = j - ng;
            let fi = i - ng;

            let fx_l = &self.flux_x[[fj, fi]];
            let fx_r = &self.flux_x[[fj, fi + 1]];
            let fy_b = &self.flux_y[[fj, fi]];
            let fy_t = &self.flux_y[[fj + 1, fi]];

            u_curr.h -= dt_dx * (fx_r.mass - fx_l.mass) + dt_dy * (fy_t.mass - fy_b.mass);
            u_curr.hu -= dt_dx * (fx_r.x_mom - fx_l.x_mom) + dt_dy * (fy_t.x_mom - fy_b.x_mom);
            u_curr.hv -= dt_dx * (fx_r.y_mom - fx_l.y_mom) + dt_dy * (fy_t.y_mom - fy_b.y_mom);
            u_curr.hc -= dt_dx * (fx_r.sed - fx_l.sed) + dt_dy * (fy_t.sed - fy_b.sed);

            let p_new = u_curr.to_primitive(z, h_min);

            unsafe {
                ctx.write_next(idx, p_new.h, p_new.u, p_new.v, p_new.c);
            }
        });
    }

    fn apply_sources(&self, ctx: &StepContext, dt: f64) {
        let g = self.params.gravity;
        let h_min = self.params.h_min;

        self.mesh.active_indices.par_iter().for_each(|&(j, i)| {
            let idx = j * ctx.stride + i;

            unsafe {
                let h = ctx.read_h_next(idx);
                if h <= h_min * (1.0 + 1e-12) {
                    ctx.write_velocity(idx, 0.0, 0.0);
                    return;
                }

                let u_old = *ctx.u_next_ptr.add(idx);
                let v_old = *ctx.v_next_ptr.add(idx);
                let n = ctx.read_roughness(idx);

                let (u_fric, v_fric) = apply_friction_implicit(u_old, v_old, h, n, dt, g, h_min);

                let (wind_u, wind_v) = ctx.read_wind(j, i);
                let w_sq = wind_u * wind_u + wind_v * wind_v;

                if !(w_sq > self.constants.wind_threshold) {
                    ctx.write_velocity(idx, u_fric, v_fric);
                    return;
                }

                let h_safe = h.max(h_min);
                let w_mag = w_sq.sqrt().max(self.constants.wind_mag_min);
                let cd = wind_drag_coefficient_lp81(w_mag);
                let factor = self.rho_ratio * cd * w_mag * dt / h_safe;

                ctx.write_velocity(idx, u_fric + factor * wind_u, v_fric + factor * wind_v);
            }
        });
    }

    pub fn get_metrics(&self) -> &SimulationMetrics {
        &self.metrics
    }

    pub fn reset_metrics_timers(&mut self) {
        self.metrics.reset_timers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_max_depth() {
        let zb = vec![-50.0, -100.0, -30.0, 5.0];
        let max_depth = zb
            .iter()
            .map(|&z| if z < 0.0 { -z } else { 0.0 })
            .fold(0.0, f64::max);
        assert_eq!(max_depth, 100.0);
    }

    #[test]
    fn test_machine_eps_calculation() {
        let dx = 10.0;
        let g = 9.81;
        let h_max = 100.0;
        let eps = (dx / (g * h_max).sqrt()) * 1e-6;
        assert!(eps > 1e-10 && eps < 1e-3);
    }

    #[test]
    fn test_nan_defense() {
        let w_sq = f64::NAN;
        let threshold = 1e-12;
        assert!(!(w_sq > threshold));

        let w_sq_valid = 1e-10;
        assert!(!(w_sq_valid > threshold));
    }
}
