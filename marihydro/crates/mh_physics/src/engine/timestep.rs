// crates/mh_physics/src/engine/timestep.rs

//! 时间步长控制模块
//!
//! 提供基于CFL条件的自适应时间步长控制，支持Backend泛型化。
//!
//! ## CFL条件
//!
//! 时间步长需满足：Δt ≤ C·min(Δx_i/(|u_i|+√(gh_i)))

use crate::adapter::PhysicsMesh;
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;
use mh_runtime::prelude::*;
use rayon::prelude::*;
use std::marker::PhantomData;

#[inline]
#[track_caller]
fn scalar_from_config_or_panic<S: RuntimeScalar>(value: f64, context: &'static str) -> S {
    S::from_config(value).unwrap_or_else(|| {
        panic!(
            "[mh_physics::engine::timestep] config scalar conversion failed: context={context}, value={value}"
        )
    })
}

/// CFL计算器
#[derive(Clone, Debug)]
pub struct CflCalculator<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    g: B::Scalar,
    cfl: B::Scalar,
    dt_min: B::Scalar,
    dt_max: B::Scalar,
    cached_dx_min: Option<B::Scalar>,
    min_wave_speed: B::Scalar,
    marker: PhantomData<B>,
}

impl<B: Backend> CflCalculator<B>
where
    B::Scalar: RuntimeScalar,
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(g: B::Scalar, params: &NumericalParams<B::Scalar>) -> Self {
        Self {
            g,
            cfl: params.cfl,
            dt_min: params.dt_min,
            dt_max: params.dt_max,
            cached_dx_min: None,
            min_wave_speed: params.min_wave_speed,
            marker: PhantomData,
        }
    }

    pub fn precompute_dx_min(&mut self, mesh: &PhysicsMesh) {
        self.cached_dx_min = Some(self.compute_min_char_length(mesh));
    }

    pub fn dx_min(&self) -> Option<B::Scalar> {
        self.cached_dx_min
    }

    pub fn compute_dt(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        params: &NumericalParams<B::Scalar>
    ) -> B::Scalar {
        let n_cells = mesh.cell_count();
        if n_cells == 0 {
            return self.dt_max;
        }

        let min_length = self.cached_dx_min
            .unwrap_or_else(|| self.compute_min_char_length(mesh));

        if !min_length.is_finite() || min_length <= B::Scalar::ZERO {
            return self.dt_min;
        }

        let max_speed = self.compute_max_wave_speed_parallel(state, params);

        if !max_speed.is_finite() || max_speed <= B::Scalar::ZERO {
            return self.dt_min;
        }
        if max_speed < self.min_wave_speed {
            return self.dt_max.max(self.dt_min);
        }

        let dt = self.cfl * min_length / max_speed;
        self.clamp_dt(dt)
    }

    pub fn compute_from_max_speed(&self, max_speed: B::Scalar) -> B::Scalar {
        if !max_speed.is_finite() || max_speed <= B::Scalar::ZERO {
            return self.dt_min;
        }
        if max_speed < self.min_wave_speed {
            return self.dt_max.max(self.dt_min);
        }

        let min_length = self.cached_dx_min.unwrap_or(B::Scalar::ZERO);
        if !min_length.is_finite() || min_length <= B::Scalar::ZERO {
            return self.dt_min;
        }
        let dt = self.cfl * min_length / max_speed;
        self.clamp_dt(dt)
    }

    #[inline]
    fn clamp_dt(&self, dt: B::Scalar) -> B::Scalar {
        if !dt.is_finite() || dt <= B::Scalar::ZERO {
            return self.dt_min;
        }
        dt.min(self.dt_max).max(self.dt_min)
    }

    fn compute_max_wave_speed_parallel(
        &self,
        state: &ShallowWaterState<B>,
        params: &NumericalParams<B::Scalar>,
    ) -> B::Scalar {
        let n = state.h.len();
        if n == 0 {
            return B::Scalar::ZERO;
        }

        (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let h = state.h[i];
                if params.is_dry(h) {
                    return None;
                }

                let (u, v) = params.safe_velocity_components(state.hu[i], state.hv[i], h);
                let speed = (u * u + v * v).sqrt();
                let c = (self.g * h.max(B::Scalar::ZERO)).sqrt();
                let wave_speed = speed + c;

                if wave_speed.is_finite() {
                    Some(wave_speed)
                } else {
                    None
                }
            })
            .reduce(
                || B::Scalar::ZERO,
                |lhs, rhs| if lhs > rhs { lhs } else { rhs },
            )
    }

    fn compute_min_char_length(&self, mesh: &PhysicsMesh) -> B::Scalar {
        let n = mesh.cell_count();
        if n == 0 {
            return B::Scalar::ZERO;
        }

        let min_val = (0..n)
            .into_par_iter()
            .filter_map(|i| {
                let area = mesh.cell_area(CellIndex(i)).unwrap_or(0.0);
                let perimeter = mesh.cell_perimeter(CellIndex(i)).unwrap_or(0.0);

                if !area.is_finite() || !perimeter.is_finite() || perimeter < 1e-14 || area <= 0.0 {
                    return None;
                }

                let dx = 2.0 * area / perimeter;
                if dx.is_finite() && dx > 0.0 {
                    Some(dx)
                } else {
                    None
                }
            })
            .reduce(|| f64::INFINITY, f64::min);

        if !min_val.is_finite() || min_val <= 0.0 {
            return B::Scalar::ZERO;
        }

        scalar_from_config_or_panic::<B::Scalar>(
            min_val,
            "CflCalculator.compute_min_char_length.min_val",
        )
    }
}

/// 时间步长控制器
#[derive(Debug)]
pub struct TimeStepController<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    calculator: CflCalculator<B>,
    current_dt: B::Scalar,
    growth_factor: B::Scalar,
    shrink_factor: B::Scalar,
    max_growth_factor: B::Scalar,
    stable_steps: usize,
    stable_growth_threshold: usize,
    adaptive_growth: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> TimeStepController<B>
where
    B::Scalar: RuntimeScalar,
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(g: B::Scalar, params: &NumericalParams<B::Scalar>) -> Self {
        Self {
            calculator: CflCalculator::new(g, params),
            current_dt: params.dt_max,
            growth_factor: scalar_from_config_or_panic::<B::Scalar>(
                1.1,
                "TimeStepController.new.growth_factor",
            ),
            shrink_factor: scalar_from_config_or_panic::<B::Scalar>(
                0.5,
                "TimeStepController.new.shrink_factor",
            ),
            max_growth_factor: scalar_from_config_or_panic::<B::Scalar>(
                1.5,
                "TimeStepController.new.max_growth_factor",
            ),
            stable_steps: 0,
            stable_growth_threshold: 10,
            adaptive_growth: true,
            marker: PhantomData,
        }
    }

    pub fn precompute_mesh_characteristics(&mut self, mesh: &PhysicsMesh) {
        self.calculator.precompute_dx_min(mesh);
    }

    pub fn dx_min(&self) -> Option<B::Scalar> {
        self.calculator.dx_min()
    }

    pub fn update(
        &mut self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        params: &NumericalParams<B::Scalar>,
    ) -> B::Scalar {
        let suggested = self.calculator.compute_dt(state, mesh, params);
        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };

        let grown = self.current_dt * growth;
        let mut new_dt = if suggested < grown {
            suggested
        } else {
            grown
        };

        if !new_dt.is_finite() || new_dt <= B::Scalar::ZERO {
            new_dt = self.calculator.dt_min;
        }

        let threshold = scalar_from_config_or_panic::<B::Scalar>(
            0.95,
            "TimeStepController.update.threshold",
        );
        if new_dt >= self.current_dt * threshold {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }

        self.current_dt = new_dt;
        self.current_dt
    }

    pub fn update_from_max_speed(&mut self, max_speed: B::Scalar) -> B::Scalar {
        let suggested = self.calculator.compute_from_max_speed(max_speed);
        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };

        let grown = self.current_dt * growth;
        let mut new_dt = if suggested < grown {
            suggested
        } else {
            grown
        };

        if !new_dt.is_finite() || new_dt <= B::Scalar::ZERO {
            new_dt = self.calculator.dt_min;
        }

        let threshold = scalar_from_config_or_panic::<B::Scalar>(
            0.95,
            "TimeStepController.update_from_max_speed.threshold",
        );
        if new_dt >= self.current_dt * threshold {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }

        self.current_dt = new_dt;
        self.current_dt
    }

    fn compute_adaptive_growth(&self) -> B::Scalar {
        if self.stable_steps >= self.stable_growth_threshold {
            self.growth_factor.min(self.max_growth_factor)
        } else if self.stable_steps >= self.stable_growth_threshold / 2 {
            self.growth_factor
        } else {
            let one = B::Scalar::ONE;
            let half = scalar_from_config_or_panic::<B::Scalar>(
                0.5,
                "TimeStepController.compute_adaptive_growth.half",
            );
            one + (self.growth_factor - one) * half
        }
    }

    pub fn shrink(&mut self) {
        self.current_dt = self.current_dt * self.shrink_factor;
        if self.current_dt < self.calculator.dt_min {
            self.current_dt = self.calculator.dt_min;
        }
        self.stable_steps = 0;
    }

    pub fn force_shrink(&mut self, factor: B::Scalar) {
        self.current_dt = self.current_dt * factor;
        if self.current_dt < self.calculator.dt_min {
            self.current_dt = self.calculator.dt_min;
        }
        self.stable_steps = 0;
    }

    pub fn current_dt(&self) -> B::Scalar {
        self.current_dt
    }

    pub fn set_dt(&mut self, dt: B::Scalar) {
        let dt_min = self.calculator.dt_min;
        let dt_max = self.calculator.dt_max;
        self.current_dt = if dt < dt_min {
            dt_min
        } else if dt > dt_max {
            dt_max
        } else {
            dt
        };
        self.stable_steps = 0;
    }

    pub fn set_growth_factor(&mut self, factor: B::Scalar) {
        let one = B::Scalar::ONE;
        self.growth_factor = if factor > one {
            factor
        } else {
            one
        };
    }

    pub fn set_shrink_factor(&mut self, factor: B::Scalar) {
        let zero = B::Scalar::ZERO;
        let one = B::Scalar::ONE;
        self.shrink_factor = if factor < zero {
            zero
        } else if factor > one {
            one
        } else {
            factor
        };
    }

    pub fn set_adaptive_growth(&mut self, enabled: bool) {
        self.adaptive_growth = enabled;
    }

    pub fn adapt_from_iterations(&mut self, iterations: usize, target_iterations: usize) -> B::Scalar {
        let ratio = scalar_from_config_or_panic::<B::Scalar>(
            iterations as f64,
            "TimeStepController.adapt_from_iterations.iterations",
        ) / scalar_from_config_or_panic::<B::Scalar>(
            target_iterations.max(1) as f64,
            "TimeStepController.adapt_from_iterations.target_iterations",
        );

        let one = B::Scalar::ONE;
        let two =
            scalar_from_config_or_panic::<B::Scalar>(2.0, "TimeStepController.adapt_from_iterations.two");

        if ratio < one / two {
            let growth = (one
                + (one - ratio * two)
                    * scalar_from_config_or_panic::<B::Scalar>(
                        0.2,
                        "TimeStepController.adapt_from_iterations.growth_gain",
                    ))
                .min(self.max_growth_factor);
            self.current_dt = self.current_dt * growth;
            self.stable_steps += 1;
        } else if ratio > one + one / two {
            let shrink = (one
                - (ratio - one - one / two)
                    * scalar_from_config_or_panic::<B::Scalar>(
                        0.3,
                        "TimeStepController.adapt_from_iterations.shrink_gain",
                    ))
            .max(scalar_from_config_or_panic::<B::Scalar>(
                0.5,
                "TimeStepController.adapt_from_iterations.shrink_floor",
            ));
            self.current_dt = self.current_dt * shrink;
            self.stable_steps = 0;
        } else if ratio > one {
            self.stable_steps = self.stable_steps.saturating_sub(1);
        }

        let dt_min = self.calculator.dt_min;
        let dt_max = self.calculator.dt_max;
        self.current_dt = self.current_dt.min(dt_max).max(dt_min);
        self.current_dt
    }

    pub fn apply_source_limits(&mut self, limits: &[Option<B::Scalar>]) -> B::Scalar {
        let mut min_dt = self.current_dt;

        for &limit in limits {
            if let Some(dt_limit) = limit {
                min_dt = min_dt.min(dt_limit);
            }
        }

        let threshold = scalar_from_config_or_panic::<B::Scalar>(
            0.9,
            "TimeStepController.apply_source_limits.threshold",
        );
        if min_dt < self.current_dt * threshold {
            self.stable_steps = 0;
        }

        let dt_min = self.calculator.dt_min;
        let dt_max = self.calculator.dt_max;
        self.current_dt = min_dt.min(dt_max).max(dt_min);
        self.current_dt
    }

    pub fn coriolis_stability_limit(&self, f: B::Scalar) -> Option<B::Scalar> {
        if f.abs()
            < scalar_from_config_or_panic::<B::Scalar>(
                1e-14,
                "TimeStepController.coriolis_stability_limit.zero_tol",
            )
        {
            None
        } else {
            let pi = scalar_from_config_or_panic::<B::Scalar>(
                std::f64::consts::PI,
                "TimeStepController.coriolis_stability_limit.pi",
            );
            Some(pi / f.abs())
        }
    }

    pub fn friction_stability_limit(&self, max_cf: B::Scalar) -> Option<B::Scalar> {
        if max_cf
            < scalar_from_config_or_panic::<B::Scalar>(
                1e-14,
                "TimeStepController.friction_stability_limit.zero_tol",
            )
        {
            None
        } else {
            Some(
                scalar_from_config_or_panic::<B::Scalar>(
                    2.0,
                    "TimeStepController.friction_stability_limit.two",
                ) / max_cf,
            )
        }
    }

    pub fn cfl(&self) -> B::Scalar {
        self.calculator.cfl
    }

    pub fn set_cfl(&mut self, cfl: B::Scalar) {
        let zero = B::Scalar::ZERO;
        let one = B::Scalar::ONE;
        self.calculator.cfl = if cfl <= zero {
            zero
        } else if cfl > one {
            one
        } else {
            cfl
        };
    }

    pub fn stats(&self) -> TimeStepStats<B> {
        TimeStepStats {
            current_dt: self.current_dt,
            dx_min: self.calculator.dx_min(),
            stable_steps: self.stable_steps,
            adaptive_growth_enabled: self.adaptive_growth,
        }
    }
}

/// 时间步统计
#[derive(Clone, Debug)]
pub struct TimeStepStats<B: Backend> {
    pub current_dt: B::Scalar,
    pub dx_min: Option<B::Scalar>,
    pub stable_steps: usize,
    pub adaptive_growth_enabled: bool,
}

/// 构建器
pub struct TimeStepControllerBuilder<B: Backend> {
    g: B::Scalar,
    cfl: B::Scalar,
    dt_min: B::Scalar,
    dt_max: B::Scalar,
    growth_factor: B::Scalar,
    shrink_factor: B::Scalar,
    adaptive_growth: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> TimeStepControllerBuilder<B>
where
    B::Scalar: RuntimeScalar,
{
    pub fn new(g: B::Scalar) -> Self {
        Self {
            g,
            cfl: scalar_from_config_or_panic::<B::Scalar>(
                0.5,
                "TimeStepControllerBuilder.new.cfl",
            ),
            dt_min: scalar_from_config_or_panic::<B::Scalar>(
                1e-6,
                "TimeStepControllerBuilder.new.dt_min",
            ),
            dt_max: scalar_from_config_or_panic::<B::Scalar>(
                1.0,
                "TimeStepControllerBuilder.new.dt_max",
            ),
            growth_factor: scalar_from_config_or_panic::<B::Scalar>(
                1.1,
                "TimeStepControllerBuilder.new.growth_factor",
            ),
            shrink_factor: scalar_from_config_or_panic::<B::Scalar>(
                0.5,
                "TimeStepControllerBuilder.new.shrink_factor",
            ),
            adaptive_growth: true,
            marker: PhantomData,
        }
    }

    pub fn with_cfl(mut self, cfl: B::Scalar) -> Self {
        self.cfl = cfl;
        self
    }

    pub fn with_dt_limits(mut self, dt_min: B::Scalar, dt_max: B::Scalar) -> Self {
        self.dt_min = dt_min;
        self.dt_max = dt_max;
        self
    }

    pub fn with_growth_factor(mut self, factor: B::Scalar) -> Self {
        self.growth_factor = factor;
        self
    }

    pub fn with_shrink_factor(mut self, factor: B::Scalar) -> Self {
        self.shrink_factor = factor;
        self
    }

    pub fn with_adaptive_growth(mut self, enabled: bool) -> Self {
        self.adaptive_growth = enabled;
        self
    }

    pub fn build(self) -> TimeStepController<B> {
        let mut params = NumericalParams::<B::Scalar>::default();
        params.cfl = self.cfl;
        params.dt_min = self.dt_min;
        params.dt_max = self.dt_max;

        let mut controller = TimeStepController::<B>::new(self.g, &params);
        controller.growth_factor = self.growth_factor;
        controller.shrink_factor = self.shrink_factor;
        controller.adaptive_growth = self.adaptive_growth;

        controller
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_cfl_calculator_f64() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.config_scalar(9.81, "timestep.test.gravity");
        let calc = CflCalculator::<CpuBackend<f64>>::new(g, &params);
        assert!(calc.cached_dx_min.is_none());
    }

    #[test]
    fn test_controller_f32() {
        let params = NumericalParams::<f32>::default();
        let backend = CpuBackend::<f32>::new();
        let g = backend.config_scalar(9.81, "timestep.test.gravity");
        let mut controller = TimeStepController::<CpuBackend<f32>>::new(g, &params);
        controller.set_dt(backend.config_scalar(0.5, "timestep.test.dt"));
        assert!((controller.current_dt().to_f64().unwrap() - 0.5).abs() < 1e-10f64);
    }

    #[test]
    fn test_coriolis_limit() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.config_scalar(9.81, "timestep.test.gravity");
        let controller = TimeStepController::<CpuBackend<f64>>::new(g, &params);

        let limit = controller.coriolis_stability_limit(backend.config_scalar(1e-4, "timestep.test.coriolis"));
        assert!(limit.is_some());
        assert!(limit.unwrap().to_f64().unwrap() > 0.0);
    }
}
