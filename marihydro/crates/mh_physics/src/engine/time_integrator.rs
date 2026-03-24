// crates/mh_physics/src/engine/time_integrator.rs

//! SSP Runge-Kutta时间积分器
//!
//! 实现强稳定保持Runge-Kutta方法，支持Backend泛型化。
//! 时间参数使用B::Scalar，确保与状态变量精度一致。

use crate::state::{RhsBuffers, ShallowWaterState};
use crate::Backend;
use mh_foundation::{MhError, MhResult};
use num_traits::Float;

/// RHS计算器trait（Backend泛型版本）
pub trait RhsComputer<B: Backend> {
    fn compute_rhs(
        &mut self,
        state: &ShallowWaterState<B>,
        time: B::Scalar,
        output: &mut RhsBuffers<B>,
    ) -> MhResult<B::Scalar>;
}

/// 时间积分器trait（Backend泛型版本）
pub trait TimeIntegrator<B: Backend>: Send + Sync {
    fn name(&self) -> &'static str;
    fn order(&self) -> u8;
    fn stages(&self) -> u8;
    fn max_cfl(&self) -> f64;

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar>;

    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize);
}

/// 一阶前向欧拉
pub struct ForwardEuler<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    rhs: RhsBuffers<B>,
}

impl<B: Backend> ForwardEuler<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(backend: B, n_cells: usize, n_tracers: usize) -> Self {
        Self {
            rhs: RhsBuffers::with_tracers(backend, n_cells, n_tracers),
        }
    }
}

impl<B: Backend> TimeIntegrator<B> for ForwardEuler<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    fn name(&self) -> &'static str {
        "ForwardEuler"
    }

    fn order(&self) -> u8 {
        1
    }

    fn stages(&self) -> u8 {
        1
    }

    fn max_cfl(&self) -> f64 {
        0.5
    }

    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        if self.rhs.n_cells() != n_cells || self.rhs.n_tracers() != n_tracers {
            self.rhs.resize(n_cells, n_tracers);
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar> {
        self.rhs.reset();
        let max_wave_speed = rhs_computer.compute_rhs(state, time, &mut self.rhs)?;
        state.add_scaled_rhs(&self.rhs, dt);
        state.enforce_positivity();
        Ok(max_wave_speed)
    }
}

/// SSP-RK2
pub struct SspRk2<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    state_1: ShallowWaterState<B>,
    rhs_1: RhsBuffers<B>,
    rhs_2: RhsBuffers<B>,
}

impl<B: Backend> SspRk2<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(backend: B, n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::<B>::new_with_backend(backend.clone(), n_cells),
            rhs_1: RhsBuffers::with_tracers(backend.clone(), n_cells, n_tracers),
            rhs_2: RhsBuffers::with_tracers(backend, n_cells, n_tracers),
        }
    }
}

impl<B: Backend> TimeIntegrator<B> for SspRk2<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    fn name(&self) -> &'static str {
        "SSP-RK2"
    }

    fn order(&self) -> u8 {
        2
    }

    fn stages(&self) -> u8 {
        2
    }

    fn max_cfl(&self) -> f64 {
        1.0
    }

    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        if self.state_1.n_cells() != n_cells {
            let backend = self.state_1.backend().clone();
            self.state_1 = ShallowWaterState::<B>::new_with_backend(backend, n_cells);
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
        } else if self.rhs_1.n_tracers() != n_tracers || self.rhs_2.n_tracers() != n_tracers {
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar> {
        let half = state.backend().config_scalar(0.5, "SspRk2.advance.half");

        self.rhs_1.reset();
        let max_wave_speed_1 = rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?;
        self.state_1.sync_tracer_layout_from(state);
        self.state_1
            .copy_from_unchecked(state)
            .map_err(|err| MhError::invalid_input(format!("时间积分状态复制失败: {err}")))?;
        self.state_1.add_scaled_rhs(&self.rhs_1, dt);
        self.state_1.enforce_positivity();

        self.rhs_2.reset();
        let time_plus_dt = time + dt;
        let max_wave_speed_2 = rhs_computer.compute_rhs(&self.state_1, time_plus_dt, &mut self.rhs_2)?;
        self.state_1.add_scaled_rhs(&self.rhs_2, dt);
        self.state_1.enforce_positivity();

        state.axpy(half, half, &self.state_1);
        state.enforce_positivity();

        Ok(max_wave_speed_1.max(max_wave_speed_2))
    }
}

/// SSP-RK3（推荐）
pub struct SspRk3<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    state_1: ShallowWaterState<B>,
    state_2: ShallowWaterState<B>,
    rhs_1: RhsBuffers<B>,
    rhs_2: RhsBuffers<B>,
    rhs_3: RhsBuffers<B>,
}

impl<B: Backend> SspRk3<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(backend: B, n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::<B>::new_with_backend(backend.clone(), n_cells),
            state_2: ShallowWaterState::<B>::new_with_backend(backend.clone(), n_cells),
            rhs_1: RhsBuffers::with_tracers(backend.clone(), n_cells, n_tracers),
            rhs_2: RhsBuffers::with_tracers(backend.clone(), n_cells, n_tracers),
            rhs_3: RhsBuffers::with_tracers(backend, n_cells, n_tracers),
        }
    }
}

impl<B: Backend> TimeIntegrator<B> for SspRk3<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    fn name(&self) -> &'static str {
        "SSP-RK3"
    }

    fn order(&self) -> u8 {
        3
    }

    fn stages(&self) -> u8 {
        3
    }

    fn max_cfl(&self) -> f64 {
        1.0
    }

    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        if self.state_1.n_cells() != n_cells {
            let backend = self.state_1.backend().clone();
            self.state_1 = ShallowWaterState::<B>::new_with_backend(backend.clone(), n_cells);
            self.state_2 = ShallowWaterState::<B>::new_with_backend(backend, n_cells);
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
            self.rhs_3.resize(n_cells, n_tracers);
        } else if self.rhs_1.n_tracers() != n_tracers
            || self.rhs_2.n_tracers() != n_tracers
            || self.rhs_3.n_tracers() != n_tracers
        {
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
            self.rhs_3.resize(n_cells, n_tracers);
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar> {
        let coef_075 = state
            .backend()
            .config_scalar(0.75, "SspRk3.advance.coef_075");
        let coef_025 = state
            .backend()
            .config_scalar(0.25, "SspRk3.advance.coef_025");
        let coef_one_third = state
            .backend()
            .config_scalar(1.0 / 3.0, "SspRk3.advance.coef_one_third");
        let coef_two_thirds = state
            .backend()
            .config_scalar(2.0 / 3.0, "SspRk3.advance.coef_two_thirds");
        let coef_050 = state
            .backend()
            .config_scalar(0.5, "SspRk3.advance.coef_050");

        self.rhs_1.reset();
        let max_wave_speed_1 = rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?;
        self.state_1.sync_tracer_layout_from(state);
        self.state_2.sync_tracer_layout_from(state);
        self.state_1
            .copy_from_unchecked(state)
            .map_err(|err| MhError::invalid_input(format!("时间积分状态复制失败: {err}")))?;
        self.state_1.add_scaled_rhs(&self.rhs_1, dt);
        self.state_1.enforce_positivity();

        self.rhs_2.reset();
        let time_plus_dt = time + dt;
        let max_wave_speed_2 = rhs_computer.compute_rhs(&self.state_1, time_plus_dt, &mut self.rhs_2)?;
        self.state_1.add_scaled_rhs(&self.rhs_2, dt);
        self.state_1.enforce_positivity();
        self.state_2.linear_combine(coef_075, state, coef_025, &self.state_1);
        self.state_2.enforce_positivity();

        self.rhs_3.reset();
        let time_plus_half_dt = time + dt * coef_050;
        let max_wave_speed_3 = rhs_computer.compute_rhs(&self.state_2, time_plus_half_dt, &mut self.rhs_3)?;
        self.state_2.add_scaled_rhs(&self.rhs_3, dt);
        self.state_2.enforce_positivity();

        state.axpy(coef_one_third, coef_two_thirds, &self.state_2);
        state.enforce_positivity();

        Ok(max_wave_speed_1.max(max_wave_speed_2).max(max_wave_speed_3))
    }
}

/// 积分器类型枚举
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeIntegratorKind {
    ForwardEuler,
    SspRk2,
    #[default]
    SspRk3,
}

pub fn create_integrator<B: Backend + Clone>(
    kind: TimeIntegratorKind,
    backend: B,
    n_cells: usize,
    n_tracers: usize,
) -> TimeIntegratorEnum<B> {
    TimeIntegratorEnum::<B>::new(kind, backend, n_cells, n_tracers)
}

/// 积分器枚举包装
pub struct TimeIntegratorEnum<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    kind: TimeIntegratorKind,
    euler: Option<ForwardEuler<B>>,
    rk2: Option<SspRk2<B>>,
    rk3: Option<SspRk3<B>>,
}

impl<B: Backend> TimeIntegratorEnum<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(kind: TimeIntegratorKind, backend: B, n_cells: usize, n_tracers: usize) -> Self {
        match kind {
            TimeIntegratorKind::ForwardEuler => Self {
                kind,
                euler: Some(ForwardEuler::<B>::new(backend, n_cells, n_tracers)),
                rk2: None,
                rk3: None,
            },
            TimeIntegratorKind::SspRk2 => Self {
                kind,
                euler: None,
                rk2: Some(SspRk2::<B>::new(backend.clone(), n_cells, n_tracers)),
                rk3: None,
            },
            TimeIntegratorKind::SspRk3 => Self {
                kind,
                euler: None,
                rk2: None,
                rk3: Some(SspRk3::<B>::new(backend, n_cells, n_tracers)),
            },
        }
    }

    pub fn name(&self) -> &'static str {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => "ForwardEuler",
            TimeIntegratorKind::SspRk2 => "SSP-RK2",
            TimeIntegratorKind::SspRk3 => "SSP-RK3",
        }
    }

    pub fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar> {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => self
                .euler
                .as_mut()
                .unwrap()
                .advance(state, time, dt, rhs_computer),
            TimeIntegratorKind::SspRk2 => self
                .rk2
                .as_mut()
                .unwrap()
                .advance(state, time, dt, rhs_computer),
            TimeIntegratorKind::SspRk3 => self
                .rk3
                .as_mut()
                .unwrap()
                .advance(state, time, dt, rhs_computer),
        }
    }

    pub fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => {
                self.euler.as_mut().unwrap().ensure_size(n_cells, n_tracers);
            }
            TimeIntegratorKind::SspRk2 => {
                self.rk2.as_mut().unwrap().ensure_size(n_cells, n_tracers);
            }
            TimeIntegratorKind::SspRk3 => {
                self.rk3.as_mut().unwrap().ensure_size(n_cells, n_tracers);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::{CpuBackend, RuntimeScalar};

    struct TestRhs;

    impl<B: Backend> RhsComputer<B> for TestRhs {
        fn compute_rhs(
            &mut self,
            state: &ShallowWaterState<B>,
            _time: B::Scalar,
            output: &mut RhsBuffers<B>,
        ) -> MhResult<B::Scalar> {
            let n = state.n_cells();
            let decay = state
                .backend()
                .config_scalar(-0.1, "TimeIntegrator.tests.TestRhs.decay");
            for i in 0..n {
                output.dh_dt[i] = state.h[i] * decay;
            }
            Ok(B::Scalar::ONE)
        }
    }

    #[test]
    fn test_forward_euler_f64() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend.clone(), 10);
        state.h.fill(1.0);

        let mut integrator = ForwardEuler::<CpuBackend<f64>>::new(backend, 10, 0);
        let mut rhs = TestRhs;
        let dt = 0.1;

        integrator.advance(&mut state, 0.0, dt, &mut rhs).unwrap();
        assert!((state.h[0] - 0.99).abs() < 1e-10);
    }

    #[test]
    fn test_ssp_rk3_f32() {
        let backend = CpuBackend::<f32>::new();
        let mut state = ShallowWaterState::<CpuBackend<f32>>::new_with_backend(backend.clone(), 5);
        state.h.fill(2.0f32);

        let mut integrator = SspRk3::<CpuBackend<f32>>::new(backend, 5, 0);
        let mut rhs = TestRhs;
        let dt = 0.05f32;

        integrator.advance(&mut state, 0.0, dt, &mut rhs).unwrap();
        assert!(state.h[0] < 2.0f32);
    }
}
