// crates/mh_physics/src/engine/time_integrator.rs

//! SSP Runge-Kutta 时间积分器
//!
//! 实现强稳定保持 Runge-Kutta 方法，支持 Backend 泛型。
//! 时间参数使用 B::Scalar，确保与状态变量精度一致。

use crate::state::{RhsBuffers, ShallowWaterState};
use crate::Backend;
use mh_foundation::error::MhResult;
use mh_runtime::RuntimeScalar;
use num_traits::{FromPrimitive, Float};

/// RHS 计算器 trait（Backend 泛型版本）
pub trait RhsComputer<B: Backend> {
    /// 计算右端项
    ///
    /// # 参数
    /// - `state`: 当前状态
    /// - `time`: 当前时间（Backend 标量类型）
    /// - `output`: 输出缓冲区
    ///
    /// # 返回
    /// 返回最大波速，用于 CFL 条件
    fn compute_rhs(
        &mut self,
        state: &ShallowWaterState<B>,
        time: B::Scalar,
        output: &mut RhsBuffers<B::Scalar>,
    ) -> MhResult<B::Scalar>;
}

/// 时间积分器 trait（Backend 泛型版本）
pub trait TimeIntegrator<B: Backend>: Send + Sync {
    /// 积分器名称
    fn name(&self) -> &'static str;

    /// 时间精度阶数
    fn order(&self) -> u8;

    /// Runge-Kutta 级数
    fn stages(&self) -> u8;

    /// 最大稳定 CFL 数
    fn max_cfl(&self) -> f64;

    /// 推进一个时间步
    ///
    /// # 参数
    /// - `state`: 要更新的状态（in-place 修改）
    /// - `time`: 当前时间（Backend 标量类型）
    /// - `dt`: 时间步长（Backend 标量类型）
    /// - `rhs_computer`: 右端项计算器
    ///
    /// # 返回
    /// 返回实际使用的最大波速
    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar>;

    /// 确保内部缓冲区大小正确
    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize);
}

/// 一阶前向欧拉（保留用于调试和对比）
pub struct ForwardEuler<B: Backend> {
    rhs: RhsBuffers<B::Scalar>,
}

impl<B: Backend> ForwardEuler<B> {
    /// 创建前向欧拉积分器
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            rhs: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
        }
    }
}

impl<B: Backend> TimeIntegrator<B> for ForwardEuler<B> {
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
        if self.rhs.n_cells() != n_cells {
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

        // U^{n+1} = U^n + dt * L(U^n)
        state.add_scaled_rhs(&self.rhs, dt);
        state.enforce_positivity();

        Ok(max_wave_speed)
    }
}

/// SSP-RK2 (二阶 Heun 方法)
pub struct SspRk2<B: Backend> {
    state_1: ShallowWaterState<B>,
    rhs_1: RhsBuffers<B::Scalar>,
    rhs_2: RhsBuffers<B::Scalar>,
}

impl<B: Backend> SspRk2<B> {
    /// 创建 SSP-RK2 积分器
    pub fn new(backend: B, n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::<B>::new_with_backend(backend, n_cells),
            rhs_1: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
            rhs_2: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
        }
    }
}

impl<B: Backend> TimeIntegrator<B> for SspRk2<B> {
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
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar> {
        let half = B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ONE);

        // Stage 1: U^(1) = U^n + dt * L(U^n)
        self.rhs_1.reset();
        let max_wave_speed_1 = rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?;

        self.state_1.copy_from(state);
        self.state_1.add_scaled_rhs(&self.rhs_1, dt);
        self.state_1.enforce_positivity();

        // Stage 2: U^{n+1} = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))
        self.rhs_2.reset();
        let time_plus_dt = time + dt;
        let max_wave_speed_2 = rhs_computer.compute_rhs(&self.state_1, time_plus_dt, &mut self.rhs_2)?;

        self.state_1.add_scaled_rhs(&self.rhs_2, dt);
        self.state_1.enforce_positivity();

        // 线性组合：U^{n+1} = 0.5 * U^n + 0.5 * U^(1)
        state.axpy(half, half, &self.state_1);
        state.enforce_positivity();

        Ok(max_wave_speed_1.max(max_wave_speed_2))
    }
}

/// SSP-RK3 (三阶) - 主推荐方案
pub struct SspRk3<B: Backend> {
    state_1: ShallowWaterState<B>,
    state_2: ShallowWaterState<B>,
    rhs_1: RhsBuffers<B::Scalar>,
    rhs_2: RhsBuffers<B::Scalar>,
    rhs_3: RhsBuffers<B::Scalar>,
}

impl<B: Backend> SspRk3<B> {
    /// 创建 SSP-RK3 积分器
    pub fn new(backend: B, n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::<B>::new_with_backend(backend.clone(), n_cells),
            state_2: ShallowWaterState::<B>::new_with_backend(backend.clone(), n_cells),
            rhs_1: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
            rhs_2: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
            rhs_3: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
        }
    }
}

impl<B: Backend> TimeIntegrator<B> for SspRk3<B> {
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
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: B::Scalar,
        dt: B::Scalar,
        rhs_computer: &mut R,
    ) -> MhResult<B::Scalar> {
        let coef_075 = B::Scalar::from_f64(0.75).unwrap_or(B::Scalar::ONE);
        let coef_025 = B::Scalar::from_f64(0.25).unwrap_or(B::Scalar::ZERO);
        let coef_one_third = B::Scalar::from_f64(1.0 / 3.0).unwrap_or(B::Scalar::ZERO);
        let coef_two_thirds = B::Scalar::from_f64(2.0 / 3.0).unwrap_or(B::Scalar::ONE);
        let coef_050 = B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ONE);

        // Stage 1: U^(1) = U^n + dt * L(U^n)
        self.rhs_1.reset();
        let max_wave_speed_1 = rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?;

        self.state_1.copy_from(state);
        self.state_1.add_scaled_rhs(&self.rhs_1, dt);
        self.state_1.enforce_positivity();

        // Stage 2: U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
        self.rhs_2.reset();
        let time_plus_dt = time + dt;
        let max_wave_speed_2 = rhs_computer.compute_rhs(&self.state_1, time_plus_dt, &mut self.rhs_2)?;

        self.state_1.add_scaled_rhs(&self.rhs_2, dt);
        self.state_1.enforce_positivity();

        self.state_2.linear_combine(coef_075, state, coef_025, &self.state_1);
        self.state_2.enforce_positivity();

        // Stage 3: U^{n+1} = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))
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

/// 时间积分器类型枚举
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeIntegratorKind {
    ForwardEuler,
    SspRk2,
    #[default]
    SspRk3,
}

impl std::fmt::Display for TimeIntegratorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForwardEuler => write!(f, "ForwardEuler"),
            Self::SspRk2 => write!(f, "SSP-RK2"),
            Self::SspRk3 => write!(f, "SSP-RK3"),
        }
    }
}

/// 工厂函数：创建时间积分器（保持 API 兼容性）
pub fn create_integrator<B: Backend + Clone>(
    kind: TimeIntegratorKind,
    backend: B,
    n_cells: usize,
    n_tracers: usize,
) -> TimeIntegratorEnum<B> {
    TimeIntegratorEnum::<B>::new(kind, backend, n_cells, n_tracers)
}

/// 时间积分器枚举包装器
pub struct TimeIntegratorEnum<B: Backend> {
    kind: TimeIntegratorKind,
    euler: Option<ForwardEuler<B>>,
    rk2: Option<SspRk2<B>>,
    rk3: Option<SspRk3<B>>,
}

impl<B: Backend> TimeIntegratorEnum<B> {
    /// 创建新的时间积分器
    pub fn new(kind: TimeIntegratorKind, backend: B, n_cells: usize, n_tracers: usize) -> Self {
        match kind {
            TimeIntegratorKind::ForwardEuler => Self {
                kind,
                euler: Some(ForwardEuler::<B>::new(n_cells, n_tracers)),
                rk2: None,
                rk3: None,
            },
            TimeIntegratorKind::SspRk2 => Self {
                kind,
                euler: None,
                rk2: Some(SspRk2::<B>::new(backend, n_cells, n_tracers)),
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

    /// 积分器名称
    pub fn name(&self) -> &'static str {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => "ForwardEuler",
            TimeIntegratorKind::SspRk2 => "SSP-RK2",
            TimeIntegratorKind::SspRk3 => "SSP-RK3",
        }
    }

    /// 推进一个时间步
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

    /// 确保内部缓冲区大小正确
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
    use mh_runtime::CpuBackend;
    use num_traits::FromPrimitive;

    struct ExponentialDecayRhs;

    impl<B: Backend> RhsComputer<B> for ExponentialDecayRhs {
        fn compute_rhs(
            &mut self,
            state: &ShallowWaterState<B>,
            _time: B::Scalar,
            output: &mut RhsBuffers<B::Scalar>,
        ) -> MhResult<B::Scalar> {
            for i in 0..state.n_cells() {
                output.dh_dt[i] = state.h[i] * B::Scalar::from_f64(-1.0).unwrap();
            }
            Ok(B::Scalar::ONE)
        }
    }

    #[test]
    fn test_forward_euler_basic() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend.clone(), 10);
        state.h.fill(1.0_f64);
        
        let mut integrator = ForwardEuler::<CpuBackend<f64>>::new(10, 0);
        let mut rhs = ExponentialDecayRhs;
        let dt = 0.01_f64;

        for _ in 0..100 {
            integrator.advance(&mut state, 0.0_f64, dt, &mut rhs).unwrap();
        }

        let expected = (-1.0_f64).exp();
        let actual = state.h[0];
        assert!((actual - expected).abs() < 0.02);
    }
}