// crates/mh_physics/src/engine/time_integrator.rs

//! SSP Runge-Kutta 时间积分器
//!
//! 实现强稳定保持 (Strong Stability Preserving) Runge-Kutta 方法，
//! 用于浅水方程的时间推进。
//!
//! ## 理论背景
//!
//! SSP-RK 方法是经典 Runge-Kutta 方法的变体，专门设计用于与空间离散化
//! 中的 TVD (Total Variation Diminishing) 性质兼容。
//!
//! ### Shu-Osher 形式
//!
//! SSP-RK 方法可以写成 Shu-Osher 形式的凸组合：
//!
//! $$ U^{(k)} = \sum_{i=0}^{k-1} \left[ \alpha_{ki} U^{(i)} + \Delta t \, \beta_{ki} L(U^{(i)}) \right] $$
//!
//! 其中 $L(U)$ 是空间算子（RHS），$\alpha_{ki} \geq 0$，$\sum_i \alpha_{ki} = 1$。
//!
//! ### SSP-RK2 (Heun 方法)
//!
//! ```text
//! U* = U^n + \Delta t L(U^n)
//! U^{n+1} = 0.5 U^n + 0.5 (U* + \Delta t L(U*))
//! ```
//!
//! - SSP 系数: $c = 1.0$
//! - CFL 条件: $\Delta t \leq c \cdot \Delta t_{FE}$
//!
//! ### SSP-RK3 (Shu-Osher)
//!
//! ```text
//! U^{(1)} = U^n + \Delta t L(U^n)
//! U^{(2)} = 3/4 U^n + 1/4 (U^{(1)} + \Delta t L(U^{(1)}))
//! U^{n+1} = 1/3 U^n + 2/3 (U^{(2)} + \Delta t L(U^{(2)}))
//! ```
//!
//! - SSP 系数: $c = 1.0$
//! - 三阶精度，最优 SSP 系数
//!
//! ## 参考文献
//!
//! 1. Gottlieb, S., Shu, C.-W., & Tadmor, E. (2001). Strong stability-preserving
//!    high-order time discretization methods. SIAM Review, 43(1), 89-112.
//!
//! 2. Shu, C.-W., & Osher, S. (1988). Efficient implementation of essentially
//!    non-oscillatory shock-capturing schemes. Journal of Computational Physics,
//!    77(2), 439-471.

use crate::state::{RhsBuffers, ShallowWaterState};
use crate::Backend;
use mh_foundation::error::MhResult;
use num_traits::FromPrimitive;

/// RHS 计算器 trait (泛型版本)
///
/// 实现此 trait 的类型可以计算右端项 dU/dt = L(U)
pub trait RhsComputer<B: Backend> {
    /// 计算右端项
    ///
    /// # 参数
    /// - `state`: 当前状态
    /// - `time`: 当前时间
    /// - `output`: 输出缓冲区
    ///
    /// # 返回
    /// 返回最大波速，用于 CFL 条件
    fn compute_rhs(
        &mut self,
        state: &ShallowWaterState<B>,
        time: f64, // ALLOW_F64: 时间步长参数
        output: &mut RhsBuffers<B::Scalar>,
    ) -> MhResult<f64>;
}

/// 时间积分器 trait (泛型版本)
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
    /// - `time`: 当前时间
    /// - `dt`: 时间步长
    /// - `rhs_computer`: 右端项计算器
    ///
    /// # 返回
    /// 返回实际使用的最大波速
    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: f64, // ALLOW_F64: 时间步长参数
        dt: f64,   // ALLOW_F64: 时间步长参数
        rhs_computer: &mut R,
    ) -> MhResult<f64>;

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
        time: f64, // ALLOW_F64: 时间步长参数
        dt: f64,   // ALLOW_F64: 时间步长参数
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
        self.rhs.reset();
        let max_wave_speed = rhs_computer.compute_rhs(state, time, &mut self.rhs)?;

        // 将 f64 转换为 B::Scalar
        let dt_scalar = B::Scalar::from_f64(dt).unwrap();

        // U^{n+1} = U^n + dt * L(U^n)
        state.add_scaled_rhs(&self.rhs, dt_scalar);
        state.enforce_positivity();

        Ok(max_wave_speed)
    }
}

/// SSP-RK2 (二阶 Heun 方法)
///
/// ```text
/// U^(1) = U^n + dt * L(U^n)
/// U^{n+1} = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))
/// ```
pub struct SspRk2<B: Backend> {
    state_1: ShallowWaterState<B>,
    rhs_1: RhsBuffers<B::Scalar>,
    rhs_2: RhsBuffers<B::Scalar>,
}

impl<B: Backend + Default> SspRk2<B> {
    /// 创建 SSP-RK2 积分器
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::<B>::new_with_backend(B::default(), n_cells),
            rhs_1: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
            rhs_2: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
        }
    }
}

impl<B: Backend + Default> TimeIntegrator<B> for SspRk2<B> {
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
            self.state_1 = ShallowWaterState::<B>::new_with_backend(B::default(), n_cells);
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: f64, // ALLOW_F64: 时间步长参数
        dt: f64,   // ALLOW_F64: 时间步长参数
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
        // 将 f64 转换为 B::Scalar
        let dt_scalar = B::Scalar::from_f64(dt).unwrap();
        let half = B::Scalar::from_f64(0.5).unwrap();

        // Stage 1: U^(1) = U^n + dt * L(U^n)
        self.rhs_1.reset();
        let max_wave_speed_1 = rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?;

        self.state_1.copy_from(state);
        self.state_1.add_scaled_rhs(&self.rhs_1, dt_scalar);
        self.state_1.enforce_positivity();

        // Stage 2: U^{n+1} = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))
        self.rhs_2.reset();
        let max_wave_speed_2 = rhs_computer.compute_rhs(&self.state_1, time + dt, &mut self.rhs_2)?;

        self.state_1.add_scaled_rhs(&self.rhs_2, dt_scalar);
        self.state_1.enforce_positivity();

        // 线性组合：U^{n+1} = 0.5 * U^n + 0.5 * U^(1)
        state.axpy(half, half, &self.state_1);
        state.enforce_positivity();

        Ok(max_wave_speed_1.max(max_wave_speed_2))
    }
}

/// SSP-RK3 (三阶) - 主推荐方案
///
/// Shu-Osher 形式：
/// ```text
/// U^(1) = U^n + dt * L(U^n)
/// U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
/// U^{n+1} = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))
/// ```
pub struct SspRk3<B: Backend> {
    state_1: ShallowWaterState<B>,
    state_2: ShallowWaterState<B>,
    rhs_1: RhsBuffers<B::Scalar>,
    rhs_2: RhsBuffers<B::Scalar>,
    rhs_3: RhsBuffers<B::Scalar>,
}

impl<B: Backend + Default> SspRk3<B> {
    /// 创建 SSP-RK3 积分器
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::<B>::new_with_backend(B::default(), n_cells),
            state_2: ShallowWaterState::<B>::new_with_backend(B::default(), n_cells),
            rhs_1: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
            rhs_2: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
            rhs_3: RhsBuffers::<B::Scalar>::with_tracers(n_cells, n_tracers),
        }
    }
}

impl<B: Backend + Default> TimeIntegrator<B> for SspRk3<B> {
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
            self.state_1 = ShallowWaterState::<B>::new_with_backend(B::default(), n_cells);
            self.state_2 = ShallowWaterState::<B>::new_with_backend(B::default(), n_cells);
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
            self.rhs_3.resize(n_cells, n_tracers);
        }
    }

    fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: f64, // ALLOW_F64: 时间步长参数
        dt: f64,   // ALLOW_F64: 时间步长参数
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
        let mut max_wave_speed = 0.0f64;

        // 将 f64 转换为 B::Scalar
        let dt_scalar = B::Scalar::from_f64(dt).unwrap();
        let coef_075 = B::Scalar::from_f64(0.75).unwrap();
        let coef_025 = B::Scalar::from_f64(0.25).unwrap();
        let coef_one_third = B::Scalar::from_f64(1.0 / 3.0).unwrap();
        let coef_two_thirds = B::Scalar::from_f64(2.0 / 3.0).unwrap();

        // Stage 1: U^(1) = U^n + dt * L(U^n)
        self.rhs_1.reset();
        max_wave_speed = max_wave_speed.max(rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?);

        self.state_1.copy_from(state);
        self.state_1.add_scaled_rhs(&self.rhs_1, dt_scalar);
        self.state_1.enforce_positivity();

        // Stage 2: U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
        self.rhs_2.reset();
        max_wave_speed = max_wave_speed.max(
            rhs_computer.compute_rhs(&self.state_1, time + dt, &mut self.rhs_2)?,
        );

        self.state_1.add_scaled_rhs(&self.rhs_2, dt_scalar);
        self.state_1.enforce_positivity();

        self.state_2.linear_combine(coef_075, state, coef_025, &self.state_1);
        self.state_2.enforce_positivity();

        // Stage 3: U^{n+1} = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))
        self.rhs_3.reset();
        max_wave_speed = max_wave_speed.max(
            rhs_computer.compute_rhs(&self.state_2, time + 0.5 * dt, &mut self.rhs_3)?,
        );

        self.state_2.add_scaled_rhs(&self.rhs_3, dt_scalar);
        self.state_2.enforce_positivity();

        state.axpy(coef_one_third, coef_two_thirds, &self.state_2);
        state.enforce_positivity();

        Ok(max_wave_speed)
    }
}

/// 时间积分器类型枚举
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TimeIntegratorKind {
    /// 一阶前向欧拉
    ForwardEuler,
    /// 二阶 SSP-RK
    SspRk2,
    /// 三阶 SSP-RK (默认推荐)
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

/// 创建时间积分器（返回具体类型）
pub fn create_integrator<B: Backend + Default>(
    kind: TimeIntegratorKind,
    n_cells: usize,
    n_tracers: usize,
) -> TimeIntegratorEnum<B> {
    TimeIntegratorEnum::<B>::new(kind, n_cells, n_tracers)
}

/// 时间积分器枚举包装器 - 替代 Box<dyn TimeIntegrator>
///
/// 使用枚举分发避免 E0038 (trait不是dyn兼容) 问题
pub struct TimeIntegratorEnum<B: Backend> {
    kind: TimeIntegratorKind,
    euler: Option<ForwardEuler<B>>,
    rk2: Option<SspRk2<B>>,
    rk3: Option<SspRk3<B>>,
}

impl<B: Backend + Default> TimeIntegratorEnum<B> {
    /// 创建新的时间积分器
    pub fn new(kind: TimeIntegratorKind, n_cells: usize, n_tracers: usize) -> Self {
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
                rk2: Some(SspRk2::<B>::new(n_cells, n_tracers)),
                rk3: None,
            },
            TimeIntegratorKind::SspRk3 => Self {
                kind,
                euler: None,
                rk2: None,
                rk3: Some(SspRk3::<B>::new(n_cells, n_tracers)),
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

    /// 时间精度阶数
    pub fn order(&self) -> u8 {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => 1,
            TimeIntegratorKind::SspRk2 => 2,
            TimeIntegratorKind::SspRk3 => 3,
        }
    }

    /// Runge-Kutta 级数
    pub fn stages(&self) -> u8 {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => 1,
            TimeIntegratorKind::SspRk2 => 2,
            TimeIntegratorKind::SspRk3 => 3,
        }
    }

    /// 最大稳定 CFL 数
    pub fn max_cfl(&self) -> f64 {
        match self.kind {
            TimeIntegratorKind::ForwardEuler => 0.5,
            TimeIntegratorKind::SspRk2 => 1.0,
            TimeIntegratorKind::SspRk3 => 1.0,
        }
    }

    /// 推进一个时间步
    pub fn advance<R: RhsComputer<B>>(
        &mut self,
        state: &mut ShallowWaterState<B>,
        time: f64, // ALLOW_F64: 时间步长参数
        dt: f64,   // ALLOW_F64: 时间步长参数
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
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

    /// 获取积分器类型
    pub fn kind(&self) -> TimeIntegratorKind {
        self.kind
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    use num_traits::FromPrimitive;

    /// 简单 ODE: du/dt = -u, 精确解 u(t) = u0 * exp(-t)
    struct ExponentialDecayRhs;

    impl<B: Backend> RhsComputer<B> for ExponentialDecayRhs {
        fn compute_rhs(
            &mut self,
            state: &ShallowWaterState<B>,
            _time: f64,
            output: &mut RhsBuffers<B::Scalar>,
        ) -> MhResult<f64> {
            for i in 0..state.n_cells() {
                output.dh_dt[i] = state.h[i] * B::Scalar::from_f64(-1.0).unwrap();
            }
            Ok(1.0)
        }
    }

    #[test]
    fn test_forward_euler_basic() {
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(CpuBackend::<f64>::new(), 10);
        state.h.fill(1.0_f64);
        
        let mut integrator = ForwardEuler::<CpuBackend<f64>>::new(10, 0);
        let mut rhs = ExponentialDecayRhs;

        let dt = 0.01;
        for _ in 0..100 {
            integrator.advance(&mut state, 0.0, dt, &mut rhs).unwrap();
        }

        // 约等于 exp(-1) ≈ 0.368
        let expected = (-1.0f64).exp();
        let actual = state.h[0];
        assert!(
            (actual - expected).abs() < 0.02,
            "Expected ~{}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_ssp_rk2_basic() {
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(CpuBackend::<f64>::new(), 10);
        state.h.fill(1.0_f64);
        
        let mut integrator = SspRk2::<CpuBackend<f64>>::new(10, 0);
        let mut rhs = ExponentialDecayRhs;

        let dt = 0.01;
        for _ in 0..100 {
            integrator.advance(&mut state, 0.0, dt, &mut rhs).unwrap();
        }

        let expected = (-1.0f64).exp();
        let actual = state.h[0];
        // RK2应该比Euler更精确
        assert!(
            (actual - expected).abs() < 0.01,
            "Expected ~{}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_ssp_rk3_convergence_order() {
        // 验证收敛阶
        let n = 1;
        let t_final: f64 = 1.0;
        let exact = (-t_final).exp();

        let mut errors = Vec::new();
        let dts = [0.1, 0.05, 0.025, 0.0125];

        for &dt in &dts {
            let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(CpuBackend::<f64>::new(), n);
            state.h[0] = 1.0_f64;

            let mut integrator = SspRk3::<CpuBackend<f64>>::new(n, 0);
            let mut rhs = ExponentialDecayRhs;

            let steps = (t_final / dt) as usize;
            for _ in 0..steps {
                integrator.advance(&mut state, 0.0, dt, &mut rhs).unwrap();
            }

            errors.push((state.h[0] - exact).abs());
        }

        // 检查收敛阶：error ratio 应该接近 2^order = 8
        if errors.len() >= 2 {
            let ratio = errors[0] / errors[1];
            // SSP-RK3 应该有约 8 的收敛比
            assert!(ratio > 6.0, "Expected ratio ~8, got {}", ratio);
        }
    }

    #[test]
    fn test_integrator_creation() {
        let integrator = create_integrator::<CpuBackend<f64>>(TimeIntegratorKind::SspRk3, 100, 2);
        assert_eq!(integrator.name(), "SSP-RK3");
        assert_eq!(integrator.order(), 3);
        assert_eq!(integrator.stages(), 3);
    }

    #[test]
    fn test_integrator_enum_all_types() {
        let euler = create_integrator::<CpuBackend<f64>>(TimeIntegratorKind::ForwardEuler, 10, 0);
        assert_eq!(euler.name(), "ForwardEuler");
        assert_eq!(euler.order(), 1);

        let rk2 = create_integrator::<CpuBackend<f64>>(TimeIntegratorKind::SspRk2, 10, 0);
        assert_eq!(rk2.name(), "SSP-RK2");
        assert_eq!(rk2.order(), 2);

        let rk3 = create_integrator::<CpuBackend<f64>>(TimeIntegratorKind::SspRk3, 10, 0);
        assert_eq!(rk3.name(), "SSP-RK3");
        assert_eq!(rk3.order(), 3);
    }

    #[test]
    fn test_time_integrator_kind_display() {
        assert_eq!(format!("{}", TimeIntegratorKind::ForwardEuler), "ForwardEuler");
        assert_eq!(format!("{}", TimeIntegratorKind::SspRk2), "SSP-RK2");
        assert_eq!(format!("{}", TimeIntegratorKind::SspRk3), "SSP-RK3");
    }

    #[test]
    fn test_time_integrator_kind_default() {
        let default = TimeIntegratorKind::default();
        assert_eq!(default, TimeIntegratorKind::SspRk3);
    }
}