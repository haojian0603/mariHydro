// File: src-tauri/src/marihydro/physics/engine/time_integrator.rs
//! SSP Runge-Kutta 时间积分器
//! 
//! 实现强稳定保持 (Strong Stability Preserving) Runge-Kutta 方法，
//! 用于浅水方程的时间推进。

use crate::marihydro::core::error::MhResult;
use crate::marihydro::domain::state::ShallowWaterState;
use rayon::prelude::*;

/// RHS 缓冲区 - 存储右端项 dU/dt
#[derive(Clone)]
pub struct RhsBuffers {
    /// 水深变化率
    pub dh_dt: Vec<f64>,
    /// x动量变化率
    pub dhu_dt: Vec<f64>,
    /// y动量变化率
    pub dhv_dt: Vec<f64>,
    /// 标量示踪剂变化率
    pub tracer_rhs: Vec<Vec<f64>>,
}

impl RhsBuffers {
    /// 创建新的 RHS 缓冲区
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            dh_dt: vec![0.0; n_cells],
            dhu_dt: vec![0.0; n_cells],
            dhv_dt: vec![0.0; n_cells],
            tracer_rhs: (0..n_tracers).map(|_| vec![0.0; n_cells]).collect(),
        }
    }

    /// 重置所有缓冲区为零
    pub fn reset(&mut self) {
        self.dh_dt.fill(0.0);
        self.dhu_dt.fill(0.0);
        self.dhv_dt.fill(0.0);
        for tracer in &mut self.tracer_rhs {
            tracer.fill(0.0);
        }
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize, n_tracers: usize) {
        self.dh_dt.resize(n_cells, 0.0);
        self.dhu_dt.resize(n_cells, 0.0);
        self.dhv_dt.resize(n_cells, 0.0);
        self.tracer_rhs.resize_with(n_tracers, || vec![0.0; n_cells]);
        for tracer in &mut self.tracer_rhs {
            tracer.resize(n_cells, 0.0);
        }
    }

    /// 获取单元数
    pub fn n_cells(&self) -> usize {
        self.dh_dt.len()
    }
}

/// RHS 计算器 trait
/// 
/// 实现此 trait 的类型可以计算右端项 dU/dt = L(U)
pub trait RhsComputer {
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
        state: &ShallowWaterState,
        time: f64,
        output: &mut RhsBuffers,
    ) -> MhResult<f64>;
}

/// 时间积分器 trait
pub trait TimeIntegrator: Send + Sync {
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
    fn advance<R: RhsComputer>(
        &mut self,
        state: &mut ShallowWaterState,
        time: f64,
        dt: f64,
        rhs_computer: &mut R,
    ) -> MhResult<f64>;
    
    /// 确保内部缓冲区大小正确
    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize);
}

/// 一阶前向欧拉（保留用于调试和对比）
pub struct ForwardEuler {
    rhs: RhsBuffers,
}

impl ForwardEuler {
    /// 创建前向欧拉积分器
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            rhs: RhsBuffers::new(n_cells, n_tracers),
        }
    }
}

impl TimeIntegrator for ForwardEuler {
    fn name(&self) -> &'static str { "ForwardEuler" }
    fn order(&self) -> u8 { 1 }
    fn stages(&self) -> u8 { 1 }
    fn max_cfl(&self) -> f64 { 0.5 }
    
    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        if self.rhs.n_cells() != n_cells {
            self.rhs.resize(n_cells, n_tracers);
        }
    }
    
    fn advance<R: RhsComputer>(
        &mut self,
        state: &mut ShallowWaterState,
        time: f64,
        dt: f64,
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
        self.rhs.reset();
        let max_wave_speed = rhs_computer.compute_rhs(state, time, &mut self.rhs)?;
        
        // U^{n+1} = U^n + dt * L(U^n)
        state.add_scaled_rhs(&self.rhs, dt);
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
pub struct SspRk2 {
    state_1: ShallowWaterState,
    rhs_1: RhsBuffers,
    rhs_2: RhsBuffers,
}

impl SspRk2 {
    /// 创建 SSP-RK2 积分器
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::new(n_cells),
            rhs_1: RhsBuffers::new(n_cells, n_tracers),
            rhs_2: RhsBuffers::new(n_cells, n_tracers),
        }
    }
}

impl TimeIntegrator for SspRk2 {
    fn name(&self) -> &'static str { "SSP-RK2" }
    fn order(&self) -> u8 { 2 }
    fn stages(&self) -> u8 { 2 }
    fn max_cfl(&self) -> f64 { 1.0 }
    
    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        if self.state_1.n_cells() != n_cells {
            self.state_1 = ShallowWaterState::new(n_cells);
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
        }
    }
    
    fn advance<R: RhsComputer>(
        &mut self,
        state: &mut ShallowWaterState,
        time: f64,
        dt: f64,
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
        // Stage 1: U^(1) = U^n + dt * L(U^n)
        self.rhs_1.reset();
        let max_wave_speed_1 = rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?;
        
        self.state_1.copy_from(state);
        self.state_1.add_scaled_rhs(&self.rhs_1, dt);
        self.state_1.enforce_positivity();
        
        // Stage 2: U^{n+1} = 0.5 * U^n + 0.5 * (U^(1) + dt * L(U^(1)))
        self.rhs_2.reset();
        let max_wave_speed_2 = rhs_computer.compute_rhs(&self.state_1, time + dt, &mut self.rhs_2)?;
        
        self.state_1.add_scaled_rhs(&self.rhs_2, dt);
        self.state_1.enforce_positivity();
        
        // 线性组合：U^{n+1} = 0.5 * U^n + 0.5 * U^(1)
        state.linear_combine(0.5, state, 0.5, &self.state_1);
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
pub struct SspRk3 {
    state_1: ShallowWaterState,
    state_2: ShallowWaterState,
    rhs_1: RhsBuffers,
    rhs_2: RhsBuffers,
    rhs_3: RhsBuffers,
}

impl SspRk3 {
    /// 创建 SSP-RK3 积分器
    pub fn new(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            state_1: ShallowWaterState::new(n_cells),
            state_2: ShallowWaterState::new(n_cells),
            rhs_1: RhsBuffers::new(n_cells, n_tracers),
            rhs_2: RhsBuffers::new(n_cells, n_tracers),
            rhs_3: RhsBuffers::new(n_cells, n_tracers),
        }
    }
}

impl TimeIntegrator for SspRk3 {
    fn name(&self) -> &'static str { "SSP-RK3" }
    fn order(&self) -> u8 { 3 }
    fn stages(&self) -> u8 { 3 }
    fn max_cfl(&self) -> f64 { 1.0 }
    
    fn ensure_size(&mut self, n_cells: usize, n_tracers: usize) {
        if self.state_1.n_cells() != n_cells {
            self.state_1 = ShallowWaterState::new(n_cells);
            self.state_2 = ShallowWaterState::new(n_cells);
            self.rhs_1.resize(n_cells, n_tracers);
            self.rhs_2.resize(n_cells, n_tracers);
            self.rhs_3.resize(n_cells, n_tracers);
        }
    }
    
    fn advance<R: RhsComputer>(
        &mut self,
        state: &mut ShallowWaterState,
        time: f64,
        dt: f64,
        rhs_computer: &mut R,
    ) -> MhResult<f64> {
        let mut max_wave_speed = 0.0f64;
        
        // Stage 1: U^(1) = U^n + dt * L(U^n)
        self.rhs_1.reset();
        max_wave_speed = max_wave_speed.max(rhs_computer.compute_rhs(state, time, &mut self.rhs_1)?);
        
        self.state_1.copy_from(state);
        self.state_1.add_scaled_rhs(&self.rhs_1, dt);
        self.state_1.enforce_positivity();
        
        // Stage 2: U^(2) = 3/4 * U^n + 1/4 * (U^(1) + dt * L(U^(1)))
        self.rhs_2.reset();
        max_wave_speed = max_wave_speed.max(
            rhs_computer.compute_rhs(&self.state_1, time + dt, &mut self.rhs_2)?
        );
        
        self.state_1.add_scaled_rhs(&self.rhs_2, dt);
        self.state_1.enforce_positivity();
        
        // state_2 = 3/4 * state + 1/4 * state_1
        self.state_2.linear_combine(0.75, state, 0.25, &self.state_1);
        self.state_2.enforce_positivity();
        
        // Stage 3: U^{n+1} = 1/3 * U^n + 2/3 * (U^(2) + dt * L(U^(2)))
        self.rhs_3.reset();
        max_wave_speed = max_wave_speed.max(
            rhs_computer.compute_rhs(&self.state_2, time + 0.5 * dt, &mut self.rhs_3)?
        );
        
        self.state_2.add_scaled_rhs(&self.rhs_3, dt);
        self.state_2.enforce_positivity();
        
        // state = 1/3 * state + 2/3 * state_2
        state.linear_combine(1.0 / 3.0, state, 2.0 / 3.0, &self.state_2);
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

/// 创建时间积分器
pub fn create_integrator(
    kind: TimeIntegratorKind,
    n_cells: usize,
    n_tracers: usize,
) -> Box<dyn TimeIntegrator> {
    match kind {
        TimeIntegratorKind::ForwardEuler => Box::new(ForwardEuler::new(n_cells, n_tracers)),
        TimeIntegratorKind::SspRk2 => Box::new(SspRk2::new(n_cells, n_tracers)),
        TimeIntegratorKind::SspRk3 => Box::new(SspRk3::new(n_cells, n_tracers)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 简单 ODE: du/dt = -u, 精确解 u(t) = u0 * exp(-t)
    struct ExponentialDecayRhs;

    impl RhsComputer for ExponentialDecayRhs {
        fn compute_rhs(
            &mut self,
            state: &ShallowWaterState,
            _time: f64,
            output: &mut RhsBuffers,
        ) -> MhResult<f64> {
            for i in 0..state.n_cells() {
                output.dh_dt[i] = -state.h[i];
            }
            Ok(1.0)
        }
    }

    #[test]
    fn test_forward_euler_basic() {
        let n = 10;
        let mut state = ShallowWaterState::new(n);
        state.h.fill(1.0);
        
        let mut integrator = ForwardEuler::new(n, 0);
        let mut rhs = ExponentialDecayRhs;
        
        let dt = 0.01;
        for _ in 0..100 {
            integrator.advance(&mut state, 0.0, dt, &mut rhs).unwrap();
        }
        
        // 约等于 exp(-1) ≈ 0.368
        let expected = (-1.0f64).exp();
        let actual = state.h[0];
        assert!((actual - expected).abs() < 0.02, "Expected ~{}, got {}", expected, actual);
    }

    #[test]
    fn test_ssp_rk3_convergence_order() {
        // 验证收敛阶
        let n = 1;
        let t_final = 1.0;
        let exact = (-t_final).exp();
        
        let mut errors = Vec::new();
        let dts = [0.1, 0.05, 0.025, 0.0125];
        
        for &dt in &dts {
            let mut state = ShallowWaterState::new(n);
            state.h[0] = 1.0;
            
            let mut integrator = SspRk3::new(n, 0);
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
        let integrator = create_integrator(TimeIntegratorKind::SspRk3, 100, 2);
        assert_eq!(integrator.name(), "SSP-RK3");
        assert_eq!(integrator.order(), 3);
        assert_eq!(integrator.stages(), 3);
    }
}
