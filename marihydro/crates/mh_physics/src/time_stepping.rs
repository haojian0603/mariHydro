// crates/mh_physics/src/time_stepping.rs

//! 高精度时间步进控制
//!
//! 提供工业级时间积分方案，支持：
//! - 多种 Runge-Kutta 方法 (RK2-TVD, RK3-SSP, RK4)
//! - 自适应时间步控制
//! - CFL 自动调节
//! - 误差估计与控制
//!
//! # 设计原则
//!
//! 1. **TVD 保持**：确保总变差不增
//! 2. **SSP 性质**：强稳定性保持
//! 3. **自适应步长**：基于误差估计
//! 4. **工业精度**：比肩 Delft3D/MIKE


// ============================================================================
// 时间步进方法
// ============================================================================

/// 时间积分方法
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeIntegrationMethod {
    /// 一阶欧拉（显式）
    Euler,
    /// 二阶 TVD Runge-Kutta (Heun)
    Rk2Tvd,
    /// 三阶 SSP Runge-Kutta (Shu-Osher)
    Rk3Ssp,
    /// 经典四阶 Runge-Kutta
    Rk4,
    /// Adams-Bashforth 二阶
    Ab2,
    /// 隐式 Crank-Nicolson
    CrankNicolson,
}

impl TimeIntegrationMethod {
    /// 获取阶数
    pub fn order(&self) -> usize {
        match self {
            Self::Euler => 1,
            Self::Rk2Tvd => 2,
            Self::Rk3Ssp => 3,
            Self::Rk4 => 4,
            Self::Ab2 => 2,
            Self::CrankNicolson => 2,
        }
    }

    /// 是否需要多步存储
    pub fn is_multistep(&self) -> bool {
        matches!(self, Self::Ab2)
    }

    /// 每步需要的右端项计算次数
    pub fn stages(&self) -> usize {
        match self {
            Self::Euler => 1,
            Self::Rk2Tvd => 2,
            Self::Rk3Ssp => 3,
            Self::Rk4 => 4,
            Self::Ab2 => 1,
            Self::CrankNicolson => 2, // 迭代
        }
    }

    /// 稳定性系数（最大允许 CFL）
    pub fn stability_limit(&self) -> f64 {
        match self {
            Self::Euler => 1.0,
            Self::Rk2Tvd => 1.0,
            Self::Rk3Ssp => 1.0,
            Self::Rk4 => 2.785, // 经验值
            Self::Ab2 => 0.5,   // 多步方法较低
            Self::CrankNicolson => f64::INFINITY, // 无条件稳定
        }
    }
}

impl Default for TimeIntegrationMethod {
    fn default() -> Self {
        Self::Rk3Ssp // 推荐默认方法
    }
}

// ============================================================================
// 时间步控制器配置
// ============================================================================

/// 时间步控制配置
#[derive(Debug, Clone)]
pub struct TimeStepConfig {
    /// 积分方法
    pub method: TimeIntegrationMethod,
    /// 初始时间步 (s)
    pub initial_dt: f64,
    /// 最小时间步 (s)
    pub min_dt: f64,
    /// 最大时间步 (s)
    pub max_dt: f64,
    /// 目标 CFL 数
    pub target_cfl: f64,
    /// 最大允许 CFL 数
    pub max_cfl: f64,
    /// 安全系数
    pub safety_factor: f64,
    /// 最大步长增长因子
    pub max_growth: f64,
    /// 最大步长收缩因子
    pub max_shrink: f64,
    /// 自适应步长控制
    pub adaptive: bool,
    /// 误差容限（自适应模式）
    pub error_tolerance: f64,
    /// 拒绝步长时的收缩因子
    pub rejection_shrink: f64,
}

impl Default for TimeStepConfig {
    fn default() -> Self {
        Self {
            method: TimeIntegrationMethod::Rk3Ssp,
            initial_dt: 0.1,
            min_dt: 1e-6,
            max_dt: 60.0,
            target_cfl: 0.5,
            max_cfl: 0.9,
            safety_factor: 0.9,
            max_growth: 1.5,
            max_shrink: 0.5,
            adaptive: true,
            error_tolerance: 1e-4,
            rejection_shrink: 0.5,
        }
    }
}

impl TimeStepConfig {
    /// 显式浅水方程配置
    pub fn shallow_water() -> Self {
        Self {
            target_cfl: 0.5,
            max_cfl: 0.9,
            ..Default::default()
        }
    }

    /// 高精度配置
    pub fn high_accuracy() -> Self {
        Self {
            method: TimeIntegrationMethod::Rk4,
            target_cfl: 0.3,
            max_cfl: 0.5,
            error_tolerance: 1e-6,
            ..Default::default()
        }
    }

    /// 快速配置（牺牲精度换速度）
    pub fn fast() -> Self {
        Self {
            method: TimeIntegrationMethod::Rk2Tvd,
            target_cfl: 0.8,
            max_cfl: 1.0,
            error_tolerance: 1e-3,
            ..Default::default()
        }
    }

    /// 归一化配置，修正非法或不一致的参数
    pub fn normalized(&self) -> Self {
        let mut cfg = self.clone();

        if !cfg.min_dt.is_finite() || cfg.min_dt <= 0.0 {
            cfg.min_dt = 1e-6;
        }
        if !cfg.max_dt.is_finite() || cfg.max_dt < cfg.min_dt {
            cfg.max_dt = cfg.min_dt;
        }
        if !cfg.initial_dt.is_finite() {
            cfg.initial_dt = cfg.min_dt;
        }
        cfg.initial_dt = cfg.initial_dt.clamp(cfg.min_dt, cfg.max_dt);

        if !cfg.target_cfl.is_finite() || cfg.target_cfl <= 0.0 {
            cfg.target_cfl = 0.5;
        }
        if !cfg.max_cfl.is_finite() || cfg.max_cfl < cfg.target_cfl {
            cfg.max_cfl = cfg.target_cfl;
        }

        if !cfg.safety_factor.is_finite() || cfg.safety_factor <= 0.0 {
            cfg.safety_factor = 0.9;
        }
        if cfg.safety_factor > 1.0 {
            cfg.safety_factor = 1.0;
        }

        if !cfg.max_growth.is_finite() || cfg.max_growth < 1.0 {
            cfg.max_growth = 1.0;
        }
        if !cfg.max_shrink.is_finite() || cfg.max_shrink <= 0.0 {
            cfg.max_shrink = 0.5;
        }
        if cfg.max_shrink > 1.0 {
            cfg.max_shrink = 1.0;
        }

        if !cfg.error_tolerance.is_finite() || cfg.error_tolerance <= 0.0 {
            cfg.error_tolerance = 1e-4;
        }
        if !cfg.rejection_shrink.is_finite() || cfg.rejection_shrink <= 0.0 {
            cfg.rejection_shrink = 0.5;
        }
        if cfg.rejection_shrink > 1.0 {
            cfg.rejection_shrink = 1.0;
        }

        cfg
    }
}

// ============================================================================
// 时间步控制器
// ============================================================================

/// 时间步控制器状态
#[derive(Debug, Clone)]
pub struct TimeStepState {
    /// 当前时间步
    pub current_dt: f64,
    /// 建议下一步
    pub suggested_dt: f64,
    /// 当前 CFL 数
    pub current_cfl: f64,
    /// 累计步数
    pub step_count: usize,
    /// 拒绝步数
    pub rejected_count: usize,
    /// 最后一步是否被拒绝
    pub last_rejected: bool,
    /// 误差估计
    pub error_estimate: f64,
}

impl Default for TimeStepState {
    fn default() -> Self {
        Self {
            current_dt: 0.1,
            suggested_dt: 0.1,
            current_cfl: 0.0,
            step_count: 0,
            rejected_count: 0,
            last_rejected: false,
            error_estimate: 0.0,
        }
    }
}

/// 时间步控制器
///
/// 管理时间步大小的自适应调节
pub struct TimeStepController {
    /// 配置
    config: TimeStepConfig,
    /// 状态
    state: TimeStepState,
    /// 历史 CFL（用于平滑）
    cfl_history: Vec<f64>,
    /// 历史误差
    error_history: Vec<f64>,
}

impl TimeStepController {
    /// 创建控制器
    pub fn new(config: TimeStepConfig) -> Self {
        let config = config.normalized();
        let initial_dt = config.initial_dt;
        Self {
            config,
            state: TimeStepState {
                current_dt: initial_dt,
                suggested_dt: initial_dt,
                ..Default::default()
            },
            cfl_history: Vec::with_capacity(10),
            error_history: Vec::with_capacity(10),
        }
    }

    /// 获取当前时间步
    pub fn current_dt(&self) -> f64 {
        self.state.current_dt
    }

    /// 获取状态
    pub fn state(&self) -> &TimeStepState {
        &self.state
    }

    /// 获取配置
    pub fn config(&self) -> &TimeStepConfig {
        &self.config
    }

    /// 基于 CFL 计算时间步
    ///
    /// # 参数
    ///
    /// * `max_wave_speed` - 最大波速 (m/s)
    /// * `min_cell_size` - 最小网格尺寸 (m)
    pub fn compute_dt_from_cfl(&self, max_wave_speed: f64, min_cell_size: f64) -> f64 {
        if max_wave_speed <= 0.0 || min_cell_size <= 0.0 {
            return self.config.min_dt;
        }

        let dt_cfl = self.config.target_cfl * min_cell_size / max_wave_speed;
        let dt_safe = self.config.safety_factor * dt_cfl;
        
        dt_safe.clamp(self.config.min_dt, self.config.max_dt)
    }

    /// 更新时间步（基于 CFL）
    ///
    /// # 参数
    ///
    /// * `max_wave_speed` - 最大波速
    /// * `min_cell_size` - 最小网格尺寸
    pub fn update_from_cfl(&mut self, max_wave_speed: f64, min_cell_size: f64) {
        let new_dt = self.compute_dt_from_cfl(max_wave_speed, min_cell_size);
        
        // 计算实际 CFL
        let actual_cfl = if min_cell_size > 0.0 && max_wave_speed > 0.0 {
            self.state.current_dt * max_wave_speed / min_cell_size
        } else {
            0.0
        };
        
        self.state.current_cfl = actual_cfl;
        
        // 记录历史
        self.cfl_history.push(actual_cfl);
        if self.cfl_history.len() > 10 {
            self.cfl_history.remove(0);
        }
        
        // 限制步长变化
        let dt_limited = self.limit_step_change(new_dt);
        
        self.state.suggested_dt = dt_limited;
        self.state.current_dt = dt_limited;
        self.state.step_count += 1;
    }

    /// 基于误差估计更新时间步（嵌入式 RK 方法）
    ///
    /// # 参数
    ///
    /// * `error` - 误差估计
    ///
    /// # 返回
    ///
    /// 是否接受该步（true = 接受，false = 拒绝需重算）
    pub fn update_from_error(&mut self, error: f64) -> bool {
        self.state.error_estimate = error;
        
        // 记录历史
        self.error_history.push(error);
        if self.error_history.len() > 10 {
            self.error_history.remove(0);
        }
        
        let tol = self.config.error_tolerance;
        let order = self.config.method.order() as f64;
        
        if error <= tol {
            // 接受步长
            self.state.last_rejected = false;
            self.state.step_count += 1;
            
            // 计算新步长（PI 控制器）
            let factor = if error > 0.0 {
                self.config.safety_factor * (tol / error).powf(1.0 / (order + 1.0))
            } else {
                self.config.max_growth
            };
            
            let factor_limited = factor.clamp(1.0 / self.config.max_shrink, self.config.max_growth);
            let new_dt = (self.state.current_dt * factor_limited)
                .clamp(self.config.min_dt, self.config.max_dt);
            
            self.state.suggested_dt = new_dt;
            self.state.current_dt = new_dt;
            
            true
        } else {
            // 拒绝步长
            self.state.last_rejected = true;
            self.state.rejected_count += 1;
            
            // 缩小步长
            let factor = self.config.rejection_shrink
                .max(self.config.safety_factor * (tol / error).powf(1.0 / order));
            
            let new_dt = (self.state.current_dt * factor)
                .clamp(self.config.min_dt, self.state.current_dt * self.config.max_shrink);
            
            self.state.current_dt = new_dt;
            self.state.suggested_dt = new_dt;
            
            false
        }
    }

    /// 限制步长变化速率
    fn limit_step_change(&self, new_dt: f64) -> f64 {
        let current = self.state.current_dt;
        
        // 限制增长
        let dt_max_growth = current * self.config.max_growth;
        let dt_max_shrink = current * self.config.max_shrink;
        
        new_dt.clamp(dt_max_shrink, dt_max_growth)
            .clamp(self.config.min_dt, self.config.max_dt)
    }

    /// 强制设置时间步
    pub fn set_dt(&mut self, dt: f64) {
        self.state.current_dt = dt.clamp(self.config.min_dt, self.config.max_dt);
        self.state.suggested_dt = self.state.current_dt;
    }

    /// 重置控制器
    pub fn reset(&mut self) {
        self.state = TimeStepState {
            current_dt: self.config.initial_dt,
            suggested_dt: self.config.initial_dt,
            ..Default::default()
        };
        self.cfl_history.clear();
        self.error_history.clear();
    }

    /// 获取平均 CFL
    pub fn average_cfl(&self) -> f64 {
        if self.cfl_history.is_empty() {
            0.0
        } else {
            self.cfl_history.iter().sum::<f64>() / self.cfl_history.len() as f64
        }
    }

    /// 获取拒绝率
    pub fn rejection_rate(&self) -> f64 {
        let total = self.state.step_count + self.state.rejected_count;
        if total == 0 {
            0.0
        } else {
            self.state.rejected_count as f64 / total as f64
        }
    }
}

// ============================================================================
// Runge-Kutta 方法实现
// ============================================================================

/// RK 阶段系数（Butcher 表）
pub struct ButcherTableau {
    /// 阶段数
    pub stages: usize,
    /// 节点 c
    pub c: Vec<f64>,
    /// 系数矩阵 A (stages x stages)
    pub a: Vec<Vec<f64>>,
    /// 权重 b
    pub b: Vec<f64>,
    /// 嵌入权重 b* (误差估计用)
    pub b_star: Option<Vec<f64>>,
    /// 阶数
    pub order: usize,
}

impl ButcherTableau {
    /// RK2-TVD (Heun)
    pub fn rk2_tvd() -> Self {
        Self {
            stages: 2,
            c: vec![0.0, 1.0],
            a: vec![
                vec![0.0, 0.0],
                vec![1.0, 0.0],
            ],
            b: vec![0.5, 0.5],
            b_star: None,
            order: 2,
        }
    }

    /// RK3-SSP (Shu-Osher)
    pub fn rk3_ssp() -> Self {
        Self {
            stages: 3,
            c: vec![0.0, 1.0, 0.5],
            a: vec![
                vec![0.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![0.25, 0.25, 0.0],
            ],
            b: vec![1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            b_star: None,
            order: 3,
        }
    }

    /// 经典 RK4
    pub fn rk4_classic() -> Self {
        Self {
            stages: 4,
            c: vec![0.0, 0.5, 0.5, 1.0],
            a: vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![0.5, 0.0, 0.0, 0.0],
                vec![0.0, 0.5, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
            ],
            b: vec![1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            b_star: None,
            order: 4,
        }
    }

    /// RK4(5) Dormand-Prince (嵌入式)
    pub fn dopri45() -> Self {
        Self {
            stages: 7,
            c: vec![0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0, 1.0],
            a: vec![
                vec![0.0; 7],
                vec![0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
                vec![
                    19372.0 / 6561.0,
                    -25360.0 / 2187.0,
                    64448.0 / 6561.0,
                    -212.0 / 729.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                vec![
                    9017.0 / 3168.0,
                    -355.0 / 33.0,
                    46732.0 / 5247.0,
                    49.0 / 176.0,
                    -5103.0 / 18656.0,
                    0.0,
                    0.0,
                ],
                vec![
                    35.0 / 384.0,
                    0.0,
                    500.0 / 1113.0,
                    125.0 / 192.0,
                    -2187.0 / 6784.0,
                    11.0 / 84.0,
                    0.0,
                ],
            ],
            b: vec![
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ],
            b_star: Some(vec![
                5179.0 / 57600.0,
                0.0,
                7571.0 / 16695.0,
                393.0 / 640.0,
                -92097.0 / 339200.0,
                187.0 / 2100.0,
                1.0 / 40.0,
            ]),
            order: 5,
        }
    }

    /// 获取对应方法的 Butcher 表
    pub fn for_method(method: TimeIntegrationMethod) -> Option<Self> {
        match method {
            TimeIntegrationMethod::Rk2Tvd => Some(Self::rk2_tvd()),
            TimeIntegrationMethod::Rk3Ssp => Some(Self::rk3_ssp()),
            TimeIntegrationMethod::Rk4 => Some(Self::rk4_classic()),
            _ => None,
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_properties() {
        assert_eq!(TimeIntegrationMethod::Rk3Ssp.order(), 3);
        assert_eq!(TimeIntegrationMethod::Rk3Ssp.stages(), 3);
        assert_eq!(TimeIntegrationMethod::Rk4.stability_limit(), 2.785);
    }

    #[test]
    fn test_controller_cfl() {
        let config = TimeStepConfig::default();
        let mut controller = TimeStepController::new(config);
        
        // 波速 10 m/s, 网格 1 m
        controller.update_from_cfl(10.0, 1.0);
        
        // 预期 dt ≈ 0.9 * 0.5 * 1.0 / 10.0 = 0.045
        assert!(controller.current_dt() > 0.0);
        assert!(controller.current_dt() < 0.1);
    }

    #[test]
    fn test_controller_error() {
        let config = TimeStepConfig {
            error_tolerance: 1e-4,
            ..Default::default()
        };
        let mut controller = TimeStepController::new(config);
        
        // 误差小于容限，应接受
        assert!(controller.update_from_error(1e-5));
        assert!(!controller.state().last_rejected);
        
        // 误差大于容限，应拒绝
        assert!(!controller.update_from_error(1e-2));
        assert!(controller.state().last_rejected);
    }

    #[test]
    fn test_butcher_tableau() {
        let rk3 = ButcherTableau::rk3_ssp();
        assert_eq!(rk3.stages, 3);
        assert_eq!(rk3.order, 3);
        assert_eq!(rk3.b.len(), 3);
        
        // 检查权重和为 1
        let sum: f64 = rk3.b.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dopri45() {
        let dopri = ButcherTableau::dopri45();
        assert_eq!(dopri.stages, 7);
        assert_eq!(dopri.order, 5);
        assert!(dopri.b_star.is_some());
    }

    #[test]
    fn test_step_limiting() {
        let config = TimeStepConfig {
            initial_dt: 1.0,
            max_growth: 1.5,
            max_shrink: 0.5,
            ..Default::default()
        };
        let mut controller = TimeStepController::new(config);
        
        // 尝试大幅增加步长
        controller.update_from_cfl(0.1, 100.0); // 很小的波速
        
        // 应该被限制在 1.5 倍
        assert!(controller.current_dt() <= 1.5);
    }

    #[test]
    fn test_rejection_rate() {
        let config = TimeStepConfig::default();
        let mut controller = TimeStepController::new(config);
        
        controller.update_from_error(1e-5); // 接受
        controller.update_from_error(1e-5); // 接受
        controller.update_from_error(1.0);   // 拒绝
        
        // 2 成功 + 1 拒绝 = 33% 拒绝率
        let rate = controller.rejection_rate();
        assert!((rate - 1.0 / 3.0).abs() < 0.01);
    }
}
