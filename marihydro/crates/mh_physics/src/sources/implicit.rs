// crates/mh_physics/src/sources/implicit.rs

//! 隐式源项处理模块
//!
//! 提供隐式时间积分方法，用于处理刚性源项（如摩擦、扩散）。
//!
//! # 物理背景
//!
//! 某些源项（如底摩擦）在小水深时可能变得非常刚性，
//! 导致显式方法需要极小的时间步长。隐式处理可以避免这个问题。
//!
//! # 算法
//!
//! 对于动量衰减方程 d(hu)/dt = -γ * hu:
//!
//! - **显式 Euler**: u_new = u - dt*γ*u
//! - **隐式 Euler**: u_new = u / (1 + dt*γ)
//! - **Crank-Nicolson**: u_new = u * (1 - dt*γ/2) / (1 + dt*γ/2)
//! - **解析衰减**: u_new = u * exp(-γ*dt)

use glam::DVec2;

/// 隐式处理方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImplicitMethod {
    /// 显式 Euler（不推荐用于刚性问题）
    Explicit,
    /// 隐式 Euler（一阶，无条件稳定）
    #[default]
    ImplicitEuler,
    /// Crank-Nicolson（二阶，A-稳定）
    CrankNicolson,
    /// 解析衰减（用于线性阻尼）
    AnalyticDecay,
}

impl ImplicitMethod {
    /// 获取方法名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Explicit => "Explicit",
            Self::ImplicitEuler => "ImplicitEuler",
            Self::CrankNicolson => "CrankNicolson",
            Self::AnalyticDecay => "AnalyticDecay",
        }
    }

    /// 是否是隐式方法
    pub fn is_implicit(&self) -> bool {
        !matches!(self, Self::Explicit)
    }

    /// 计算衰减因子
    ///
    /// 返回 u_new / u 的比值
    #[inline]
    pub fn decay_factor(&self, gamma: f64, dt: f64) -> f64 {
        match self {
            Self::Explicit => 1.0 - dt * gamma,
            Self::ImplicitEuler => 1.0 / (1.0 + dt * gamma),
            Self::CrankNicolson => {
                let half_dt_gamma = 0.5 * dt * gamma;
                (1.0 - half_dt_gamma) / (1.0 + half_dt_gamma)
            }
            Self::AnalyticDecay => (-gamma * dt).exp(),
        }
    }
}

/// 隐式源项配置
#[derive(Debug, Clone, Copy)]
pub struct ImplicitConfig {
    /// 处理方法
    pub method: ImplicitMethod,
    /// 最大迭代次数（用于非线性隐式）
    pub max_iterations: usize,
    /// 收敛容差
    pub tolerance: f64,
    /// 欠松弛因子
    pub relaxation: f64,
}

impl Default for ImplicitConfig {
    fn default() -> Self {
        Self {
            method: ImplicitMethod::default(),
            max_iterations: 5,
            tolerance: 1e-8,
            relaxation: 1.0,
        }
    }
}

impl ImplicitConfig {
    /// 创建使用指定方法的配置
    pub fn with_method(method: ImplicitMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// 设置最大迭代次数
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// 设置容差
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// 设置欠松弛因子
    pub fn with_relaxation(mut self, omega: f64) -> Self {
        self.relaxation = omega.clamp(0.1, 2.0);
        self
    }
}

/// 阻尼系数计算器 trait
///
/// 将各种摩擦公式统一为阻尼系数 γ
pub trait DampingCoefficient {
    /// 计算阻尼系数 γ（单位：1/s）
    ///
    /// 动量方程中的阻尼项形式为 -(γ/h) * (hu, hv)
    fn compute_gamma(&self, h: f64, speed: f64, cell_idx: usize) -> f64;

    /// 名称
    fn name(&self) -> &'static str;
}

/// Manning 阻尼系数
#[derive(Debug, Clone)]
pub struct ManningDamping {
    /// 重力加速度
    pub g: f64,
    /// Manning 系数
    pub n: f64,
    /// 预计算 g*n²
    gn2: f64,
}

impl ManningDamping {
    /// 创建 Manning 阻尼
    pub fn new(g: f64, n: f64) -> Self {
        Self {
            g,
            n,
            gn2: g * n * n,
        }
    }
}

impl DampingCoefficient for ManningDamping {
    fn compute_gamma(&self, h: f64, speed: f64, _cell_idx: usize) -> f64 {
        if h < 1e-6 {
            return 0.0;
        }
        // γ = g * n² * |u| / h^(4/3)
        self.gn2 * speed / h.powf(4.0 / 3.0)
    }

    fn name(&self) -> &'static str {
        "Manning"
    }
}

/// Chezy 阻尼系数
#[derive(Debug, Clone)]
pub struct ChezyDamping {
    /// 重力加速度
    pub g: f64,
    /// Chezy 系数
    pub c: f64,
    /// 预计算 g/C²
    g_c2: f64,
}

impl ChezyDamping {
    /// 创建 Chezy 阻尼
    pub fn new(g: f64, c: f64) -> Self {
        Self {
            g,
            c,
            g_c2: g / (c * c),
        }
    }
}

impl DampingCoefficient for ChezyDamping {
    fn compute_gamma(&self, h: f64, speed: f64, _cell_idx: usize) -> f64 {
        if h < 1e-6 {
            return 0.0;
        }
        // γ = g * |u| / (C² * h)
        self.g_c2 * speed / h
    }

    fn name(&self) -> &'static str {
        "Chezy"
    }
}

/// 隐式动量衰减求解器
///
/// 求解 d(hu)/dt = -γ * hu
#[derive(Debug)]
pub struct ImplicitMomentumDecay {
    config: ImplicitConfig,
}

impl Default for ImplicitMomentumDecay {
    fn default() -> Self {
        Self {
            config: ImplicitConfig::default(),
        }
    }
}

impl ImplicitMomentumDecay {
    /// 创建求解器
    pub fn new(config: ImplicitConfig) -> Self {
        Self { config }
    }

    /// 获取配置
    pub fn config(&self) -> &ImplicitConfig {
        &self.config
    }

    /// 应用隐式衰减到单个单元
    ///
    /// # 参数
    /// - `hu`, `hv`: 动量分量
    /// - `gamma`: 阻尼系数
    /// - `dt`: 时间步长
    ///
    /// # 返回
    /// 更新后的 (hu, hv)
    pub fn apply(&self, hu: f64, hv: f64, gamma: f64, dt: f64) -> (f64, f64) {
        if gamma.abs() < 1e-20 {
            return (hu, hv);
        }

        let factor = self.config.method.decay_factor(gamma, dt);
        (hu * factor, hv * factor)
    }

    /// 应用隐式衰减到向量
    pub fn apply_vec(&self, momentum: DVec2, gamma: f64, dt: f64) -> DVec2 {
        let (hu, hv) = self.apply(momentum.x, momentum.y, gamma, dt);
        DVec2::new(hu, hv)
    }

    /// 批量应用隐式衰减
    pub fn apply_batch<D: DampingCoefficient>(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        damping: &D,
        dt: f64,
        h_dry: f64,
    ) {
        for i in 0..h.len() {
            if h[i] < h_dry {
                // 干区：动量归零
                hu[i] = 0.0;
                hv[i] = 0.0;
                continue;
            }

            let h_safe = h[i].max(1e-6);
            let speed = ((hu[i] / h_safe).powi(2) + (hv[i] / h_safe).powi(2)).sqrt();
            let gamma = damping.compute_gamma(h[i], speed, i);

            let (new_hu, new_hv) = self.apply(hu[i], hv[i], gamma, dt);
            hu[i] = new_hu;
            hv[i] = new_hv;
        }
    }

    /// 应用隐式衰减到状态数组
    pub fn apply_to_state(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        gamma: &[f64],
        dt: f64,
        h_dry: f64,
    ) {
        for i in 0..h.len() {
            if h[i] < h_dry {
                hu[i] = 0.0;
                hv[i] = 0.0;
                continue;
            }

            let (new_hu, new_hv) = self.apply(hu[i], hv[i], gamma[i], dt);
            hu[i] = new_hu;
            hv[i] = new_hv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_method_default() {
        let method = ImplicitMethod::default();
        assert_eq!(method, ImplicitMethod::ImplicitEuler);
    }

    #[test]
    fn test_implicit_method_names() {
        assert_eq!(ImplicitMethod::Explicit.name(), "Explicit");
        assert_eq!(ImplicitMethod::ImplicitEuler.name(), "ImplicitEuler");
        assert_eq!(ImplicitMethod::CrankNicolson.name(), "CrankNicolson");
        assert_eq!(ImplicitMethod::AnalyticDecay.name(), "AnalyticDecay");
    }

    #[test]
    fn test_implicit_method_is_implicit() {
        assert!(!ImplicitMethod::Explicit.is_implicit());
        assert!(ImplicitMethod::ImplicitEuler.is_implicit());
        assert!(ImplicitMethod::CrankNicolson.is_implicit());
        assert!(ImplicitMethod::AnalyticDecay.is_implicit());
    }

    #[test]
    fn test_decay_factor_explicit() {
        let factor = ImplicitMethod::Explicit.decay_factor(1.0, 0.5);
        // 1 - 0.5 * 1 = 0.5
        assert!((factor - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_factor_implicit_euler() {
        let factor = ImplicitMethod::ImplicitEuler.decay_factor(1.0, 1.0);
        // 1 / (1 + 1) = 0.5
        assert!((factor - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_factor_crank_nicolson() {
        let factor = ImplicitMethod::CrankNicolson.decay_factor(2.0, 1.0);
        // (1 - 1) / (1 + 1) = 0
        assert!(factor.abs() < 1e-10);
    }

    #[test]
    fn test_decay_factor_analytic() {
        let factor = ImplicitMethod::AnalyticDecay.decay_factor(1.0, 1.0);
        // exp(-1) ≈ 0.368
        assert!((factor - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_config_default() {
        let config = ImplicitConfig::default();
        assert_eq!(config.method, ImplicitMethod::ImplicitEuler);
        assert_eq!(config.max_iterations, 5);
        assert!((config.tolerance - 1e-8).abs() < 1e-10);
        assert!((config.relaxation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_config_builder() {
        let config = ImplicitConfig::with_method(ImplicitMethod::CrankNicolson)
            .with_max_iterations(10)
            .with_tolerance(1e-6)
            .with_relaxation(0.8);

        assert_eq!(config.method, ImplicitMethod::CrankNicolson);
        assert_eq!(config.max_iterations, 10);
        assert!((config.tolerance - 1e-6).abs() < 1e-10);
        assert!((config.relaxation - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_manning_damping() {
        let damping = ManningDamping::new(9.81, 0.03);

        let h = 1.0;
        let speed = 1.0;
        let gamma = damping.compute_gamma(h, speed, 0);

        // γ = g * n² * |u| / h^(4/3) = 9.81 * 0.0009 * 1 / 1 = 0.008829
        assert!((gamma - 9.81 * 0.0009).abs() < 1e-6);
    }

    #[test]
    fn test_manning_damping_dry() {
        let damping = ManningDamping::new(9.81, 0.03);
        let gamma = damping.compute_gamma(1e-7, 1.0, 0);
        assert_eq!(gamma, 0.0);
    }

    #[test]
    fn test_chezy_damping() {
        let damping = ChezyDamping::new(9.81, 50.0);

        let h = 1.0;
        let speed = 1.0;
        let gamma = damping.compute_gamma(h, speed, 0);

        // γ = g * |u| / (C² * h) = 9.81 / (2500 * 1) = 0.003924
        assert!((gamma - 9.81 / 2500.0).abs() < 1e-6);
    }

    #[test]
    fn test_chezy_damping_dry() {
        let damping = ChezyDamping::new(9.81, 50.0);
        let gamma = damping.compute_gamma(1e-7, 1.0, 0);
        assert_eq!(gamma, 0.0);
    }

    #[test]
    fn test_implicit_momentum_decay_creation() {
        let solver = ImplicitMomentumDecay::default();
        assert_eq!(solver.config().method, ImplicitMethod::ImplicitEuler);
    }

    #[test]
    fn test_implicit_momentum_decay_apply() {
        let solver = ImplicitMomentumDecay::new(ImplicitConfig {
            method: ImplicitMethod::ImplicitEuler,
            ..Default::default()
        });

        // γ = 1, dt = 1  =>  factor = 0.5
        let (hu, hv) = solver.apply(1.0, 0.5, 1.0, 1.0);
        assert!((hu - 0.5).abs() < 1e-10);
        assert!((hv - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_momentum_decay_apply_vec() {
        let solver = ImplicitMomentumDecay::default();
        let momentum = DVec2::new(1.0, 0.5);
        let result = solver.apply_vec(momentum, 1.0, 1.0);

        assert!((result.x - 0.5).abs() < 1e-10);
        assert!((result.y - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_momentum_decay_zero_gamma() {
        let solver = ImplicitMomentumDecay::default();
        let (hu, hv) = solver.apply(1.0, 0.5, 0.0, 1.0);

        // 无衰减
        assert!((hu - 1.0).abs() < 1e-10);
        assert!((hv - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_analytic_decay() {
        let solver = ImplicitMomentumDecay::new(ImplicitConfig {
            method: ImplicitMethod::AnalyticDecay,
            ..Default::default()
        });

        // γ = 1, dt = 1  =>  factor = e^(-1) ≈ 0.368
        let (hu, _) = solver.apply(1.0, 0.0, 1.0, 1.0);
        assert!((hu - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_crank_nicolson() {
        let solver = ImplicitMomentumDecay::new(ImplicitConfig {
            method: ImplicitMethod::CrankNicolson,
            ..Default::default()
        });

        // γ = 2, dt = 1  =>  factor = (1-1)/(1+1) = 0
        let (hu, _) = solver.apply(1.0, 0.0, 2.0, 1.0);
        assert!(hu.abs() < 1e-10);
    }
}
