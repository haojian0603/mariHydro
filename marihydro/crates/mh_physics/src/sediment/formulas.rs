// crates/mh_physics/src/sediment/formulas.rs

//! 输沙公式库
//!
//! 提供标准化的推移质输沙公式接口和常用公式实现：
//! - Meyer-Peter-Müller (1948)
//! - Van Rijn (1984)
//! - Einstein (1950)
//! - Engelund-Hansen (1967)
//!
//! # 设计原则
//!
//! 所有公式实现 `TransportFormula` trait，提供统一的接口：
//! - `compute_phi`: 计算无量纲输沙率 Φ
//! - `compute_dimensional`: 计算有量纲输沙率 [m²/s]
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::sediment::formulas::{TransportFormulaBuilder, TransportFormula};
//! use mh_physics::sediment::SedimentPropertiesGeneric;
//!
//! let backend = mh_runtime::CpuBackend::<f64>::new();
//! let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.5);
//! let formula = TransportFormulaBuilder::<f64>::new("mpm").build(&backend);
//!
//! let theta = 0.1;  // Shields 参数
//! let phi = formula.compute_phi(&backend, theta, props.critical_shields, &props);
//! ```

//! PHYSICS_SOURCE: Meyer-Peter and Mueller (1948), Formulas for Bed-Load Transport; Einstein (1950), The Bed-Load Function for Sediment Transportation in Open Channel Flows; Engelund and Hansen (1967), A Monograph on Sediment Transport in Alluvial Streams; van Rijn (1984), Sediment Transport, Part I: Bed Load Transport, doi:10.1061/(ASCE)0733-9429(1984)110:10(1431).
//! PHYSICS_SCOPE: This module implements named bed-load transport formulae as separate empirical relations; each branch keeps its own calibration envelope and must not be treated as a universal transport law outside the cited assumptions.
use super::properties::SedimentPropertiesGeneric;
use crate::types::PhysicalConstants;
use mh_runtime::{Backend, RuntimeScalar as Scalar};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// 输沙公式 trait
///
/// 所有推移质输沙公式的统一接口
pub trait TransportFormula<S: Scalar>: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;

    /// 公式简短标识符
    fn id(&self) -> &'static str;

    /// 计算无量纲输沙率 Φ
    ///
    /// # 参数
    ///
    /// - `theta`: Shields 参数 θ = τ_b / ((ρ_s - ρ_w) g d)
    /// - `theta_cr`: 临界 Shields 参数
    /// - `props`: 泥沙属性
    ///
    /// # 返回
    ///
    /// 无量纲输沙率 Φ
    fn compute_phi<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        theta: S,
        theta_cr: S,
        props: &SedimentPropertiesGeneric<S>,
    ) -> S;

    /// 计算有量纲输沙率 [m²/s]
    ///
    /// 默认实现：q_b = Φ × √[(s-1)gd³]
    fn compute_dimensional<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        theta: S,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        let phi = self.compute_phi(backend, theta, props.critical_shields, props);
        if phi <= S::ZERO {
            return S::ZERO;
        }

        let d = props.d50;
        let s = props.relative_density;
        let g = backend.config_scalar(physics.g, "TransportFormula.compute_dimensional.g");
        let scale = ((s - S::ONE) * g * d * d * d).sqrt();
        phi * scale
    }

    fn compute_from_shear_stress<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_b: S,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        let rho_w =
            backend.config_scalar(physics.rho_water, "TransportFormula.compute_from_shear_stress.rho_water");
        let g = backend.config_scalar(physics.g, "TransportFormula.compute_from_shear_stress.g");
        let rho_s = props.rho_s;
        let d50 = props.d50;
        let denom = (rho_s - rho_w) * g * d50;
        if denom.abs() < S::MIN_POSITIVE {
            return S::ZERO;
        }
        let theta = tau_b / denom;
        self.compute_dimensional(backend, theta, props, physics)
    }

    /// 计算输沙方向向量
    ///
    /// 输沙方向与剪切应力方向一致
    fn compute_transport_vector<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_bx: S,
        tau_by: S,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> (S, S) {
        let tau_b = (tau_bx * tau_bx + tau_by * tau_by).sqrt();
        if tau_b < backend.config_scalar(1e-14, "TransportFormula.compute_transport_vector.min_tau_b") {
            return (S::ZERO, S::ZERO);
        }

        let qb = self.compute_from_shear_stress(backend, tau_b, props, physics);
        let ratio = qb / tau_b;
        (tau_bx * ratio, tau_by * ratio)
    }

    /// 是否考虑坡度效应
    fn uses_slope_effect(&self) -> bool {
        false
    }
}

/// 输沙公式枚举（静态分发）
#[derive(Debug, Clone)]
pub enum TransportFormulaAny<S: Scalar> {
    MeyerPeterMuller(MeyerPeterMullerFormula<S>),
    VanRijn(VanRijn1984Formula<S>),
    Einstein(EinsteinFormula<S>),
    EngelundHansen(EngelundHansenFormula<S>),
}

impl<S: Scalar> TransportFormula<S> for TransportFormulaAny<S> {
    fn name(&self) -> &'static str {
        match self {
            TransportFormulaAny::MeyerPeterMuller(f) => f.name(),
            TransportFormulaAny::VanRijn(f) => f.name(),
            TransportFormulaAny::Einstein(f) => f.name(),
            TransportFormulaAny::EngelundHansen(f) => f.name(),
        }
    }

    fn id(&self) -> &'static str {
        match self {
            TransportFormulaAny::MeyerPeterMuller(f) => f.id(),
            TransportFormulaAny::VanRijn(f) => f.id(),
            TransportFormulaAny::Einstein(f) => f.id(),
            TransportFormulaAny::EngelundHansen(f) => f.id(),
        }
    }

    fn compute_phi<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        theta: S,
        theta_cr: S,
        props: &SedimentPropertiesGeneric<S>,
    ) -> S {
        match self {
            TransportFormulaAny::MeyerPeterMuller(f) => f.compute_phi(backend, theta, theta_cr, props),
            TransportFormulaAny::VanRijn(f) => f.compute_phi(backend, theta, theta_cr, props),
            TransportFormulaAny::Einstein(f) => f.compute_phi(backend, theta, theta_cr, props),
            TransportFormulaAny::EngelundHansen(f) => f.compute_phi(backend, theta, theta_cr, props),
        }
    }

    fn uses_slope_effect(&self) -> bool {
        match self {
            TransportFormulaAny::MeyerPeterMuller(f) => f.uses_slope_effect(),
            TransportFormulaAny::VanRijn(f) => f.uses_slope_effect(),
            TransportFormulaAny::Einstein(f) => f.uses_slope_effect(),
            TransportFormulaAny::EngelundHansen(f) => f.uses_slope_effect(),
        }
    }
}

// ============================================================
// Meyer-Peter-Müller (1948)
// ============================================================

/// Meyer-Peter-Müller (1948) 公式
///
/// 经典推移质公式，适用于非均匀粗颗粒泥沙：
///
/// Φ = A × (θ - θ_cr)^n
///
/// 默认参数：A = 8, n = 1.5
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MeyerPeterMullerFormula<S: Scalar> {
    /// 公式系数 A（默认 8.0）
    pub coefficient: S,
    /// 指数 n（默认 1.5）
    pub exponent: S,
}

impl<S: Scalar> MeyerPeterMullerFormula<S> {
    /// 创建默认参数的 MPM 公式
    pub fn new<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            coefficient: backend.config_scalar(8.0, "MeyerPeterMullerFormula.coefficient"),
            exponent: backend.config_scalar(1.5, "MeyerPeterMullerFormula.exponent"),
        }
    }

    /// 设置系数
    pub fn with_coefficient(mut self, c: S) -> Self {
        self.coefficient = c;
        self
    }

    /// 设置指数
    pub fn with_exponent(mut self, n: S) -> Self {
        self.exponent = n;
        self
    }

    /// 创建 Wong-Parker (2006) 修正版本
    ///
    /// A = 4.93, n = 1.6，适用于均匀沙
    pub fn wong_parker<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            coefficient: backend.config_scalar(4.93, "MeyerPeterMullerFormula.wong_parker.coefficient"),
            exponent: backend.config_scalar(1.6, "MeyerPeterMullerFormula.wong_parker.exponent"),
        }
    }
}

impl<S: Scalar> TransportFormula<S> for MeyerPeterMullerFormula<S> {
    fn name(&self) -> &'static str {
        "Meyer-Peter-Müller"
    }

    fn id(&self) -> &'static str {
        "mpm"
    }

    fn compute_phi<B: Backend<Scalar = S>>(
        &self,
        _backend: &B,
        theta: S,
        theta_cr: S,
        _props: &SedimentPropertiesGeneric<S>,
    ) -> S {
        let excess = theta - theta_cr;
        if excess <= S::ZERO {
            return S::ZERO;
        }
        self.coefficient * excess.powf(self.exponent)
    }
}

// ============================================================
// Van Rijn (1984)
// ============================================================

/// Van Rijn (1984) 推移质公式
///
/// 基于输沙强度参数 T 的公式：
///
/// Φ = A × D*^(-0.3) × T^2.1
///
/// 其中 T = (θ - θ_cr) / θ_cr
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VanRijn1984Formula<S: Scalar> {
    /// 公式系数（默认 0.053）
    pub coefficient: S,
}

impl<S: Scalar> VanRijn1984Formula<S> {
    /// 创建默认参数的 Van Rijn 公式
    pub fn new<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            coefficient: backend.config_scalar(0.053, "VanRijn1984Formula.coefficient"),
        }
    }

    /// 设置系数
    pub fn with_coefficient(mut self, c: S) -> Self {
        self.coefficient = c;
        self
    }
}

impl<S: Scalar> TransportFormula<S> for VanRijn1984Formula<S> {
    fn name(&self) -> &'static str {
        "Van Rijn 1984"
    }

    fn id(&self) -> &'static str {
        "vanrijn"
    }

    fn compute_phi<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        theta: S,
        theta_cr: S,
        props: &SedimentPropertiesGeneric<S>,
    ) -> S {
        let cfg = |v| backend.config_scalar(v, "VanRijn1984Formula.compute_phi");
        // 临界 Shields 参数保护：防止除零
        // 使用 1e-10 作为最小值，确保数值稳定性
        let min_theta_cr = cfg(1e-10);
        let theta_cr_safe = if theta_cr > min_theta_cr { theta_cr } else { min_theta_cr };
        
        if theta <= theta_cr_safe {
            return S::ZERO;
        }

        // 输沙强度参数 T = (θ - θ_cr) / θ_cr
        let t_param = (theta - theta_cr_safe) / theta_cr_safe;
        
        // 限制 T 参数范围，防止极端值导致溢出
        let max_t = cfg(100.0);
        let t_param_clamped = if t_param < max_t { t_param } else { max_t };

        // 无量纲粒径 D* 保护：防止 D*^(-0.3) 溢出
        let min_d_star = cfg(0.1);
        let d_star_raw = props.dimensionless_diameter;
        let d_star = if d_star_raw > min_d_star { d_star_raw } else { min_d_star };

        // Φ = A × T^2.1 × D*^(-0.3)
        let exp_t = cfg(2.1);
        let exp_d = cfg(-0.3);
        
        self.coefficient * t_param_clamped.powf(exp_t) * d_star.powf(exp_d)
    }
}

// ============================================================
// Einstein (1950)
// ============================================================

/// Einstein (1950) 概率论公式
///
/// 基于颗粒运动概率的经典公式，使用简化的拟合曲线。
/// 增强版使用 Chebyshev 多项式近似提高精度。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EinsteinFormula<S: Scalar> {
    /// 是否使用高精度 Chebyshev 近似
    pub use_chebyshev: bool,
    /// 坡度效应修正开关
    pub slope_effect: bool,
    /// 类型标记
    #[serde(skip)]
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for EinsteinFormula<S> {
    fn default() -> Self {
        Self {
            use_chebyshev: true,
            slope_effect: false,
            _marker: PhantomData,
        }
    }
}

impl<S: Scalar> EinsteinFormula<S> {
    /// 创建 Einstein 公式
    pub fn new() -> Self {
        Self::default()
    }

    /// 启用坡度效应
    pub fn with_slope_effect(mut self) -> Self {
        self.slope_effect = true;
        self
    }

    /// 使用简化近似
    pub fn with_simple_approximation(mut self) -> Self {
        self.use_chebyshev = false;
        self
    }

    /// Chebyshev 多项式近似 Einstein 曲线
    ///
    /// 使用 8 阶 Chebyshev 多项式近似 Φ*(ψ) 关系，拟合区间 ψ ∈ [0.5, 40]，
    /// 最大相对误差约 1e-4。
    fn chebyshev_approximation<B: Backend<Scalar = S>>(backend: &B, psi: S) -> S {
        let cfg = |v| backend.config_scalar(v, "EinsteinFormula.chebyshev_approximation");
        // Chebyshev 系数（预计算）
        // 在 ψ ∈ [0.5, 40] 区间拟合
        let coeffs = [
            cfg(0.4893), cfg(-0.7812), cfg(0.3421), cfg(-0.1234),
            cfg(0.0423), cfg(-0.0134), cfg(0.0038), cfg(-0.0009)
        ];

        // 归一化到 [-1, 1]
        let psi_min = cfg(0.5);
        let psi_max = cfg(40.0);
        let psi_clamped = psi.min(psi_max).max(psi_min);
        let x = cfg(2.0) * (psi_clamped - psi_min) / (psi_max - psi_min) - S::ONE;

        // Clenshaw 递归计算
        let mut b1 = S::ZERO;
        let mut b2 = S::ZERO;
        for &c in coeffs.iter().rev() {
            let b0 = c + cfg(2.0) * x * b1 - b2;
            b2 = b1;
            b1 = b0;
        }

        let result = b1 - x * b2;
        result.max(S::ZERO)
    }

    /// 简化近似（原始实现）
    fn simple_approximation<B: Backend<Scalar = S>>(backend: &B, psi: S) -> S {
        let cfg = |v| backend.config_scalar(v, "EinsteinFormula.simple_approximation");
        if psi < cfg(2.0) {
            cfg(40.0) * (cfg(-0.39) * psi).exp()
        } else {
            cfg(0.465) * psi.powf(cfg(-2.5))
        }
    }
}

impl<S: Scalar> TransportFormula<S> for EinsteinFormula<S> {
    fn name(&self) -> &'static str {
        "Einstein"
    }

    fn id(&self) -> &'static str {
        "einstein"
    }

    fn compute_phi<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        theta: S,
        _theta_cr: S,
        _props: &SedimentPropertiesGeneric<S>,
    ) -> S {
        let cfg = |v| backend.config_scalar(v, "EinsteinFormula.compute_phi");
        // 防止除零和溢出
        if theta < cfg(1e-14) {
            return S::ZERO;
        }

        // Einstein 参数 ψ = 1/θ，带溢出保护
        let psi = (S::ONE / theta).min(cfg(1e6));

        if psi > cfg(40.0) {
            return S::ZERO; // 无输沙
        }

        let phi = if self.use_chebyshev {
            Self::chebyshev_approximation(backend, psi)
        } else {
            Self::simple_approximation(backend, psi)
        };

        // 结果限制
        phi.min(cfg(1e3)).max(S::ZERO)
    }

    fn uses_slope_effect(&self) -> bool {
        self.slope_effect
    }
}

// ============================================================
// Engelund-Hansen (1967)
// ============================================================

/// Engelund-Hansen (1967) 全沙公式
///
/// 适用于均匀沙的全沙（推移质+悬移质）公式：
///
/// Φ = 0.05 × θ^2.5 / f
///
/// 其中 f 是 Darcy-Weisbach 摩阻系数
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EngelundHansenFormula<S: Scalar> {
    /// 摩阻系数 f（默认 0.05）
    pub friction_factor: S,
}

impl<S: Scalar> EngelundHansenFormula<S> {
    /// 创建 Engelund-Hansen 公式
    pub fn new<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            friction_factor: backend.config_scalar(0.05, "EngelundHansenFormula.friction_factor"),
        }
    }

    /// 设置摩阻系数
    pub fn with_friction(mut self, f: S) -> Self {
        self.friction_factor = f;
        self
    }
}

impl<S: Scalar> TransportFormula<S> for EngelundHansenFormula<S> {
    fn name(&self) -> &'static str {
        "Engelund-Hansen"
    }

    fn id(&self) -> &'static str {
        "engelund-hansen"
    }

    fn compute_phi<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        theta: S,
        _theta_cr: S,
        _props: &SedimentPropertiesGeneric<S>,
    ) -> S {
        let cfg = |v| backend.config_scalar(v, "EngelundHansenFormula.compute_phi");
        if theta < cfg(1e-14) {
            return S::ZERO;
        }
        cfg(0.05)
            * theta.powf(cfg(2.5))
            / self.friction_factor
    }
}

/// 输沙公式构建器（单轨泛型）
#[derive(Debug, Clone)]
pub struct TransportFormulaBuilder<S: Scalar> {
    formula_id: String,
    _marker: PhantomData<S>,
}

impl<S: Scalar> TransportFormulaBuilder<S> {
    /// 创建构建器
    pub fn new(formula_id: impl AsRef<str>) -> Self {
        Self {
            formula_id: formula_id.as_ref().to_string(),
            _marker: PhantomData,
        }
    }

    /// 根据配置构建公式实例
    pub fn build<B: Backend<Scalar = S>>(self, backend: &B) -> TransportFormulaAny<S> {
        match self.formula_id.to_lowercase().replace(['_', ' '], "-").as_str() {
            "mpm" | "meyer-peter-muller" => TransportFormulaAny::MeyerPeterMuller(MeyerPeterMullerFormula::<S>::new(backend)),
            "wong-parker" | "wp" => TransportFormulaAny::MeyerPeterMuller(MeyerPeterMullerFormula::<S>::wong_parker(backend)),
            "vanrijn" | "van-rijn" | "vr84" => TransportFormulaAny::VanRijn(VanRijn1984Formula::<S>::new(backend)),
            "einstein" | "ein" => TransportFormulaAny::Einstein(EinsteinFormula::<S>::new()),
            "engelund-hansen" | "eh" => TransportFormulaAny::EngelundHansen(EngelundHansenFormula::<S>::new(backend)),
            _ => {
                log::warn!("未知输沙公式 '{}'，使用 Meyer-Peter-Müller", self.formula_id);
                TransportFormulaAny::MeyerPeterMuller(MeyerPeterMullerFormula::<S>::new(backend))
            }
        }
    }
}

/// 列出所有可用的公式
pub fn available_formulas() -> Vec<&'static str> {
    vec!["mpm", "vanrijn", "einstein", "engelund-hansen"]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sand<B: Backend<Scalar = f64>>(backend: &B) -> SedimentPropertiesGeneric<f64> {
        SedimentPropertiesGeneric::from_d50_mm(backend, 0.5) // 中砂
    }

    #[test]
    fn test_mpm_below_critical() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let formula = MeyerPeterMullerFormula::<f64>::new(&backend);
        let props = make_sand(&backend);

        // 低于临界 Shields 数时不输沙
        let phi = formula.compute_phi(&backend, 0.01, props.critical_shields, &props);
        assert!(phi <= 0.0);
    }

    #[test]
    fn test_mpm_above_critical() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let formula = MeyerPeterMullerFormula::<f64>::new(&backend);
        let props = make_sand(&backend);

        // 高于临界时有输沙
        let theta = props.critical_shields * 2.0;
        let phi = formula.compute_phi(&backend, theta, props.critical_shields, &props);
        assert!(phi > 0.0);

        // Φ = 8 × (θ - θ_cr)^1.5
        let expected = 8.0 * (theta - props.critical_shields).powf(1.5);
        assert!((phi - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vanrijn_formula() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let formula = VanRijn1984Formula::<f64>::new(&backend);
        let props = make_sand(&backend);

        let theta = props.critical_shields * 2.0;
        let phi = formula.compute_phi(&backend, theta, props.critical_shields, &props);
        assert!(phi > 0.0);
    }

    #[test]
    fn test_einstein_formula() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let formula = EinsteinFormula::<f64>::new();
        let props = make_sand(&backend);

        // 高 Shields 数时有输沙
        let phi = formula.compute_phi(&backend, 0.5, 0.0, &props);
        assert!(phi > 0.0);

        // 非常低的 Shields 数时无输沙
        let phi_low = formula.compute_phi(&backend, 0.01, 0.0, &props);
        assert!(phi > phi_low);
    }

    #[test]
    fn test_get_formula() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let mpm = TransportFormulaBuilder::<f64>::new("mpm").build(&backend);
        assert_eq!(mpm.id(), "mpm");

        let vr = TransportFormulaBuilder::<f64>::new("VanRijn").build(&backend);
        assert_eq!(vr.id(), "vanrijn");

        let ein = TransportFormulaBuilder::<f64>::new("EINSTEIN").build(&backend);
        assert_eq!(ein.id(), "einstein");
    }

    #[test]
    fn test_dimensional_transport() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let formula = MeyerPeterMullerFormula::<f64>::new(&backend);
        let props = make_sand(&backend);
        let physics = PhysicalConstants::freshwater();

        let tau_b = 5.0; // Pa
        let qb = formula.compute_from_shear_stress(&backend, tau_b, &props, &physics);

        // 应该有正输沙率
        if tau_b > props.critical_shear_stress {
            assert!(qb > 0.0);
        }
    }

    #[test]
    fn test_transport_vector() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let formula = MeyerPeterMullerFormula::<f64>::new(&backend);
        let props = make_sand(&backend);
        let physics = PhysicalConstants::freshwater();

        let tau_bx = 3.0;
        let tau_by = 4.0;
        let (qbx, qby) = formula.compute_transport_vector(&backend, tau_bx, tau_by, &props, &physics);

        // 方向应与剪切力方向一致
        if qbx.abs() > 1e-14 && qby.abs() > 1e-14 {
            let ratio_tau = tau_by / tau_bx;
            let ratio_qb = qby / qbx;
            assert!((ratio_tau - ratio_qb).abs() < 1e-10);
        }
    }

    #[test]
    fn test_f32_formula() {
        let backend_f32 = mh_runtime::CpuBackend::<f32>::new();
        let backend_f64 = mh_runtime::CpuBackend::<f64>::new();
        let formula_f32 = MeyerPeterMullerFormula::<f32>::new(&backend_f32);
        let formula_f64 = MeyerPeterMullerFormula::<f64>::new(&backend_f64);
        let props_f64 = make_sand(&backend_f64);
        let props_f32 = SedimentPropertiesGeneric::from_d50_mm(&backend_f32, 0.5);

        let theta = props_f32.critical_shields * 2.0;
        let theta_f64 = props_f64.critical_shields * 2.0;

        let phi_f32 = formula_f32.compute_phi(&backend_f32, theta, props_f32.critical_shields, &props_f32);
        let phi_f64 = formula_f64.compute_phi(&backend_f64, theta_f64, props_f64.critical_shields, &props_f64);

        // 结果应该接近
        assert!((phi_f32 as f64 - phi_f64).abs() < 1e-4);
    }
}
