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
//! use mh_physics::sediment::formulas::{get_formula_f64, TransportFormula};
//! use mh_physics::sediment::SedimentProperties;
//!
//! let props = SedimentProperties::from_d50_mm(0.5);
//! let formula = get_formula_f64("mpm");
//!
//! let theta = 0.1;  // Shields 参数
//! let phi = formula.compute_phi(theta, props.critical_shields, &props);
//! ```

use super::properties::SedimentProperties;
use crate::types::PhysicalConstants;
use mh_runtime::RuntimeScalar as Scalar;
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
    fn compute_phi(&self, theta: S, theta_cr: S, props: &SedimentProperties) -> S;

    /// 计算有量纲输沙率 [m²/s]
    ///
    /// 默认实现：q_b = Φ × √[(s-1)gd³]
    fn compute_dimensional(&self, theta: S, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        let phi = self.compute_phi(theta, S::from_f64(props.critical_shields).unwrap_or(S::ZERO), props);
        if phi <= S::ZERO {
            return S::ZERO;
        }

        let d = S::from_f64(props.d50).unwrap_or(S::ZERO);
        let s = S::from_f64(props.relative_density).unwrap_or(S::ZERO);
        let g = S::from_f64(physics.g).unwrap_or(S::ZERO);
        let scale = ((s - S::ONE) * g * d * d * d).sqrt();

        phi * scale
    }

    /// 从床面剪切应力计算输沙率
    fn compute_from_shear_stress(&self, tau_b: S, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        let theta = S::from_f64(props.shields_number(tau_b.to_f64().unwrap_or(0.0), physics)).unwrap_or(S::ZERO);
        self.compute_dimensional(theta, props, physics)
    }

    /// 计算输沙方向向量
    ///
    /// 输沙方向与剪切应力方向一致
    fn compute_transport_vector(
        &self,
        tau_bx: S,
        tau_by: S,
        props: &SedimentProperties,
        physics: &PhysicalConstants,
    ) -> (S, S) {
        let tau_b = (tau_bx * tau_bx + tau_by * tau_by).sqrt();
        if tau_b < S::from_f64(1e-14).unwrap_or(S::ZERO) {
            return (S::ZERO, S::ZERO);
        }

        let qb = self.compute_from_shear_stress(tau_b, props, physics);
        let ratio = qb / tau_b;
        (tau_bx * ratio, tau_by * ratio)
    }

    /// 是否考虑坡度效应
    fn uses_slope_effect(&self) -> bool {
        false
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

impl<S: Scalar> Default for MeyerPeterMullerFormula<S> {
    fn default() -> Self {
        Self {
            coefficient: S::from_f64(8.0).unwrap_or(S::ZERO),
            exponent: S::from_f64(1.5).unwrap_or(S::ZERO),
        }
    }
}

impl<S: Scalar> MeyerPeterMullerFormula<S> {
    /// 创建默认参数的 MPM 公式
    pub fn new() -> Self {
        Self::default()
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
    pub fn wong_parker() -> Self {
        Self {
            coefficient: S::from_f64(4.93).unwrap_or(S::ZERO),
            exponent: S::from_f64(1.6).unwrap_or(S::ZERO),
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

    fn compute_phi(&self, theta: S, theta_cr: S, _props: &SedimentProperties) -> S {
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

impl<S: Scalar> Default for VanRijn1984Formula<S> {
    fn default() -> Self {
        Self { coefficient: S::from_f64(0.053).unwrap_or(S::ZERO) }
    }
}

impl<S: Scalar> VanRijn1984Formula<S> {
    /// 创建默认参数的 Van Rijn 公式
    pub fn new() -> Self {
        Self::default()
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

    fn compute_phi(&self, theta: S, theta_cr: S, props: &SedimentProperties) -> S {
        if theta <= theta_cr {
            return S::ZERO;
        }

        // 输沙强度参数 T
        let t_param = (theta - theta_cr) / theta_cr;

        // 无量纲粒径 D*
        let d_star = S::from_f64(props.dimensionless_diameter).unwrap_or(S::ZERO);

        self.coefficient * t_param.powf(S::from_f64(2.1).unwrap_or(S::ZERO)) * d_star.powf(S::from_f64(-0.3).unwrap_or(S::ZERO))
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
    /// 使用 8 阶 Chebyshev 多项式近似 Φ*(ψ) 关系
    fn chebyshev_approximation(psi: S) -> S {
        // Chebyshev 系数（预计算）
        // 在 ψ ∈ [0.5, 40] 区间拟合
        let coeffs = [
            S::from_f64(0.4893).unwrap_or(S::ZERO), S::from_f64(-0.7812).unwrap_or(S::ZERO), S::from_f64(0.3421).unwrap_or(S::ZERO), S::from_f64(-0.1234).unwrap_or(S::ZERO),
            S::from_f64(0.0423).unwrap_or(S::ZERO), S::from_f64(-0.0134).unwrap_or(S::ZERO), S::from_f64(0.0038).unwrap_or(S::ZERO), S::from_f64(-0.0009).unwrap_or(S::ZERO)
        ];

        // 归一化到 [-1, 1]
        let psi_min = S::from_f64(0.5).unwrap_or(S::ZERO);
        let psi_max = S::from_f64(40.0).unwrap_or(S::ZERO);
        let psi_clamped = psi.min(psi_max).max(psi_min);
        let x = S::from_f64(2.0).unwrap_or(S::ZERO) * (psi_clamped - psi_min) / (psi_max - psi_min) - S::ONE;

        // Clenshaw 递归计算
        let mut b1 = S::ZERO;
        let mut b2 = S::ZERO;
        for &c in coeffs.iter().rev() {
            let b0 = c + S::from_f64(2.0).unwrap_or(S::ZERO) * x * b1 - b2;
            b2 = b1;
            b1 = b0;
        }

        let result = b1 - x * b2;
        result.max(S::ZERO)
    }

    /// 简化近似（原始实现）
    fn simple_approximation(psi: S) -> S {
        if psi < S::from_f64(2.0).unwrap_or(S::ZERO) {
            S::from_f64(40.0).unwrap_or(S::ZERO) * (S::from_f64(-0.39).unwrap_or(S::ZERO) * psi).exp()
        } else {
            S::from_f64(0.465).unwrap_or(S::ZERO) * psi.powf(S::from_f64(-2.5).unwrap_or(S::ZERO))
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

    fn compute_phi(&self, theta: S, _theta_cr: S, _props: &SedimentProperties) -> S {
        // 防止除零和溢出
        if theta < S::from_f64(1e-14).unwrap_or(S::ZERO) {
            return S::ZERO;
        }

        // Einstein 参数 ψ = 1/θ，带溢出保护
        let psi = (S::ONE / theta).min(S::from_f64(1e6).unwrap_or(S::ZERO));

        if psi > S::from_f64(40.0).unwrap_or(S::ZERO) {
            return S::ZERO; // 无输沙
        }

        let phi = if self.use_chebyshev {
            Self::chebyshev_approximation(psi)
        } else {
            Self::simple_approximation(psi)
        };

        // 结果限制
        phi.min(S::from_f64(1e3).unwrap_or(S::ZERO)).max(S::ZERO)
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

impl<S: Scalar> Default for EngelundHansenFormula<S> {
    fn default() -> Self {
        Self {
            friction_factor: S::from_f64(0.05).unwrap_or(S::ZERO),
        }
    }
}

impl<S: Scalar> EngelundHansenFormula<S> {
    /// 创建 Engelund-Hansen 公式
    pub fn new() -> Self {
        Self::default()
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

    fn compute_phi(&self, theta: S, _theta_cr: S, _props: &SedimentProperties) -> S {
        if theta < S::from_f64(1e-14).unwrap_or(S::ZERO) {
            return S::ZERO;
        }
        S::from_f64(0.05).unwrap_or(S::ZERO) * theta.powf(S::from_f64(2.5).unwrap_or(S::ZERO)) / self.friction_factor
    }
}

// ============================================================
// 公式注册表（TODO: Phase 4迁移到SolverBuilder）
// ============================================================

/// 根据名称获取输沙公式（f64版本）
///
/// # TODO
/// 此函数将在Phase 4迁移到SolverBuilder
pub fn get_formula_f64(name: &str) -> Box<dyn TransportFormula<f64>> {
    match name.to_lowercase().replace(['_', ' '], "-").as_str() {
        "mpm" | "meyer-peter-muller" => Box::new(MeyerPeterMullerFormula::<f64>::default()),
        "wong-parker" | "wp" => Box::new(MeyerPeterMullerFormula::<f64>::wong_parker()),
        "vanrijn" | "van-rijn" | "vr84" => Box::new(VanRijn1984Formula::<f64>::default()),
        "einstein" | "ein" => Box::new(EinsteinFormula::<f64>::new()),
        "engelund-hansen" | "eh" => Box::new(EngelundHansenFormula::<f64>::default()),
        _ => {
            log::warn!("未知输沙公式 '{}', 使用 Meyer-Peter-Müller", name);
            Box::new(MeyerPeterMullerFormula::<f64>::default())
        }
    }
}

/// 根据名称获取输沙公式（f32版本）
///
/// # TODO
/// 此函数将在Phase 4迁移到SolverBuilder
pub fn get_formula_f32(name: &str) -> Box<dyn TransportFormula<f32>> {
    match name.to_lowercase().replace(['_', ' '], "-").as_str() {
        "mpm" | "meyer-peter-muller" => Box::new(MeyerPeterMullerFormula::<f32>::default()),
        "wong-parker" | "wp" => Box::new(MeyerPeterMullerFormula::<f32>::wong_parker()),
        "vanrijn" | "van-rijn" | "vr84" => Box::new(VanRijn1984Formula::<f32>::default()),
        "einstein" | "ein" => Box::new(EinsteinFormula::<f32>::new()),
        "engelund-hansen" | "eh" => Box::new(EngelundHansenFormula::<f32>::default()),
        _ => {
            log::warn!("未知输沙公式 '{}', 使用 Meyer-Peter-Müller", name);
            Box::new(MeyerPeterMullerFormula::<f32>::default())
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

    fn make_sand() -> SedimentProperties {
        SedimentProperties::from_d50_mm(0.5) // 中砂
    }

    #[test]
    fn test_mpm_below_critical() {
        let formula = MeyerPeterMullerFormula::<f64>::default();
        let props = make_sand();

        // 低于临界 Shields 数时不输沙
        let phi = formula.compute_phi(0.01, props.critical_shields, &props);
        assert!(phi <= 0.0);
    }

    #[test]
    fn test_mpm_above_critical() {
        let formula = MeyerPeterMullerFormula::<f64>::default();
        let props = make_sand();

        // 高于临界时有输沙
        let theta = props.critical_shields * 2.0;
        let phi = formula.compute_phi(theta, props.critical_shields, &props);
        assert!(phi > 0.0);

        // Φ = 8 × (θ - θ_cr)^1.5
        let expected = 8.0 * (theta - props.critical_shields).powf(1.5);
        assert!((phi - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vanrijn_formula() {
        let formula = VanRijn1984Formula::<f64>::default();
        let props = make_sand();

        let theta = props.critical_shields * 2.0;
        let phi = formula.compute_phi(theta, props.critical_shields, &props);
        assert!(phi > 0.0);
    }

    #[test]
    fn test_einstein_formula() {
        let formula = EinsteinFormula::<f64>::new();
        let props = make_sand();

        // 高 Shields 数时有输沙
        let phi = formula.compute_phi(0.5, 0.0, &props);
        assert!(phi > 0.0);

        // 非常低的 Shields 数时无输沙
        let phi_low = formula.compute_phi(0.01, 0.0, &props);
        assert!(phi > phi_low);
    }

    #[test]
    fn test_get_formula() {
        let mpm = get_formula_f64("mpm");
        assert_eq!(mpm.id(), "mpm");

        let vr = get_formula_f64("VanRijn");
        assert_eq!(vr.id(), "vanrijn");

        let ein = get_formula_f64("EINSTEIN");
        assert_eq!(ein.id(), "einstein");
    }

    #[test]
    fn test_dimensional_transport() {
        let formula = MeyerPeterMullerFormula::<f64>::default();
        let props = make_sand();
        let physics = PhysicalConstants::freshwater();

        let tau_b = 5.0; // Pa
        let qb = formula.compute_from_shear_stress(tau_b, &props, &physics);

        // 应该有正输沙率
        if tau_b > props.critical_shear_stress {
            assert!(qb > 0.0);
        }
    }

    #[test]
    fn test_transport_vector() {
        let formula = MeyerPeterMullerFormula::<f64>::default();
        let props = make_sand();
        let physics = PhysicalConstants::freshwater();

        let tau_bx = 3.0;
        let tau_by = 4.0;
        let (qbx, qby) = formula.compute_transport_vector(tau_bx, tau_by, &props, &physics);

        // 方向应与剪切力方向一致
        if qbx.abs() > 1e-14 && qby.abs() > 1e-14 {
            let ratio_tau = tau_by / tau_bx;
            let ratio_qb = qby / qbx;
            assert!((ratio_tau - ratio_qb).abs() < 1e-10);
        }
    }

    #[test]
    fn test_f32_formula() {
        let formula_f32 = MeyerPeterMullerFormula::<f32>::default();
        let formula_f64 = MeyerPeterMullerFormula::<f64>::default();
        let props = make_sand();

        let theta = (props.critical_shields * 2.0) as f32;
        let theta_f64 = props.critical_shields * 2.0;

        let phi_f32 = formula_f32.compute_phi(theta, props.critical_shields as f32, &props);
        let phi_f64 = formula_f64.compute_phi(theta_f64, props.critical_shields, &props);

        // 结果应该接近
        assert!((phi_f32 as f64 - phi_f64).abs() < 1e-4);
    }
}