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
//! use mh_physics::sediment::formulas::{get_formula, TransportFormula};
//! use mh_physics::sediment::SedimentProperties;
//!
//! let props = SedimentProperties::from_d50_mm(0.5);
//! let formula = get_formula("mpm");
//!
//! let theta = 0.1;  // Shields 参数
//! let phi = formula.compute_phi(theta, props.critical_shields, &props);
//! ```

use super::properties::SedimentProperties;
use mh_foundation::Scalar;
use serde::{Deserialize, Serialize};

/// 重力加速度
const G: Scalar = 9.81;

/// 输沙公式 trait
///
/// 所有推移质输沙公式的统一接口
pub trait TransportFormula: Send + Sync {
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
    fn compute_phi(&self, theta: Scalar, theta_cr: Scalar, props: &SedimentProperties) -> Scalar;

    /// 计算有量纲输沙率 [m²/s]
    ///
    /// 默认实现：q_b = Φ × √[(s-1)gd³]
    fn compute_dimensional(&self, theta: Scalar, props: &SedimentProperties) -> Scalar {
        let phi = self.compute_phi(theta, props.critical_shields, props);
        if phi <= 0.0 {
            return 0.0;
        }

        let d = props.d50;
        let s = props.relative_density;
        let scale = ((s - 1.0) * G * d * d * d).sqrt();

        phi * scale
    }

    /// 从床面剪切应力计算输沙率
    fn compute_from_shear_stress(&self, tau_b: Scalar, props: &SedimentProperties) -> Scalar {
        let theta = props.shields_number(tau_b);
        self.compute_dimensional(theta, props)
    }

    /// 计算输沙方向向量
    ///
    /// 输沙方向与剪切应力方向一致
    fn compute_transport_vector(
        &self,
        tau_bx: Scalar,
        tau_by: Scalar,
        props: &SedimentProperties,
    ) -> (Scalar, Scalar) {
        let tau_b = (tau_bx * tau_bx + tau_by * tau_by).sqrt();
        if tau_b < 1e-14 {
            return (0.0, 0.0);
        }

        let qb = self.compute_from_shear_stress(tau_b, props);
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
pub struct MeyerPeterMullerFormula {
    /// 公式系数 A（默认 8.0）
    pub coefficient: Scalar,
    /// 指数 n（默认 1.5）
    pub exponent: Scalar,
}

impl Default for MeyerPeterMullerFormula {
    fn default() -> Self {
        Self {
            coefficient: 8.0,
            exponent: 1.5,
        }
    }
}

impl MeyerPeterMullerFormula {
    /// 创建默认参数的 MPM 公式
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置系数
    pub fn with_coefficient(mut self, c: Scalar) -> Self {
        self.coefficient = c;
        self
    }

    /// 设置指数
    pub fn with_exponent(mut self, n: Scalar) -> Self {
        self.exponent = n;
        self
    }

    /// 创建 Wong-Parker (2006) 修正版本
    ///
    /// A = 4.93, n = 1.6，适用于均匀沙
    pub fn wong_parker() -> Self {
        Self {
            coefficient: 4.93,
            exponent: 1.6,
        }
    }
}

impl TransportFormula for MeyerPeterMullerFormula {
    fn name(&self) -> &'static str {
        "Meyer-Peter-Müller"
    }

    fn id(&self) -> &'static str {
        "mpm"
    }

    fn compute_phi(&self, theta: Scalar, theta_cr: Scalar, _props: &SedimentProperties) -> Scalar {
        let excess = theta - theta_cr;
        if excess <= 0.0 {
            return 0.0;
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
pub struct VanRijn1984Formula {
    /// 公式系数（默认 0.053）
    pub coefficient: Scalar,
}

impl Default for VanRijn1984Formula {
    fn default() -> Self {
        Self { coefficient: 0.053 }
    }
}

impl VanRijn1984Formula {
    /// 创建默认参数的 Van Rijn 公式
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置系数
    pub fn with_coefficient(mut self, c: Scalar) -> Self {
        self.coefficient = c;
        self
    }
}

impl TransportFormula for VanRijn1984Formula {
    fn name(&self) -> &'static str {
        "Van Rijn 1984"
    }

    fn id(&self) -> &'static str {
        "vanrijn"
    }

    fn compute_phi(&self, theta: Scalar, theta_cr: Scalar, props: &SedimentProperties) -> Scalar {
        if theta <= theta_cr {
            return 0.0;
        }

        // 输沙强度参数 T
        let t_param = (theta - theta_cr) / theta_cr;

        // 无量纲粒径 D*
        let d_star = props.dimensionless_diameter;

        self.coefficient * t_param.powf(2.1) * d_star.powf(-0.3)
    }
}

// ============================================================
// Einstein (1950)
// ============================================================

/// Einstein (1950) 概率论公式
///
/// 基于颗粒运动概率的经典公式，使用简化的拟合曲线。
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct EinsteinFormula;

impl EinsteinFormula {
    /// 创建 Einstein 公式
    pub fn new() -> Self {
        Self
    }
}

impl TransportFormula for EinsteinFormula {
    fn name(&self) -> &'static str {
        "Einstein"
    }

    fn id(&self) -> &'static str {
        "einstein"
    }

    fn compute_phi(&self, theta: Scalar, _theta_cr: Scalar, _props: &SedimentProperties) -> Scalar {
        if theta < 1e-14 {
            return 0.0;
        }

        // Einstein 参数 ψ = 1/θ
        let psi = 1.0 / theta;

        if psi > 40.0 {
            return 0.0; // 无输沙
        }

        // 简化的 Einstein 曲线近似
        if psi < 2.0 {
            40.0 * (-0.39 * psi).exp()
        } else {
            0.465 * psi.powf(-2.5)
        }
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
pub struct EngelundHansenFormula {
    /// 摩阻系数 f（默认 0.05）
    pub friction_factor: Scalar,
}

impl Default for EngelundHansenFormula {
    fn default() -> Self {
        Self {
            friction_factor: 0.05,
        }
    }
}

impl EngelundHansenFormula {
    /// 创建 Engelund-Hansen 公式
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置摩阻系数
    pub fn with_friction(mut self, f: Scalar) -> Self {
        self.friction_factor = f;
        self
    }
}

impl TransportFormula for EngelundHansenFormula {
    fn name(&self) -> &'static str {
        "Engelund-Hansen"
    }

    fn id(&self) -> &'static str {
        "engelund-hansen"
    }

    fn compute_phi(&self, theta: Scalar, _theta_cr: Scalar, _props: &SedimentProperties) -> Scalar {
        if theta < 1e-14 {
            return 0.0;
        }
        0.05 * theta.powf(2.5) / self.friction_factor
    }
}

// ============================================================
// 公式注册表
// ============================================================

/// 根据名称获取输沙公式
///
/// # 参数
///
/// - `name`: 公式名称或标识符（不区分大小写）
///
/// # 支持的名称
///
/// - "mpm", "meyer-peter-muller": Meyer-Peter-Müller
/// - "vanrijn", "van-rijn", "vr84": Van Rijn 1984
/// - "einstein": Einstein
/// - "engelund-hansen", "eh": Engelund-Hansen
///
/// # 返回
///
/// 对应的公式实现，未知名称返回 MPM 并输出警告
pub fn get_formula(name: &str) -> Box<dyn TransportFormula> {
    match name.to_lowercase().replace(['_', ' '], "-").as_str() {
        "mpm" | "meyer-peter-muller" => Box::new(MeyerPeterMullerFormula::default()),
        "wong-parker" | "wp" => Box::new(MeyerPeterMullerFormula::wong_parker()),
        "vanrijn" | "van-rijn" | "vr84" => Box::new(VanRijn1984Formula::default()),
        "einstein" | "ein" => Box::new(EinsteinFormula::new()),
        "engelund-hansen" | "eh" => Box::new(EngelundHansenFormula::default()),
        _ => {
            log::warn!("Unknown transport formula '{}', using Meyer-Peter-Müller", name);
            Box::new(MeyerPeterMullerFormula::default())
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
        let formula = MeyerPeterMullerFormula::default();
        let props = make_sand();

        // 低于临界 Shields 数时不输沙
        let phi = formula.compute_phi(0.01, props.critical_shields, &props);
        assert!(phi <= 0.0);
    }

    #[test]
    fn test_mpm_above_critical() {
        let formula = MeyerPeterMullerFormula::default();
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
        let formula = VanRijn1984Formula::default();
        let props = make_sand();

        let theta = props.critical_shields * 2.0;
        let phi = formula.compute_phi(theta, props.critical_shields, &props);
        assert!(phi > 0.0);
    }

    #[test]
    fn test_einstein_formula() {
        let formula = EinsteinFormula::new();
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
        let mpm = get_formula("mpm");
        assert_eq!(mpm.id(), "mpm");

        let vr = get_formula("VanRijn");
        assert_eq!(vr.id(), "vanrijn");

        let ein = get_formula("EINSTEIN");
        assert_eq!(ein.id(), "einstein");
    }

    #[test]
    fn test_dimensional_transport() {
        let formula = MeyerPeterMullerFormula::default();
        let props = make_sand();

        let tau_b = 5.0; // Pa
        let qb = formula.compute_from_shear_stress(tau_b, &props);

        // 应该有正输沙率
        if tau_b > props.critical_shear_stress {
            assert!(qb > 0.0);
        }
    }

    #[test]
    fn test_transport_vector() {
        let formula = MeyerPeterMullerFormula::default();
        let props = make_sand();

        let tau_bx = 3.0;
        let tau_by = 4.0;
        let (qbx, qby) = formula.compute_transport_vector(tau_bx, tau_by, &props);

        // 方向应与剪切力方向一致
        if qbx.abs() > 1e-14 && qby.abs() > 1e-14 {
            let ratio_tau = tau_by / tau_bx;
            let ratio_qb = qby / qbx;
            assert!((ratio_tau - ratio_qb).abs() < 1e-10);
        }
    }
}
