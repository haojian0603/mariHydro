// marihydro\crates\mh_physics\src\sediment\suspended\resuspension.rs
//! 再悬浮/侵蚀源项
//!
//! 实现床面与水体之间的泥沙交换：
//! - 侵蚀（resuspension/erosion）：床面泥沙进入水体
//! - 沉降（deposition）：悬浮泥沙沉降到床面
//!
//! # 常用公式
//!
//! - Smith & McLean (1977): 适用于河流
//! - Garcia & Parker (1991): 适用于强输沙
//! - Van Rijn (1984): 通用公式

use crate::sediment::properties::SedimentProperties;
use crate::types::PhysicalConstants;

/// 侵蚀公式 trait
pub trait ErosionFormula: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;
    
    /// 计算侵蚀率 E [kg/m²/s]
    ///
    /// # 参数
    /// - `tau_b`: 床面剪切应力 [Pa]
    /// - `tau_cr`: 临界剪切应力 [Pa]
    /// - `props`: 泥沙属性
    /// - `physics`: 物理常数
    fn erosion_rate(&self, tau_b: f64, tau_cr: f64, props: &SedimentProperties, physics: &PhysicalConstants) -> f64;
    
    /// 计算沉降率 D [kg/m²/s]
    ///
    /// D = ws × C_b × ρ_s
    ///
    /// # 参数
    /// - `c_b`: 近底浓度 [kg/m³]
    /// - `ws`: 沉降速度 [m/s]
    fn deposition_rate(&self, c_b: f64, ws: f64) -> f64 {
        ws * c_b
    }
    
    /// 计算净侵蚀/沉降率 [kg/m²/s]
    ///
    /// E - D > 0: 侵蚀主导
    /// E - D < 0: 沉降主导
    fn net_exchange(&self, tau_b: f64, c_b: f64, ws: f64, props: &SedimentProperties, physics: &PhysicalConstants) -> f64 {
        let e = self.erosion_rate(tau_b, props.critical_shear_stress, props, physics);
        let d = self.deposition_rate(c_b, ws);
        e - d
    }
}

/// Smith & McLean (1977) 侵蚀公式
///
/// 适用于河流环境：
/// E = γ₀ × ρ_s × ws × (T / (1 + γ₀ × T))
///
/// 其中 T = (τ_b - τ_cr) / τ_cr
#[derive(Debug, Clone, Copy)]
pub struct SmithMcLean {
    /// 再悬浮系数 γ₀（默认 0.0024）
    pub gamma0: f64,
}

impl Default for SmithMcLean {
    fn default() -> Self {
        Self { gamma0: 0.0024 }
    }
}

impl SmithMcLean {
    /// 创建新实例
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置再悬浮系数
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma0 = gamma;
        self
    }
}

impl ErosionFormula for SmithMcLean {
    fn name(&self) -> &'static str {
        "Smith-McLean"
    }
    
    fn erosion_rate(&self, tau_b: f64, tau_cr: f64, props: &SedimentProperties, _physics: &PhysicalConstants) -> f64 {
        if tau_b <= tau_cr {
            return 0.0;
        }
        
        // 输沙强度参数
        let t_param = (tau_b - tau_cr) / tau_cr;
        
        // 近底参考浓度（体积分数）
        let c_b_vol = self.gamma0 * t_param / (1.0 + self.gamma0 * t_param);
        
        // 转换为质量浓度 [kg/m³]，假设沉降速度约为 ws
        // E = c_b × ws × ρ_s
        let rho_s = props.rho_s;
        let ws = props.settling_velocity;
        
        c_b_vol * rho_s * ws
    }
}

/// Garcia & Parker (1991) 侵蚀公式
///
/// 适用于强输沙河流：
/// E = A × Z^5 / (1 + (A/0.3) × Z^5) × ws × ρ_s
///
/// 其中 Z = u* × Re_p^0.6 / ws
#[derive(Debug, Clone, Copy)]
pub struct GarciaParker {
    /// 公式系数 A（默认 1.3e-7）
    pub coefficient_a: f64,
}

impl Default for GarciaParker {
    fn default() -> Self {
        Self { coefficient_a: 1.3e-7 }
    }
}

impl GarciaParker {
    /// 创建新实例
    pub fn new() -> Self {
        Self::default()
    }
}

impl ErosionFormula for GarciaParker {
    fn name(&self) -> &'static str {
        "Garcia-Parker"
    }
    
    fn erosion_rate(&self, tau_b: f64, tau_cr: f64, props: &SedimentProperties, physics: &PhysicalConstants) -> f64 {
        if tau_b <= tau_cr {
            return 0.0;
        }
        
        // 剪切速度
        let u_star = (tau_b / physics.rho_water).sqrt();
        
        // 颗粒雷诺数
        let re_p = props.d50 * u_star / physics.nu_water;
        
        // 沉降速度
        let ws = props.settling_velocity.max(1e-10);
        
        // Z 参数
        let z = u_star * re_p.powf(0.6) / ws;
        
        // 近底浓度
        let z5 = z.powi(5);
        let c_b = self.coefficient_a * z5 / (1.0 + self.coefficient_a / 0.3 * z5);
        
        // 侵蚀率
        c_b * ws * props.rho_s
    }
}

/// 悬移质源项（综合侵蚀和沉降）
pub struct ResuspensionSource {
    /// 侵蚀公式
    formula: Box<dyn ErosionFormula>,
    /// 泥沙属性
    properties: SedimentProperties,
    /// 沉降速度 [m/s]
    settling_velocity: f64,
}

impl ResuspensionSource {
    /// 创建新的源项计算器
    pub fn new(properties: SedimentProperties) -> Self {
        Self {
            formula: Box::new(SmithMcLean::default()),
            properties: properties.clone(),
            settling_velocity: properties.settling_velocity,
        }
    }
    
    /// 设置侵蚀公式
    pub fn with_formula<F: ErosionFormula + 'static>(mut self, formula: F) -> Self {
        self.formula = Box::new(formula);
        self
    }
    
    /// 设置沉降速度
    pub fn with_settling_velocity(mut self, ws: f64) -> Self {
        self.settling_velocity = ws;
        self
    }
    
    /// 计算单元的源项 [kg/m³/s]
    ///
    /// 正值表示增加（侵蚀），负值表示减少（沉降）
    pub fn compute_source(&self, tau_b: f64, concentration: f64, water_depth: f64, physics: &PhysicalConstants) -> f64 {
        if water_depth < 1e-6 {
            return 0.0;
        }
        
        // 床面交换 [kg/m²/s]
        let net_flux = self.formula.net_exchange(
            tau_b,
            concentration,
            self.settling_velocity,
            &self.properties,
            physics,
        );
        
        // 转换为体积源项 [kg/m³/s]
        net_flux / water_depth
    }
    
    /// 获取沉降速度
    pub fn settling_velocity(&self) -> f64 {
        self.settling_velocity
    }
    
    /// 获取泥沙属性
    pub fn properties(&self) -> &SedimentProperties {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_props() -> SedimentProperties {
        SedimentProperties::from_d50_mm(0.2)
    }
    
    fn make_physics() -> PhysicalConstants {
        PhysicalConstants::freshwater()
    }
    
    #[test]
    fn test_smith_mclean_below_critical() {
        let sm = SmithMcLean::default();
        let props = make_props();
        let physics = make_physics();
        
        // 低于临界应力，无侵蚀
        let e = sm.erosion_rate(0.1, props.critical_shear_stress, &props, &physics);
        assert_eq!(e, 0.0);
    }
    
    #[test]
    fn test_smith_mclean_above_critical() {
        let sm = SmithMcLean::default();
        let props = make_props();
        let physics = make_physics();
        
        // 高于临界应力，有侵蚀
        let e = sm.erosion_rate(2.0, props.critical_shear_stress, &props, &physics);
        assert!(e > 0.0);
    }
    
    #[test]
    fn test_deposition_rate() {
        let sm = SmithMcLean::default();
        
        let c_b = 1.0; // kg/m³
        let ws = 0.01; // m/s
        
        let d = sm.deposition_rate(c_b, ws);
        assert!((d - 0.01).abs() < 1e-10);
    }
    
    #[test]
    fn test_resuspension_source() {
        let props = make_props();
        let physics = make_physics();
        
        let source = ResuspensionSource::new(props.clone());
        
        // 无剪切力时应该是纯沉降（负值）
        let s = source.compute_source(0.0, 1.0, 1.0, &physics);
        assert!(s < 0.0);
        
        // 高剪切力时可能是侵蚀主导（正值）
        let s = source.compute_source(5.0, 0.1, 1.0, &physics);
        // 取决于具体参数，但应该是有限值
        assert!(s.is_finite());
    }
}
