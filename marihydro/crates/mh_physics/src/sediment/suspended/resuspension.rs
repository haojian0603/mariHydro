// crates/mh_physics/src/sediment/suspended/resuspension.rs

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
use mh_core::Scalar;
use std::marker::PhantomData;

/// 侵蚀公式 trait
pub trait ErosionFormula<S: Scalar>: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;
    
    /// 计算侵蚀率 E [kg/m²/s]
    ///
    /// # 参数
    /// - `tau_b`: 床面剪切应力 [Pa]
    /// - `tau_cr`: 临界剪切应力 [Pa]
    /// - `props`: 泥沙属性（配置参数，f64存储）
    /// - `physics`: 物理常数
    fn erosion_rate(&self, tau_b: S, tau_cr: S, props: &SedimentProperties, physics: &PhysicalConstants) -> S;
    
    /// 计算沉降率 D [kg/m²/s]
    ///
    /// D = ws × C_b × ρ_s
    ///
    /// # 参数
    /// - `c_b`: 近底浓度 [kg/m³]
    /// - `ws`: 沉降速度 [m/s]
    fn deposition_rate(&self, c_b: S, ws: S) -> S {
        ws * c_b
    }
    
    /// 计算净侵蚀/沉降率 [kg/m²/s]
    ///
    /// E - D > 0: 侵蚀主导
    /// E - D < 0: 沉降主导
    fn net_exchange(&self, tau_b: S, c_b: S, ws: S, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        let e = self.erosion_rate(tau_b, props.critical_shear_stress, props, physics);
        let d = self.deposition_rate(c_b, ws);
        e - d
    }
}

// ============================================================
// Smith & McLean (1977) 公式
// ============================================================

/// Smith & McLean (1977) 侵蚀公式
///
/// 适用于河流环境：
/// E = γ₀ × ρ_s × ws × (T / (1 + γ₀ × T))
///
/// 其中 T = (τ_b - τ_cr) / τ_cr
#[derive(Debug, Clone, Copy)]
pub struct SmithMcLean<S: Scalar> {
    /// 再悬浮系数 γ₀（默认 0.0024）
    pub gamma0: S,
}

impl<S: Scalar> Default for SmithMcLean<S> {
    fn default() -> Self {
        Self { gamma0: S::from_f64(0.0024) }
    }
}

impl<S: Scalar> SmithMcLean<S> {
    /// 创建新实例
    pub fn new() -> Self {
        Self::default()
    }
    
    /// 设置再悬浮系数
    pub fn with_gamma(mut self, gamma: S) -> Self {
        self.gamma0 = gamma;
        self
    }
}

impl<S: Scalar> ErosionFormula<S> for SmithMcLean<S> {
    fn name(&self) -> &'static str {
        "Smith-McLean"
    }
    
    fn erosion_rate(&self, tau_b: S, tau_cr: S, props: &SedimentProperties, _physics: &PhysicalConstants) -> S {
        if tau_b <= tau_cr {
            return S::ZERO;
        }
        
        // 输沙强度参数
        let t_param = (tau_b - tau_cr) / tau_cr;
        
        // 近底参考浓度（体积分数）
        let c_b_vol = self.gamma0 * t_param / (S::ONE + self.gamma0 * t_param);
        
        // 转换为质量浓度 [kg/m³]
        // 注：props.rho_s为f64配置参数，运行时转换
        let rho_s = S::from_f64(props.rho_s);
        let ws = S::from_f64(props.settling_velocity);
        
        c_b_vol * rho_s * ws
    }
}

// ============================================================
// Garcia & Parker (1991) 公式
// ============================================================

/// Garcia & Parker (1991) 侵蚀公式
///
/// 适用于强输沙河流：
/// E = A × Z^5 / (1 + (A/0.3) × Z^5) × ws × ρ_s
///
/// 其中 Z = u* × Re_p^0.6 / ws
#[derive(Debug, Clone, Copy)]
pub struct GarciaParker<S: Scalar> {
    /// 公式系数 A（默认 1.3e-7）
    pub coefficient_a: S,
}

impl<S: Scalar> Default for GarciaParker<S> {
    fn default() -> Self {
        Self { coefficient_a: S::from_f64(1.3e-7) }
    }
}

impl<S: Scalar> GarciaParker<S> {
    /// 创建新实例
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: Scalar> ErosionFormula<S> for GarciaParker<S> {
    fn name(&self) -> &'static str {
        "Garcia-Parker"
    }
    
    fn erosion_rate(&self, tau_b: S, tau_cr: S, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        if tau_b <= tau_cr {
            return S::ZERO;
        }
        
        // 剪切速度
        let rho_water = S::from_f64(physics.rho_water);
        let u_star = (tau_b / rho_water).sqrt();
        
        // 颗粒雷诺数
        let d50 = S::from_f64(props.d50);
        let nu_water = S::from_f64(physics.nu_water);
        let re_p = d50 * u_star / nu_water;
        
        // 沉降速度（确保不为零）
        let ws = S::from_f64(props.settling_velocity).max(S::from_f64(1e-10));
        
        // Z 参数
        let z = u_star * re_p.powf(S::from_f64(0.6)) / ws;
        
        // 近底浓度
        let z5 = z.powi(5);
        let c_b = self.coefficient_a * z5 / (S::ONE + self.coefficient_a / S::from_f64(0.3) * z5);
        
        // 侵蚀率
        let rho_s = S::from_f64(props.rho_s);
        c_b * ws * rho_s
    }
}

// ============================================================
// 悬移质源项（综合侵蚀和沉降）
// ============================================================

/// 悬移质源项（完全泛型化）
pub struct ResuspensionSource<S: Scalar> {
    /// 侵蚀公式
    formula: Box<dyn ErosionFormula<S>>,
    /// 泥沙属性（配置参数，f64存储）
    properties: SedimentProperties,
    /// 沉降速度 [m/s]
    settling_velocity: S,
    /// 类型标记
    _marker: PhantomData<S>,
}

impl<S: Scalar> ResuspensionSource<S> {
    /// 创建新的源项计算器
    pub fn new(properties: SedimentProperties) -> Self {
        Self {
            formula: Box::new(SmithMcLean::default()),
            properties: properties.clone(),
            settling_velocity: S::from_f64(properties.settling_velocity),
            _marker: PhantomData,
        }
    }
    
    /// 设置侵蚀公式
    pub fn with_formula<F: ErosionFormula<S> + 'static>(mut self, formula: F) -> Self {
        self.formula = Box::new(formula);
        self
    }
    
    /// 设置沉降速度
    pub fn with_settling_velocity(mut self, ws: S) -> Self {
        self.settling_velocity = ws;
        self
    }
    
    /// 计算单元的源项 [kg/m³/s]
    ///
    /// 正值表示增加（侵蚀），负值表示减少（沉降）
    pub fn compute_source(
        &self,
        tau_b: S,
        concentration: S,
        water_depth: S,
        physics: &PhysicalConstants,
    ) -> S {
        if water_depth < S::from_f64(1e-6) {
            return S::ZERO;
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
    pub fn settling_velocity(&self) -> S {
        self.settling_velocity
    }
    
    /// 获取泥沙属性引用
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
        let sm: SmithMcLean<f64> = SmithMcLean::default();
        let props = make_props();
        let physics = make_physics();
        
        // 低于临界应力，无侵蚀
        let e = sm.erosion_rate(0.1, props.critical_shear_stress, &props, &physics);
        assert_eq!(e, 0.0);
    }
    
    #[test]
    fn test_smith_mclean_above_critical() {
        let sm: SmithMcLean<f64> = SmithMcLean::default();
        let props = make_props();
        let physics = make_physics();
        
        // 高于临界应力，有侵蚀
        let e = sm.erosion_rate(2.0, props.critical_shear_stress, &props, &physics);
        assert!(e > 0.0);
    }
    
    #[test]
    fn test_deposition_rate() {
        let sm: SmithMcLean<f64> = SmithMcLean::default();
        
        let c_b = 1.0; // kg/m³
        let ws = 0.01; // m/s
        
        let d = sm.deposition_rate(c_b, ws);
        assert!((d - 0.01).abs() < 1e-10);
    }
    
    #[test]
    fn test_resuspension_source() {
        let props = make_props();
        let physics = make_physics();
        
        let source: ResuspensionSource<f64> = ResuspensionSource::new(props.clone());
        
        // 无剪切力时应该是纯沉降（负值）
        let s = source.compute_source(0.0, 1.0, 1.0, &physics);
        assert!(s < 0.0);
        
        // 高剪切力时可能是侵蚀主导（正值）
        let s = source.compute_source(5.0, 0.1, 1.0, &physics);
        // 取决于具体参数，但应该是有限值
        assert!(s.is_finite());
    }

    #[test]
    fn test_f32_precision() {
        let props = make_props();
        let physics = make_physics();
        
        let source_f32: ResuspensionSource<f32> = ResuspensionSource::new(props.clone());
        let source_f64: ResuspensionSource<f64> = ResuspensionSource::new(props.clone());
        
        let tau_b = 2.0;
        let conc = 0.5;
        let depth = 1.0;
        
        let s_f32 = source_f32.compute_source(
            f32::from_f64(tau_b),
            f32::from_f64(conc),
            f32::from_f64(depth),
            &physics,
        );
        let s_f64 = source_f64.compute_source(tau_b, conc, depth, &physics);
        
        // f32和f64结果应接近
        assert!((s_f32 as f64 - s_f64).abs() < 1e-3);
    }
}




