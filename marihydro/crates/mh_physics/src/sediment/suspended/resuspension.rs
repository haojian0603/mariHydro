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

use crate::prelude::*;
use crate::sediment::properties::SedimentPropertiesGeneric;
use crate::types::PhysicalConstants;

/// 侵蚀公式 trait
pub trait ErosionFormula<S: RuntimeScalar>: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;
    
    /// 计算侵蚀率 E [kg/m²/s]
    ///
    /// # 参数
    /// - `tau_b`: 床面剪切应力 [Pa]
    /// - `tau_cr`: 临界剪切应力 [Pa]
    /// - `props`: 泥沙属性（配置参数，f64存储）
    /// - `physics`: 物理常数
    fn erosion_rate<B: Backend<Scalar = S>>(
        &self,
        _backend: &B,
        tau_b: S,
        tau_cr: S,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S;
    
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
    fn net_exchange<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_b: S,
        c_b: S,
        ws: S,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        let tau_cr = props.critical_shear_stress;
        let e = self.erosion_rate(backend, tau_b, tau_cr, props, physics);
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
pub struct SmithMcLean<S: RuntimeScalar> {
    /// 再悬浮系数 γ₀（默认 0.0024）
    pub gamma0: S,
}

impl<S: RuntimeScalar> SmithMcLean<S> {
    /// 创建新实例
    pub fn new<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            gamma0: backend.config_scalar(0.0024, "SmithMcLean.gamma0"),
        }
    }
    
    /// 设置再悬浮系数
    pub fn with_gamma(mut self, gamma: S) -> Self {
        self.gamma0 = gamma;
        self
    }
}

impl<S: RuntimeScalar> ErosionFormula<S> for SmithMcLean<S> {
    fn name(&self) -> &'static str {
        "Smith-McLean"
    }
    
    fn erosion_rate<B: Backend<Scalar = S>>(
        &self,
        _backend: &B,
        tau_b: S,
        tau_cr: S,
        props: &SedimentPropertiesGeneric<S>,
        _physics: &PhysicalConstants,
    ) -> S {
        if tau_b <= tau_cr {
            return S::ZERO;
        }
        
        // 输沙强度参数
        let t_param = (tau_b - tau_cr) / tau_cr;
        
        // 近底参考浓度（体积分数）
        let c_b_vol = self.gamma0 * t_param / (S::ONE + self.gamma0 * t_param);
        
        // 转换为质量浓度 [kg/m³]
        // 注：props.rho_s为f64配置参数，运行时转换
        let rho_s = props.rho_s;
        let ws = props.settling_velocity;
        
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
pub struct GarciaParker<S: RuntimeScalar> {
    /// 公式系数 A（默认 1.3e-7）
    pub coefficient_a: S,
}

impl<S: RuntimeScalar> GarciaParker<S> {
    /// 创建新实例
    pub fn new<B: Backend<Scalar = S>>(backend: &B) -> Self {
        Self {
            coefficient_a: backend.config_scalar(1.3e-7, "GarciaParker.coefficient_a"),
        }
    }
}

impl<S: RuntimeScalar> ErosionFormula<S> for GarciaParker<S> {
    fn name(&self) -> &'static str {
        "Garcia-Parker"
    }
    
    fn erosion_rate<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_b: S,
        tau_cr: S,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        if tau_b <= tau_cr {
            return S::ZERO;
        }
        let cfg = |v| backend.config_scalar(v, "GarciaParker.erosion_rate");
        
        // 剪切速度
        let rho_water = cfg(physics.rho_water);
        let u_star = (tau_b / rho_water).sqrt();
        
        // 颗粒雷诺数
        let d50 = props.d50;
        let nu_water = cfg(physics.nu_water);
        let re_p = d50 * u_star / nu_water;
        
        // 沉降速度（确保不为零）
        let ws = props.settling_velocity.max(cfg(1e-10));
        
        // Z 参数
        let z = u_star * re_p.powf(cfg(0.6)) / ws;
        
        // 近底浓度
        let z5 = z.powi(5);
        let c_b = self.coefficient_a * z5
            / (S::ONE + self.coefficient_a / cfg(0.3) * z5);
        
        // 侵蚀率
        let rho_s = props.rho_s;
        c_b * ws * rho_s
    }
}

// ============================================================
// 悬移质源项（综合侵蚀和沉降）
// ============================================================

/// 悬移质源项（完全泛型化）
pub struct ResuspensionSourceGeneric<B: Backend, F: ErosionFormula<B::Scalar>> {
    /// 侵蚀公式
    formula: F,
    /// 泥沙属性（配置参数，f64存储）
    properties: SedimentPropertiesGeneric<B::Scalar>,
    /// 沉降速度 [m/s]
    settling_velocity: B::Scalar,
    /// 后端
    backend: B,
}

impl<B: Backend> ResuspensionSourceGeneric<B, SmithMcLean<B::Scalar>> {
    /// 创建新的源项计算器
    pub fn new(backend: B, properties: SedimentPropertiesGeneric<B::Scalar>) -> Self {
        let settling_velocity = properties.settling_velocity;
        let formula = SmithMcLean::new(&backend);
        Self {
            formula,
            properties: properties.clone(),
            settling_velocity,
            backend,
        }
    }
}

impl<B: Backend, F: ErosionFormula<B::Scalar>> ResuspensionSourceGeneric<B, F> {
    /// 设置侵蚀公式
    pub fn with_formula<F2: ErosionFormula<B::Scalar>>(self, formula: F2) -> ResuspensionSourceGeneric<B, F2> {
        ResuspensionSourceGeneric {
            formula,
            properties: self.properties,
            settling_velocity: self.settling_velocity,
            backend: self.backend,
        }
    }

    /// 设置沉降速度
    pub fn with_settling_velocity(mut self, ws: B::Scalar) -> Self {
        self.settling_velocity = ws;
        self
    } 

    /// 计算单元的源项 [kg/m³/s]
    ///
    /// 正值表示增加（侵蚀），负值表示减少（沉降）
    pub fn compute_source(
        &self,
        tau_b: B::Scalar,
        concentration: B::Scalar,
        water_depth: B::Scalar,
        physics: &PhysicalConstants,
    ) -> B::Scalar {
        if water_depth
            < self
                .backend
                .config_scalar(1e-6, "ResuspensionSourceGeneric.compute_source.min_depth")
        {
            return B::Scalar::ZERO;
        }
        
        // 床面交换 [kg/m²/s]
        let net_flux = self.formula.net_exchange(
            &self.backend,
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
    pub fn settling_velocity(&self) -> B::Scalar {
        self.settling_velocity
    }
    
    /// 获取泥沙属性引用
    pub fn properties(&self) -> &SedimentPropertiesGeneric<B::Scalar> {
        &self.properties
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_props<B: Backend<Scalar = f64>>(backend: &B) -> SedimentPropertiesGeneric<f64> {
        SedimentPropertiesGeneric::from_d50_mm(backend, 0.2)
    }
    
    fn make_physics() -> PhysicalConstants {
        PhysicalConstants::freshwater()
    }
    
    #[test]
    fn test_smith_mclean_below_critical() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let sm: SmithMcLean<f64> = SmithMcLean::new(&backend);
        let props = make_props(&backend);
        let physics = make_physics();
        
        // 低于临界应力，无侵蚀
        let e = sm.erosion_rate(&backend, 0.1, props.critical_shear_stress, &props, &physics);
        assert_eq!(e, 0.0);
    }
    
    #[test]
    fn test_smith_mclean_above_critical() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let sm: SmithMcLean<f64> = SmithMcLean::new(&backend);
        let props = make_props(&backend);
        let physics = make_physics();
        
        // 高于临界应力，有侵蚀
        let e = sm.erosion_rate(&backend, 2.0, props.critical_shear_stress, &props, &physics);
        assert!(e > 0.0);
    }
    
    #[test]
    fn test_deposition_rate() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let sm: SmithMcLean<f64> = SmithMcLean::new(&backend);
        
        let c_b = 1.0; // kg/m³
        let ws = 0.01; // m/s
        
        let d = sm.deposition_rate(c_b, ws);
        assert!((d - 0.01).abs() < 1e-10);
    }
    
    #[test]
    fn test_resuspension_source() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        
        let source = ResuspensionSourceGeneric::<crate::core::CpuBackend<f64>, SmithMcLean<f64>>::new(
            backend,
            props.clone(),
        );
        
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
        let backend_f32 = crate::core::CpuBackend::<f32>::new();
        let backend_f64 = crate::core::CpuBackend::<f64>::new();
        let props_f64 = make_props(&backend_f64);
        let props_f32 = SedimentPropertiesGeneric::from_d50_mm(&backend_f32, 0.2);
        let physics = make_physics();
        
        let source_f32 = ResuspensionSourceGeneric::<crate::core::CpuBackend<f32>, SmithMcLean<f32>>::new(
            backend_f32,
            props_f32,
        );
        let source_f64 = ResuspensionSourceGeneric::<crate::core::CpuBackend<f64>, SmithMcLean<f64>>::new(
            backend_f64,
            props_f64.clone(),
        );
        
        let tau_b = 2.0;
        let conc = 0.5;
        let depth = 1.0;
        
        let s_f32 = source_f32.compute_source(
            tau_b as f32,
            conc as f32,
            depth as f32,
            &physics,
        );
        let s_f64 = source_f64.compute_source(tau_b, conc, depth, &physics);
        
        // f32和f64结果应接近
        assert!((s_f32 as f64 - s_f64).abs() < 1e-3);
    }
}




