// crates/mh_physics/src/sediment/suspended/settling.rs

//! 沉降速度计算模块
//!
//! 提供多种沉降速度公式：
//! - Stokes: 低雷诺数（细颗粒）
//! - Van Rijn: 通用公式
//! - Dietrich: 经验公式
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::sediment::suspended::settling::{SettlingVelocity, StokesSettling};
//!
//! let props = SedimentProperties::from_d50_mm(0.2);
//! let ws = StokesSettling.compute(&props, &physics);
//! ```

use crate::sediment::properties::SedimentProperties;
use crate::types::PhysicalConstants;
use mh_runtime::RuntimeScalar as Scalar;
use std::marker::PhantomData;

/// 沉降速度公式 trait
pub trait SettlingFormula<S: Scalar>: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;
    
    /// 计算沉降速度 [m/s]
    fn compute(&self, props: &SedimentProperties, physics: &PhysicalConstants) -> S;
}

/// 沉降速度结果
#[derive(Debug, Clone, Copy)]
pub struct SettlingVelocity<S: Scalar> {
    /// 沉降速度 [m/s]
    pub ws: S,
    /// 使用的公式名称
    pub formula: &'static str,
}

impl<S: Scalar> SettlingVelocity<S> {
    /// 自动选择最佳公式计算沉降速度
    pub fn auto(props: &SedimentProperties, physics: &PhysicalConstants) -> Self {
        // 根据无量纲粒径选择公式
        let d_star = props.dimensionless_diameter;
        
        if d_star < 1.0 {
            // 细颗粒使用 Stokes
            let formula = StokesSettling::<S>::new();
            Self {
                ws: formula.compute(props, physics),
                formula: formula.name(),
            }
        } else if d_star < 100.0 {
            // 中等粒径使用 Van Rijn
            let formula = VanRijnSettling::<S>::new();
            Self {
                ws: formula.compute(props, physics),
                formula: formula.name(),
            }
        } else {
            // 粗颗粒使用 Dietrich
            let formula = DietrichSettling::<S>::new();
            Self {
                ws: formula.compute(props, physics),
                formula: formula.name(),
            }
        }
    }
    
    /// 使用指定公式计算
    pub fn with_formula<F: SettlingFormula<S>>(formula: &F, props: &SedimentProperties, physics: &PhysicalConstants) -> Self {
        Self {
            ws: formula.compute(props, physics),
            formula: formula.name(),
        }
    }
    
    /// 直接指定沉降速度
    pub fn fixed(ws: S) -> Self {
        Self {
            ws,
            formula: "fixed",
        }
    }
}

/// Stokes 沉降公式（低雷诺数，细颗粒）
///
/// ws = (s - 1) × g × d² / (18 × ν)
///
/// 适用范围：Re_p < 1，D* < 1
#[derive(Debug, Clone, Copy)]
pub struct StokesSettling<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for StokesSettling<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> StokesSettling<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: Scalar> SettlingFormula<S> for StokesSettling<S> {
    fn name(&self) -> &'static str {
        "Stokes"
    }
    
    fn compute(&self, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        let s = S::from_config(props.relative_density).unwrap_or(S::ZERO);
        let d = S::from_config(props.d50).unwrap_or(S::ZERO);
        let nu = S::from_config(physics.nu_water).unwrap_or(S::ZERO);
        let g = S::from_config(physics.g).unwrap_or(S::ZERO);
        let eighteen = S::from_config(18.0).unwrap_or(S::ZERO);
        let one = S::ONE;
        
        (s - one) * g * d * d / (eighteen * nu)
    }
}

/// Van Rijn (1984) 沉降公式
///
/// 分段公式，适用于广泛粒径范围
#[derive(Debug, Clone, Copy)]
pub struct VanRijnSettling<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for VanRijnSettling<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> VanRijnSettling<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: Scalar> SettlingFormula<S> for VanRijnSettling<S> {
    fn name(&self) -> &'static str {
        "Van Rijn"
    }
    
    fn compute(&self, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        let s = S::from_config(props.relative_density).unwrap_or(S::ZERO);
        let d = S::from_config(props.d50).unwrap_or(S::ZERO);
        let d_star = props.dimensionless_diameter;
        let nu = S::from_config(physics.nu_water).unwrap_or(S::ZERO);
        let g = S::from_config(physics.g).unwrap_or(S::ZERO);
        let one = S::ONE;
        
        if d_star < 1.0 {
            // Stokes 区
            let eighteen = S::from_config(18.0).unwrap_or(S::ZERO);
            (s - one) * g * d * d / (eighteen * nu)
        } else if d_star <= 100.0 {
            // 过渡区
            let eighteen = S::from_config(18.0).unwrap_or(S::ZERO);
            let ws_stokes = (s - one) * g * d * d / (eighteen * nu);
            let ws_newton = S::from_config(1.1).unwrap_or(S::ZERO) * ((s - one) * g * d).sqrt();
            // 线性插值
            let f = S::from_config((d_star - 1.0) / 99.0).unwrap_or(S::ZERO);
            ws_stokes * (one - f) + ws_newton * f
        } else {
            // Newton 区
            S::from_config(1.1).unwrap_or(S::ZERO) * ((s - one) * g * d).sqrt()
        }
    }
}

/// Dietrich (1982) 经验沉降公式
///
/// 基于大量实验数据的经验公式
#[derive(Debug, Clone, Copy)]
pub struct DietrichSettling<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> Default for DietrichSettling<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Scalar> DietrichSettling<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: Scalar> SettlingFormula<S> for DietrichSettling<S> {
    fn name(&self) -> &'static str {
        "Dietrich"
    }
    
    fn compute(&self, props: &SedimentProperties, physics: &PhysicalConstants) -> S {
        let s = S::from_config(props.relative_density).unwrap_or(S::ZERO);
        let d = S::from_config(props.d50).unwrap_or(S::ZERO);
        let nu = S::from_config(physics.nu_water).unwrap_or(S::ZERO);
        let g = S::from_config(physics.g).unwrap_or(S::ZERO);
        let one = S::ONE;
        
        // 无量纲粒径
        let d_star = d * ((s - one) * g / (nu * nu)).powf(one / S::from_config(3.0).unwrap_or(S::ZERO));
        
        // Dietrich 公式
        let ln_d_star = d_star.ln();
        let ln_d_star_sq = ln_d_star * ln_d_star;
        let ln_d_star_cubed = ln_d_star_sq * ln_d_star;
        let ln_d_star_fourth = ln_d_star_cubed * ln_d_star;
        
        let r1 = S::from_config(-3.76715).unwrap_or(S::ZERO) 
            + S::from_config(1.92944).unwrap_or(S::ZERO) * ln_d_star 
            - S::from_config(0.09815).unwrap_or(S::ZERO) * ln_d_star_sq
            - S::from_config(0.00575).unwrap_or(S::ZERO) * ln_d_star_cubed 
            + S::from_config(0.00056).unwrap_or(S::ZERO) * ln_d_star_fourth;
        let r2 = (ln_d_star - r1).exp();
        
        // 形状因子修正（球形）
        let csf = S::from_config(1.0).unwrap_or(S::ZERO); // 球形 Corey 形状因子
        let tanh_arg = one - (S::from_config(-0.2).unwrap_or(S::ZERO) * d_star).exp();
        let r3 = S::from_config(0.65).unwrap_or(S::ZERO) - csf / S::from_config(2.83).unwrap_or(S::ZERO) * tanh_arg.tanh();
        
        // 修正的 W*
        let w_star = r2 * S::from_config(10.0).unwrap_or(S::ZERO).powf(-r3);
        
        // 转换为有量纲速度
        let ws = w_star * ((s - one) * g * nu).powf(one / S::from_config(3.0).unwrap_or(S::ZERO));
        
        ws
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
    fn test_stokes_settling() {
        let props = make_props();
        let physics = make_physics();
        let stokes = StokesSettling::<f64>::new();
        
        let ws = stokes.compute(&props, &physics);
        assert!(ws > 0.0);
        assert!(ws < 1.0); // 沉降速度应该是合理的
    }
    
    #[test]
    fn test_van_rijn_settling() {
        let props = make_props();
        let physics = make_physics();
        let van_rijn = VanRijnSettling::<f64>::new();
        
        let ws = van_rijn.compute(&props, &physics);
        assert!(ws > 0.0);
        assert!(ws < 1.0);
    }
    
    #[test]
    fn test_dietrich_settling() {
        let props = make_props();
        let physics = make_physics();
        let dietrich = DietrichSettling::<f64>::new();
        
        let ws = dietrich.compute(&props, &physics);
        assert!(ws > 0.0);
    }
    
    #[test]
    fn test_auto_selection() {
        let props = make_props();
        let physics = make_physics();
        
        let settling = SettlingVelocity::<f64>::auto(&props, &physics);
        assert!(settling.ws > 0.0);
        assert!(!settling.formula.is_empty());
    }
    
    #[test]
    fn test_fixed_velocity() {
        let settling = SettlingVelocity::<f64>::fixed(0.01);
        assert!((settling.ws - 0.01).abs() < 1e-10);
        assert_eq!(settling.formula, "fixed");
    }
    
    #[test]
    fn test_fine_sand_uses_appropriate_formula() {
        // 细砂 D* < 100
        let props = SedimentProperties::from_d50_mm(0.1);
        let physics = make_physics();
        
        let settling = SettlingVelocity::<f64>::auto(&props, &physics);
        // 应该使用 Van Rijn 或 Stokes
        assert!(settling.formula == "Van Rijn" || settling.formula == "Stokes");
    }
    
    #[test]
    fn test_f32_precision() {
        let props = make_props();
        let physics = make_physics();
        
        let settling_f32 = SettlingVelocity::<f32>::auto(&props, &physics);
        let settling_f64 = SettlingVelocity::<f64>::auto(&props, &physics);
        
        // f32 and f64 results should be close
        assert!((settling_f32.ws as f64 - settling_f64.ws).abs() < 1e-4);
    }
}