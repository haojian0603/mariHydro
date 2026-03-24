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
//! let backend = mh_runtime::CpuBackend::<f64>::new();
//! let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.2);
//! let ws = StokesSettling.compute(&props, &physics);
//! ```

use crate::prelude::*;
use crate::sediment::properties::SedimentPropertiesGeneric;
use crate::types::PhysicalConstants;
use std::marker::PhantomData;

/// 沉降速度公式 trait
pub trait SettlingFormula<S: RuntimeScalar>: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;
    
    /// 计算沉降速度 [m/s]
    fn compute<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S;
}

/// 沉降速度结果
#[derive(Debug, Clone, Copy)]
pub struct SettlingVelocity<S: RuntimeScalar> {
    /// 沉降速度 [m/s]
    pub ws: S,
    /// 使用的公式名称
    pub formula: &'static str,
}

impl<S: RuntimeScalar> SettlingVelocity<S> {
    #[inline]
    fn normalized_d_star<B: Backend<Scalar = S>>(
        backend: &B,
        value: S,
    ) -> S {
        let min_d_star = backend.config_scalar(1e-12, "SettlingVelocity.normalized_d_star");
        if value.is_finite() && value > min_d_star {
            value
        } else {
            min_d_star
        }
    }

    /// 自动选择最佳公式计算沉降速度
    pub fn auto<B: Backend<Scalar = S>>(
        backend: &B,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> Self {
        // 根据无量纲粒径选择公式
        let d_star = SettlingVelocity::normalized_d_star(backend, props.dimensionless_diameter);
        let one = backend.config_scalar(1.0, "SettlingVelocity.auto.one");
        let hundred = backend.config_scalar(100.0, "SettlingVelocity.auto.hundred");
        
        if d_star < one {
            // 细颗粒使用 Stokes
            let formula = StokesSettling::<S>::new();
            Self {
                ws: formula.compute(backend, props, physics),
                formula: formula.name(),
            }
        } else if d_star < hundred {
            // 中等粒径使用 Van Rijn
            let formula = VanRijnSettling::<S>::new();
            Self {
                ws: formula.compute(backend, props, physics),
                formula: formula.name(),
            }
        } else {
            // 粗颗粒使用 Dietrich
            let formula = DietrichSettling::<S>::new();
            Self {
                ws: formula.compute(backend, props, physics),
                formula: formula.name(),
            }
        }
    }
    
    /// 使用指定公式计算
    pub fn with_formula<B: Backend<Scalar = S>, F: SettlingFormula<S>>(
        backend: &B,
        formula: &F,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> Self {
        Self {
            ws: formula.compute(backend, props, physics),
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
pub struct StokesSettling<S: RuntimeScalar> {
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> Default for StokesSettling<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RuntimeScalar> StokesSettling<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: RuntimeScalar> SettlingFormula<S> for StokesSettling<S> {
    fn name(&self) -> &'static str {
        "Stokes"
    }
    
    fn compute<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        let s = props.relative_density;
        let d = props.d50;
        let cfg = |v| backend.config_scalar(v, "StokesSettling.compute");
        let nu = cfg(physics.nu_water);
        let g = cfg(physics.g);
        let eighteen = cfg(18.0);
        let one = S::ONE;
        
        (s - one) * g * d * d / (eighteen * nu)
    }
}

/// Van Rijn (1984) 沉降公式
///
/// 分段公式，适用于广泛粒径范围
#[derive(Debug, Clone, Copy)]
pub struct VanRijnSettling<S: RuntimeScalar> {
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> Default for VanRijnSettling<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RuntimeScalar> VanRijnSettling<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: RuntimeScalar> SettlingFormula<S> for VanRijnSettling<S> {
    fn name(&self) -> &'static str {
        "Van Rijn"
    }
    
    fn compute<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        let s = props.relative_density;
        let d = props.d50;
        let d_star = SettlingVelocity::normalized_d_star(backend, props.dimensionless_diameter);
        let cfg = |v| backend.config_scalar(v, "VanRijnSettling.compute");
        let nu = cfg(physics.nu_water);
        let g = cfg(physics.g);
        let one = S::ONE;
        let d_star_1 = cfg(1.0);
        let d_star_100 = cfg(100.0);
        
        if d_star < d_star_1 {
            // Stokes 区
            let eighteen = cfg(18.0);
            (s - one) * g * d * d / (eighteen * nu)
        } else if d_star <= d_star_100 {
            // 过渡区
            let eighteen = cfg(18.0);
            let ws_stokes = (s - one) * g * d * d / (eighteen * nu);
            let ws_newton = cfg(1.1) * ((s - one) * g * d).sqrt();
            // 线性插值
            let f = (d_star - d_star_1) / cfg(99.0);
            ws_stokes * (one - f) + ws_newton * f
        } else {
            // Newton 区
            cfg(1.1) * ((s - one) * g * d).sqrt()
        }
    }
}

/// Dietrich (1982) 经验沉降公式
///
/// 基于大量实验数据的经验公式
#[derive(Debug, Clone, Copy)]
pub struct DietrichSettling<S: RuntimeScalar> {
    _marker: PhantomData<S>,
}

impl<S: RuntimeScalar> Default for DietrichSettling<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: RuntimeScalar> DietrichSettling<S> {
    pub fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<S: RuntimeScalar> SettlingFormula<S> for DietrichSettling<S> {
    fn name(&self) -> &'static str {
        "Dietrich"
    }
    
    fn compute<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> S {
        let s = props.relative_density;
        let d = props.d50;
        let cfg = |v| backend.config_scalar(v, "DietrichSettling.compute");
        let nu = cfg(physics.nu_water);
        let g = cfg(physics.g);
        let one = S::ONE;
        
        // 无量纲粒径
        let d_star = SettlingVelocity::normalized_d_star(
            backend,
            d * ((s - one) * g / (nu * nu)).powf(one / cfg(3.0)),
        );
        if !d_star.is_finite() {
            return cfg(0.0);
        }
        
        // Dietrich 公式
        let ln_d_star = d_star.ln();
        let ln_d_star_sq = ln_d_star * ln_d_star;
        let ln_d_star_cubed = ln_d_star_sq * ln_d_star;
        let ln_d_star_fourth = ln_d_star_cubed * ln_d_star;
        
        let r1 = cfg(-3.76715)
            + cfg(1.92944) * ln_d_star
            - cfg(0.09815) * ln_d_star_sq
            - cfg(0.00575) * ln_d_star_cubed
            + cfg(0.00056) * ln_d_star_fourth;
        let r2 = (ln_d_star - r1).exp();
        
        // 形状因子修正（球形）
        let csf = cfg(1.0); // 球形 Corey 形状因子
        let tanh_arg = one - (cfg(-0.2) * d_star).exp();
        let r3 = cfg(0.65) - csf / cfg(2.83) * tanh_arg.tanh();
        
        // 修正的 W*
        let w_star = r2 * cfg(10.0).powf(-r3);
        
        // 转换为有量纲速度
        let ws = w_star * ((s - one) * g * nu).powf(one / cfg(3.0));
        
        ws
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
    fn test_stokes_settling() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        let stokes = StokesSettling::<f64>::new();
        
        let ws = stokes.compute(&backend, &props, &physics);
        assert!(ws > 0.0);
        assert!(ws < 1.0); // 沉降速度应该是合理的
    }
    
    #[test]
    fn test_van_rijn_settling() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        let van_rijn = VanRijnSettling::<f64>::new();
        
        let ws = van_rijn.compute(&backend, &props, &physics);
        assert!(ws > 0.0);
        assert!(ws < 1.0);
    }
    
    #[test]
    fn test_dietrich_settling() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        let dietrich = DietrichSettling::<f64>::new();
        
        let ws = dietrich.compute(&backend, &props, &physics);
        assert!(ws > 0.0);
    }
    
    #[test]
    fn test_auto_selection() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = make_props(&backend);
        let physics = make_physics();
        
        let settling = SettlingVelocity::<f64>::auto(&backend, &props, &physics);
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
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.1);
        let physics = make_physics();
        
        let settling = SettlingVelocity::<f64>::auto(&backend, &props, &physics);
        // 应该使用 Van Rijn 或 Stokes
        assert!(settling.formula == "Van Rijn" || settling.formula == "Stokes");
    }
    
    #[test]
    fn test_f32_precision() {
        let backend_f64 = crate::core::CpuBackend::<f64>::new();
        let props_f64 = make_props(&backend_f64);
        let physics = make_physics();
        let backend_f32 = crate::core::CpuBackend::<f32>::new();
        let props_f32 = SedimentPropertiesGeneric::from_d50_mm(&backend_f32, 0.2);
        
        let settling_f32 = SettlingVelocity::<f32>::auto(&backend_f32, &props_f32, &physics);
        let settling_f64 = SettlingVelocity::<f64>::auto(&backend_f64, &props_f64, &physics);
        
        // f32 and f64 results should be close
        assert!((settling_f32.ws as f64 - settling_f64.ws).abs() < 1e-4);
    }
}
