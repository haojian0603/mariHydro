// crates/mh_physics/src/sediment/suspended/settling.rs

//! 沉降速度计算模块
//!
//! 提供多种沉降速度公式：
//! - Stokes: 低雷诺数（细颗粒）
//! - Van Rijn: 通用分段公式
//!
//! PHYSICS_SOURCE: Stokes 1851, On the effect of the internal friction of fluids on the motion of pendulums; Van Rijn 1984, Sediment transport, part II: suspended load transport.
//! PHYSICS_SCOPE: 主链当前只实现 Stokes 与 Van Rijn 沉降速度关系。Dietrich 公式需要颗粒圆度、Corey 形状因子等额外输入，当前状态结构尚未建模，因此明确不进入主链导出与自动选择。
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
    /// 自动选择最佳公式计算沉降速度
    pub fn auto<B: Backend<Scalar = S>>(
        backend: &B,
        props: &SedimentPropertiesGeneric<S>,
        physics: &PhysicalConstants,
    ) -> Self {
        let fine_limit = backend.config_scalar(100e-6, "SettlingVelocity.auto.fine_limit");

        if props.d50 <= fine_limit {
            // 细颗粒使用 Stokes，Van Rijn 在该粒径段退化到 Stokes 区。
            let formula = StokesSettling::<S>::new();
            Self {
                ws: formula.compute(backend, props, physics),
                formula: formula.name(),
            }
        } else {
            // 其余粒径统一使用 Van Rijn 分段公式，粗颗粒分支同样由该公式覆盖。
            let formula = VanRijnSettling::<S>::new();
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
/// 采用 Van Rijn (1984) 的三段关系：
/// - D <= 100 μm: Stokes 区
/// - 100 μm < D <= 1000 μm: 过渡区
/// - D > 1000 μm: Newton 区
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
        let cfg = |v| backend.config_scalar(v, "VanRijnSettling.compute");
        let nu = cfg(physics.nu_water);
        let g = cfg(physics.g);
        let one = S::ONE;
        let g_prime = (s - one) * g;
        let fine_limit = cfg(100e-6);
        let coarse_limit = cfg(1000e-6);

        if d <= fine_limit {
            // Stokes 区
            let eighteen = cfg(18.0);
            g_prime * d * d / (eighteen * nu)
        } else if d <= coarse_limit {
            // 过渡区，使用 Van Rijn 的显式中段关系。
            let rd = d * (g_prime * d).sqrt() / nu;
            cfg(10.0) * nu / d * ((one + cfg(0.01) * rd * rd).sqrt() - one)
        } else {
            // Newton 区
            cfg(1.1) * (g_prime * d).sqrt()
        }
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
        // 0.1 mm 位于 Stokes/Van Rijn 交界，自动选择应显式落到主链真实公式之一。
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.1);
        let physics = make_physics();
        
        let settling = SettlingVelocity::<f64>::auto(&backend, &props, &physics);
        assert!(settling.formula == "Van Rijn" || settling.formula == "Stokes");
    }

    #[test]
    fn test_coarse_sand_uses_van_rijn_instead_of_dietrich() {
        let backend = crate::core::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 1.5);
        let physics = make_physics();

        let settling = SettlingVelocity::<f64>::auto(&backend, &props, &physics);
        assert_eq!(settling.formula, "Van Rijn");
        assert!(settling.ws > 0.0);
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
