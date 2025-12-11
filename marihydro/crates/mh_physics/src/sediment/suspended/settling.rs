//! 沉降速度公式
//!
//! 实现常用沉降速度公式：
//! - Stokes (细颗粒, Re < 1)
//! - Dietrich (中等颗粒)  
//! - Van Rijn (粗颗粒)
//!
//! # 沉降速度公式选择
//!
//! | 粒径范围 | 推荐公式 |
//! |---------|---------|
//! | d < 0.1 mm | Stokes |
//! | 0.1 - 1 mm | Dietrich |
//! | d > 1 mm | Van Rijn |

use crate::sediment::properties::SedimentProperties;
use crate::types::PhysicalConstants;

/// 沉降速度公式 trait
pub trait SettlingFormula: Send + Sync {
    /// 公式名称
    fn name(&self) -> &'static str;
    
    /// 计算沉降速度 [m/s]
    ///
    /// # 参数
    /// - `d`: 颗粒直径 [m]
    /// - `rho_s`: 泥沙密度 [kg/m³]
    /// - `rho_w`: 水密度 [kg/m³]
    /// - `nu`: 运动粘度 [m²/s]
    /// - `g`: 重力加速度 [m/s²]
    fn compute(&self, d: f64, rho_s: f64, rho_w: f64, nu: f64, g: f64) -> f64;
    
    /// 从泥沙属性和物理常数计算
    fn compute_from_props(&self, props: &SedimentProperties, physics: &PhysicalConstants) -> f64 {
        self.compute(
            props.d50,
            props.rho_s,
            physics.rho_water,
            physics.nu_water,
            physics.g,
        )
    }
}

/// 沉降速度计算器（自动选择公式）
#[derive(Debug, Clone)]
pub struct SettlingVelocity {
    /// 预计算的沉降速度 [m/s]
    pub ws: f64,
    /// 使用的公式名称
    pub formula_used: &'static str,
}

impl SettlingVelocity {
    /// 自动选择公式计算沉降速度
    pub fn auto(props: &SedimentProperties, physics: &PhysicalConstants) -> Self {
        let d = props.d50;
        
        // 根据粒径选择公式
        let (ws, formula) = if d < 0.0001 {
            // d < 0.1 mm: Stokes
            let stokes = StokesSettling;
            (stokes.compute_from_props(props, physics), stokes.name())
        } else if d < 0.001 {
            // 0.1 - 1 mm: Dietrich
            let dietrich = DietrichSettling::default();
            (dietrich.compute_from_props(props, physics), dietrich.name())
        } else {
            // d > 1 mm: Van Rijn
            let vanrijn = VanRijnSettling;
            (vanrijn.compute_from_props(props, physics), vanrijn.name())
        };
        
        Self { ws, formula_used: formula }
    }
    
    /// 使用指定公式计算
    pub fn with_formula<F: SettlingFormula>(formula: &F, props: &SedimentProperties, physics: &PhysicalConstants) -> Self {
        Self {
            ws: formula.compute_from_props(props, physics),
            formula_used: formula.name(),
        }
    }
}

/// Stokes 沉降公式（细颗粒）
///
/// ws = (s-1) g d² / (18 ν)
///
/// 适用于 Re_p < 1 的细颗粒
#[derive(Debug, Clone, Copy, Default)]
pub struct StokesSettling;

impl SettlingFormula for StokesSettling {
    fn name(&self) -> &'static str {
        "Stokes"
    }
    
    fn compute(&self, d: f64, rho_s: f64, rho_w: f64, nu: f64, g: f64) -> f64 {
        let s = rho_s / rho_w;
        (s - 1.0) * g * d * d / (18.0 * nu)
    }
}

/// Dietrich (1982) 沉降公式
///
/// 使用无量纲粒径 D* 的经验公式，适用范围广
#[derive(Debug, Clone, Copy)]
pub struct DietrichSettling {
    /// 形状因子（球形=1.0，天然砂≈0.7）
    pub shape_factor: f64,
}

impl Default for DietrichSettling {
    fn default() -> Self {
        Self { shape_factor: 0.7 }
    }
}

impl SettlingFormula for DietrichSettling {
    fn name(&self) -> &'static str {
        "Dietrich"
    }
    
    fn compute(&self, d: f64, rho_s: f64, rho_w: f64, nu: f64, g: f64) -> f64 {
        let s = rho_s / rho_w;
        
        // 无量纲粒径
        let d_star = d * ((s - 1.0) * g / (nu * nu)).powf(1.0 / 3.0);
        
        // Dietrich 经验公式
        let r1 = -3.76715 + 1.92944 * d_star.ln() - 0.09815 * d_star.ln().powi(2)
                 - 0.00575 * d_star.ln().powi(3) + 0.00056 * d_star.ln().powi(4);
        
        let w_star = r1.exp();
        
        // 形状修正
        let w_star_corrected = w_star * self.shape_factor;
        
        // 转换为有量纲
        w_star_corrected * ((s - 1.0) * g * nu).powf(1.0 / 3.0)
    }
}

/// Van Rijn (1984) 沉降公式
///
/// 使用分段公式，适用于粗颗粒
#[derive(Debug, Clone, Copy, Default)]
pub struct VanRijnSettling;

impl SettlingFormula for VanRijnSettling {
    fn name(&self) -> &'static str {
        "Van Rijn"
    }
    
    fn compute(&self, d: f64, rho_s: f64, rho_w: f64, nu: f64, g: f64) -> f64 {
        let s = rho_s / rho_w;
        let d_star = d * ((s - 1.0) * g / (nu * nu)).powf(1.0 / 3.0);
        
        let ws = if d_star <= 10.0 {
            // d < ~0.1 mm
            (s - 1.0) * g * d * d / (18.0 * nu)
        } else if d_star <= 1000.0 {
            // 0.1 mm < d < ~1 mm
            10.0 * nu / d * ((1.0 + 0.01 * (s - 1.0) * g * d.powi(3) / (nu * nu)).sqrt() - 1.0)
        } else {
            // d > ~1 mm
            1.1 * ((s - 1.0) * g * d).sqrt()
        };
        
        ws.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_sand() -> SedimentProperties {
        SedimentProperties::from_d50_mm(0.2) // 细砂
    }
    
    fn make_physics() -> PhysicalConstants {
        PhysicalConstants::freshwater()
    }
    
    #[test]
    fn test_stokes_settling() {
        let stokes = StokesSettling;
        let props = SedimentProperties::from_d50_mm(0.05); // 很细的砂
        let physics = make_physics();
        
        let ws = stokes.compute_from_props(&props, &physics);
        assert!(ws > 0.0);
        assert!(ws < 0.1); // 沉降速度应该很小
    }
    
    #[test]
    fn test_dietrich_settling() {
        let dietrich = DietrichSettling::default();
        let props = make_sand();
        let physics = make_physics();
        
        let ws = dietrich.compute_from_props(&props, &physics);
        assert!(ws > 0.0);
    }
    
    #[test]
    fn test_vanrijn_settling() {
        let vanrijn = VanRijnSettling;
        let props = SedimentProperties::from_d50_mm(1.0); // 粗砂
        let physics = make_physics();
        
        let ws = vanrijn.compute_from_props(&props, &physics);
        assert!(ws > 0.0);
    }
    
    #[test]
    fn test_auto_selection() {
        let physics = make_physics();
        
        // 细颗粒应该用 Stokes
        let fine = SedimentProperties::from_d50_mm(0.05);
        let sv = SettlingVelocity::auto(&fine, &physics);
        assert_eq!(sv.formula_used, "Stokes");
        
        // 中等颗粒应该用 Dietrich
        let medium = SedimentProperties::from_d50_mm(0.3);
        let sv = SettlingVelocity::auto(&medium, &physics);
        assert_eq!(sv.formula_used, "Dietrich");
        
        // 粗颗粒应该用 Van Rijn
        let coarse = SedimentProperties::from_d50_mm(2.0);
        let sv = SettlingVelocity::auto(&coarse, &physics);
        assert_eq!(sv.formula_used, "Van Rijn");
    }
}
