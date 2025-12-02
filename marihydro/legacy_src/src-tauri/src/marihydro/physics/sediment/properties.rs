// src-tauri/src/marihydro/physics/sediment/properties.rs
//! 泥沙物理属性
//! 包含粒径、密度、沉降速度等参数

use serde::{Deserialize, Serialize};

/// 重力加速度
const G: f64 = 9.81;
/// 水的运动粘度
const NU_WATER: f64 = 1.0e-6;
/// 水的密度
const RHO_WATER: f64 = 1000.0;

/// 泥沙类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SedimentType {
    /// 粘土 (d < 0.004 mm)
    Clay,
    /// 粉砂 (0.004-0.063 mm)
    Silt,
    /// 细砂 (0.063-0.25 mm)
    FineSand,
    /// 中砂 (0.25-0.5 mm)
    MediumSand,
    /// 粗砂 (0.5-2 mm)
    CoarseSand,
    /// 砾石 (> 2 mm)
    Gravel,
    /// 自定义
    Custom,
}

impl SedimentType {
    /// 根据粒径自动分类
    pub fn from_diameter(d50_mm: f64) -> Self {
        if d50_mm < 0.004 {
            Self::Clay
        } else if d50_mm < 0.063 {
            Self::Silt
        } else if d50_mm < 0.25 {
            Self::FineSand
        } else if d50_mm < 0.5 {
            Self::MediumSand
        } else if d50_mm < 2.0 {
            Self::CoarseSand
        } else {
            Self::Gravel
        }
    }
}

/// 泥沙物理属性
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SedimentProperties {
    /// 中值粒径 d50 [m]
    pub d50: f64,
    /// 泥沙密度 [kg/m³]
    pub rho_s: f64,
    /// 相对密度 s = ρs/ρw
    pub relative_density: f64,
    /// 沉降速度 [m/s]
    pub settling_velocity: f64,
    /// 临界起动剪切应力 [Pa]
    pub critical_shear_stress: f64,
    /// 临界希尔兹数
    pub critical_shields: f64,
    /// 床面孔隙率
    pub porosity: f64,
    /// 静止摩擦角 [度]
    pub angle_of_repose: f64,
    /// 无量纲粒径 D*
    pub dimensionless_diameter: f64,
}

impl SedimentProperties {
    /// 从 d50 (mm) 创建，自动计算其他属性
    pub fn from_d50_mm(d50_mm: f64) -> Self {
        let d50 = d50_mm * 1e-3;  // mm -> m
        let rho_s = 2650.0;       // 典型石英密度
        let s = rho_s / RHO_WATER;
        
        // 无量纲粒径
        let d_star = Self::compute_dimensionless_diameter(d50, s);
        
        // 沉降速度
        let ws = Self::compute_settling_velocity(d50, s, d_star);
        
        // 临界希尔兹数
        let theta_cr = Self::compute_critical_shields(d_star);
        
        // 临界剪切应力
        let tau_cr = theta_cr * (rho_s - RHO_WATER) * G * d50;
        
        Self {
            d50,
            rho_s,
            relative_density: s,
            settling_velocity: ws,
            critical_shear_stress: tau_cr,
            critical_shields: theta_cr,
            porosity: 0.4,
            angle_of_repose: 32.0,
            dimensionless_diameter: d_star,
        }
    }

    /// 自定义参数创建
    pub fn custom(d50: f64, rho_s: f64) -> Self {
        let s = rho_s / RHO_WATER;
        let d_star = Self::compute_dimensionless_diameter(d50, s);
        let ws = Self::compute_settling_velocity(d50, s, d_star);
        let theta_cr = Self::compute_critical_shields(d_star);
        let tau_cr = theta_cr * (rho_s - RHO_WATER) * G * d50;
        
        Self {
            d50,
            rho_s,
            relative_density: s,
            settling_velocity: ws,
            critical_shear_stress: tau_cr,
            critical_shields: theta_cr,
            porosity: 0.4,
            angle_of_repose: 32.0,
            dimensionless_diameter: d_star,
        }
    }

    /// 计算无量纲粒径 D* = d × [(s-1)g/ν²]^(1/3)
    fn compute_dimensionless_diameter(d: f64, s: f64) -> f64 {
        let factor = (s - 1.0) * G / (NU_WATER * NU_WATER);
        d * factor.powf(1.0 / 3.0)
    }

    /// 计算沉降速度 (Van Rijn, 1984)
    fn compute_settling_velocity(d: f64, s: f64, d_star: f64) -> f64 {
        if d_star < 1.0 {
            // Stokes 沉降
            (s - 1.0) * G * d * d / (18.0 * NU_WATER)
        } else if d_star <= 100.0 {
            // 过渡区
            let ws_stokes = (s - 1.0) * G * d * d / (18.0 * NU_WATER);
            let ws_newton = 1.1 * ((s - 1.0) * G * d).sqrt();
            // 插值
            let f = (d_star - 1.0) / 99.0;
            ws_stokes * (1.0 - f) + ws_newton * f
        } else {
            // Newton 沉降
            1.1 * ((s - 1.0) * G * d).sqrt()
        }
    }

    /// 计算临界希尔兹数 (Soulsby-Whitehouse, 1997)
    fn compute_critical_shields(d_star: f64) -> f64 {
        0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - (-0.02 * d_star).exp())
    }

    /// 计算床面剪切应力对应的希尔兹数
    pub fn shields_number(&self, tau_b: f64) -> f64 {
        tau_b / ((self.rho_s - RHO_WATER) * G * self.d50)
    }

    /// 判断是否起动
    pub fn is_mobile(&self, tau_b: f64) -> bool {
        self.shields_number(tau_b) > self.critical_shields
    }

    /// 获取超临界希尔兹数
    pub fn excess_shields(&self, tau_b: f64) -> f64 {
        (self.shields_number(tau_b) - self.critical_shields).max(0.0)
    }
}

/// 多粒径泥沙级配
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SedimentClass {
    /// 粒径组（按升序）
    pub sizes: Vec<SedimentProperties>,
    /// 各粒径组的体积分数（和为1）
    pub fractions: Vec<f64>,
}

impl SedimentClass {
    /// 创建单粒径
    pub fn uniform(d50_mm: f64) -> Self {
        Self {
            sizes: vec![SedimentProperties::from_d50_mm(d50_mm)],
            fractions: vec![1.0],
        }
    }

    /// 创建多粒径级配
    pub fn graded(sizes_mm: &[f64], fractions: &[f64]) -> Self {
        assert_eq!(sizes_mm.len(), fractions.len());
        let sum: f64 = fractions.iter().sum();
        
        Self {
            sizes: sizes_mm.iter().map(|&d| SedimentProperties::from_d50_mm(d)).collect(),
            fractions: fractions.iter().map(|&f| f / sum).collect(),
        }
    }

    /// 获取 d50
    pub fn d50(&self) -> f64 {
        let mut cumulative = 0.0;
        for (props, &frac) in self.sizes.iter().zip(&self.fractions) {
            cumulative += frac;
            if cumulative >= 0.5 {
                return props.d50;
            }
        }
        self.sizes.last().map(|p| p.d50).unwrap_or(0.0)
    }

    /// 获取平均沉降速度
    pub fn mean_settling_velocity(&self) -> f64 {
        self.sizes
            .iter()
            .zip(&self.fractions)
            .map(|(p, &f)| p.settling_velocity * f)
            .sum()
    }

    /// 粒径组数量
    pub fn n_classes(&self) -> usize {
        self.sizes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sediment_classification() {
        assert_eq!(SedimentType::from_diameter(0.001), SedimentType::Clay);
        assert_eq!(SedimentType::from_diameter(0.03), SedimentType::Silt);
        assert_eq!(SedimentType::from_diameter(0.15), SedimentType::FineSand);
        assert_eq!(SedimentType::from_diameter(0.35), SedimentType::MediumSand);
        assert_eq!(SedimentType::from_diameter(1.0), SedimentType::CoarseSand);
        assert_eq!(SedimentType::from_diameter(5.0), SedimentType::Gravel);
    }

    #[test]
    fn test_sediment_properties() {
        let props = SedimentProperties::from_d50_mm(0.2);
        
        assert!((props.d50 - 0.0002).abs() < 1e-10);
        assert!(props.settling_velocity > 0.0);
        assert!(props.critical_shields > 0.0 && props.critical_shields < 0.1);
        assert!(props.dimensionless_diameter > 1.0);
    }

    #[test]
    fn test_shields_number() {
        let props = SedimentProperties::from_d50_mm(0.5);
        
        // 典型床面剪切应力
        let tau = 0.5; // Pa
        let theta = props.shields_number(tau);
        
        // 希尔兹数应该在合理范围内
        assert!(theta > 0.0 && theta < 1.0);
    }

    #[test]
    fn test_graded_sediment() {
        let graded = SedimentClass::graded(&[0.1, 0.2, 0.5], &[0.3, 0.5, 0.2]);
        
        assert_eq!(graded.n_classes(), 3);
        assert!(graded.d50() > 0.0);
        assert!(graded.mean_settling_velocity() > 0.0);
    }
}
