// crates/mh_physics/src/sediment/properties.rs

//! 泥沙物理属性
//!
//! 包含粒径、密度、沉降速度等参数。
//!
//! # 设计说明
//!
//! - **泛型化**: `SedimentPropertiesGeneric<S>` 支持任意精度标量
//! - **配置驱动**: 所有物理常数从 `PhysicalConstants` 获取

use crate::prelude::*;
use serde::{Deserialize, Serialize};

use crate::types::PhysicalConstants;

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
    /// 根据粒径自动分类（mm 为单位）
    ///
    /// # 参数
    /// - `d50_mm`: 中值粒径 [mm]
    pub fn from_diameter(d50_mm: f64) -> Self {
        let d = d50_mm;
        if d < 0.004 {
            Self::Clay
        } else if d < 0.063 {
            Self::Silt
        } else if d < 0.25 {
            Self::FineSand
        } else if d < 0.5 {
            Self::MediumSand
        } else if d < 2.0 {
            Self::CoarseSand
        } else {
            Self::Gravel
        }
    }
    
}

/// 泥沙物理属性（泛型版本）
///
/// # 类型参数
///
/// - `S`: 标量类型，实现 `RuntimeScalar`
///
/// # 示例
///
/// ```ignore
/// use mh_physics::sediment::SedimentPropertiesGeneric;
///
/// // 使用 f64
/// let backend = mh_runtime::CpuBackend::<f64>::new();
/// let props: SedimentPropertiesGeneric<f64> = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.5);
///
/// // 使用 f32
/// let backend_f32 = mh_runtime::CpuBackend::<f32>::new();
/// let props_f32: SedimentPropertiesGeneric<f32> = SedimentPropertiesGeneric::from_d50_mm(&backend_f32, 0.5);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SedimentPropertiesGeneric<S: RuntimeScalar> {
    /// 中值粒径 d50 [m]
    pub d50: S,
    /// 泥沙密度 [kg/m³]
    pub rho_s: S,
    /// 相对密度 s = ρs/ρw
    pub relative_density: S,
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 临界起动剪切应力 [Pa]
    pub critical_shear_stress: S,
    /// 临界希尔兹数
    pub critical_shields: S,
    /// 床面孔隙率
    pub porosity: S,
    /// 静止摩擦角 [度]
    pub angle_of_repose: S,
    /// 无量纲粒径 D*
    pub dimensionless_diameter: S,
}

impl<S: RuntimeScalar> SedimentPropertiesGeneric<S> {
    /// 从 d50 (mm) 创建，自动计算其他属性
    pub fn from_d50_mm<B: Backend<Scalar = S>>(backend: &B, d50_mm: f64) -> Self {
        Self::from_d50_mm_with_physics(backend, d50_mm, &PhysicalConstants::freshwater())
    }

    /// 从 d50 (mm) 创建，使用指定物理常数
    pub fn from_d50_mm_with_physics<B: Backend<Scalar = S>>(
        backend: &B,
        d50_mm: f64,
        physics: &PhysicalConstants,
    ) -> Self {
        let d50 = d50_mm * 1e-3;  // mm -> m
        let rho_s = 2650.0;       // 典型石英密度
        let s = rho_s / physics.rho_water;
        
        // 无量纲粒径
        let d_star = Self::compute_dimensionless_diameter_f64(d50, s, physics);
        
        // 沉降速度
        let ws = Self::compute_settling_velocity_f64(d50, s, d_star, physics);
        
        // 临界希尔兹数
        let theta_cr = Self::compute_critical_shields_f64(d_star);
        
        // 临界剪切应力
        let tau_cr = theta_cr * (rho_s - physics.rho_water) * physics.g * d50;
        
        Self {
            d50: backend.scalar_from_f64(d50),
            rho_s: backend.scalar_from_f64(rho_s),
            relative_density: backend.scalar_from_f64(s),
            settling_velocity: backend.scalar_from_f64(ws),
            critical_shear_stress: backend.scalar_from_f64(tau_cr),
            critical_shields: backend.scalar_from_f64(theta_cr),
            porosity: backend.scalar_from_f64(0.4),
            angle_of_repose: backend.scalar_from_f64(32.0),
            dimensionless_diameter: backend.scalar_from_f64(d_star),
        }
    }

    /// 自定义参数创建
    pub fn custom<B: Backend<Scalar = S>>(backend: &B, d50: f64, rho_s: f64) -> Self {
        Self::custom_with_physics(backend, d50, rho_s, &PhysicalConstants::freshwater())
    }

    /// 自定义参数创建，使用指定物理常数
    pub fn custom_with_physics<B: Backend<Scalar = S>>(
        backend: &B,
        d50: f64,
        rho_s: f64,
        physics: &PhysicalConstants,
    ) -> Self {
        let s = rho_s / physics.rho_water;
        let d_star = Self::compute_dimensionless_diameter_f64(d50, s, physics);
        let ws = Self::compute_settling_velocity_f64(d50, s, d_star, physics);
        let theta_cr = Self::compute_critical_shields_f64(d_star);
        let tau_cr = theta_cr * (rho_s - physics.rho_water) * physics.g * d50;
        
        Self {
            d50: backend.scalar_from_f64(d50),
            rho_s: backend.scalar_from_f64(rho_s),
            relative_density: backend.scalar_from_f64(s),
            settling_velocity: backend.scalar_from_f64(ws),
            critical_shear_stress: backend.scalar_from_f64(tau_cr),
            critical_shields: backend.scalar_from_f64(theta_cr),
            porosity: backend.scalar_from_f64(0.4),
            angle_of_repose: backend.scalar_from_f64(32.0),
            dimensionless_diameter: backend.scalar_from_f64(d_star),
        }
    }

    // ========== 内部 f64 计算方法 (Layer 2) ==========
    
    /// 计算无量纲粒径 D* = d × [(s-1)g/ν²]^(1/3)
    fn compute_dimensionless_diameter_f64(d: f64, s: f64, physics: &PhysicalConstants) -> f64 {
        let factor = (s - 1.0) * physics.g / (physics.nu_water * physics.nu_water);
        d * factor.powf(1.0 / 3.0)
    }

    /// 计算沉降速度 (Van Rijn, 1984)
    fn compute_settling_velocity_f64(d: f64, s: f64, d_star: f64, physics: &PhysicalConstants) -> f64 {
        if d_star < 1.0 {
            // Stokes 沉降
            (s - 1.0) * physics.g * d * d / (18.0 * physics.nu_water)
        } else if d_star <= 100.0 {
            // 过渡区
            let ws_stokes = (s - 1.0) * physics.g * d * d / (18.0 * physics.nu_water);
            let ws_newton = 1.1 * ((s - 1.0) * physics.g * d).sqrt();
            // 插值
            let f = (d_star - 1.0) / 99.0;
            ws_stokes * (1.0 - f) + ws_newton * f
        } else {
            // Newton 沉降
            1.1 * ((s - 1.0) * physics.g * d).sqrt()
        }
    }

    /// 计算临界希尔兹数 (Soulsby-Whitehouse, 1997)
    fn compute_critical_shields_f64(d_star: f64) -> f64 {
        0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - (-0.02 * d_star).exp())
    }

    /// 计算床面剪切应力对应的希尔兹数
    ///
    /// # 注意
    ///
    /// 该方法需要 `PhysicalConstants` 使用相同的标量类型，
    /// 当前 PhysicalConstants 使用 f64，因此需要转换。
    pub fn shields_number<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_b: S,
        physics: &PhysicalConstants,
    ) -> S {
        let rho_w = backend.scalar_from_f64(physics.rho_water);
        let g = backend.scalar_from_f64(physics.g);
        let denom = (self.rho_s - rho_w) * g * self.d50;
        if denom.abs() < S::MIN_POSITIVE {
            return S::ZERO;
        }
        tau_b / denom
    }

    /// 判断是否起动
    pub fn is_mobile<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_b: S,
        physics: &PhysicalConstants,
    ) -> bool {
        let shields = self.shields_number(backend, tau_b, physics);
        shields > self.critical_shields
    }

    /// 获取超临界希尔兹数
    pub fn excess_shields<B: Backend<Scalar = S>>(
        &self,
        backend: &B,
        tau_b: S,
        physics: &PhysicalConstants,
    ) -> S {
        let shields = self.shields_number(backend, tau_b, physics);
        if shields > self.critical_shields {
            shields - self.critical_shields
        } else {
            S::ZERO
        }
    }
}

// ============================================================================
// SedimentClass - 多粒径泥沙级配（泛型版本）
// ============================================================================

/// 多粒径泥沙级配（泛型版本）
///
/// # 类型参数
///
/// - `S`: 标量类型，实现 `RuntimeScalar`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SedimentClassGeneric<S: RuntimeScalar> {
    /// 粒径组（按升序）
    pub sizes: Vec<SedimentPropertiesGeneric<S>>,
    /// 各粒径组的体积分数（和为1）
    pub fractions: Vec<S>,
}

impl<S: RuntimeScalar> SedimentClassGeneric<S> {
    /// 创建单粒径
    pub fn uniform<B: Backend<Scalar = S>>(backend: &B, d50_mm: f64) -> Self {
        Self {
            sizes: vec![SedimentPropertiesGeneric::from_d50_mm(backend, d50_mm)],
            fractions: vec![S::ONE],
        }
    }

    /// 创建多粒径级配
    pub fn graded<B: Backend<Scalar = S>>(backend: &B, sizes_mm: &[f64], fractions: &[f64]) -> Self {
        assert_eq!(sizes_mm.len(), fractions.len());
        let sum: f64 = fractions.iter().sum();
        
        Self {
            sizes: sizes_mm.iter()
                .map(|&d| SedimentPropertiesGeneric::from_d50_mm(backend, d))
                .collect(),
            fractions: fractions.iter()
                .map(|&f| backend.scalar_from_f64(f / sum))
                .collect(),
        }
    }

    /// 获取加权平均d50
    pub fn mean_d50(&self) -> S {
        self.sizes.iter()
            .zip(self.fractions.iter())
            .map(|(s, f)| s.d50 * *f)
            .fold(S::ZERO, |acc, x| acc + x)
    }

    /// 获取粒径组数量
    pub fn n_classes(&self) -> usize {
        self.sizes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sediment_type_classification() {
        assert_eq!(SedimentType::from_diameter(0.001), SedimentType::Clay);
        assert_eq!(SedimentType::from_diameter(0.01), SedimentType::Silt);
        assert_eq!(SedimentType::from_diameter(0.1), SedimentType::FineSand);
        assert_eq!(SedimentType::from_diameter(0.3), SedimentType::MediumSand);
        assert_eq!(SedimentType::from_diameter(1.0), SedimentType::CoarseSand);
        assert_eq!(SedimentType::from_diameter(5.0), SedimentType::Gravel);
    }

    #[test]
    fn test_sediment_properties_from_d50() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.5);
        
        assert!((props.d50 - 0.0005).abs() < 1e-10);
        assert!((props.rho_s - 2650.0).abs() < 1e-10);
        assert!((props.relative_density - 2.65).abs() < 1e-10);
        assert!(props.settling_velocity > 0.0);
        assert!(props.critical_shear_stress > 0.0);
        assert!(props.critical_shields > 0.0);
    }

    #[test]
    fn test_shields_number() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.5);
        let physics = PhysicalConstants::freshwater();
        let tau_b = 1.0; // Pa
        
        let theta = props.shields_number(&backend, tau_b, &physics);
        assert!(theta > 0.0);
    }

    #[test]
    fn test_is_mobile() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.5);
        let physics = PhysicalConstants::freshwater();
        
        // 低剪切应力不起动
        assert!(!props.is_mobile(&backend, 0.01, &physics));
        
        // 高剪切应力起动
        assert!(props.is_mobile(&backend, 10.0, &physics));
    }

    #[test]
    fn test_excess_shields() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let props = SedimentPropertiesGeneric::from_d50_mm(&backend, 0.5);
        let physics = PhysicalConstants::freshwater();
        
        // 低于临界，返回接近0的值
        let excess_low = props.excess_shields(&backend, 0.01, &physics);
        assert!(excess_low.abs() < 1e-10);
        
        // 高于临界，返回正值
        assert!(props.excess_shields(&backend, 10.0, &physics) > 0.0);
    }

    #[test]
    fn test_sediment_class_uniform() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let class = SedimentClassGeneric::uniform(&backend, 0.5);
        
        assert_eq!(class.n_classes(), 1);
        assert!((class.fractions[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sediment_class_graded() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let class = SedimentClassGeneric::graded(
            &backend,
            &[0.1, 0.5, 1.0],
            &[0.3, 0.5, 0.2]
        );
        
        assert_eq!(class.n_classes(), 3);
        
        let sum: f64 = class.fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sediment_class_mean_d50() {
        let backend = mh_runtime::CpuBackend::<f64>::new();
        let class = SedimentClassGeneric::uniform(&backend, 0.5);
        let mean = class.mean_d50();
        
        assert!((mean - 0.0005).abs() < 1e-10);
    }
}
