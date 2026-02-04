// crates/mh_physics/src/sources/coriolis.rs

//! 科氏力源项
//!
//! 实现地球自转产生的科氏力效应。
//!
//! # 算法
//!
//! 科氏参数: f = 2 ω sin(φ)
//! 
//! 精确旋转:
//! ```text
//! [hu']   [cos(fΔt)   sin(fΔt) ] [hu]
//! [hv'] = [-sin(fΔt)  cos(fΔt) ] [hv]
//! ```
//!
//! 线性近似:
//! ```text
//! d(hu)/dt = f hv
//! d(hv)/dt = -f hu
//! ```

use super::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::prelude::*;

// 注意：CpuBackend 已在上方导入

/// 地球角速度 [rad/s]
// ALLOW_F64: 物理常数
pub const EARTH_ANGULAR_VELOCITY: f64 = 7.2921e-5;

/// 科氏力源项配置
#[derive(Debug, Clone)]
pub struct CoriolisConfig {
    /// 是否启用
    pub enabled: bool,
    /// 科氏参数 f = 2ω sin(lat) [rad/s]
    pub f: f64, // ALLOW_F64: Layer 4 配置参数
    /// 是否使用精确旋转（否则使用线性近似）
    pub use_exact_rotation: bool,
}

impl CoriolisConfig {
    /// 创建新的科氏力配置
    ///
    /// # Arguments
    /// * `f` - 科氏参数 [rad/s]
    // ALLOW_F64: 物理参数
    pub fn new(f: f64) -> Self {
        Self {
            enabled: true,
            f,
            use_exact_rotation: true,
        }
    }

    /// 从纬度创建配置
    ///
    /// # Arguments
    /// * `lat_deg` - 纬度 [度]
    // ALLOW_F64: 物理参数
    pub fn from_latitude(lat_deg: f64) -> Self {
        let f = 2.0 * EARTH_ANGULAR_VELOCITY * (lat_deg * std::f64::consts::PI / 180.0).sin();
        Self::new(f)
    }

    /// 禁用精确旋转（使用线性近似）
    pub fn with_linear_approximation(mut self) -> Self {
        self.use_exact_rotation = false;
        self
    }

    /// 设置启用状态
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

impl Default for CoriolisConfig {
    fn default() -> Self {
        // 默认北纬 30 度
        Self::from_latitude(30.0)
    }
}

impl SourceTerm for CoriolisConfig {
    fn name(&self) -> &'static str {
        "Coriolis"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState<CpuBackend<f64>>,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];
        if !h.is_finite() || !ctx.dt.is_finite() || ctx.dt <= 0.0 || ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;

        let (hu_new, hv_new) = if self.use_exact_rotation {
            // 精确旋转
            let theta = self.f * dt;
            let (sin_t, cos_t) = if theta.abs() < 1e-3 {
                // 🔥 增强小角度优化（6阶泰勒展开）
                // sin(x) ≈ x - x³/6 + x⁵/120
                // cos(x) ≈ 1 - x²/2 + x⁴/24
                let t2 = theta * theta;
                let t4 = t2 * t2;
                let sin_t = theta * (1.0 - t2 / 6.0 + t4 / 120.0);
                let cos_t = 1.0 - t2 * 0.5 + t4 / 24.0;
                (sin_t, cos_t)
            } else {
                theta.sin_cos()
            };
            (hu * cos_t + hv * sin_t, -hu * sin_t + hv * cos_t)
        } else {
            // 线性近似
            let dhu = self.f * hv * dt;
            let dhv = -self.f * hu * dt;
            (hu + dhu, hv + dhv)
        };

        // 返回变化率（源项形式）
        SourceContribution::momentum(
            (hu_new - hu) / dt,
            (hv_new - hv) / dt,
        )
    }

    fn is_explicit(&self) -> bool {
        // 科氏力是显式的（不含耗散）
        true
    }
}

/// 科氏力便捷构造器
pub struct CoriolisSource;

impl CoriolisSource {
    /// 创建新的科氏力配置
    // ALLOW_F64: 物理参数
    pub fn new(f: f64) -> CoriolisConfig {
        CoriolisConfig::new(f)
    }

    /// 从纬度创建配置
    // ALLOW_F64: 物理参数
    pub fn from_latitude(lat_deg: f64) -> CoriolisConfig {
        CoriolisConfig::from_latitude(lat_deg)
    }

    /// 创建默认配置
    pub fn default_config() -> CoriolisConfig {
        CoriolisConfig::default()
    }
}

// =============================================================================
// 泛型科氏力源项
// =============================================================================

use super::traits::{
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use mh_runtime::{Backend, RuntimeScalar};

/// 泛型科氏力配置
#[derive(Debug, Clone)]
pub struct CoriolisConfigGeneric<S: RuntimeScalar> {
    /// 是否启用
    pub enabled: bool,
    /// 科氏参数 f = 2ω sin(lat) [rad/s]
    pub f: S,
    /// 是否使用精确旋转
    pub use_exact_rotation: bool,
}

impl<S: RuntimeScalar> CoriolisConfigGeneric<S> {
    /// 创建新的科氏力配置
    pub fn new(f: S) -> Self {
        Self {
            enabled: true,
            f,
            use_exact_rotation: true,
        }
    }

    /// 从纬度创建配置（需要 f64 输入）
    pub fn from_latitude(lat_deg: f64) -> Self {
        let f_f64 = 2.0 * EARTH_ANGULAR_VELOCITY * (lat_deg * PI / 180.0).sin();
        Self::new(S::from_f64(f_f64).unwrap_or(S::ZERO))
    }

    /// 禁用精确旋转
    pub fn with_linear_approximation(mut self) -> Self {
        self.use_exact_rotation = false;
        self
    }
}

impl<S: RuntimeScalar> Default for CoriolisConfigGeneric<S> {
    fn default() -> Self {
        Self::from_latitude(30.0)
    }
}

/// 泛型科氏力源项
pub struct CoriolisGeneric<B: Backend> {
    config: CoriolisConfigGeneric<B::Scalar>,
    #[allow(dead_code)]
    backend: B,
}

impl<B: Backend> CoriolisGeneric<B> {
    /// 创建新的泛型科氏力源项
    pub fn new(backend: B, config: CoriolisConfigGeneric<B::Scalar>) -> Self {
        Self { config, backend }
    }

    /// 从纬度创建
    pub fn from_latitude(backend: B, lat_deg: f64) -> Self {
        Self::new(backend, CoriolisConfigGeneric::from_latitude(lat_deg))
    }
}

// 使用宏生成 f32/f64 实现
macro_rules! impl_coriolis_generic {
    ($scalar:ty) => {
        impl SourceTermGeneric<CpuBackend<$scalar>> for CoriolisGeneric<CpuBackend<$scalar>> {
            fn name(&self) -> &'static str { "Coriolis" }

            fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }

            fn is_enabled(&self) -> bool { self.config.enabled }

            fn compute_cell(
                &self,
                cell: usize,
                state: &ShallowWaterState<CpuBackend<$scalar>>,
                ctx: &SourceContextGeneric<$scalar>,
            ) -> SourceContributionGeneric<$scalar> {
                let h = state.h[cell];
                if !h.is_finite() || !ctx.dt.is_finite() || ctx.dt <= (0.0 as $scalar) || ctx.is_dry(h) {
                    return SourceContributionGeneric::default();
                }

                let hu = state.hu[cell];
                let hv = state.hv[cell];
                let dt = ctx.dt;
                let f = self.config.f;

                let (hu_new, hv_new) = if self.config.use_exact_rotation {
                    let theta = f * dt;
                    // 🔥 增强小角度优化（6阶泰勒展开）
                    let (sin_t, cos_t) = if theta.abs() < (1e-3 as $scalar) {
                        let t2 = theta * theta;
                        let t4 = t2 * t2;
                        // sin(x) ≈ x - x³/6 + x⁵/120
                        // cos(x) ≈ 1 - x²/2 + x⁴/24
                        (theta * ((1.0 as $scalar) - t2 / (6.0 as $scalar) + t4 / (120.0 as $scalar)), 
                         (1.0 as $scalar) - t2 * (0.5 as $scalar) + t4 / (24.0 as $scalar))
                    } else {
                        (theta.sin(), theta.cos())
                    };
                    (hu * cos_t + hv * sin_t, -hu * sin_t + hv * cos_t)
                } else {
                    let dhu = f * hv * dt;
                    let dhv = -f * hu * dt;
                    (hu + dhu, hv + dhv)
                };

                SourceContributionGeneric::momentum(
                    (hu_new - hu) / dt,
                    (hv_new - hv) / dt,
                )
            }

            fn accumulate(
                &self,
                state: &ShallowWaterState<CpuBackend<$scalar>>,
                _rhs_h: &mut Vec<$scalar>,
                rhs_hu: &mut Vec<$scalar>,
                rhs_hv: &mut Vec<$scalar>,
                ctx: &SourceContextGeneric<$scalar>,
            ) {
                if !self.config.enabled {
                    return;
                }

                for cell in 0..state.n_cells() {
                    let contrib = self.compute_cell(cell, state, ctx);
                    rhs_hu[cell] += contrib.s_hu;
                    rhs_hv[cell] += contrib.s_hv;
                }
            }
        }
    };
}

impl_coriolis_generic!(f32);
impl_coriolis_generic!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64, u: f64, v: f64) -> ShallowWaterState<CpuBackend<f64>> {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.hu[i] = h * u;
            state.hv[i] = h * v;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_coriolis_creation() {
        let config = CoriolisConfig::new(1e-4);
        assert!(config.enabled);
        assert_eq!(config.f, 1e-4);
        assert!(config.use_exact_rotation);
    }

    #[test]
    fn test_coriolis_from_latitude() {
        let equator = CoriolisConfig::from_latitude(0.0);
        assert!(equator.f.abs() < 1e-10);

        let north_pole = CoriolisConfig::from_latitude(90.0);
        assert!((north_pole.f - 2.0 * EARTH_ANGULAR_VELOCITY).abs() < 1e-10);

        let north_30 = CoriolisConfig::from_latitude(30.0);
        assert!((north_30.f - EARTH_ANGULAR_VELOCITY).abs() < 1e-10);
    }

    #[test]
    fn test_coriolis_dry_cell() {
        let config = CoriolisConfig::new(1e-4);
        // 使用 1e-7 作为干单元（小于默认 h_dry = 1e-6）
        let state = create_test_state(10, 1e-7, 1.0, 1.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_coriolis_still_water() {
        let config = CoriolisConfig::new(1e-4);
        let state = create_test_state(10, 1.0, 0.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_coriolis_x_flow_exact() {
        // 仅 x 方向流动，科氏力应该产生 y 方向变化
        let config = CoriolisConfig::new(1e-4);
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        // hu 和 hv 都应该变化（旋转效果）
        assert!(contrib.is_valid());
    }

    #[test]
    fn test_coriolis_y_flow_exact() {
        // 仅 y 方向流动
        let config = CoriolisConfig::new(1e-4);
        let state = create_test_state(10, 1.0, 0.0, 1.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.is_valid());
    }

    #[test]
    fn test_coriolis_exact_vs_linear() {
        let f = 1e-4;
        let dt = 100.0; // 较大时间步以看出差异

        let exact = CoriolisConfig::new(f);
        let linear = CoriolisConfig::new(f).with_linear_approximation();

        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, dt, &params);

        let contrib_exact = exact.compute_cell(&state, 0, &ctx);
        let contrib_linear = linear.compute_cell(&state, 0, &ctx);

        // 两种方法应该给出不同结果（除了小时间步）
        assert!((contrib_exact.s_hu - contrib_linear.s_hu).abs() > 1e-10 ||
                (contrib_exact.s_hv - contrib_linear.s_hv).abs() > 1e-10);
    }

    #[test]
    fn test_coriolis_momentum_conservation() {
        // 精确旋转应该保持动量大小
        let config = CoriolisConfig::new(1e-4);
        let state = create_test_state(10, 1.0, 1.0, 0.5);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 100.0, &params);

        let hu = state.hu[0];
        let hv = state.hv[0];
        let initial_mag = (hu * hu + hv * hv).sqrt();

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        let hu_new = hu + contrib.s_hu * ctx.dt;
        let hv_new = hv + contrib.s_hv * ctx.dt;
        let final_mag = (hu_new * hu_new + hv_new * hv_new).sqrt();

        // 动量大小应该保持（精确旋转）
        assert!((final_mag - initial_mag).abs() < 1e-10);
    }

    #[test]
    fn test_coriolis_small_angle() {
        // 测试小角度泰勒展开的准确性
        let f = 1e-4;
        let dt = 0.001; // 非常小的时间步

        let config = CoriolisConfig::new(f);
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, dt, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        // 小角度近似应该接近线性结果
        // dhu/dt ≈ f * hv = 0
        // dhv/dt ≈ -f * hu = -f * 1 = -1e-4
        assert!(contrib.s_hu.abs() < 1e-6);
        assert!((contrib.s_hv + f).abs() < 1e-6);
    }

    #[test]
    fn test_convenience_constructors() {
        let c1 = CoriolisSource::new(1e-4);
        assert_eq!(c1.f, 1e-4);

        let c2 = CoriolisSource::from_latitude(45.0);
        let expected_f = 2.0 * EARTH_ANGULAR_VELOCITY * (45.0 * PI / 180.0).sin();
        assert!((c2.f - expected_f).abs() < 1e-10);
    }

    #[test]
    fn test_source_term_trait() {
        let config = CoriolisConfig::new(1e-4);
        
        assert_eq!(config.name(), "Coriolis");
        assert!(config.is_enabled());
        assert!(config.is_explicit());
        assert!(!config.is_locally_implicit());
    }
}
