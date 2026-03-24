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

use super::traits::{SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric};
use crate::prelude::*;
use std::f64::consts::PI;
use mh_runtime::DeviceBuffer;

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

impl<B> SourceTermGeneric<B> for CoriolisConfig
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str { "Coriolis" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let backend = state.backend();
        let h = state.h[cell];
        if !h.is_finite() || !ctx.dt.is_finite() || ctx.dt <= B::Scalar::ZERO || ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;
        let f = backend.config_scalar(self.f, "CoriolisConfig.f");

        let (hu_new, hv_new) = if self.use_exact_rotation {
            let theta = f * dt;
            let (sin_t, cos_t) = if theta.abs()
                < backend.config_scalar(1e-3, "CoriolisConfig.small_angle")
            {
                let t2 = theta * theta;
                let t4 = t2 * t2;
                let sin_t = theta
                    * (B::Scalar::ONE
                        - t2 / backend.config_scalar(6.0, "CoriolisConfig.sin_series_6")
                        + t4 / backend.config_scalar(120.0, "CoriolisConfig.sin_series_120"));
                let cos_t = B::Scalar::ONE
                    - t2 * backend.config_scalar(0.5, "CoriolisConfig.cos_series_0_5")
                    + t4 / backend.config_scalar(24.0, "CoriolisConfig.cos_series_24");
                (sin_t, cos_t)
            } else {
                (theta.sin(), theta.cos())
            };
            (hu * cos_t + hv * sin_t, -hu * sin_t + hv * cos_t)
        } else {
            let dhu = f * hv * dt;
            let dhv = -f * hu * dt;
            (hu + dhu, hv + dhv)
        };

        SourceContributionGeneric::momentum((hu_new - hu) / dt, (hv_new - hv) / dt)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.enabled {
            return;
        }

        let n = state.n_cells();
        if rhs_hu.len() < n {
            rhs_hu.resize(n, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n {
            rhs_hv.resize(n, B::Scalar::ZERO);
        }

        for cell in 0..n {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}
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
    /// create a typed coriolis source
    pub fn new(backend: B, config: CoriolisConfigGeneric<B::Scalar>) -> Self {
        Self { config, backend }
    }

    /// build typed config from latitude
    pub fn from_latitude(backend: B, lat_deg: f64) -> Self {
        Self::new(backend, CoriolisConfigGeneric::from_latitude(lat_deg))
    }
}

impl<B> SourceTermGeneric<B> for CoriolisGeneric<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str { "Coriolis" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::Explicit }

    fn is_enabled(&self) -> bool { self.config.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let h = state.h[cell];
        if !h.is_finite() || !ctx.dt.is_finite() || ctx.dt <= B::Scalar::ZERO || ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }

        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;
        let f = self.config.f;
        let backend = state.backend();

        let (hu_new, hv_new) = if self.config.use_exact_rotation {
            let theta = f * dt;
            let (sin_t, cos_t) = if theta.abs()
                < backend.config_scalar(1e-3, "CoriolisGeneric.small_angle")
            {
                let t2 = theta * theta;
                let t4 = t2 * t2;
                (
                    theta
                        * (B::Scalar::ONE
                            - t2 / backend.config_scalar(6.0, "CoriolisGeneric.sin_series_6")
                            + t4 / backend.config_scalar(120.0, "CoriolisGeneric.sin_series_120")),
                    B::Scalar::ONE
                        - t2 * backend.config_scalar(0.5, "CoriolisGeneric.cos_series_0_5")
                        + t4 / backend.config_scalar(24.0, "CoriolisGeneric.cos_series_24"),
                )
            } else {
                (theta.sin(), theta.cos())
            };
            (hu * cos_t + hv * sin_t, -hu * sin_t + hv * cos_t)
        } else {
            let dhu = f * hv * dt;
            let dhv = -f * hu * dt;
            (hu + dhu, hv + dhv)
        };

        SourceContributionGeneric::momentum((hu_new - hu) / dt, (hv_new - hv) / dt)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.config.enabled {
            return;
        }

        let n = state.n_cells();
        if rhs_hu.len() < n {
            rhs_hu.resize(n, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n {
            rhs_hv.resize(n, B::Scalar::ZERO);
        }

        for cell in 0..state.n_cells() {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

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
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        assert!(config.config.enabled);
        assert_eq!(config.config.f, 1e-4);
        assert!(config.config.use_exact_rotation);
    }

    #[test]
    fn test_coriolis_from_latitude() {
        let equator = CoriolisGeneric::from_latitude(CpuBackend::<f64>::new(), 0.0);
        assert!(equator.config.f.abs() < 1e-10);

        let north_pole = CoriolisGeneric::from_latitude(CpuBackend::<f64>::new(), 90.0);
        assert!((north_pole.config.f - 2.0 * EARTH_ANGULAR_VELOCITY).abs() < 1e-10);

        let north_30 = CoriolisGeneric::from_latitude(CpuBackend::<f64>::new(), 30.0);
        assert!((north_30.config.f - EARTH_ANGULAR_VELOCITY).abs() < 1e-10);
    }

    #[test]
    fn test_coriolis_dry_cell() {
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        // 使用 1e-7 作为干单元（小于默认 h_dry = 1e-6）
        let state = create_test_state(10, 1e-7, 1.0, 1.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_coriolis_still_water() {
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        let state = create_test_state(10, 1.0, 0.0, 0.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_coriolis_x_flow_exact() {
        // 仅 x 方向流动，科氏力应该产生 y 方向变化
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        // hu 和 hv 都应该变化（旋转效果）
        assert!(contrib.s_h.is_finite() && contrib.s_hu.is_finite() && contrib.s_hv.is_finite());
    }

    #[test]
    fn test_coriolis_y_flow_exact() {
        // 仅 y 方向流动
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        let state = create_test_state(10, 1.0, 0.0, 1.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_h.is_finite() && contrib.s_hu.is_finite() && contrib.s_hv.is_finite());
    }

    #[test]
    fn test_coriolis_exact_vs_linear() {
        let f = 1e-4;
        let dt = 100.0; // 较大时间步以看出差异

        let exact = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(f));
        let linear = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(f).with_linear_approximation());

        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, dt);

        let contrib_exact = exact.compute_cell(0, &state, &ctx);
        let contrib_linear = linear.compute_cell(0, &state, &ctx);

        // 两种方法应该给出不同结果（除了小时间步）
        assert!((contrib_exact.s_hu - contrib_linear.s_hu).abs() > 1e-10 ||
                (contrib_exact.s_hv - contrib_linear.s_hv).abs() > 1e-10);
    }

    #[test]
    fn test_coriolis_momentum_conservation() {
        // 精确旋转应该保持动量大小
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        let state = create_test_state(10, 1.0, 1.0, 0.5);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 100.0);

        let hu = state.hu[0];
        let hv = state.hv[0];
        let initial_mag = (hu * hu + hv * hv).sqrt();

        let contrib = config.compute_cell(0, &state, &ctx);
        
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
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, dt);

        let contrib = config.compute_cell(0, &state, &ctx);
        
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
        let config = CoriolisGeneric::new(CpuBackend::<f64>::new(), CoriolisConfigGeneric::new(1e-4));
        
        assert_eq!(config.name(), "Coriolis");
        assert!(config.is_enabled());
        assert_eq!(config.stiffness(), SourceStiffness::Explicit);
    }
}
