// crates/mh_physics/src/sources/atmosphere.rs

//! 大气源项
//!
//! 实现风应力和大气压力梯度的源项。
//!
//! # 风应力
//!
//! 风应力公式:
//! ```text
//! τ_w = ρ_air * C_d * |W| * W / ρ_water
//! ```
//!
//! 其中 C_d 是风阻系数，可使用:
//! - Large & Pond (1981)
//! - Wu (1982)
//!
//! # 压力梯度
//!
//! 大气压力梯度产生的流速变化:
//! ```text
//! ∂u/∂t = -(1/ρ) * ∂p/∂x
//! ∂v/∂t = -(1/ρ) * ∂p/∂y
//! ```

use super::traits::{
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterState;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use std::sync::{Arc, RwLock};

// 注意：CpuBackend 已在上方导入

/// 最大风速限制 [m/s]
const MAX_WIND_SPEED: f64 = 100.0;

/// Large and Pond (1981) 风阻系数
///
/// 适用于中等风速（4-25 m/s）
#[inline]
pub fn wind_drag_coefficient_lp81(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 11.0 {
        1.2e-3
    } else if w < 25.0 {
        (0.49 + 0.065 * w) * 1e-3
    } else {
        2.11e-3
    }
}

/// Wu (1982) 风阻系数
///
/// 更通用的公式，适用于更广泛的风速范围
#[inline]
pub fn wind_drag_coefficient_wu82(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    (0.8 + 0.065 * w) * 1e-3
}

/// Garratt (1977) 风阻系数
#[inline]
pub fn wind_drag_coefficient_garratt77(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    (0.75 + 0.067 * w) * 1e-3
}

/// Smith (1980) 风阻系数
#[inline]
pub fn wind_drag_coefficient_smith80(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    (0.63 + 0.066 * w) * 1e-3
}

/// Yelland & Taylor (1996) 风阻系数（低风速改进）
#[inline]
pub fn wind_drag_coefficient_yelland96(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 3.0 {
        1.0e-3
    } else if w < 6.0 {
        let x = w.max(1e-6);
        (0.29 + 3.1 / x + 7.7 / (x * x)) * 1e-3
    } else {
        wind_drag_coefficient_lp81(w)
    }
}

/// COARE 3.0 中性风阻系数近似
#[inline]
pub fn wind_drag_coefficient_coare30(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 10.0 {
        1.14e-3
    } else {
        (0.49 + 0.065 * w) * 1e-3
    }
}

/// COARE 3.5 中性风阻系数近似
#[inline]
pub fn wind_drag_coefficient_coare35(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 6.0 {
        0.92e-3
    } else if w < 20.0 {
        (0.61 + 0.063 * w) * 1e-3
    } else {
        2.0e-3
    }
}

/// 风阻系数计算方法
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum DragCoefficientMethod {
    /// Large & Pond (1981)
    #[default]
    LargePond1981,
    /// Wu (1982)
    Wu1982,
    /// Garratt (1977)
    Garratt1977,
    /// Smith (1980)
    Smith1980,
    /// Yelland & Taylor (1996)
    YellandTaylor1996,
    /// COARE 3.0（中性近似）
    Coare30,
    /// COARE 3.5（中性近似）
    Coare35,
    /// 常数（直接存储 f64）
    Constant(u64), // 使用 u64 存储位模式以保持 Copy + Eq
}

impl DragCoefficientMethod {
    /// 创建常数风阻系数
    pub fn constant(cd: f64) -> Self {
        Self::Constant(cd.to_bits())
    }

    /// 计算风阻系数
    pub fn compute(&self, wind_speed: f64) -> f64 {
        match self {
            Self::LargePond1981 => wind_drag_coefficient_lp81(wind_speed),
            Self::Wu1982 => wind_drag_coefficient_wu82(wind_speed),
            Self::Garratt1977 => wind_drag_coefficient_garratt77(wind_speed),
            Self::Smith1980 => wind_drag_coefficient_smith80(wind_speed),
            Self::YellandTaylor1996 => wind_drag_coefficient_yelland96(wind_speed),
            Self::Coare30 => wind_drag_coefficient_coare30(wind_speed),
            Self::Coare35 => wind_drag_coefficient_coare35(wind_speed),
            Self::Constant(bits) => f64::from_bits(*bits).max(0.0),
        }
    }
}

/// 风应力源项配置
#[derive(Debug, Clone)]
pub struct WindStressConfig {
    /// 是否启用
    pub enabled: bool,
    /// 空气密度 [kg/m³]
    pub rho_air: f64,
    /// 水密度 [kg/m³]
    pub rho_water: f64,
    /// 风阻系数计算方法
    pub drag_method: DragCoefficientMethod,
    /// 风速 x 分量 [m/s]（每个单元）
    pub wind_u: Vec<f64>,
    /// 风速 y 分量 [m/s]（每个单元）
    pub wind_v: Vec<f64>,
    /// 最小水深（小于此值不计算风应力）
    pub h_min: f64,
}

impl WindStressConfig {
    /// 创建新配置
    pub fn new(n_cells: usize, rho_air: f64, rho_water: f64) -> Self {
        Self {
            enabled: true,
            rho_air,
            rho_water,
            drag_method: DragCoefficientMethod::default(),
            wind_u: vec![0.0; n_cells],
            wind_v: vec![0.0; n_cells],
            h_min: 1e-4,
        }
    }

    /// 创建默认参数配置
    pub fn default_config(n_cells: usize) -> Self {
        Self::new(n_cells, 1.225, 1000.0)
    }

    /// 设置均匀风场
    pub fn with_uniform_wind(mut self, wind_u: f64, wind_v: f64) -> Self {
        self.wind_u.fill(wind_u);
        self.wind_v.fill(wind_v);
        self
    }

    /// 设置风阻系数方法
    pub fn with_drag_method(mut self, method: DragCoefficientMethod) -> Self {
        self.drag_method = method;
        self
    }

    /// 更新单元风速
    pub fn set_wind(&mut self, cell: usize, u: f64, v: f64) {
        if cell < self.wind_u.len() {
            self.wind_u[cell] = u;
            self.wind_v[cell] = v;
        }
    }

    /// 更新所有单元风速
    pub fn set_wind_field(&mut self, wind_u: &[f64], wind_v: &[f64]) {
        let n = self.wind_u.len().min(wind_u.len()).min(wind_v.len());
        self.wind_u[..n].copy_from_slice(&wind_u[..n]);
        self.wind_v[..n].copy_from_slice(&wind_v[..n]);
    }

    /// 预计算密度比
    #[inline]
    fn density_ratio(&self) -> f64 {
        self.rho_air / self.rho_water
    }
}

impl<B> SourceTermGeneric<B> for WindStressConfig
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str {
        "WindStress"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let backend = state.backend();
        let h = state.h[cell];
        let h_min = backend.config_scalar(self.h_min, "WindStressConfig.h_min");
        if h < h_min || ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }
        if !self.rho_water.is_finite() || self.rho_water <= 0.0 {
            return SourceContributionGeneric::default();
        }

        let wu = self.wind_u.get(cell).copied().unwrap_or(0.0);
        let wv = self.wind_v.get(cell).copied().unwrap_or(0.0);
        if !wu.is_finite() || !wv.is_finite() {
            return SourceContributionGeneric::default();
        }
        let wind_speed = (wu * wu + wv * wv).sqrt();
        if wind_speed < 1e-10 {
            return SourceContributionGeneric::default();
        }

        let cd = self.drag_method.compute(wind_speed);
        let factor = backend.config_scalar(
            self.density_ratio() * cd * wind_speed,
            "WindStressConfig.factor",
        );
        let wu = backend.config_scalar(wu, "WindStressConfig.wind_u");
        let wv = backend.config_scalar(wv, "WindStressConfig.wind_v");

        SourceContributionGeneric::momentum(factor * wu, factor * wv)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !SourceTermGeneric::<B>::is_enabled(self) {
            return;
        }

        let n_cells = state.n_cells();
        if rhs_h.len() < n_cells {
            rhs_h.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hu.len() < n_cells {
            rhs_hu.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n_cells {
            rhs_hv.resize(n_cells, B::Scalar::ZERO);
        }

        for cell in 0..n_cells {
            let contrib = SourceTermGeneric::<B>::compute_cell(self, cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

/// 可更新的风应力源项（用于运行时强迫更新）
#[derive(Clone)]
pub struct WindStressRuntimeSource {
    config: Arc<RwLock<WindStressConfig>>,
}

impl WindStressRuntimeSource {
    pub fn new(config: Arc<RwLock<WindStressConfig>>) -> Self {
        Self { config }
    }

    pub fn config(&self) -> Arc<RwLock<WindStressConfig>> {
        self.config.clone()
    }
}

impl<B> SourceTermGeneric<B> for WindStressRuntimeSource
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str {
        "WindStress"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.config.read().map(|cfg| cfg.enabled).unwrap_or(false)
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let guard = match self.config.read() {
            Ok(cfg) => cfg,
            Err(_) => return SourceContributionGeneric::default(),
        };
        SourceTermGeneric::compute_cell(&*guard, cell, state, ctx)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        let guard = match self.config.read() {
            Ok(cfg) => cfg,
            Err(_) => return,
        };
        SourceTermGeneric::accumulate(&*guard, state, rhs_h, rhs_hu, rhs_hv, ctx)
    }
}

/// 大气压力梯度源项配置
#[derive(Debug, Clone)]
pub struct PressureGradientConfig {
    /// 是否启用
    pub enabled: bool,
    /// 水密度 [kg/m³]
    pub rho_water: f64,
    /// 压力梯度 x 分量 [Pa/m]（每个单元）
    pub dpdx: Vec<f64>,
    /// 压力梯度 y 分量 [Pa/m]（每个单元）
    pub dpdy: Vec<f64>,
    /// 最小水深
    pub h_min: f64,
}

impl PressureGradientConfig {
    /// 创建新配置
    pub fn new(n_cells: usize, rho_water: f64) -> Self {
        Self {
            enabled: true,
            rho_water,
            dpdx: vec![0.0; n_cells],
            dpdy: vec![0.0; n_cells],
            h_min: 1e-4,
        }
    }

    /// 创建默认参数配置
    pub fn default_config(n_cells: usize) -> Self {
        Self::new(n_cells, 1000.0)
    }

    /// 设置均匀压力梯度
    pub fn with_uniform_gradient(mut self, dpdx: f64, dpdy: f64) -> Self {
        self.dpdx.fill(dpdx);
        self.dpdy.fill(dpdy);
        self
    }

    /// 更新压力梯度场
    pub fn set_gradient_field(&mut self, dpdx: &[f64], dpdy: &[f64]) {
        let n = self.dpdx.len().min(dpdx.len()).min(dpdy.len());
        self.dpdx[..n].copy_from_slice(&dpdx[..n]);
        self.dpdy[..n].copy_from_slice(&dpdy[..n]);
    }
}

impl<B> SourceTermGeneric<B> for PressureGradientConfig
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str {
        "PressureGradient"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let backend = state.backend();
        let h = state.h[cell];
        let h_min = backend.config_scalar(self.h_min, "PressureGradientConfig.h_min");

        if h < h_min || ctx.is_dry(h) {
            return SourceContributionGeneric::default();
        }
        if !self.rho_water.is_finite() || self.rho_water <= 0.0 {
            return SourceContributionGeneric::default();
        }

        let dpdx = self.dpdx.get(cell).copied().unwrap_or(0.0);
        let dpdy = self.dpdy.get(cell).copied().unwrap_or(0.0);
        if !dpdx.is_finite() || !dpdy.is_finite() {
            return SourceContributionGeneric::default();
        }

        let rho_water = backend.config_scalar(
            self.rho_water,
            "PressureGradientConfig.rho_water",
        );
        let dpdx = backend.config_scalar(dpdx, "PressureGradientConfig.dpdx");
        let dpdy = backend.config_scalar(dpdy, "PressureGradientConfig.dpdy");
        let factor = -h / rho_water;
        SourceContributionGeneric::momentum(factor * dpdx, factor * dpdy)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !SourceTermGeneric::<B>::is_enabled(self) {
            return;
        }

        let n_cells = state.n_cells();
        if rhs_h.len() < n_cells {
            rhs_h.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hu.len() < n_cells {
            rhs_hu.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n_cells {
            rhs_hv.resize(n_cells, B::Scalar::ZERO);
        }

        for cell in 0..n_cells {
            let contrib = SourceTermGeneric::<B>::compute_cell(self, cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

pub struct WindStressSource;

impl WindStressSource {
    /// 创建新配置
    pub fn new(n_cells: usize, rho_air: f64, rho_water: f64) -> WindStressConfig {
        WindStressConfig::new(n_cells, rho_air, rho_water)
    }

    /// 创建默认配置
    pub fn default_config(n_cells: usize) -> WindStressConfig {
        WindStressConfig::default_config(n_cells)
    }
}

/// 压力梯度便捷构造器
pub struct PressureGradientSource;

impl PressureGradientSource {
    /// 创建新配置
    pub fn new(n_cells: usize, rho_water: f64) -> PressureGradientConfig {
        PressureGradientConfig::new(n_cells, rho_water)
    }

    /// 创建默认配置
    pub fn default_config(n_cells: usize) -> PressureGradientConfig {
        PressureGradientConfig::default_config(n_cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    fn create_test_state(n_cells: usize, h: f64) -> ShallowWaterState<CpuBackend<f64>> {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_wind_drag_lp81_low_speed() {
        let cd = wind_drag_coefficient_lp81(5.0);
        assert!((cd - 1.2e-3).abs() < 1e-10);
    }

    #[test]
    fn test_wind_drag_lp81_medium_speed() {
        let cd = wind_drag_coefficient_lp81(15.0);
        // (0.49 + 0.065 * 15) * 1e-3 = (0.49 + 0.975) * 1e-3 = 1.465e-3
        assert!((cd - 1.465e-3).abs() < 1e-6);
    }

    #[test]
    fn test_wind_drag_lp81_high_speed() {
        let cd = wind_drag_coefficient_lp81(30.0);
        assert!((cd - 2.11e-3).abs() < 1e-10);
    }

    #[test]
    fn test_wind_drag_wu82() {
        let cd = wind_drag_coefficient_wu82(10.0);
        // (0.8 + 0.065 * 10) * 1e-3 = 1.45e-3
        assert!((cd - 1.45e-3).abs() < 1e-6);
    }

    #[test]
    fn test_drag_method_constant() {
        let method = DragCoefficientMethod::constant(0.001);
        let cd = method.compute(10.0);
        assert!((cd - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_wind_stress_config_creation() {
        let config = WindStressConfig::new(10, 1.225, 1000.0);
        assert!(config.enabled);
        assert!((config.rho_air - 1.225).abs() < 1e-10);
        assert!((config.rho_water - 1000.0).abs() < 1e-10);
        assert_eq!(config.wind_u.len(), 10);
    }

    #[test]
    fn test_wind_stress_uniform_wind() {
        let config = WindStressConfig::new(10, 1.225, 1000.0)
            .with_uniform_wind(5.0, 3.0);

        assert!((config.wind_u[0] - 5.0).abs() < 1e-10);
        assert!((config.wind_v[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_wind_stress_compute() {
        let config = WindStressConfig::new(10, 1.225, 1000.0)
            .with_uniform_wind(10.0, 0.0);

        let state = create_test_state(10, 2.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = SourceTermGeneric::compute_cell(&config, 0, &state, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu > 0.0); // 正向风应力
        assert!((contrib.s_hv).abs() < 1e-10); // 无 y 方向风
    }

    #[test]
    fn test_wind_stress_dry_cell() {
        let config = WindStressConfig::new(10, 1.225, 1000.0)
            .with_uniform_wind(10.0, 0.0);

        let state = create_test_state(10, 1e-7);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = SourceTermGeneric::compute_cell(&config, 0, &state, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_pressure_gradient_config_creation() {
        let config = PressureGradientConfig::new(10, 1025.0);
        assert!(config.enabled);
        assert!((config.rho_water - 1025.0).abs() < 1e-10);
    }

    #[test]
    fn test_pressure_gradient_compute() {
        let config = PressureGradientConfig::new(10, 1000.0)
            .with_uniform_gradient(100.0, 0.0); // 100 Pa/m

        let state = create_test_state(10, 2.0);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = SourceTermGeneric::compute_cell(&config, 0, &state, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        // S_hu = -h/ρ * dp/dx = -2.0/1000.0 * 100 = -0.2
        assert!((contrib.s_hu - (-0.2)).abs() < 1e-10);
        assert!((contrib.s_hv).abs() < 1e-10);
    }

    #[test]
    fn test_pressure_gradient_dry_cell() {
        let config = PressureGradientConfig::new(10, 1000.0)
            .with_uniform_gradient(100.0, 50.0);

        let state = create_test_state(10, 1e-7);
        let backend = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend, 0.0, 1.0);

        let contrib = SourceTermGeneric::compute_cell(&config, 0, &state, &ctx);

        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_source_term_trait_wind() {
        let config = WindStressConfig::default_config(10);
        assert_eq!(
            SourceTermGeneric::<CpuBackend<f64>>::name(&config),
            "WindStress"
        );
        assert_eq!(
            SourceTermGeneric::<CpuBackend<f64>>::stiffness(&config),
            SourceStiffness::Explicit
        );
    }

    #[test]
    fn test_source_term_trait_pressure() {
        let config = PressureGradientConfig::default_config(10);
        assert_eq!(
            SourceTermGeneric::<CpuBackend<f64>>::name(&config),
            "PressureGradient"
        );
        assert_eq!(
            SourceTermGeneric::<CpuBackend<f64>>::stiffness(&config),
            SourceStiffness::Explicit
        );
    }
}
