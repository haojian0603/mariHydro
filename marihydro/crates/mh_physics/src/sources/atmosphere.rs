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

use super::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;
use glam::DVec2;

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

/// 风阻系数计算方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DragCoefficientMethod {
    /// Large & Pond (1981)
    #[default]
    LargePond1981,
    /// Wu (1982)
    Wu1982,
    /// 常数
    Constant(u32), // 使用 u32 存储 (value * 1e6) 以便 Copy
}

impl DragCoefficientMethod {
    /// 创建常数风阻系数
    pub fn constant(cd: f64) -> Self {
        Self::Constant((cd * 1e6) as u32)
    }

    /// 计算风阻系数
    pub fn compute(&self, wind_speed: f64) -> f64 {
        match self {
            Self::LargePond1981 => wind_drag_coefficient_lp81(wind_speed),
            Self::Wu1982 => wind_drag_coefficient_wu82(wind_speed),
            Self::Constant(cd_scaled) => *cd_scaled as f64 * 1e-6,
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

impl SourceTerm for WindStressConfig {
    fn name(&self) -> &'static str {
        "WindStress"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];

        // 干单元不计算
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let wu = self.wind_u.get(cell).copied().unwrap_or(0.0);
        let wv = self.wind_v.get(cell).copied().unwrap_or(0.0);

        let wind_speed = (wu * wu + wv * wv).sqrt();
        if wind_speed < 1e-10 {
            return SourceContribution::ZERO;
        }

        // 计算风阻系数
        let cd = self.drag_method.compute(wind_speed);

        // 风应力: τ/ρ_water = (ρ_air/ρ_water) * C_d * |W| * W
        let factor = self.density_ratio() * cd * wind_speed;

        SourceContribution::momentum(factor * wu, factor * wv)
    }

    fn is_explicit(&self) -> bool {
        true
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

impl SourceTerm for PressureGradientConfig {
    fn name(&self) -> &'static str {
        "PressureGradient"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];

        // 干单元不计算
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let dpdx = self.dpdx.get(cell).copied().unwrap_or(0.0);
        let dpdy = self.dpdy.get(cell).copied().unwrap_or(0.0);

        // 加速度: a = -(1/ρ) * ∇p
        // 动量源项: S = h * a = -h/ρ * ∇p
        let factor = -h / self.rho_water;

        SourceContribution::momentum(factor * dpdx, factor * dpdy)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

/// 风应力便捷构造器
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
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
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
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu > 0.0); // 正向风应力
        assert!((contrib.s_hv).abs() < 1e-10); // 无 y 方向风
    }

    #[test]
    fn test_wind_stress_dry_cell() {
        let config = WindStressConfig::new(10, 1.225, 1000.0)
            .with_uniform_wind(10.0, 0.0);

        let state = create_test_state(10, 1e-7);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

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
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

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
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_source_term_trait_wind() {
        let config = WindStressConfig::default_config(10);
        assert_eq!(config.name(), "WindStress");
        assert!(config.is_explicit());
    }

    #[test]
    fn test_source_term_trait_pressure() {
        let config = PressureGradientConfig::default_config(10);
        assert_eq!(config.name(), "PressureGradient");
        assert!(config.is_explicit());
    }
}
