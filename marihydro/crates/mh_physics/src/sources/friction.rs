// crates/mh_physics/src/sources/friction.rs

//! 摩擦源项
//!
//! 实现 Manning 和 Chezy 摩擦模型，采用隐式半解析方法处理。
//!
//! # 算法
//!
//! Manning 摩擦:
//! ```text
//! S_f = g n² |V| / h^(4/3)
//! decay = 1 / (1 + dt * c_f * |V|)
//! ```
//!
//! Chezy 摩擦:
//! ```text
//! S_f = g |V| / C²
//! decay = 1 / (1 + dt * c_f * |V|)
//! ```
//!
//! 使用隐式处理避免大摩擦系数时的数值不稳定。

use super::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;

/// Manning 摩擦配置
#[derive(Debug, Clone)]
pub struct ManningFrictionConfig {
    /// 是否启用
    pub enabled: bool,
    /// 重力加速度 [m/s²]
    pub g: f64,
    /// 预计算 g * n² (均匀场时)
    precomputed_gn2: Option<f64>,
    /// Manning 系数场 [s/m^(1/3)]
    pub manning_n: Vec<f64>,
    /// 摩擦计算的最小水深
    pub h_friction_min: f64,
}

impl ManningFrictionConfig {
    /// 创建均匀 Manning 系数配置
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> Self {
        let gn2 = g * default_n * default_n;
        Self {
            enabled: true,
            g,
            precomputed_gn2: Some(gn2),
            manning_n: vec![default_n; n_cells],
            h_friction_min: 1e-4,
        }
    }

    /// 创建空间变化 Manning 系数配置
    pub fn with_field(g: f64, manning_n: Vec<f64>) -> Self {
        Self {
            enabled: true,
            g,
            precomputed_gn2: None,
            manning_n,
            h_friction_min: 1e-4,
        }
    }

    /// 创建默认配置 (g=9.81, n=0.025)
    pub fn default_config(n_cells: usize) -> Self {
        Self::new(9.81, n_cells, 0.025)
    }

    /// 设置最小摩擦水深
    pub fn with_min_depth(mut self, h_min: f64) -> Self {
        self.h_friction_min = h_min;
        self
    }

    /// 计算摩擦系数 c_f = g n² / h^(1/3)
    #[inline]
    fn compute_cf(&self, h: f64, cell: usize) -> f64 {
        let h_safe = h.max(self.h_friction_min);
        if let Some(gn2) = self.precomputed_gn2 {
            gn2 / h_safe.cbrt()
        } else {
            let n = self.manning_n.get(cell).copied().unwrap_or(0.025);
            self.g * n * n / h_safe.cbrt()
        }
    }
}

impl SourceTerm for ManningFrictionConfig {
    fn name(&self) -> &'static str {
        "ManningFriction"
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
        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;

        // 干单元处理
        if ctx.is_dry(h) {
            return SourceContribution::momentum(-hu / dt, -hv / dt);
        }

        // 计算速度
        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 {
            return SourceContribution::ZERO;
        }

        // 计算摩擦系数
        let cf = self.compute_cf(h, cell);
        let speed = speed_sq.sqrt();

        // 隐式衰减
        let decay = 1.0 / (1.0 + dt * cf * speed);
        let factor = (decay - 1.0) / dt;

        SourceContribution::momentum(hu * factor, hv * factor)
    }

    fn compute_all(
        &self,
        state: &ShallowWaterState,
        ctx: &SourceContext,
        _output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) {
        if !self.is_enabled() {
            return;
        }

        let dt = ctx.dt;
        let n_cells = state.h.len();

        for i in 0..n_cells {
            let h = state.h[i];
            let hu = state.hu[i];
            let hv = state.hv[i];

            if ctx.is_dry(h) {
                output_hu[i] += -hu / dt;
                output_hv[i] += -hv / dt;
                continue;
            }

            let speed_sq = (hu * hu + hv * hv) / (h * h);
            if speed_sq < 1e-20 {
                continue;
            }

            let cf = self.compute_cf(h, i);
            let speed = speed_sq.sqrt();
            let decay = 1.0 / (1.0 + dt * cf * speed);
            let factor = (decay - 1.0) / dt;

            output_hu[i] += hu * factor;
            output_hv[i] += hv * factor;
        }
    }

    fn is_explicit(&self) -> bool {
        false
    }

    fn is_locally_implicit(&self) -> bool {
        true
    }
}

/// Chezy 摩擦配置
#[derive(Debug, Clone)]
pub struct ChezyFrictionConfig {
    /// 是否启用
    pub enabled: bool,
    /// 重力加速度 [m/s²]
    pub g: f64,
    /// Chezy 系数 [m^(1/2)/s]
    pub chezy_c: f64,
    /// 预计算 cf = g / C²
    cf: f64,
}

impl ChezyFrictionConfig {
    /// 创建新的 Chezy 摩擦配置
    pub fn new(g: f64, chezy_c: f64) -> Self {
        let cf = g / (chezy_c * chezy_c);
        Self {
            enabled: true,
            g,
            chezy_c,
            cf,
        }
    }

    /// 创建默认配置 (g=9.81, C=50)
    pub fn default_config() -> Self {
        Self::new(9.81, 50.0)
    }
}

impl SourceTerm for ChezyFrictionConfig {
    fn name(&self) -> &'static str {
        "ChezyFriction"
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
        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;

        // 干单元处理
        if ctx.is_dry(h) {
            return SourceContribution::momentum(-hu / dt, -hv / dt);
        }

        // 计算速度
        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 {
            return SourceContribution::ZERO;
        }

        let speed = speed_sq.sqrt();

        // 隐式衰减
        let decay = 1.0 / (1.0 + dt * self.cf * speed);
        let factor = (decay - 1.0) / dt;

        SourceContribution::momentum(hu * factor, hv * factor)
    }

    fn is_explicit(&self) -> bool {
        false
    }

    fn is_locally_implicit(&self) -> bool {
        true
    }
}

/// 摩擦衰减计算器
///
/// 提供摩擦相关的辅助计算函数。
pub struct FrictionCalculator {
    g: f64,
    h_min: f64,
    h_friction: f64,
}

impl FrictionCalculator {
    /// 创建新的计算器
    pub fn new(g: f64, h_min: f64, h_friction: f64) -> Self {
        Self { g, h_min, h_friction }
    }

    /// 从数值参数创建
    pub fn from_params(g: f64, params: &crate::types::NumericalParams) -> Self {
        Self::new(g, params.h_dry, params.h_dry)
    }

    /// 计算 Manning 摩擦系数
    #[inline]
    pub fn manning_cf(&self, h: f64, n: f64) -> f64 {
        let h_safe = h.max(self.h_friction);
        self.g * n * n / h_safe.cbrt()
    }

    /// 计算 Chezy 摩擦系数
    #[inline]
    pub fn chezy_cf(&self, chezy_c: f64) -> f64 {
        self.g / (chezy_c * chezy_c)
    }

    /// 计算衰减因子
    #[inline]
    pub fn decay_factor(&self, cf: f64, speed: f64, dt: f64) -> f64 {
        1.0 / (1.0 + dt * cf * speed)
    }

    /// 应用隐式摩擦
    #[inline]
    pub fn apply_implicit(&self, hu: f64, hv: f64, h: f64, cf: f64, dt: f64) -> (f64, f64) {
        if h < self.h_min {
            return (0.0, 0.0);
        }

        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 {
            return (hu, hv);
        }

        let speed = speed_sq.sqrt();
        let decay = self.decay_factor(cf, speed, dt);
        (hu * decay, hv * decay)
    }
}

/// Manning 摩擦便捷构造器
pub struct ManningFriction;

impl ManningFriction {
    /// 创建均匀 Manning 系数配置
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> ManningFrictionConfig {
        ManningFrictionConfig::new(g, n_cells, default_n)
    }

    /// 创建空间变化 Manning 系数配置
    pub fn with_field(g: f64, manning_n: Vec<f64>) -> ManningFrictionConfig {
        ManningFrictionConfig::with_field(g, manning_n)
    }

    /// 创建默认配置 (g=9.81, n=0.025)
    pub fn default_config(n_cells: usize) -> ManningFrictionConfig {
        ManningFrictionConfig::default_config(n_cells)
    }
}

/// Chezy 摩擦便捷构造器
pub struct ChezyFriction;

impl ChezyFriction {
    /// 创建 Chezy 摩擦配置
    pub fn new(g: f64, chezy_c: f64) -> ChezyFrictionConfig {
        ChezyFrictionConfig::new(g, chezy_c)
    }

    /// 创建默认配置 (g=9.81, C=50)
    pub fn default_config() -> ChezyFrictionConfig {
        ChezyFrictionConfig::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64, u: f64, v: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.hu[i] = h * u;
            state.hv[i] = h * v;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_manning_config_creation() {
        let config = ManningFrictionConfig::new(9.81, 100, 0.03);
        assert!(config.enabled);
        assert_eq!(config.g, 9.81);
        assert!(config.precomputed_gn2.is_some());
    }

    #[test]
    fn test_manning_dry_cell() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 0.0001, 1.0, 1.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 0.1, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        // 干单元应该衰减动量
        assert!(contrib.s_h.abs() < 1e-10);
        assert!(contrib.s_hu < 0.0); // 负的源项，减少动量
        assert!(contrib.s_hv < 0.0);
    }

    #[test]
    fn test_manning_still_water() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 1.0, 0.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 0.1, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        // 静水应该没有摩擦
        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_manning_flowing_water() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 0.1, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        // 流动水应该有摩擦减速
        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu < 0.0); // x方向减速
        assert!(contrib.s_hv.abs() < 1e-10); // y方向无速度
    }

    #[test]
    fn test_manning_implicit() {
        // 验证隐式处理不会产生负动量
        let config = ManningFrictionConfig::new(9.81, 10, 0.1); // 高摩擦
        let state = create_test_state(10, 0.1, 1.0, 0.5); // 浅水
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params); // 大时间步

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        // 隐式处理应该给出有限的源项
        assert!(contrib.is_valid());
        assert!(contrib.s_hu < 0.0);
    }

    #[test]
    fn test_chezy_config_creation() {
        let config = ChezyFrictionConfig::new(9.81, 50.0);
        assert!(config.enabled);
        assert_eq!(config.chezy_c, 50.0);
        assert!((config.cf - 9.81 / 2500.0).abs() < 1e-10);
    }

    #[test]
    fn test_chezy_flowing_water() {
        let config = ChezyFrictionConfig::new(9.81, 50.0);
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 0.1, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu < 0.0);
    }

    #[test]
    fn test_friction_calculator() {
        let calc = FrictionCalculator::new(9.81, 0.001, 0.001);
        
        let cf = calc.manning_cf(1.0, 0.03);
        assert!(cf > 0.0);
        
        let decay = calc.decay_factor(cf, 1.0, 0.1);
        assert!(decay > 0.0 && decay < 1.0);
    }

    #[test]
    fn test_friction_calculator_apply() {
        let calc = FrictionCalculator::new(9.81, 0.001, 0.001);
        
        let (hu_new, hv_new) = calc.apply_implicit(1.0, 0.5, 1.0, 0.01, 0.1);
        
        // 应该减少但不变号
        assert!(hu_new > 0.0 && hu_new < 1.0);
        assert!(hv_new > 0.0 && hv_new < 0.5);
    }

    #[test]
    fn test_manning_batch_compute() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 1.0, 1.0, 0.5);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 0.1, &params);

        let mut out_h = vec![0.0; 10];
        let mut out_hu = vec![0.0; 10];
        let mut out_hv = vec![0.0; 10];

        config.compute_all(&state, &ctx, &mut out_h, &mut out_hu, &mut out_hv);

        // 所有单元应该有相同的负源项
        for i in 0..10 {
            assert!(out_h[i].abs() < 1e-10);
            assert!(out_hu[i] < 0.0);
            assert!(out_hv[i] < 0.0);
        }
    }

    #[test]
    fn test_source_term_trait() {
        let manning = ManningFrictionConfig::new(9.81, 10, 0.03);
        let chezy = ChezyFrictionConfig::new(9.81, 50.0);

        assert_eq!(manning.name(), "ManningFriction");
        assert_eq!(chezy.name(), "ChezyFriction");

        assert!(!manning.is_explicit());
        assert!(manning.is_locally_implicit());
    }

    #[test]
    fn test_convenience_constructors() {
        let manning = ManningFriction::new(9.81, 100, 0.025);
        assert!(manning.enabled);

        let chezy = ChezyFriction::default_config();
        assert_eq!(chezy.chezy_c, 50.0);
    }
}
