// crates/mh_physics/src/sources/vegetation.rs

//! 植被阻力源项
//!
//! 实现浅水方程中的植被阻力效应。
//!
//! # 植被阻力模型
//!
//! 植被对水流的阻力使用阻力公式：
//! ```text
//! F_v = 0.5 * ρ * C_d * A_v * |u| * u
//! ```
//!
//! 其中：
//! - C_d 是阻力系数（通常 1.0-1.5）
//! - A_v 是植被投影面积密度 [m²/m³]
//! - u 是流速向量
//!
//! # 植被参数化
//!
//! 提供多种植被模型：
//! - 刚性植被（固定高度）
//! - 柔性植被（随流弯曲）
//! - 淹没/露出植被

use super::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;

/// 植被类型
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum VegetationType {
    /// 无植被
    #[default]
    None,
    /// 刚性植被（如树干、芦苇杆）
    Rigid {
        /// 阻力系数
        cd: f64, // ALLOW_F64: Layer 4 配置参数
        /// 茎直径 [m]
        diameter: f64, // ALLOW_F64: Layer 4 配置参数
        /// 茎密度 [1/m²]
        density: f64, // ALLOW_F64: Layer 4 配置参数
        /// 植被高度 [m]
        height: f64, // ALLOW_F64: Layer 4 配置参数
    },
    /// 柔性植被（如水草）
    Flexible {
        /// 基础阻力系数
        cd_base: f64, // ALLOW_F64: Layer 4 配置参数
        /// 弯曲模量
        flex_modulus: f64, // ALLOW_F64: Layer 4 配置参数
        /// 叶面积指数 [m²/m²]
        lai: f64, // ALLOW_F64: Layer 4 配置参数
        /// 植被高度 [m]
        height: f64, // ALLOW_F64: Layer 4 配置参数
    },
    /// 通用植被（使用体积阻力系数）
    Generic {
        /// 体积阻力系数 [1/m]
        av_cd: f64, // ALLOW_F64: Layer 4 配置参数
        /// 植被高度 [m]
        height: f64, // ALLOW_F64: Layer 4 配置参数
    },
}


impl VegetationType {
    /// 创建刚性植被
    // ALLOW_F64: 物理参数
    pub fn rigid(cd: f64, diameter: f64, density: f64, height: f64) -> Self {
        Self::Rigid {
            cd: cd.max(0.1).min(3.0),
            diameter: diameter.max(0.001),
            density: density.max(0.0),
            height: height.max(0.0),
        }
    }

    /// 创建典型芦苇植被
    pub fn reed() -> Self {
        Self::rigid(1.2, 0.01, 50.0, 2.0) // 直径1cm，50根/m²，高2m
    }

    /// 创建典型红树林
    pub fn mangrove() -> Self {
        Self::rigid(1.0, 0.05, 10.0, 3.0) // 直径5cm，10根/m²，高3m
    }

    /// 创建柔性水草
    // ALLOW_F64: 物理参数
    pub fn flexible(cd_base: f64, lai: f64, height: f64) -> Self {
        Self::Flexible {
            cd_base: cd_base.max(0.1),
            flex_modulus: 1.0,
            lai: lai.max(0.0),
            height: height.max(0.0),
        }
    }

    /// 创建通用植被
    // ALLOW_F64: 物理参数
    pub fn generic(av_cd: f64, height: f64) -> Self {
        Self::Generic {
            av_cd: av_cd.max(0.0),
            height: height.max(0.0),
        }
    }

    /// 计算有效阻力系数 × 投影面积密度
    ///
    /// 返回 C_d * A_v [1/m]
    // ALLOW_F64: 源项计算
    pub fn effective_drag(&self, water_depth: f64, velocity: f64) -> f64 {
        match *self {
            Self::None => 0.0,
            Self::Rigid { cd, diameter, density, height } => {
                // 植被高度可能被水深限制
                let effective_height = height.min(water_depth);
                if effective_height <= 0.0 {
                    return 0.0;
                }
                // A_v = 直径 × 密度 / 高度（在水深范围内的部分）
                let av = diameter * density * effective_height / water_depth;
                cd * av
            }
            Self::Flexible { cd_base, flex_modulus, lai, height } => {
                let effective_height = height.min(water_depth);
                if effective_height <= 0.0 {
                    return 0.0;
                }
                // 柔性植被的阻力随流速降低（弯曲效应）
                let bend_factor = 1.0 / (1.0 + flex_modulus * velocity.abs());
                let av = lai * effective_height / water_depth;
                cd_base * bend_factor * av
            }
            Self::Generic { av_cd, height } => {
                let effective_height = height.min(water_depth);
                if effective_height <= 0.0 {
                    return 0.0;
                }
                av_cd * effective_height / water_depth
            }
        }
    }

    /// 获取植被高度
    // ALLOW_F64: 源项计算
    pub fn height(&self) -> f64 {
        match *self {
            Self::None => 0.0,
            Self::Rigid { height, .. } => height,
            Self::Flexible { height, .. } => height,
            Self::Generic { height, .. } => height,
        }
    }

    /// 是否为淹没植被
    // ALLOW_F64: 与 ConservedState 配合
    pub fn is_submerged(&self, water_depth: f64) -> bool {
        water_depth >= self.height()
    }
}

/// 植被阻力源项配置
#[derive(Debug, Clone)]
pub struct VegetationConfig {
    /// 是否启用
    pub enabled: bool,
    /// 每个单元的植被类型
    pub vegetation: Vec<VegetationType>,
    /// 水密度 [kg/m³]
    pub rho_water: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小流速（避免除零）
    pub vel_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl VegetationConfig {
    /// 创建新配置
    // ALLOW_F64: 物理参数
    pub fn new(n_cells: usize, rho_water: f64) -> Self {
        Self {
            enabled: true,
            vegetation: vec![VegetationType::None; n_cells],
            rho_water,
            h_min: 1e-4,
            vel_min: 1e-6,
        }
    }

    /// 创建默认配置
    pub fn default_config(n_cells: usize) -> Self {
        Self::new(n_cells, 1000.0)
    }

    /// 设置单元植被
    pub fn set_vegetation(&mut self, cell: usize, veg: VegetationType) {
        if cell < self.vegetation.len() {
            self.vegetation[cell] = veg;
        }
    }

    /// 设置区域植被（所有单元相同）
    pub fn with_uniform_vegetation(mut self, veg: VegetationType) -> Self {
        self.vegetation.fill(veg);
        self
    }

    /// 设置芦苇区域
    pub fn with_reed_zone(mut self, cells: &[usize]) -> Self {
        let veg = VegetationType::reed();
        for &cell in cells {
            if cell < self.vegetation.len() {
                self.vegetation[cell] = veg;
            }
        }
        self
    }

    /// 设置红树林区域
    pub fn with_mangrove_zone(mut self, cells: &[usize]) -> Self {
        let veg = VegetationType::mangrove();
        for &cell in cells {
            if cell < self.vegetation.len() {
                self.vegetation[cell] = veg;
            }
        }
        self
    }
}

impl SourceTerm for VegetationConfig {
    fn name(&self) -> &'static str {
        "Vegetation"
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

        let veg = self.vegetation.get(cell).copied().unwrap_or(VegetationType::None);
        if matches!(veg, VegetationType::None) {
            return SourceContribution::ZERO;
        }

        let u = state.hu[cell] / h;
        let v = state.hv[cell] / h;
        let vel = (u * u + v * v).sqrt();

        if vel < self.vel_min {
            return SourceContribution::ZERO;
        }

        // 计算有效阻力
        let cd_av = veg.effective_drag(h, vel);
        if cd_av <= 0.0 {
            return SourceContribution::ZERO;
        }

        // 植被阻力: F = -0.5 * C_d * A_v * |u| * u
        // 这是单位体积的阻力，需要乘以水深得到单位面积的阻力
        // S_hu = -0.5 * C_d * A_v * h * |u| * u
        let factor = -0.5 * cd_av * h * vel;

        SourceContribution::momentum(factor * u, factor * v)
    }

    fn is_explicit(&self) -> bool {
        // 植被阻力可能很大，使用隐式处理更稳定
        false
    }
}

/// 植被阻力便捷构造器
pub struct VegetationSource;

impl VegetationSource {
    /// 创建新配置
    // ALLOW_F64: 物理参数
    pub fn new(n_cells: usize, rho_water: f64) -> VegetationConfig {
        VegetationConfig::new(n_cells, rho_water)
    }

    /// 创建默认配置
    pub fn default_config(n_cells: usize) -> VegetationConfig {
        VegetationConfig::default_config(n_cells)
    }
}

/// 植被阻力隐式处理
///
/// 使用半隐式方法处理植被阻力，避免数值不稳定
#[derive(Debug, Clone)]
pub struct VegetationImplicit {
    /// 配置
    pub config: VegetationConfig,
    /// 阻力衰减因子（预计算）
    decay_factors: Vec<f64>, // ALLOW_F64: 源项计算
}

impl VegetationImplicit {
    /// 创建新实例
    pub fn new(config: VegetationConfig) -> Self {
        let n = config.vegetation.len();
        Self {
            config,
            decay_factors: vec![0.0; n],
        }
    }

    /// 计算衰减因子
    ///
    /// 返回 exp(-Δt * 0.5 * C_d * A_v * |u|)
    // ALLOW_F64: 时间参数与模拟进度配合
    pub fn compute_decay_factors(&mut self, state: &ShallowWaterState, dt: f64) {
        let n = self.decay_factors.len().min(state.h.len());

        for i in 0..n {
            let h = state.h[i];
            if h < self.config.h_min {
                self.decay_factors[i] = 1.0;
                continue;
            }

            let veg = self.config.vegetation[i];
            if matches!(veg, VegetationType::None) {
                self.decay_factors[i] = 1.0;
                continue;
            }

            let u = state.hu[i] / h;
            let v = state.hv[i] / h;
            let vel = (u * u + v * v).sqrt();

            let cd_av = veg.effective_drag(h, vel);
            let decay_rate = 0.5 * cd_av * vel;

            self.decay_factors[i] = (-dt * decay_rate).exp().max(0.0).min(1.0);
        }
    }

    /// 应用隐式衰减
    pub fn apply_decay(&self, state: &mut ShallowWaterState) {
        let n = self.decay_factors.len().min(state.h.len());

        for i in 0..n {
            let factor = self.decay_factors[i];
            state.hu[i] *= factor;
            state.hv[i] *= factor;
        }
    }

    /// 获取衰减因子
    // ALLOW_F64: 源项计算
    pub fn get_decay_factor(&self, cell: usize) -> f64 {
        self.decay_factors.get(cell).copied().unwrap_or(1.0)
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
    fn test_vegetation_type_none() {
        let veg = VegetationType::None;
        assert_eq!(veg.height(), 0.0);
        assert_eq!(veg.effective_drag(1.0, 1.0), 0.0);
    }

    #[test]
    fn test_vegetation_type_rigid() {
        let veg = VegetationType::rigid(1.0, 0.01, 100.0, 1.0);
        match veg {
            VegetationType::Rigid { cd, diameter, density, height } => {
                assert!((cd - 1.0).abs() < 1e-10);
                assert!((diameter - 0.01).abs() < 1e-10);
                assert!((density - 100.0).abs() < 1e-10);
                assert!((height - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected Rigid type"),
        }
    }

    #[test]
    fn test_vegetation_type_reed() {
        let veg = VegetationType::reed();
        assert!((veg.height() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_vegetation_type_mangrove() {
        let veg = VegetationType::mangrove();
        assert!((veg.height() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_vegetation_effective_drag_rigid() {
        // 直径0.01m, 密度100/m², 高度1m
        let veg = VegetationType::rigid(1.0, 0.01, 100.0, 1.0);

        // 水深2m（完全淹没）
        // A_v = 0.01 * 100 * 1.0 / 2.0 = 0.5
        // C_d * A_v = 1.0 * 0.5 = 0.5
        let drag = veg.effective_drag(2.0, 1.0);
        assert!((drag - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_vegetation_effective_drag_partial() {
        // 水深0.5m，植被高度1m（部分淹没）
        let veg = VegetationType::rigid(1.0, 0.01, 100.0, 1.0);

        // effective_height = 0.5
        // A_v = 0.01 * 100 * 0.5 / 0.5 = 1.0
        let drag = veg.effective_drag(0.5, 1.0);
        assert!((drag - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vegetation_is_submerged() {
        let veg = VegetationType::rigid(1.0, 0.01, 100.0, 1.0);
        assert!(!veg.is_submerged(0.5)); // 未淹没
        assert!(veg.is_submerged(1.5));  // 已淹没
    }

    #[test]
    fn test_vegetation_config_creation() {
        let config = VegetationConfig::new(10, 1000.0);
        assert!(config.enabled);
        assert_eq!(config.vegetation.len(), 10);
    }

    #[test]
    fn test_vegetation_config_uniform() {
        let config = VegetationConfig::new(10, 1000.0)
            .with_uniform_vegetation(VegetationType::reed());

        for v in &config.vegetation {
            assert!((v.height() - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_vegetation_source_compute() {
        let mut config = VegetationConfig::new(10, 1000.0);
        config.set_vegetation(0, VegetationType::rigid(1.0, 0.01, 100.0, 1.0));

        let state = create_test_state(10, 2.0, 1.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu < 0.0); // 阻力与流向相反
        assert!((contrib.s_hv).abs() < 1e-10); // 无 y 方向速度
    }

    #[test]
    fn test_vegetation_source_no_vegetation() {
        let config = VegetationConfig::new(10, 1000.0);

        let state = create_test_state(10, 2.0, 1.0, 0.5);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_vegetation_source_dry_cell() {
        let config = VegetationConfig::new(10, 1000.0)
            .with_uniform_vegetation(VegetationType::reed());

        let state = create_test_state(10, 1e-7, 0.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_source_term_trait() {
        let config = VegetationConfig::default_config(10);
        assert_eq!(config.name(), "Vegetation");
        assert!(!config.is_explicit()); // 使用隐式处理
    }

    #[test]
    fn test_vegetation_implicit_creation() {
        let config = VegetationConfig::new(10, 1000.0);
        let implicit = VegetationImplicit::new(config);
        assert_eq!(implicit.decay_factors.len(), 10);
    }

    #[test]
    fn test_vegetation_implicit_no_vegetation() {
        let config = VegetationConfig::new(10, 1000.0);
        let mut implicit = VegetationImplicit::new(config);

        let state = create_test_state(10, 2.0, 1.0, 0.0);
        implicit.compute_decay_factors(&state, 0.1);

        // 无植被时衰减因子为1
        assert!((implicit.get_decay_factor(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vegetation_implicit_with_vegetation() {
        let mut config = VegetationConfig::new(10, 1000.0);
        config.set_vegetation(0, VegetationType::rigid(1.0, 0.01, 100.0, 1.0));
        let mut implicit = VegetationImplicit::new(config);

        let state = create_test_state(10, 2.0, 1.0, 0.0);
        implicit.compute_decay_factors(&state, 0.1);

        // 有植被时衰减因子小于1
        let factor = implicit.get_decay_factor(0);
        assert!(factor < 1.0);
        assert!(factor > 0.0);
    }
}
