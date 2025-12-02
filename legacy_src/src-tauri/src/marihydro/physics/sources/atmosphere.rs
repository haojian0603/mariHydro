// src-tauri/src/marihydro/physics/sources/atmosphere.rs
// 优化版本：预计算因子，并行化梯度计算
//
// 注意: SourceTerm trait 已被枚举化重构，计算逻辑已移至
// core/traits/source.rs 中的 SourceTermKind::WindStress 和 SourceTermKind::PressureGradient
// 此文件保留辅助函数和公共API

// 从核心模块导入配置类型
pub use crate::marihydro::core::traits::source::{WindStressConfig, PressureGradientConfig};

const MAX_WIND_SPEED: f64 = 100.0;

/// Large and Pond (1981) 风阻系数
#[inline(always)]
pub fn wind_drag_coefficient_lp81(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 11.0 { 1.2e-3 }
    else if w < 25.0 { (0.49 + 0.065 * w) * 1e-3 }
    else { 2.11e-3 }
}

/// Wu (1982) 风阻系数
#[inline(always)]
pub fn wind_drag_coefficient_wu82(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    (0.8 + 0.065 * w) * 1e-3
}

/// 风应力源项包装器（便捷构造）
pub struct WindStressSource;

impl WindStressSource {
    /// 创建新的风应力配置
    pub fn new(n_cells: usize, rho_air: f64, rho_water: f64) -> WindStressConfig {
        WindStressConfig::new(n_cells, rho_air, rho_water)
    }

    /// 创建默认参数的风应力配置
    pub fn default_config(n_cells: usize) -> WindStressConfig {
        WindStressConfig::new(n_cells, 1.225, 1000.0)
    }
}

/// 压力梯度源项包装器（便捷构造）
pub struct PressureGradientSource;

impl PressureGradientSource {
    /// 创建新的压力梯度配置
    pub fn new(n_cells: usize, rho_water: f64) -> PressureGradientConfig {
        PressureGradientConfig::new(n_cells, rho_water)
    }

    /// 创建默认参数的压力梯度配置
    pub fn default_config(n_cells: usize) -> PressureGradientConfig {
        PressureGradientConfig::new(n_cells, 1000.0)
    }
}
