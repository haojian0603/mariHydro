// src-tauri/src/marihydro/physics/sources/friction.rs
// Optimized: 2025-11-30
//
// 注意: SourceTerm trait 已被枚举化重构，计算逻辑已移至
// core/traits/source.rs 中的 SourceTermKind::ManningFriction 和 SourceTermKind::ChezyFriction
// 此文件保留辅助函数和公共API

// 从核心模块导入配置类型
pub use crate::marihydro::core::traits::source::{ManningFrictionConfig, ChezyFrictionConfig};

/// Manning 摩擦源项包装器（便捷构造）
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
        ManningFrictionConfig::new(9.81, n_cells, 0.025)
    }
}

/// Chezy 摩擦源项包装器（便捷构造）
pub struct ChezyFriction;

impl ChezyFriction {
    /// 创建 Chezy 摩擦配置
    pub fn new(g: f64, chezy_c: f64) -> ChezyFrictionConfig {
        ChezyFrictionConfig::new(g, chezy_c)
    }

    /// 创建默认配置 (g=9.81, C=50)
    pub fn default_config() -> ChezyFrictionConfig {
        ChezyFrictionConfig::new(9.81, 50.0)
    }
}
