// src-tauri/src/marihydro/physics/sources/coriolis.rs
//
// 注意: SourceTerm trait 已被枚举化重构，计算逻辑已移至
// core/traits/source.rs 中的 SourceTermKind::Coriolis
// 此文件保留辅助函数和公共API

// 从核心模块导入配置类型
pub use crate::marihydro::core::traits::source::CoriolisConfig;
use std::f64::consts::PI;

/// 科氏力源项包装器（便捷构造）
pub struct CoriolisSource;

impl CoriolisSource {
    /// 从科氏参数创建配置
    pub fn new(f: f64) -> CoriolisConfig {
        CoriolisConfig::new(f)
    }

    /// 从纬度创建配置
    pub fn from_latitude(lat_deg: f64) -> CoriolisConfig {
        CoriolisConfig::from_latitude(lat_deg)
    }

    /// 计算科氏参数
    pub fn coriolis_parameter(lat_deg: f64) -> f64 {
        let omega = 7.2921e-5;
        2.0 * omega * (lat_deg * PI / 180.0).sin()
    }

    /// 检查时间步长是否稳定
    pub fn is_stable(f: f64, dt: f64) -> bool {
        (f * dt).abs() < 0.1
    }

    /// 计算最大稳定时间步长
    pub fn max_stable_dt(f: f64, safety: f64) -> f64 {
        if f.abs() < 1e-14 {
            f64::INFINITY
        } else {
            safety * 0.1 / f.abs()
        }
    }
}
