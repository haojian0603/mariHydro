// src-tauri/src/marihydro/physics/schemes/wetting_drying/mod.rs

//! 干湿处理子系统
//!
//! 解决浅水方程在水深趋近于零时的数值不稳定问题。
//!
//! # 功能组件
//!
//! - [`handler`]: 核心干湿状态判定与处理
//! - [`transitions`]: 干湿过渡平滑函数
//! - [`momentum`]: 干区动量守恒处理
//!
//! # 物理背景
//!
//! 浅水方程在干湿边界需要特殊处理：
//! 1. **干单元检测** - 水深低于阈值的单元标记为干
//! 2. **干湿界面通量** - 限制流出干单元的通量
//! 3. **负水深修正** - 保证质量守恒的负水深处理
//! 4. **动量衰减** - 干区动量平滑衰减
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use crate::marihydro::physics::schemes::wetting_drying::{
//!     WettingDryingHandler, WetState, SmoothingType
//! };
//!
//! let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);
//!
//! // 判定单元状态
//! let state = handler.classify(0.0005);
//! assert_eq!(state, WetState::PartiallyWet);
//!
//! // 获取过渡因子
//! let factor = handler.smoothing_factor(0.0005);
//! ```
//!
//! # 关联问题
//!
//! - P5-001: 完整的干湿处理模块
//! - P5-015: 干区边界通量处理
//! - P5-016: 动量守恒

pub mod handler;
pub mod momentum;
pub mod transitions;

// 重导出常用类型
pub use handler::{WetState, WettingDryingHandler};
pub use momentum::{MomentumCorrector, MomentumCorrectionMethod};
pub use transitions::{SmoothingType, TransitionFunction};

use glam::DVec2;

/// 干湿界面通量结构
#[derive(Debug, Clone, Copy, Default)]
pub struct DryWetFlux {
    /// 质量通量
    pub mass: f64,
    /// x方向动量通量
    pub momentum_x: f64,
    /// y方向动量通量
    pub momentum_y: f64,
}

impl DryWetFlux {
    /// 零通量
    pub const ZERO: Self = Self {
        mass: 0.0,
        momentum_x: 0.0,
        momentum_y: 0.0,
    };

    /// 创建通量
    #[inline]
    pub fn new(mass: f64, momentum_x: f64, momentum_y: f64) -> Self {
        Self {
            mass,
            momentum_x,
            momentum_y,
        }
    }

    /// 应用缩放因子
    #[inline]
    pub fn scale(&mut self, factor: f64) {
        self.mass *= factor;
        self.momentum_x *= factor;
        self.momentum_y *= factor;
    }

    /// 返回缩放后的通量（不修改原值）
    #[inline]
    pub fn scaled(self, factor: f64) -> Self {
        Self {
            mass: self.mass * factor,
            momentum_x: self.momentum_x * factor,
            momentum_y: self.momentum_y * factor,
        }
    }
}

/// 干湿界面状态重构结果
#[derive(Debug, Clone, Copy)]
pub struct FaceReconstruction {
    /// 左侧水深
    pub h_left: f64,
    /// 右侧水深
    pub h_right: f64,
    /// 左侧速度
    pub vel_left: DVec2,
    /// 右侧速度
    pub vel_right: DVec2,
    /// 干湿状态
    pub state: WetState,
    /// 通量限制因子
    pub flux_limiter: f64,
}

impl FaceReconstruction {
    /// 对称的湿单元界面
    pub fn both_wet(h_l: f64, h_r: f64, vel_l: DVec2, vel_r: DVec2) -> Self {
        Self {
            h_left: h_l,
            h_right: h_r,
            vel_left: vel_l,
            vel_right: vel_r,
            state: WetState::Wet,
            flux_limiter: 1.0,
        }
    }

    /// 完全干燥界面
    pub fn both_dry() -> Self {
        Self {
            h_left: 0.0,
            h_right: 0.0,
            vel_left: DVec2::ZERO,
            vel_right: DVec2::ZERO,
            state: WetState::Dry,
            flux_limiter: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dry_wet_flux() {
        let mut flux = DryWetFlux::new(1.0, 0.5, 0.3);
        flux.scale(0.5);
        assert!((flux.mass - 0.5).abs() < 1e-10);
        assert!((flux.momentum_x - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_face_reconstruction() {
        let rec = FaceReconstruction::both_dry();
        assert_eq!(rec.state, WetState::Dry);
        assert_eq!(rec.flux_limiter, 0.0);
    }
}
