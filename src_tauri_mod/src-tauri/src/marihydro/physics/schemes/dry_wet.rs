// src-tauri/src/marihydro/physics/schemes/dry_wet.rs

//! 统一干湿处理（已废弃）
//!
//! ⚠️ **此模块已废弃**，请使用 `wetting_drying` 模块替代。
//!
//! 迁移指南：
//! ```rust,ignore
//! // 旧代码
//! use crate::marihydro::physics::schemes::dry_wet::{DryWetHandler, WetDryState};
//!
//! // 新代码
//! use crate::marihydro::physics::schemes::wetting_drying::{
//!     WettingDryingHandler, WetState, MomentumCorrector
//! };
//! ```
//!
//! 新模块提供：
//! - 更完整的干湿状态处理
//! - 多种过渡函数选择
//! - 动量守恒修正
//! - 并行批量处理支持

#![deprecated(
    since = "0.3.0",
    note = "请使用 physics::schemes::wetting_drying 模块替代"
)]

use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WetDryState {
    BothWet,
    LeftDry,
    RightDry,
    BothDry,
}

pub struct DryWetHandler {
    h_dry: f64,
    h_min: f64,
    h_wet: f64,
}

impl DryWetHandler {
    pub fn new(params: &NumericalParams) -> Self {
        Self {
            h_dry: params.h_dry,
            h_min: params.h_min,
            h_wet: params.h_wet,
        }
    }

    pub fn from_thresholds(h_dry: f64, h_min: f64, h_wet: f64) -> Self {
        Self {
            h_dry,
            h_min,
            h_wet,
        }
    }

    #[inline]
    pub fn classify(&self, h_l: f64, h_r: f64) -> WetDryState {
        match (h_l < self.h_dry, h_r < self.h_dry) {
            (false, false) => WetDryState::BothWet,
            (true, false) => WetDryState::LeftDry,
            (false, true) => WetDryState::RightDry,
            (true, true) => WetDryState::BothDry,
        }
    }

    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.h_dry
    }
    #[inline]
    pub fn safe_depth(&self, h: f64) -> f64 {
        h.max(self.h_min)
    }

    #[inline]
    pub fn safe_velocity(&self, hu: f64, hv: f64, h: f64) -> DVec2 {
        if self.is_dry(h) {
            DVec2::ZERO
        } else {
            DVec2::new(hu / self.safe_depth(h), hv / self.safe_depth(h))
        }
    }

    #[inline]
    pub fn transition_weight(&self, h: f64) -> f64 {
        if h <= self.h_dry {
            0.0
        } else if h >= self.h_wet {
            1.0
        } else {
            (h - self.h_dry) / (self.h_wet - self.h_dry)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WetDryFaceReconstruction {
    pub h_left: f64,
    pub h_right: f64,
    pub vel_left: DVec2,
    pub vel_right: DVec2,
    pub state: WetDryState,
}

impl WetDryFaceReconstruction {
    pub fn reconstruct(
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        z_l: f64,
        z_r: f64,
        handler: &DryWetHandler,
    ) -> Self {
        let state = handler.classify(h_l, h_r);
        match state {
            WetDryState::BothDry => Self {
                h_left: 0.0,
                h_right: 0.0,
                vel_left: DVec2::ZERO,
                vel_right: DVec2::ZERO,
                state,
            },
            WetDryState::BothWet => Self {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                state,
            },
            WetDryState::LeftDry => {
                let h_star = (h_r + z_r - z_l).max(0.0);
                Self {
                    h_left: 0.0,
                    h_right: h_r,
                    vel_left: DVec2::ZERO,
                    vel_right: if h_star < handler.h_dry {
                        DVec2::ZERO
                    } else {
                        vel_r
                    },
                    state,
                }
            }
            WetDryState::RightDry => {
                let h_star = (h_l + z_l - z_r).max(0.0);
                Self {
                    h_left: h_l,
                    h_right: 0.0,
                    vel_left: if h_star < handler.h_dry {
                        DVec2::ZERO
                    } else {
                        vel_l
                    },
                    vel_right: DVec2::ZERO,
                    state,
                }
            }
        }
    }
}
