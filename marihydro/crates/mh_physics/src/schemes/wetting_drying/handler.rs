// crates/mh_physics/src/schemes/wetting_drying/handler.rs

//! 干湿处理核心模块
//!
//! 提供单元干湿状态判定、状态修正和界面通量限制。

use crate::state::{ConservedState, ShallowWaterState};
use crate::types::NumericalParams;

/// 单元干湿状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WetState {
    /// 完全干燥 (h <= h_dry)
    Dry,
    /// 干湿过渡区 (h_dry < h < h_wet)
    PartiallyWet,
    /// 完全湿润 (h >= h_wet)
    Wet,
}

impl WetState {
    /// 从水深判断状态
    #[inline]
    pub fn from_depth(h: f64, h_dry: f64, h_wet: f64) -> Self {
        if h <= h_dry {
            Self::Dry
        } else if h >= h_wet {
            Self::Wet
        } else {
            Self::PartiallyWet
        }
    }

    /// 是否完全干燥
    #[inline]
    pub fn is_dry(self) -> bool {
        matches!(self, Self::Dry)
    }

    /// 是否完全湿润
    #[inline]
    pub fn is_wet(self) -> bool {
        matches!(self, Self::Wet)
    }

    /// 是否在过渡区
    #[inline]
    pub fn is_transitional(self) -> bool {
        matches!(self, Self::PartiallyWet)
    }
}

/// 干湿处理配置
#[derive(Debug, Clone, Copy)]
pub struct WettingDryingConfig {
    /// 干燥判定阈值 [m]，低于此值视为干
    pub h_dry: f64,
    /// 湿润恢复阈值 [m]，高于此值视为湿
    pub h_wet: f64,
    /// 干区最小水深 [m]，用于数值稳定
    pub h_min: f64,
    /// 是否启用负水深修正
    pub fix_negative_depth: bool,
    /// 干燥时动量衰减系数 [0, 1]
    pub momentum_decay: f64,
}

impl Default for WettingDryingConfig {
    fn default() -> Self {
        Self {
            h_dry: 1e-4,
            h_wet: 1e-3,
            h_min: 1e-6,
            fix_negative_depth: true,
            momentum_decay: 0.0,
        }
    }
}

impl WettingDryingConfig {
    /// 从 NumericalParams 创建
    pub fn from_params(params: &NumericalParams) -> Self {
        Self {
            h_dry: params.h_dry,
            h_wet: params.h_wet,
            h_min: params.h_min,
            ..Default::default()
        }
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.h_min <= 0.0 {
            return Err("h_min must be positive");
        }
        if self.h_dry < self.h_min {
            return Err("h_dry must be >= h_min");
        }
        if self.h_wet <= self.h_dry {
            return Err("h_wet must be > h_dry");
        }
        if !(0.0..=1.0).contains(&self.momentum_decay) {
            return Err("momentum_decay must be in [0, 1]");
        }
        Ok(())
    }
}

/// 干湿处理器
///
/// 负责单元干湿状态判定、状态修正和界面通量限制。
#[derive(Debug, Clone)]
pub struct WettingDryingHandler {
    config: WettingDryingConfig,
}

impl WettingDryingHandler {
    /// 创建新的处理器
    pub fn new(config: WettingDryingConfig) -> Self {
        Self { config }
    }

    /// 从 NumericalParams 创建
    pub fn from_params(params: &NumericalParams) -> Self {
        Self::new(WettingDryingConfig::from_params(params))
    }

    /// 获取配置
    pub fn config(&self) -> &WettingDryingConfig {
        &self.config
    }

    /// 判断单元干湿状态
    #[inline]
    pub fn get_state(&self, h: f64) -> WetState {
        WetState::from_depth(h, self.config.h_dry, self.config.h_wet)
    }

    /// 计算干湿过渡权重 [0, 1]
    ///
    /// 0 = 完全干, 1 = 完全湿
    #[inline]
    pub fn wet_fraction(&self, h: f64) -> f64 {
        if h <= self.config.h_dry {
            0.0
        } else if h >= self.config.h_wet {
            1.0
        } else {
            (h - self.config.h_dry) / (self.config.h_wet - self.config.h_dry)
        }
    }

    /// 平滑的干湿过渡权重 (Hermite 插值)
    #[inline]
    pub fn wet_fraction_smooth(&self, h: f64) -> f64 {
        let t = self.wet_fraction(h);
        t * t * (3.0 - 2.0 * t)
    }

    /// 修正单个单元状态
    ///
    /// 返回修正后的 (h, hu, hv)
    pub fn correct_cell(&self, h: f64, hu: f64, hv: f64) -> (f64, f64, f64) {
        // 负水深修正
        if h < 0.0 && self.config.fix_negative_depth {
            return (0.0, 0.0, 0.0);
        }

        // 干单元处理
        if h <= self.config.h_dry {
            let h_corrected = if h > 0.0 { h } else { 0.0 };
            let decay = self.config.momentum_decay;
            return (h_corrected, hu * decay, hv * decay);
        }

        (h, hu, hv)
    }

    /// 修正整个状态场
    pub fn correct_state(&self, state: &mut ShallowWaterState) {
        for i in 0..state.n_cells() {
            let (h, hu, hv) = self.correct_cell(state.h[i], state.hu[i], state.hv[i]);
            state.h[i] = h;
            state.hu[i] = hu;
            state.hv[i] = hv;
        }
    }

    /// 判断界面是否需要计算通量
    ///
    /// 返回 (should_compute, left_dry, right_dry)
    #[inline]
    pub fn should_compute_flux(&self, h_left: f64, h_right: f64) -> (bool, bool, bool) {
        let left_dry = h_left <= self.config.h_dry;
        let right_dry = h_right <= self.config.h_dry;

        // 两边都干则不需要计算
        let should_compute = !(left_dry && right_dry);

        (should_compute, left_dry, right_dry)
    }

    /// 限制干单元的动量通量
    ///
    /// 确保干单元不会接收导致负水深的通量
    #[inline]
    pub fn limit_dry_cell_flux(
        &self,
        h: f64,
        flux_in: f64,
        dt: f64,
        area: f64,
    ) -> f64 {
        if h <= self.config.h_dry {
            // 干单元：只允许入流
            flux_in.max(0.0)
        } else if h < self.config.h_wet {
            // 过渡区：按比例限制出流
            let w = self.wet_fraction(h);
            if flux_in < 0.0 {
                // 出流限制
                let max_outflow = h * area / dt * w;
                flux_in.max(-max_outflow)
            } else {
                flux_in
            }
        } else {
            // 湿单元：不限制
            flux_in
        }
    }

    /// 应用正性保持修正
    ///
    /// 确保更新后水深非负
    pub fn apply_positivity_preserving(
        &self,
        state: &mut ShallowWaterState,
        dh: &[f64],
        dt: f64,
    ) {
        for i in 0..state.n_cells() {
            let h_new = state.h[i] + dt * dh[i];
            if h_new < 0.0 {
                // 需要限制
                if dh[i] < 0.0 {
                    // 限制减少量
                    let limited_dh = -state.h[i] / dt;
                    state.h[i] = 0.0;
                    // 同时清零动量
                    state.hu[i] = 0.0;
                    state.hv[i] = 0.0;
                } else {
                    state.h[i] = h_new;
                }
            } else {
                state.h[i] = h_new;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wet_state() {
        let config = WettingDryingConfig::default();
        let handler = WettingDryingHandler::new(config);

        assert!(handler.get_state(0.0).is_dry());
        assert!(handler.get_state(1e-5).is_dry());
        assert!(handler.get_state(5e-4).is_transitional());
        assert!(handler.get_state(1.0).is_wet());
    }

    #[test]
    fn test_wet_fraction() {
        let config = WettingDryingConfig {
            h_dry: 1e-4,
            h_wet: 1e-3,
            ..Default::default()
        };
        let handler = WettingDryingHandler::new(config);

        assert_eq!(handler.wet_fraction(0.0), 0.0);
        assert_eq!(handler.wet_fraction(1e-4), 0.0);
        assert_eq!(handler.wet_fraction(1e-3), 1.0);
        assert_eq!(handler.wet_fraction(10.0), 1.0);

        let mid = 5.5e-4; // 中间值
        let frac = handler.wet_fraction(mid);
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_correct_cell() {
        let config = WettingDryingConfig {
            h_dry: 1e-4,
            fix_negative_depth: true,
            momentum_decay: 0.0,
            ..Default::default()
        };
        let handler = WettingDryingHandler::new(config);

        // 负水深
        let (h, hu, hv) = handler.correct_cell(-1.0, 10.0, 10.0);
        assert_eq!(h, 0.0);
        assert_eq!(hu, 0.0);
        assert_eq!(hv, 0.0);

        // 干单元
        let (h, hu, hv) = handler.correct_cell(1e-5, 10.0, 10.0);
        assert_eq!(h, 1e-5);
        assert_eq!(hu, 0.0);
        assert_eq!(hv, 0.0);

        // 湿单元
        let (h, hu, hv) = handler.correct_cell(1.0, 10.0, 10.0);
        assert_eq!(h, 1.0);
        assert_eq!(hu, 10.0);
        assert_eq!(hv, 10.0);
    }

    #[test]
    fn test_should_compute_flux() {
        let handler = WettingDryingHandler::new(WettingDryingConfig::default());

        let (compute, _, _) = handler.should_compute_flux(0.0, 0.0);
        assert!(!compute, "双干不应计算");

        let (compute, left_dry, right_dry) = handler.should_compute_flux(0.0, 1.0);
        assert!(compute);
        assert!(left_dry);
        assert!(!right_dry);

        let (compute, _, _) = handler.should_compute_flux(1.0, 1.0);
        assert!(compute, "双湿应计算");
    }

    #[test]
    fn test_config_validation() {
        let valid = WettingDryingConfig::default();
        assert!(valid.validate().is_ok());

        let invalid = WettingDryingConfig {
            h_min: -1.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid = WettingDryingConfig {
            h_wet: 1e-5,
            h_dry: 1e-4,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
}
