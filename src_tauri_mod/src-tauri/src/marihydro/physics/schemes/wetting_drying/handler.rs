// src-tauri/src/marihydro/physics/schemes/wetting_drying/handler.rs

//! 干湿处理核心模块
//!
//! 提供单元干湿状态判定、状态修正和界面通量限制。

use super::{DryWetFlux, FaceReconstruction};
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

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
    /// 干燥判定阈值 (m)，低于此值视为干
    pub h_dry: f64,
    /// 湿润恢复阈值 (m)，高于此值视为湿
    pub h_wet: f64,
    /// 干区最小水深 (m)，用于数值稳定
    pub h_min: f64,
    /// 是否启用负水深修正
    pub fix_negative_depth: bool,
    /// 干燥时动量衰减系数 [0, 1]
    pub momentum_decay: f64,
}

impl Default for WettingDryingConfig {
    fn default() -> Self {
        Self {
            h_dry: 1e-4,               // 0.1 mm
            h_wet: 1e-3,               // 1 mm
            h_min: 1e-6,               // 1 μm
            fix_negative_depth: true,
            momentum_decay: 0.0,       // 完全衰减
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
            return Err("h_min 必须为正数");
        }
        if self.h_dry < self.h_min {
            return Err("h_dry 必须大于等于 h_min");
        }
        if self.h_wet <= self.h_dry {
            return Err("h_wet 必须大于 h_dry");
        }
        if !(0.0..=1.0).contains(&self.momentum_decay) {
            return Err("momentum_decay 必须在 [0, 1] 范围内");
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

impl Default for WettingDryingHandler {
    fn default() -> Self {
        Self {
            config: WettingDryingConfig::default(),
        }
    }
}

impl WettingDryingHandler {
    /// 创建处理器
    pub fn new(h_dry: f64, h_wet: f64, h_min: f64) -> Self {
        Self {
            config: WettingDryingConfig {
                h_dry,
                h_wet,
                h_min,
                ..Default::default()
            },
        }
    }

    /// 从配置创建
    pub fn from_config(config: WettingDryingConfig) -> Self {
        Self { config }
    }

    /// 从 NumericalParams 创建
    pub fn from_params(params: &NumericalParams) -> Self {
        Self {
            config: WettingDryingConfig::from_params(params),
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &WettingDryingConfig {
        &self.config
    }

    // ================= 状态判定 =================

    /// 判定单元干湿状态
    #[inline]
    pub fn classify(&self, h: f64) -> WetState {
        if h <= self.config.h_dry {
            WetState::Dry
        } else if h < self.config.h_wet {
            WetState::PartiallyWet
        } else {
            WetState::Wet
        }
    }

    /// 是否干燥
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h <= self.config.h_dry
    }

    /// 是否完全湿润
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.config.h_wet
    }

    // ================= 数值稳定 =================

    /// 安全水深（防止除零）
    #[inline]
    pub fn safe_depth(&self, h: f64) -> f64 {
        h.max(self.config.h_min)
    }

    /// 安全速度（干区返回零）
    #[inline]
    pub fn safe_velocity(&self, hu: f64, hv: f64, h: f64) -> DVec2 {
        if self.is_dry(h) {
            DVec2::ZERO
        } else {
            let h_safe = self.safe_depth(h);
            DVec2::new(hu / h_safe, hv / h_safe)
        }
    }

    /// 安全动量（干区衰减）
    #[inline]
    pub fn safe_momentum(&self, hu: f64, hv: f64, h: f64) -> DVec2 {
        match self.classify(h) {
            WetState::Dry => DVec2::ZERO,
            WetState::Wet => DVec2::new(hu, hv),
            WetState::PartiallyWet => {
                let factor = self.smoothing_factor(h);
                DVec2::new(hu * factor, hv * factor)
            }
        }
    }

    // ================= 过渡平滑 =================

    /// 计算干湿过渡平滑因子 [0, 1]
    ///
    /// 使用 Hermite 多项式实现 C1 连续过渡
    #[inline]
    pub fn smoothing_factor(&self, h: f64) -> f64 {
        match self.classify(h) {
            WetState::Dry => 0.0,
            WetState::Wet => 1.0,
            WetState::PartiallyWet => {
                let t = (h - self.config.h_dry) / (self.config.h_wet - self.config.h_dry);
                // Hermite 平滑: 3t² - 2t³
                t * t * (3.0 - 2.0 * t)
            }
        }
    }

    /// 线性过渡因子（用于某些场景）
    #[inline]
    pub fn linear_factor(&self, h: f64) -> f64 {
        if h <= self.config.h_dry {
            0.0
        } else if h >= self.config.h_wet {
            1.0
        } else {
            (h - self.config.h_dry) / (self.config.h_wet - self.config.h_dry)
        }
    }

    // ================= 界面处理 =================

    /// 判定界面干湿状态
    ///
    /// 基于两侧单元的干湿状态
    pub fn classify_face(&self, h_left: f64, h_right: f64) -> (WetState, WetState) {
        (self.classify(h_left), self.classify(h_right))
    }

    /// 界面通量限制因子
    ///
    /// 基于两侧最小水深计算
    #[inline]
    pub fn face_flux_limiter(&self, h_left: f64, h_right: f64) -> f64 {
        let h_min = h_left.min(h_right);
        self.smoothing_factor(h_min)
    }

    /// 限制干区边界通量
    ///
    /// 遵循物理约束：
    /// 1. 干-干边界：无通量
    /// 2. 干区边界：只允许流入干区（负通量）
    /// 3. 过渡区：平滑衰减
    pub fn limit_flux(&self, h_owner: f64, h_neighbor: f64, flux: &mut DryWetFlux) {
        let owner_state = self.classify(h_owner);
        let neighbor_state = self.classify(h_neighbor);

        // 两个都干，无通量
        if owner_state == WetState::Dry && neighbor_state == WetState::Dry {
            *flux = DryWetFlux::ZERO;
            return;
        }

        // 干->湿方向：限制流出干区
        if owner_state == WetState::Dry && flux.mass > 0.0 {
            // 流出干区，禁止
            *flux = DryWetFlux::ZERO;
            return;
        } else if neighbor_state == WetState::Dry && flux.mass < 0.0 {
            // 流入邻居干区，禁止
            *flux = DryWetFlux::ZERO;
            return;
        }

        // 过渡区平滑
        let factor = self.smoothing_factor(h_owner.min(h_neighbor));
        flux.scale(factor);
    }

    /// 界面状态重构（考虑地形）
    ///
    /// 处理斜坡上的干湿界面
    pub fn reconstruct_face(
        &self,
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        z_l: f64,
        z_r: f64,
    ) -> FaceReconstruction {
        let state_l = self.classify(h_l);
        let state_r = self.classify(h_r);

        match (state_l, state_r) {
            (WetState::Dry, WetState::Dry) => FaceReconstruction::both_dry(),

            (WetState::Wet, WetState::Wet) | (WetState::Wet, WetState::PartiallyWet) |
            (WetState::PartiallyWet, WetState::Wet) | (WetState::PartiallyWet, WetState::PartiallyWet) => {
                let limiter = self.face_flux_limiter(h_l, h_r);
                FaceReconstruction {
                    h_left: h_l,
                    h_right: h_r,
                    vel_left: vel_l,
                    vel_right: vel_r,
                    state: WetState::Wet,
                    flux_limiter: limiter,
                }
            }

            (WetState::Dry, _) => {
                // 左干右湿：考虑地形坡度
                let h_star = (h_r + z_r - z_l).max(0.0);
                let use_vel = if h_star < self.config.h_dry {
                    DVec2::ZERO
                } else {
                    vel_r
                };
                FaceReconstruction {
                    h_left: 0.0,
                    h_right: h_r,
                    vel_left: DVec2::ZERO,
                    vel_right: use_vel,
                    state: WetState::PartiallyWet,
                    flux_limiter: self.smoothing_factor(h_star),
                }
            }

            (_, WetState::Dry) => {
                // 左湿右干
                let h_star = (h_l + z_l - z_r).max(0.0);
                let use_vel = if h_star < self.config.h_dry {
                    DVec2::ZERO
                } else {
                    vel_l
                };
                FaceReconstruction {
                    h_left: h_l,
                    h_right: 0.0,
                    vel_left: use_vel,
                    vel_right: DVec2::ZERO,
                    state: WetState::PartiallyWet,
                    flux_limiter: self.smoothing_factor(h_star),
                }
            }
        }
    }

    // ================= 状态修正 =================

    /// 修正负水深（单个单元）
    ///
    /// 返回质量损失（用于诊断）
    #[inline]
    pub fn fix_depth(&self, h: &mut f64, hu: &mut f64, hv: &mut f64) -> f64 {
        if *h < self.config.h_min {
            let mass_loss = *h - self.config.h_min;

            // 修正水深
            *h = self.config.h_min;

            // 衰减动量
            *hu *= self.config.momentum_decay;
            *hv *= self.config.momentum_decay;

            mass_loss
        } else {
            0.0
        }
    }

    /// 批量修正负水深
    ///
    /// 返回总质量损失
    pub fn fix_depths_batch(
        &self,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) -> f64 {
        debug_assert_eq!(h.len(), hu.len());
        debug_assert_eq!(h.len(), hv.len());

        let mut total_loss = 0.0;
        for i in 0..h.len() {
            total_loss += self.fix_depth(&mut h[i], &mut hu[i], &mut hv[i]);
        }
        total_loss
    }

    /// 并行批量修正（大规模网格）
    #[cfg(feature = "parallel")]
    pub fn fix_depths_parallel(
        &self,
        h: &mut [f64],
        hu: &mut [f64],
        hv: &mut [f64],
    ) -> f64 {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicU64, Ordering};

        let total_loss = AtomicU64::new(0);
        let config = self.config;

        h.par_iter_mut()
            .zip(hu.par_iter_mut())
            .zip(hv.par_iter_mut())
            .for_each(|((h_i, hu_i), hv_i)| {
                if *h_i < config.h_min {
                    let loss = *h_i - config.h_min;
                    *h_i = config.h_min;
                    *hu_i *= config.momentum_decay;
                    *hv_i *= config.momentum_decay;

                    // 原子累加
                    let loss_bits = loss.to_bits();
                    total_loss.fetch_add(loss_bits, Ordering::Relaxed);
                }
            });

        f64::from_bits(total_loss.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify() {
        let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);

        assert_eq!(handler.classify(0.0), WetState::Dry);
        assert_eq!(handler.classify(1e-5), WetState::Dry);
        assert_eq!(handler.classify(1e-4), WetState::Dry);
        assert_eq!(handler.classify(5e-4), WetState::PartiallyWet);
        assert_eq!(handler.classify(1e-3), WetState::Wet);
        assert_eq!(handler.classify(1.0), WetState::Wet);
    }

    #[test]
    fn test_smoothing_factor() {
        let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);

        assert_eq!(handler.smoothing_factor(0.0), 0.0);
        assert_eq!(handler.smoothing_factor(1e-4), 0.0);
        assert_eq!(handler.smoothing_factor(1e-3), 1.0);
        assert_eq!(handler.smoothing_factor(1.0), 1.0);

        // 过渡区单调递增
        let f1 = handler.smoothing_factor(2e-4);
        let f2 = handler.smoothing_factor(5e-4);
        let f3 = handler.smoothing_factor(8e-4);
        assert!(f1 < f2);
        assert!(f2 < f3);
    }

    #[test]
    fn test_safe_velocity() {
        let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);

        // 干区返回零
        let v = handler.safe_velocity(1.0, 0.5, 0.0);
        assert_eq!(v, DVec2::ZERO);

        // 湿区正常计算
        let v = handler.safe_velocity(1.0, 0.5, 1.0);
        assert!((v.x - 1.0).abs() < 1e-10);
        assert!((v.y - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_limit_flux_dry_dry() {
        let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);
        let mut flux = DryWetFlux::new(1.0, 0.5, 0.3);

        handler.limit_flux(0.0, 0.0, &mut flux);

        assert_eq!(flux.mass, 0.0);
        assert_eq!(flux.momentum_x, 0.0);
    }

    #[test]
    fn test_limit_flux_wet_to_dry() {
        let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);

        // 流出湿区进入干区（负通量表示流向邻居）
        let mut flux = DryWetFlux::new(-1.0, -0.5, 0.0);
        handler.limit_flux(1.0, 0.0, &mut flux);
        assert_eq!(flux.mass, 0.0, "禁止流入干区");
    }

    #[test]
    fn test_fix_depth() {
        let handler = WettingDryingHandler::new(1e-4, 1e-3, 1e-6);

        let mut h = -0.001;
        let mut hu = 1.0;
        let mut hv = 0.5;

        let loss = handler.fix_depth(&mut h, &mut hu, &mut hv);

        assert!(loss < 0.0);
        assert_eq!(h, 1e-6);
        assert_eq!(hu, 0.0); // momentum_decay = 0
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn test_config_validation() {
        let mut config = WettingDryingConfig::default();
        assert!(config.validate().is_ok());

        config.h_dry = 0.5e-6; // 小于 h_min
        assert!(config.validate().is_err());
    }
}
