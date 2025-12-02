// src-tauri/src/marihydro/physics/schemes/wetting_drying/momentum.rs

//! 干区动量守恒处理
//!
//! 处理干湿边界处的动量守恒问题，防止干区出现非物理速度。

use glam::DVec2;

/// 动量修正方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MomentumCorrectionMethod {
    /// 完全衰减（动量归零）
    #[default]
    FullDecay,
    /// 指数衰减
    ExponentialDecay,
    /// 线性衰减
    LinearDecay,
    /// 保守修正（仅限制速度）
    VelocityCap,
}

/// 动量修正配置
#[derive(Debug, Clone, Copy)]
pub struct MomentumCorrectorConfig {
    /// 修正方法
    pub method: MomentumCorrectionMethod,
    /// 干燥阈值
    pub h_dry: f64,
    /// 湿润阈值
    pub h_wet: f64,
    /// 最小水深
    pub h_min: f64,
    /// 最大允许速度 (m/s)
    pub max_velocity: f64,
    /// 指数衰减时间常数 (s)
    pub decay_tau: f64,
}

impl Default for MomentumCorrectorConfig {
    fn default() -> Self {
        Self {
            method: MomentumCorrectionMethod::default(),
            h_dry: 1e-4,
            h_wet: 1e-3,
            h_min: 1e-6,
            max_velocity: 50.0, // 50 m/s 上限
            decay_tau: 0.01,    // 10 ms
        }
    }
}

/// 动量修正器
///
/// 处理干区和过渡区的动量守恒
#[derive(Debug, Clone)]
pub struct MomentumCorrector {
    config: MomentumCorrectorConfig,
}

impl Default for MomentumCorrector {
    fn default() -> Self {
        Self {
            config: MomentumCorrectorConfig::default(),
        }
    }
}

impl MomentumCorrector {
    /// 创建动量修正器
    pub fn new(config: MomentumCorrectorConfig) -> Self {
        Self { config }
    }

    /// 从阈值创建
    pub fn from_thresholds(h_dry: f64, h_wet: f64, h_min: f64) -> Self {
        Self {
            config: MomentumCorrectorConfig {
                h_dry,
                h_wet,
                h_min,
                ..Default::default()
            },
        }
    }

    /// 设置修正方法
    pub fn with_method(mut self, method: MomentumCorrectionMethod) -> Self {
        self.config.method = method;
        self
    }

    /// 获取配置
    pub fn config(&self) -> &MomentumCorrectorConfig {
        &self.config
    }

    // ================= 单点修正 =================

    /// 计算干湿过渡因子
    #[inline]
    fn transition_factor(&self, h: f64) -> f64 {
        if h <= self.config.h_dry {
            0.0
        } else if h >= self.config.h_wet {
            1.0
        } else {
            let t = (h - self.config.h_dry) / (self.config.h_wet - self.config.h_dry);
            // Hermite 平滑
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// 计算安全水深
    #[inline]
    fn safe_depth(&self, h: f64) -> f64 {
        h.max(self.config.h_min)
    }

    /// 修正单个单元的动量
    ///
    /// # 参数
    /// - `h`: 水深
    /// - `hu`: x方向动量
    /// - `hv`: y方向动量
    /// - `dt`: 时间步长（用于时间相关衰减）
    ///
    /// # 返回
    /// 修正后的 (hu, hv)
    pub fn correct(&self, h: f64, hu: f64, hv: f64, dt: f64) -> (f64, f64) {
        match self.config.method {
            MomentumCorrectionMethod::FullDecay => {
                self.correct_full_decay(h, hu, hv)
            }
            MomentumCorrectionMethod::ExponentialDecay => {
                self.correct_exponential_decay(h, hu, hv, dt)
            }
            MomentumCorrectionMethod::LinearDecay => {
                self.correct_linear_decay(h, hu, hv)
            }
            MomentumCorrectionMethod::VelocityCap => {
                self.correct_velocity_cap(h, hu, hv)
            }
        }
    }

    /// 完全衰减：干区动量归零
    #[inline]
    fn correct_full_decay(&self, h: f64, hu: f64, hv: f64) -> (f64, f64) {
        if h <= self.config.h_dry {
            (0.0, 0.0)
        } else if h >= self.config.h_wet {
            (hu, hv)
        } else {
            let factor = self.transition_factor(h);
            (hu * factor, hv * factor)
        }
    }

    /// 指数衰减：时间相关的动量衰减
    #[inline]
    fn correct_exponential_decay(&self, h: f64, hu: f64, hv: f64, dt: f64) -> (f64, f64) {
        if h <= self.config.h_dry {
            // 完全干燥：快速衰减
            let decay = (-dt / self.config.decay_tau).exp();
            (hu * decay, hv * decay)
        } else if h >= self.config.h_wet {
            (hu, hv)
        } else {
            // 过渡区：部分衰减
            let factor = self.transition_factor(h);
            let partial_decay = (-(1.0 - factor) * dt / self.config.decay_tau).exp();
            (hu * partial_decay, hv * partial_decay)
        }
    }

    /// 线性衰减：基于水深的线性缩放
    #[inline]
    fn correct_linear_decay(&self, h: f64, hu: f64, hv: f64) -> (f64, f64) {
        let factor = self.transition_factor(h);
        (hu * factor, hv * factor)
    }

    /// 速度限制：仅限制最大速度
    #[inline]
    fn correct_velocity_cap(&self, h: f64, hu: f64, hv: f64) -> (f64, f64) {
        if h <= self.config.h_dry {
            (0.0, 0.0)
        } else {
            let h_safe = self.safe_depth(h);
            let u = hu / h_safe;
            let v = hv / h_safe;
            let speed = (u * u + v * v).sqrt();

            if speed > self.config.max_velocity {
                let scale = self.config.max_velocity / speed;
                (hu * scale, hv * scale)
            } else {
                (hu, hv)
            }
        }
    }

    // ================= 向量版本 =================

    /// 修正动量向量
    #[inline]
    pub fn correct_vec(&self, h: f64, momentum: DVec2, dt: f64) -> DVec2 {
        let (hu, hv) = self.correct(h, momentum.x, momentum.y, dt);
        DVec2::new(hu, hv)
    }

    /// 计算安全速度
    #[inline]
    pub fn safe_velocity(&self, h: f64, hu: f64, hv: f64) -> DVec2 {
        if h <= self.config.h_dry {
            DVec2::ZERO
        } else {
            let h_safe = self.safe_depth(h);
            let u = hu / h_safe;
            let v = hv / h_safe;
            let speed = (u * u + v * v).sqrt();

            if speed > self.config.max_velocity {
                let scale = self.config.max_velocity / speed;
                DVec2::new(u * scale, v * scale)
            } else {
                DVec2::new(u, v)
            }
        }
    }

    // ================= 批量修正 =================

    /// 批量修正动量
    pub fn correct_batch(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        dt: f64,
    ) {
        debug_assert_eq!(h.len(), hu.len());
        debug_assert_eq!(h.len(), hv.len());

        for i in 0..h.len() {
            let (new_hu, new_hv) = self.correct(h[i], hu[i], hv[i], dt);
            hu[i] = new_hu;
            hv[i] = new_hv;
        }
    }

    /// 批量计算安全速度
    pub fn safe_velocities_batch(
        &self,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        out: &mut [DVec2],
    ) {
        debug_assert_eq!(h.len(), hu.len());
        debug_assert_eq!(h.len(), hv.len());
        debug_assert_eq!(h.len(), out.len());

        for i in 0..h.len() {
            out[i] = self.safe_velocity(h[i], hu[i], hv[i]);
        }
    }

    // ================= 诊断 =================

    /// 计算动量守恒误差
    ///
    /// 返回修正前后的总动量差
    pub fn momentum_error(&self, h: &[f64], hu: &[f64], hv: &[f64], dt: f64) -> (f64, f64) {
        let mut total_hu_before = 0.0;
        let mut total_hv_before = 0.0;
        let mut total_hu_after = 0.0;
        let mut total_hv_after = 0.0;

        for i in 0..h.len() {
            total_hu_before += hu[i];
            total_hv_before += hv[i];

            let (new_hu, new_hv) = self.correct(h[i], hu[i], hv[i], dt);
            total_hu_after += new_hu;
            total_hv_after += new_hv;
        }

        (
            total_hu_before - total_hu_after,
            total_hv_before - total_hv_after,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_decay() {
        let corrector = MomentumCorrector::from_thresholds(1e-4, 1e-3, 1e-6);

        // 干区：动量归零
        let (hu, hv) = corrector.correct(0.0, 1.0, 0.5, 0.01);
        assert_eq!(hu, 0.0);
        assert_eq!(hv, 0.0);

        // 湿区：动量保持
        let (hu, hv) = corrector.correct(1.0, 1.0, 0.5, 0.01);
        assert_eq!(hu, 1.0);
        assert_eq!(hv, 0.5);

        // 过渡区：部分衰减
        let (hu, hv) = corrector.correct(5e-4, 1.0, 0.5, 0.01);
        assert!(hu > 0.0 && hu < 1.0);
        assert!(hv > 0.0 && hv < 0.5);
    }

    #[test]
    fn test_velocity_cap() {
        let corrector = MomentumCorrector::from_thresholds(1e-4, 1e-3, 1e-6)
            .with_method(MomentumCorrectionMethod::VelocityCap);

        // 正常速度：不变
        let (hu, hv) = corrector.correct(1.0, 10.0, 0.0, 0.01);
        assert_eq!(hu, 10.0);

        // 超速：限制
        let (hu, hv) = corrector.correct(1.0, 100.0, 0.0, 0.01); // 100 m/s
        let speed = (hu * hu + hv * hv).sqrt() / 1.0;
        assert!((speed - 50.0).abs() < 1e-6); // 限制到 50 m/s
    }

    #[test]
    fn test_safe_velocity() {
        let corrector = MomentumCorrector::from_thresholds(1e-4, 1e-3, 1e-6);

        // 干区：零速度
        let v = corrector.safe_velocity(0.0, 1.0, 0.5);
        assert_eq!(v, DVec2::ZERO);

        // 正常情况
        let v = corrector.safe_velocity(1.0, 2.0, 1.0);
        assert!((v.x - 2.0).abs() < 1e-10);
        assert!((v.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_decay() {
        let corrector = MomentumCorrector::new(MomentumCorrectorConfig {
            method: MomentumCorrectionMethod::ExponentialDecay,
            decay_tau: 0.01,
            ..Default::default()
        });

        // 干区随时间衰减
        let (hu1, _) = corrector.correct(0.0, 1.0, 0.0, 0.001);
        let (hu2, _) = corrector.correct(0.0, 1.0, 0.0, 0.01);
        let (hu3, _) = corrector.correct(0.0, 1.0, 0.0, 0.1);

        assert!(hu1 > hu2);
        assert!(hu2 > hu3);
    }
}
