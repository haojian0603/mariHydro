// crates/mh_physics/src/schemes/wetting_drying/handler.rs

//! 干湿处理核心模块
//!
//! 提供单元干湿状态判定、状态修正和界面通量限制。
//! 支持 `f32`/`f64` 双精度运行时切换，通过 `Backend` 泛型实现零成本抽象。
//!
//! # 架构层级
//!
//! 本模块属于 **Layer 3 (Engine)**，所有数值字段使用泛型参数 `B::Scalar`，
//! 确保与 `Backend` 精度一致。
//!
//! # 使用示例
//!
//! ```
//! use mh_physics::schemes::wetting_drying::WettingDryingHandler;
//! use mh_physics::types::NumericalParams;
//! use mh_runtime::{CpuBackend, RuntimeScalar};
//!
//! // 创建 f64 精度的处理器
//! let params_f64 = NumericalParams::<f64>::default();
//! let handler_f64 = WettingDryingHandler::<CpuBackend<f64>>::from_params(&params_f64);
//!
//! // 创建 f32 精度的处理器
//! let params_f32 = NumericalParams::<f32>::default();
//! let handler_f32 = WettingDryingHandler::<CpuBackend<f32>>::from_params(&params_f32);
//!
//! // 判定干湿状态
//! let is_dry_f64 = handler_f64.is_dry(1e-8);
//! let is_dry_f32 = handler_f32.is_dry(1e-8f32);
//! ```

use mh_runtime::{Backend, RuntimeScalar};
use num_traits::Float;

// ============================================================================
// 单元干湿状态枚举
// ============================================================================

/// 单元干湿状态分类
///
/// 根据水深阈值将单元划分为三种状态，用于指导数值处理策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WetState {
    /// 完全干燥 (`h <= h_dry`)
    Dry,
    /// 干湿过渡区 (`h_dry < h < h_wet`)
    PartiallyWet,
    /// 完全湿润 (`h >= h_wet`)
    Wet,
}

impl WetState {
    /// 根据水深判断单元状态
    ///
    /// # 参数
    /// - `h`: 单元水深
    /// - `h_dry`: 干燥判定阈值
    /// - `h_wet`: 湿润判定阈值
    #[inline]
    pub fn from_depth<S: RuntimeScalar>(h: S, h_dry: S, h_wet: S) -> Self {
        if h <= h_dry {
            Self::Dry
        } else if h >= h_wet {
            Self::Wet
        } else {
            Self::PartiallyWet
        }
    }

    /// 检查是否为完全干燥状态
    #[inline]
    pub fn is_dry(self) -> bool {
        matches!(self, Self::Dry)
    }

    /// 检查是否为完全湿润状态
    #[inline]
    pub fn is_wet(self) -> bool {
        matches!(self, Self::Wet)
    }

    /// 检查是否在过渡区
    #[inline]
    pub fn is_transitional(self) -> bool {
        matches!(self, Self::PartiallyWet)
    }
}

// ============================================================================
// 干湿处理配置
// ============================================================================

/// 干湿处理参数配置（泛型化）
///
/// 所有阈值字段使用 `S: RuntimeScalar` 泛型，支持 `f32`/`f64` 运行时切换。
#[derive(Debug, Clone, Copy)]
pub struct WettingDryingConfig<S: RuntimeScalar> {
    /// 干燥判定阈值 [m]，低于此值视为干单元
    pub h_dry: S,
    /// 湿润恢复阈值 [m]，高于此值视为湿单元
    pub h_wet: S,
    /// 干区最小水深 [m]，用于数值稳定
    pub h_min: S,
    /// 是否启用负水深修正
    pub fix_negative_depth: bool,
    /// 干燥时动量衰减系数 [0, 1]
    pub momentum_decay: S,
}

impl<S: RuntimeScalar> Default for WettingDryingConfig<S> {
    /// 默认配置，使用标准物理默认值
    fn default() -> Self {
        use num_traits::FromPrimitive;
        Self {
            h_dry: S::from_f64(1e-4).unwrap_or(S::ZERO),
            h_wet: S::from_f64(1e-3).unwrap_or(S::ZERO),
            h_min: S::from_f64(1e-6).unwrap_or(S::ZERO),
            fix_negative_depth: true,
            momentum_decay: S::ZERO,
        }
    }
}

impl<S: RuntimeScalar> WettingDryingConfig<S> {
    /// 从数值参数创建配置
    ///
    /// # 参数
    /// - `params`: 数值参数配置（已泛型化）
    #[inline]
    pub fn from_params(params: &crate::types::NumericalParams<S>) -> Self {
        Self {
            h_dry: params.h_dry,
            h_wet: params.h_wet,
            h_min: params.h_min,
            ..Default::default()
        }
    }

    /// 验证配置有效性
    ///
    /// # 返回
    /// - `Ok(())`: 配置合法
    /// - `Err(&'static str)`: 验证失败原因
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.h_min <= S::ZERO {
            return Err("h_min must be positive");
        }
        if self.h_dry < self.h_min {
            return Err("h_dry must be >= h_min");
        }
        if self.h_wet <= self.h_dry {
            return Err("h_wet must be > h_dry");
        }
        if !(S::ZERO..=S::ONE).contains(&self.momentum_decay) {
            return Err("momentum_decay must be in [0, 1]");
        }
        Ok(())
    }
}

// ============================================================================
// 干湿处理器（Backend泛型化）
// ============================================================================

/// 干湿处理器
///
/// 负责单元干湿状态判定、状态修正和界面通量限制。
/// 使用 `Backend<B>` 泛型参数，支持 `f32`/`f64` 精度切换。
///
/// # 类型参数
/// - `B: Backend`: 计算后端，提供标量类型 `B::Scalar`
#[derive(Debug, Clone)]
pub struct WettingDryingHandler<B: Backend> {
    /// 泛型化配置
    config: WettingDryingConfig<B::Scalar>,
}

impl<B: Backend> WettingDryingHandler<B> {
    /// 使用配置创建处理器
    #[inline]
    pub fn new(config: WettingDryingConfig<B::Scalar>) -> Self {
        Self { config }
    }

    /// 从数值参数创建处理器
    #[inline]
    pub fn from_params(params: &crate::types::NumericalParams<B::Scalar>) -> Self {
        Self::new(WettingDryingConfig::from_params(params))
    }

    /// 获取配置引用
    #[inline]
    pub fn config(&self) -> &WettingDryingConfig<B::Scalar> {
        &self.config
    }

    /// 判断单元是否为干单元
    #[inline]
    pub fn is_dry(&self, h: B::Scalar) -> bool {
        h <= self.config.h_dry
    }

    /// 判断单元是否为湿单元
    #[inline]
    pub fn is_wet(&self, h: B::Scalar) -> bool {
        h >= self.config.h_wet
    }

    /// 获取水深对应的干湿状态
    #[inline]
    pub fn get_state(&self, h: B::Scalar) -> WetState {
        WetState::from_depth(h, self.config.h_dry, self.config.h_wet)
    }

    /// 计算干湿过渡权重（线性插值）
    ///
    /// 返回值范围 `[0, 1]`，其中 `0` 表示完全干，`1` 表示完全湿。
    #[inline]
    pub fn wet_fraction(&self, h: B::Scalar) -> B::Scalar {
        if h <= self.config.h_dry {
            B::Scalar::ZERO
        } else if h >= self.config.h_wet {
            B::Scalar::ONE
        } else {
            (h - self.config.h_dry) / (self.config.h_wet - self.config.h_dry)
        }
    }

    /// 计算平滑的干湿过渡权重（Hermite插值）
    ///
    /// 提供 `C¹` 连续的光滑过渡，避免线性权重导致的数值振荡。
    #[inline]
    pub fn wet_fraction_smooth(&self, h: B::Scalar) -> B::Scalar {
        let t = self.wet_fraction(h);
        let two = B::Scalar::ONE + B::Scalar::ONE;
        let three = two + B::Scalar::ONE;
        t * t * (three - two * t)
    }

    /// 修正单个单元状态
    ///
    /// 返回修正后的 `(h, hu, hv)`，确保物理合理性和数值稳定性。
    #[inline]
    pub fn correct_cell(&self, h: B::Scalar, hu: B::Scalar, hv: B::Scalar) -> (B::Scalar, B::Scalar, B::Scalar) {
        // 负水深修正
        if h < B::Scalar::ZERO && self.config.fix_negative_depth {
            return (B::Scalar::ZERO, B::Scalar::ZERO, B::Scalar::ZERO);
        }

        // 干单元处理
        if h <= self.config.h_dry {
            let h_corrected = if h > B::Scalar::ZERO { h } else { B::Scalar::ZERO };
            let decay = self.config.momentum_decay;
            return (h_corrected, hu * decay, hv * decay);
        }

        (h, hu, hv)
    }

    /// 判断界面是否需要计算通量
    ///
    /// # 返回
    /// - `(should_compute, left_dry, right_dry)`: 
    ///   - `should_compute`: 是否执行黎曼求解
    ///   - `left_dry`: 左侧单元是否干燥
    ///   - `right_dry`: 右侧单元是否干燥
    #[inline]
    pub fn should_compute_flux(&self, h_left: B::Scalar, h_right: B::Scalar) -> (bool, bool, bool) {
        let left_dry = self.is_dry(h_left);
        let right_dry = self.is_dry(h_right);

        // 两边都干则不需要计算
        let should_compute = !(left_dry && right_dry);

        (should_compute, left_dry, right_dry)
    }

    /// 限制干单元的动量通量
    ///
    /// 确保干单元不会接收导致负水深的通量。
    #[inline]
    pub fn limit_dry_cell_flux(
        &self,
        h: B::Scalar,
        flux_in: B::Scalar,
        dt: B::Scalar,
        area: B::Scalar,
    ) -> B::Scalar {
        if h <= self.config.h_dry {
            // 干单元：只允许入流
            flux_in.max(B::Scalar::ZERO)
        } else if h < self.config.h_wet {
            // 过渡区：按比例限制出流
            let w = self.wet_fraction(h);
            if flux_in < B::Scalar::ZERO {
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
}

// ============================================================================
// 类型别名（向后兼容）
// ============================================================================

/// f64 精度的处理器类型别名
pub type WettingDryingHandlerF64 = WettingDryingHandler<mh_runtime::CpuBackend<f64>>;

/// f32 精度的处理器类型别名
pub type WettingDryingHandlerF32 = WettingDryingHandler<mh_runtime::CpuBackend<f32>>;

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_wet_state_f64() {
        let config = WettingDryingConfig::<f64>::default();
        let handler = WettingDryingHandler::<CpuBackend<f64>>::new(config);

        assert!(handler.get_state(0.0).is_dry());
        assert!(handler.get_state(1e-5).is_dry());
    }

    #[test]
    fn test_wet_state_f32() {
        let config = WettingDryingConfig::<f32>::default();
        let handler = WettingDryingHandler::<CpuBackend<f32>>::new(config);

        assert!(handler.get_state(0.0f32).is_dry());
        assert!(handler.get_state(1e-5f32).is_dry());
    }

    #[test]
    fn test_wet_fraction_f64() {
        let config = WettingDryingConfig::<f64> {
            h_dry: 1e-4,
            h_wet: 1e-3,
            ..Default::default()
        };
        let handler = WettingDryingHandler::<CpuBackend<f64>>::new(config);

        assert_eq!(handler.wet_fraction(0.0), 0.0);
        assert_eq!(handler.wet_fraction(1e-4), 0.0);
        assert_eq!(handler.wet_fraction(1e-3), 1.0);

        let mid = 5.5e-4;
        let frac = handler.wet_fraction(mid);
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_wet_fraction_f32() {
        let config = WettingDryingConfig::<f32> {
            h_dry: 1e-4f32,
            h_wet: 1e-3f32,
            ..Default::default()
        };
        let handler = WettingDryingHandler::<CpuBackend<f32>>::new(config);

        assert_eq!(handler.wet_fraction(0.0f32), 0.0f32);
        let mid = 5.5e-4f32;
        let frac = handler.wet_fraction(mid);
        assert!(frac > 0.0f32 && frac < 1.0f32);
    }

    #[test]
    fn test_wet_fraction_smooth_f64() {
        let config = WettingDryingConfig::<f64>::default();
        let handler = WettingDryingHandler::<CpuBackend<f64>>::new(config);

        // 平滑函数在边界处应连续
        let frac_dry = handler.wet_fraction_smooth(config.h_dry);
        let frac_wet = handler.wet_fraction_smooth(config.h_wet);
        assert!((frac_dry - 0.0).abs() < 1e-10);
        assert!((frac_wet - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correct_cell_f64() {
        let config = WettingDryingConfig::<f64>::default();
        let handler = WettingDryingHandler::<CpuBackend<f64>>::new(config);

        // 负水深修正
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
    fn test_correct_cell_f32() {
        let config = WettingDryingConfig::<f32>::default();
        let handler = WettingDryingHandler::<CpuBackend<f32>>::new(config);

        let (h, hu, hv) = handler.correct_cell(-1.0f32, 10.0f32, 10.0f32);
        assert_eq!(h, 0.0f32);
        assert_eq!(hu, 0.0f32);
        assert_eq!(hv, 0.0f32);
    }

    #[test]
    fn test_should_compute_flux_f64() {
        let handler = WettingDryingHandler::<CpuBackend<f64>>::from_params(
            &NumericalParams::<f64>::default()
        );

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
    fn test_config_validation_f64() {
        let valid = WettingDryingConfig::<f64>::default();
        assert!(valid.validate().is_ok());

        let invalid = WettingDryingConfig::<f64> {
            h_min: -1.0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());

        let invalid = WettingDryingConfig::<f64> {
            h_wet: 1e-5,
            h_dry: 1e-4,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_config_from_params() {
        let params = NumericalParams::<f64>::default();
        let config = WettingDryingConfig::<f64>::from_params(&params);

        assert_eq!(config.h_dry, params.h_dry);
        assert_eq!(config.h_wet, params.h_wet);
        assert_eq!(config.h_min, params.h_min);
    }

    #[test]
    fn test_type_aliases() {
        let handler_f64 = WettingDryingHandlerF64::from_params(
            &NumericalParams::<f64>::default()
        );
        let handler_f32 = WettingDryingHandlerF32::from_params(
            &NumericalParams::<f32>::default()
        );

        assert_eq!(std::mem::size_of_val(&handler_f64.config.h_dry), 8);
        assert_eq!(std::mem::size_of_val(&handler_f32.config.h_dry), 4);
    }
}