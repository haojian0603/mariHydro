// src-tauri/src/marihydro/core/types/numerical_params.rs

//! 数值计算参数
//!
//! 集中管理所有数值阈值和计算参数，确保一致性和可配置性。

use super::safe_types::{SafeDepth, SafeVelocity};
use serde::{Deserialize, Serialize};
use std::fmt;

/// 数值计算参数（集中管理所有阈值）
///
/// # 设计原则
///
/// 1. 水深阈值分层：h_min < h_dry < h_friction < h_wet
/// 2. 所有阈值都有物理意义和默认值
/// 3. 提供验证方法确保一致性
///
/// # 水深阈值层级说明
///
/// ```text
/// 0 ─────────┬─ h_min ──────── 数值零，避免除零
///            │                  (1e-9 m)
///            ├─ h_dry ──────── 干湿判断阈值
///            │                  (1e-6 m)
///            ├─ h_friction ─── 摩擦计算安全水深
///            │                  (1e-4 m)
///            ├─ h_wet ──────── 完全湿单元阈值
///            │                  (1e-3 m)
///            │
///            ▼ 完全参与计算
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalParams {
    // ===== 水深阈值（严格递增）=====
    /// 绝对最小水深（数值零，用于避免除零）
    /// 默认: 1e-9 m
    pub h_min: f64,

    /// 干湿判断阈值（低于此值视为干单元）
    /// 默认: 1e-6 m
    pub h_dry: f64,

    /// 摩擦计算安全水深（保证摩擦项数值稳定）
    /// 默认: 1e-4 m
    pub h_friction: f64,

    /// 完全湿单元阈值（高于此值完全参与计算）
    /// 默认: 1e-3 m
    pub h_wet: f64,

    // ===== 通量计算阈值 =====
    /// 通量分母最小值（防止除零）
    /// 默认: 1e-14
    pub flux_eps: f64,

    /// 熵修正比例（相对于当地波速）
    /// 默认: 1e-4
    pub entropy_ratio: f64,

    /// 最小波速（用于CFL计算）
    /// 默认: 1e-10 m/s
    pub min_wave_speed: f64,

    // ===== 梯度计算阈值 =====
    /// 最小二乘矩阵行列式阈值（低于此值回退到Green-Gauss）
    /// 默认: 1e-12
    pub det_min: f64,

    /// 梯度限制器参数K（Venkatakrishnan限制器）
    /// 默认: 0.3
    pub limiter_k: f64,

    // ===== 速度阈值 =====
    /// 最小有效速度
    /// 默认: 1e-10 m/s
    pub vel_min: f64,

    /// 最大允许速度（超过则触发警告）
    /// 默认: 100.0 m/s
    pub vel_max: f64,

    // ===== 湍流模型 =====
    /// 最小涡粘系数
    /// 默认: 1e-6 m²/s
    pub nu_min: f64,

    /// 最大涡粘系数
    /// 默认: 1e3 m²/s
    pub nu_max: f64,

    /// Smagorinsky常数
    /// 默认: 0.15
    pub cs: f64,

    // ===== 时间步进 =====
    /// CFL数
    /// 默认: 0.5
    pub cfl: f64,

    /// 最小时间步长
    /// 默认: 1e-8 s
    pub dt_min: f64,

    /// 最大时间步长
    /// 默认: 3600.0 s
    pub dt_max: f64,

    // ===== 迭代控制 =====
    /// 最大迭代次数
    /// 默认: 100
    pub max_iterations: usize,

    /// 收敛容差
    /// 默认: 1e-8
    pub convergence_tol: f64,
}

impl Default for NumericalParams {
    fn default() -> Self {
        Self {
            // 水深阈值（严格递增）
            h_min: 1e-9,
            h_dry: 1e-6,
            h_friction: 1e-4,
            h_wet: 1e-3,

            // 通量阈值
            flux_eps: 1e-14,
            entropy_ratio: 1e-4,
            min_wave_speed: 1e-10,

            // 梯度阈值
            det_min: 1e-12,
            limiter_k: 0.3,

            // 速度阈值
            vel_min: 1e-10,
            vel_max: 100.0,

            // 湍流
            nu_min: 1e-6,
            nu_max: 1e3,
            cs: 0.15,

            // 时间
            cfl: 0.5,
            dt_min: 1e-8,
            dt_max: 3600.0,

            // 迭代
            max_iterations: 100,
            convergence_tol: 1e-8,
        }
    }
}

impl NumericalParams {
    /// 构建器
    pub fn builder() -> NumericalParamsBuilder {
        NumericalParamsBuilder::default()
    }

    /// 创建保守配置（更严格的阈值）
    pub fn conservative() -> Self {
        Self {
            cfl: 0.3,
            h_min: 1e-8,
            h_dry: 1e-5,
            vel_max: 50.0,
            ..Default::default()
        }
    }

    /// 创建高性能配置（更宽松的阈值）
    pub fn performance() -> Self {
        Self {
            cfl: 0.8,
            h_min: 1e-10,
            h_dry: 1e-7,
            vel_max: 150.0,
            ..Default::default()
        }
    }

    /// 验证参数一致性
    pub fn validate(&self) -> Result<(), ParamsValidationError> {
        // 水深阈值必须严格递增
        if self.h_min >= self.h_dry {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_min",
                constraint: "h_min < h_dry",
                value: self.h_min,
            });
        }
        if self.h_dry >= self.h_friction {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_dry",
                constraint: "h_dry < h_friction",
                value: self.h_dry,
            });
        }
        if self.h_friction >= self.h_wet {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "h_friction",
                constraint: "h_friction < h_wet",
                value: self.h_friction,
            });
        }

        // CFL数必须在合理范围
        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(ParamsValidationError::OutOfRange {
                field: "cfl",
                min: 0.0,
                max: 1.0,
                value: self.cfl,
            });
        }

        // 时间步长范围
        if self.dt_min >= self.dt_max {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "dt_min",
                constraint: "dt_min < dt_max",
                value: self.dt_min,
            });
        }

        // dt 必须为正
        if self.dt_min <= 0.0 {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "dt_min",
                constraint: "dt_min > 0",
                value: self.dt_min,
            });
        }

        // 涡粘范围
        if self.nu_min >= self.nu_max {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "nu_min",
                constraint: "nu_min < nu_max",
                value: self.nu_min,
            });
        }

        // nu 必须为正
        if self.nu_min < 0.0 {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "nu_min",
                constraint: "nu_min >= 0",
                value: self.nu_min,
            });
        }

        // 速度阈值
        if self.vel_max <= 0.0 {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "vel_max",
                constraint: "vel_max > 0",
                value: self.vel_max,
            });
        }

        // Smagorinsky 常数
        if self.cs <= 0.0 || self.cs > 1.0 {
            return Err(ParamsValidationError::OutOfRange {
                field: "cs",
                min: 0.0,
                max: 1.0,
                value: self.cs,
            });
        }

        // 限制器参数
        if self.limiter_k <= 0.0 {
            return Err(ParamsValidationError::InvalidThreshold {
                field: "limiter_k",
                constraint: "limiter_k > 0",
                value: self.limiter_k,
            });
        }

        Ok(())
    }

    // ===== 安全辅助方法 =====

    /// 判断是否为干单元
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.h_dry
    }

    /// 判断是否为湿单元
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.h_wet
    }

    /// 判断是否在过渡区
    #[inline]
    pub fn is_transition(&self, h: f64) -> bool {
        h >= self.h_dry && h < self.h_wet
    }

    /// 创建安全水深
    #[inline]
    pub fn safe_depth(&self, h: f64) -> SafeDepth {
        SafeDepth::new(h, self.h_min)
    }

    /// 创建摩擦安全水深
    #[inline]
    pub fn friction_safe_depth(&self, h: f64) -> SafeDepth {
        SafeDepth::new(h, self.h_friction)
    }

    /// 计算安全速度
    #[inline]
    pub fn safe_velocity(&self, hu: f64, hv: f64, h: f64) -> SafeVelocity {
        SafeVelocity::from_momentum(hu, hv, h, self.h_dry, self.h_min)
    }

    /// 动态熵修正阈值
    #[inline]
    pub fn entropy_threshold(&self, local_wave_speed: f64) -> f64 {
        (self.entropy_ratio * local_wave_speed.abs()).max(self.flux_eps)
    }

    /// 计算最大允许时间步（基于CFL）
    #[inline]
    pub fn max_dt_from_cfl(&self, dx: f64, max_wave_speed: f64) -> f64 {
        let wave_speed = max_wave_speed.max(self.min_wave_speed);
        let dt = self.cfl * dx / wave_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }

    /// 限制涡粘系数
    #[inline]
    pub fn clamp_nu(&self, nu: f64) -> f64 {
        nu.clamp(self.nu_min, self.nu_max)
    }

    /// 限制速度
    #[inline]
    pub fn clamp_velocity(&self, vel: SafeVelocity) -> SafeVelocity {
        vel.clamp_speed(self.vel_max)
    }

    /// 干湿过渡权重（用于平滑过渡）
    ///
    /// 返回值 ∈ [0, 1]：
    /// - 0: 完全干
    /// - 1: 完全湿
    /// - 中间值: 过渡区
    #[inline]
    pub fn wet_fraction(&self, h: f64) -> f64 {
        if h <= self.h_dry {
            0.0
        } else if h >= self.h_wet {
            1.0
        } else {
            // 线性过渡
            (h - self.h_dry) / (self.h_wet - self.h_dry)
        }
    }

    /// 平滑的干湿过渡权重（使用 Hermite 插值）
    #[inline]
    pub fn wet_fraction_smooth(&self, h: f64) -> f64 {
        if h <= self.h_dry {
            0.0
        } else if h >= self.h_wet {
            1.0
        } else {
            // Hermite 平滑插值 (3t² - 2t³)
            let t = (h - self.h_dry) / (self.h_wet - self.h_dry);
            t * t * (3.0 - 2.0 * t)
        }
    }

    /// 检查速度是否超过警告阈值
    #[inline]
    pub fn is_velocity_excessive(&self, speed: f64) -> bool {
        speed > self.vel_max
    }

    /// 计算波速（浅水）
    #[inline]
    pub fn wave_speed(&self, h: f64, g: f64) -> f64 {
        (g * h.max(0.0)).sqrt().max(self.min_wave_speed)
    }
}

/// 参数验证错误
#[derive(Debug, Clone)]
pub enum ParamsValidationError {
    InvalidThreshold {
        field: &'static str,
        constraint: &'static str,
        value: f64,
    },
    OutOfRange {
        field: &'static str,
        min: f64,
        max: f64,
        value: f64,
    },
}

impl fmt::Display for ParamsValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidThreshold {
                field,
                constraint,
                value,
            } => {
                write!(f, "参数 {} = {} 违反约束: {}", field, value, constraint)
            }
            Self::OutOfRange {
                field,
                min,
                max,
                value,
            } => {
                write!(f, "参数 {} = {} 超出范围 [{}, {}]", field, value, min, max)
            }
        }
    }
}

impl std::error::Error for ParamsValidationError {}

/// 参数构建器
#[derive(Default)]
pub struct NumericalParamsBuilder {
    params: NumericalParams,
}

impl NumericalParamsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    // 水深阈值
    pub fn h_min(mut self, v: f64) -> Self {
        self.params.h_min = v;
        self
    }
    pub fn h_dry(mut self, v: f64) -> Self {
        self.params.h_dry = v;
        self
    }
    pub fn h_friction(mut self, v: f64) -> Self {
        self.params.h_friction = v;
        self
    }
    pub fn h_wet(mut self, v: f64) -> Self {
        self.params.h_wet = v;
        self
    }

    // 通量阈值
    pub fn flux_eps(mut self, v: f64) -> Self {
        self.params.flux_eps = v;
        self
    }
    pub fn entropy_ratio(mut self, v: f64) -> Self {
        self.params.entropy_ratio = v;
        self
    }
    pub fn min_wave_speed(mut self, v: f64) -> Self {
        self.params.min_wave_speed = v;
        self
    }

    // 梯度阈值
    pub fn det_min(mut self, v: f64) -> Self {
        self.params.det_min = v;
        self
    }
    pub fn limiter_k(mut self, v: f64) -> Self {
        self.params.limiter_k = v;
        self
    }

    // 速度阈值
    pub fn vel_min(mut self, v: f64) -> Self {
        self.params.vel_min = v;
        self
    }
    pub fn vel_max(mut self, v: f64) -> Self {
        self.params.vel_max = v;
        self
    }

    // 湍流参数
    pub fn nu_min(mut self, v: f64) -> Self {
        self.params.nu_min = v;
        self
    }
    pub fn nu_max(mut self, v: f64) -> Self {
        self.params.nu_max = v;
        self
    }
    pub fn cs(mut self, v: f64) -> Self {
        self.params.cs = v;
        self
    }

    // 时间参数
    pub fn cfl(mut self, v: f64) -> Self {
        self.params.cfl = v;
        self
    }
    pub fn dt_min(mut self, v: f64) -> Self {
        self.params.dt_min = v;
        self
    }
    pub fn dt_max(mut self, v: f64) -> Self {
        self.params.dt_max = v;
        self
    }

    // 迭代控制
    pub fn max_iterations(mut self, v: usize) -> Self {
        self.params.max_iterations = v;
        self
    }
    pub fn convergence_tol(mut self, v: f64) -> Self {
        self.params.convergence_tol = v;
        self
    }

    /// 构建并验证参数
    pub fn build(self) -> Result<NumericalParams, ParamsValidationError> {
        self.params.validate()?;
        Ok(self.params)
    }

    /// 构建但不验证（用于测试）
    pub fn build_unchecked(self) -> NumericalParams {
        self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params_are_valid() {
        let params = NumericalParams::default();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_conservative_params_are_valid() {
        let params = NumericalParams::conservative();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_performance_params_are_valid() {
        let params = NumericalParams::performance();
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_invalid_h_thresholds() {
        let params = NumericalParams {
            h_min: 1e-6,
            h_dry: 1e-9, // 违反 h_min < h_dry
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_invalid_cfl() {
        let params = NumericalParams {
            cfl: 1.5, // > 1
            ..Default::default()
        };
        assert!(params.validate().is_err());
    }

    #[test]
    fn test_dry_wet_classification() {
        let params = NumericalParams::default();

        assert!(params.is_dry(1e-8));
        assert!(!params.is_dry(1.0));

        assert!(params.is_wet(1.0));
        assert!(!params.is_wet(1e-5));

        assert!(params.is_transition(5e-4)); // 在 h_dry 和 h_wet 之间
    }

    #[test]
    fn test_wet_fraction() {
        let params = NumericalParams::default();

        // 干单元
        assert!((params.wet_fraction(0.0) - 0.0).abs() < 1e-10);

        // 湿单元
        assert!((params.wet_fraction(1.0) - 1.0).abs() < 1e-10);

        // 中间值
        let mid = (params.h_dry + params.h_wet) / 2.0;
        let frac = params.wet_fraction(mid);
        assert!(frac > 0.0 && frac < 1.0);
    }

    #[test]
    fn test_max_dt_from_cfl() {
        let params = NumericalParams::default();

        let dt = params.max_dt_from_cfl(10.0, 5.0);
        // CFL = 0.5, dx = 10, c = 5 => dt = 0.5 * 10 / 5 = 1.0
        assert!((dt - 1.0).abs() < 1e-10);

        // 极端情况：波速接近零
        let dt_slow = params.max_dt_from_cfl(10.0, 1e-15);
        assert!(dt_slow <= params.dt_max);
    }

    #[test]
    fn test_builder() {
        let params = NumericalParams::builder()
            .cfl(0.3)
            .h_min(1e-10)
            .build()
            .unwrap();

        assert!((params.cfl - 0.3).abs() < 1e-10);
        assert!((params.h_min - 1e-10).abs() < 1e-20);
    }

    #[test]
    fn test_safe_velocity() {
        let params = NumericalParams::default();

        let vel = params.safe_velocity(2.0, 3.0, 1.0);
        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);

        // 干单元应返回零速度
        let dry_vel = params.safe_velocity(2.0, 3.0, 1e-9);
        assert!((dry_vel.u - 0.0).abs() < 1e-10);
        assert!((dry_vel.v - 0.0).abs() < 1e-10);
    }
}
