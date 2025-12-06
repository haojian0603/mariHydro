// crates/mh_physics/src/schemes/wetting_drying/transitions.rs

//! 干湿过渡函数
//!
//! 提供用于干湿过渡区域的光滑过渡函数，避免数值不稳定。
//!
//! # 设计目标
//!
//! 1. **连续性**: 所有函数及其导数在过渡区域连续
//! 2. **物理一致性**: 干区（h=0）时输出为0，湿区（h>>h_dry）时输出正常值
//! 3. **数值稳定性**: 避免除零和极大值
//! 4. **效率**: 尽量使用简单的数学运算
//!
//! # 核心函数
//!
//! - [`smooth_heaviside`]: 光滑阶跃函数，用于干湿状态渐变
//! - [`porosity_factor`]: 孔隙率因子，控制浅水区流动
//! - [`velocity_damping`]: 速度阻尼函数，防止浅水区过高速度
//! - [`friction_enhancement`]: 摩阻增强因子，浅水区增加摩阻
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::schemes::wetting_drying::transitions::*;
//!
//! let h = 0.005; // 5mm 水深
//! let h_dry = 0.01; // 干湿阈值
//!
//! // 计算光滑过渡因子
//! let factor = smooth_heaviside(h, h_dry);
//! println!("过渡因子: {:.3}", factor); // 约 0.5
//!
//! // 计算速度阻尼
//! let damping = velocity_damping(h, h_dry, 1.0, 0.1);
//! let damped_vel = u * damping;
//! ```

use std::f64::consts::PI;

// ============================================================================
// 配置结构
// ============================================================================

/// 过渡函数配置
#[derive(Debug, Clone, Copy)]
pub struct TransitionConfig {
    /// 干湿阈值 [m]
    pub h_dry: f64,

    /// 过渡带宽度因子（相对于 h_dry）
    ///
    /// 过渡区域为 [h_dry * (1 - width), h_dry * (1 + width)]
    pub transition_width: f64,

    /// 最小孔隙率（防止完全阻断）
    pub min_porosity: f64,

    /// 最大速度阻尼（完全干时的阻尼）
    pub max_velocity_damping: f64,

    /// 最大摩阻增强因子
    pub max_friction_factor: f64,

    /// 过渡函数类型
    pub function_type: TransitionFunctionType,
}

/// 过渡函数类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionFunctionType {
    /// 多项式过渡（三次 Hermite）
    Polynomial,
    /// 正弦过渡
    Sinusoidal,
    /// 双曲正切过渡
    Hyperbolic,
    /// 指数过渡
    Exponential,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self {
            h_dry: 1e-6,
            transition_width: 0.5,
            min_porosity: 0.01,
            max_velocity_damping: 0.99,
            max_friction_factor: 10.0,
            function_type: TransitionFunctionType::Polynomial,
        }
    }
}

impl TransitionConfig {
    /// 创建保守配置（宽过渡带）
    pub fn conservative() -> Self {
        Self {
            h_dry: 1e-4,
            transition_width: 1.0,
            min_porosity: 0.05,
            max_velocity_damping: 0.95,
            max_friction_factor: 5.0,
            function_type: TransitionFunctionType::Hyperbolic,
        }
    }

    /// 创建高精度配置（窄过渡带）
    pub fn precise() -> Self {
        Self {
            h_dry: 1e-8,
            transition_width: 0.2,
            min_porosity: 0.001,
            max_velocity_damping: 0.999,
            max_friction_factor: 20.0,
            function_type: TransitionFunctionType::Polynomial,
        }
    }

    /// 过渡区域下边界
    #[inline]
    pub fn transition_low(&self) -> f64 {
        self.h_dry * (1.0 - self.transition_width).max(0.0)
    }

    /// 过渡区域上边界
    #[inline]
    pub fn transition_high(&self) -> f64 {
        self.h_dry * (1.0 + self.transition_width)
    }
}

// ============================================================================
// 核心过渡函数
// ============================================================================

/// 光滑 Heaviside（阶跃）函数
///
/// 在 h = h_dry 附近提供光滑过渡：
/// - h << h_dry: 返回 0（完全干）
/// - h >> h_dry: 返回 1（完全湿）
/// - h ≈ h_dry: 返回 0.5（过渡区）
///
/// # 参数
///
/// * `h` - 水深 [m]
/// * `h_dry` - 干湿阈值 [m]
///
/// # 返回
///
/// 过渡因子 [0, 1]
#[inline]
pub fn smooth_heaviside(h: f64, h_dry: f64) -> f64 {
    smooth_heaviside_with_width(h, h_dry, 0.5)
}

/// 带宽度参数的光滑 Heaviside 函数
///
/// # 参数
///
/// * `h` - 水深 [m]
/// * `h_dry` - 干湿阈值 [m]
/// * `width` - 过渡带相对宽度（推荐 0.3-1.0）
#[inline]
pub fn smooth_heaviside_with_width(h: f64, h_dry: f64, width: f64) -> f64 {
    if h_dry <= 0.0 {
        return if h > 0.0 { 1.0 } else { 0.0 };
    }

    let h_low = h_dry * (1.0 - width).max(0.0);
    let h_high = h_dry * (1.0 + width);

    if h <= h_low {
        0.0
    } else if h >= h_high {
        1.0
    } else {
        // 归一化到 [0, 1]
        let t = (h - h_low) / (h_high - h_low);
        // 三次 Hermite 插值: 3t² - 2t³
        t * t * (3.0 - 2.0 * t)
    }
}

/// 正弦过渡函数
///
/// 使用正弦函数提供光滑过渡。
#[inline]
pub fn sinusoidal_transition(h: f64, h_dry: f64, width: f64) -> f64 {
    if h_dry <= 0.0 {
        return if h > 0.0 { 1.0 } else { 0.0 };
    }

    let h_low = h_dry * (1.0 - width).max(0.0);
    let h_high = h_dry * (1.0 + width);

    if h <= h_low {
        0.0
    } else if h >= h_high {
        1.0
    } else {
        let t = (h - h_low) / (h_high - h_low);
        0.5 * (1.0 - (PI * (1.0 - t)).cos())
    }
}

/// 双曲正切过渡函数
///
/// 使用 tanh 提供更陡峭的过渡。
#[inline]
pub fn hyperbolic_transition(h: f64, h_dry: f64, steepness: f64) -> f64 {
    if h_dry <= 0.0 {
        return if h > 0.0 { 1.0 } else { 0.0 };
    }

    // tanh((h - h_dry) / (h_dry * steepness)) 映射到 [0, 1]
    let x = (h - h_dry) / (h_dry * steepness.max(0.01));
    0.5 * (1.0 + x.tanh())
}

/// 指数过渡函数
///
/// 使用指数函数，在 h=0 时快速接近 0。
#[inline]
pub fn exponential_transition(h: f64, h_dry: f64, decay_rate: f64) -> f64 {
    if h <= 0.0 {
        return 0.0;
    }
    if h_dry <= 0.0 {
        return 1.0;
    }

    // 1 - exp(-decay_rate * h / h_dry)
    let x = decay_rate * h / h_dry;
    1.0 - (-x).exp()
}

// ============================================================================
// 物理过渡函数
// ============================================================================

/// 孔隙率因子
///
/// 控制浅水区的有效流动面积，模拟浅水流动受阻效应。
///
/// # 参数
///
/// * `h` - 水深 [m]
/// * `h_dry` - 干湿阈值 [m]
/// * `min_porosity` - 最小孔隙率（防止完全阻断）
///
/// # 返回
///
/// 孔隙率因子 [min_porosity, 1.0]
///
/// # 物理意义
///
/// - φ = 1.0: 完全自由流动
/// - φ < 1.0: 部分受阻流动
/// - φ → min_porosity: 接近完全阻断
#[inline]
pub fn porosity_factor(h: f64, h_dry: f64, min_porosity: f64) -> f64 {
    let transition = smooth_heaviside(h, h_dry);
    min_porosity + (1.0 - min_porosity) * transition
}

/// 速度阻尼因子
///
/// 在浅水区阻尼速度，防止出现非物理的高速度。
///
/// # 参数
///
/// * `h` - 水深 [m]
/// * `h_dry` - 干湿阈值 [m]
/// * `transition_width` - 过渡带宽度
/// * `max_damping` - 最大阻尼系数 (0-1)
///
/// # 返回
///
/// 速度保留因子 [1 - max_damping, 1.0]
///
/// # 使用方式
///
/// ```ignore
/// let factor = velocity_damping(h, h_dry, 0.5, 0.9);
/// let u_new = u * factor;  // 阻尼后的速度
/// ```
#[inline]
pub fn velocity_damping(h: f64, h_dry: f64, transition_width: f64, max_damping: f64) -> f64 {
    let transition = smooth_heaviside_with_width(h, h_dry, transition_width);
    1.0 - max_damping * (1.0 - transition)
}

/// 摩阻增强因子
///
/// 在浅水区增强底床摩阻，模拟浅水流动的额外阻力。
///
/// # 参数
///
/// * `h` - 水深 [m]
/// * `h_dry` - 干湿阈值 [m]
/// * `max_factor` - 最大增强因子（通常 5-20）
///
/// # 返回
///
/// 摩阻系数乘数 [1.0, max_factor]
///
/// # 使用方式
///
/// ```ignore
/// let factor = friction_enhancement(h, h_dry, 10.0);
/// let tau = tau_base * factor;  // 增强后的摩阻
/// ```
#[inline]
pub fn friction_enhancement(h: f64, h_dry: f64, max_factor: f64) -> f64 {
    let transition = smooth_heaviside(h, h_dry);
    1.0 + (max_factor - 1.0) * (1.0 - transition)
}

/// 计算安全速度
///
/// 根据水深限制速度，避免干区出现非物理高速。
///
/// # 参数
///
/// * `hu` - 动量 (h*u) [m²/s]
/// * `h` - 水深 [m]
/// * `h_dry` - 干湿阈值 [m]
/// * `max_velocity` - 最大允许速度 [m/s]
///
/// # 返回
///
/// 安全的速度值 [m/s]
#[inline]
pub fn safe_velocity(hu: f64, h: f64, h_dry: f64, max_velocity: f64) -> f64 {
    if h <= h_dry {
        0.0
    } else {
        let u = hu / h;
        // 光滑过渡到零
        let factor = velocity_damping(h, h_dry, 0.5, 0.99);
        let u_damped = u * factor;
        // 限制最大速度
        u_damped.clamp(-max_velocity, max_velocity)
    }
}

/// 计算安全 Froude 数
///
/// 在干区和浅水区安全计算 Froude 数。
///
/// # 参数
///
/// * `u` - 速度 [m/s]
/// * `h` - 水深 [m]
/// * `g` - 重力加速度 [m/s²]
/// * `h_min` - 最小有效水深 [m]
///
/// # 返回
///
/// Froude 数 [-]
#[inline]
pub fn safe_froude(u: f64, h: f64, g: f64, h_min: f64) -> f64 {
    let h_eff = h.max(h_min);
    let c = (g * h_eff).sqrt();
    u.abs() / c
}

// ============================================================================
// 过渡函数工具类
// ============================================================================

/// 过渡函数计算器
///
/// 封装配置并提供统一的过渡函数接口。
#[derive(Debug, Clone)]
pub struct TransitionCalculator {
    config: TransitionConfig,
}

impl TransitionCalculator {
    /// 创建新的计算器
    pub fn new(config: TransitionConfig) -> Self {
        Self { config }
    }

    /// 从干湿阈值创建默认配置的计算器
    pub fn from_h_dry(h_dry: f64) -> Self {
        Self::new(TransitionConfig {
            h_dry,
            ..Default::default()
        })
    }

    /// 获取配置
    pub fn config(&self) -> &TransitionConfig {
        &self.config
    }

    /// 计算主过渡函数
    pub fn transition(&self, h: f64) -> f64 {
        match self.config.function_type {
            TransitionFunctionType::Polynomial => {
                smooth_heaviside_with_width(h, self.config.h_dry, self.config.transition_width)
            }
            TransitionFunctionType::Sinusoidal => {
                sinusoidal_transition(h, self.config.h_dry, self.config.transition_width)
            }
            TransitionFunctionType::Hyperbolic => {
                hyperbolic_transition(h, self.config.h_dry, 1.0 / self.config.transition_width)
            }
            TransitionFunctionType::Exponential => {
                exponential_transition(h, self.config.h_dry, 3.0)
            }
        }
    }

    /// 计算孔隙率因子
    pub fn porosity(&self, h: f64) -> f64 {
        let t = self.transition(h);
        self.config.min_porosity + (1.0 - self.config.min_porosity) * t
    }

    /// 计算速度阻尼因子
    pub fn velocity_damping(&self, h: f64) -> f64 {
        let t = self.transition(h);
        1.0 - self.config.max_velocity_damping * (1.0 - t)
    }

    /// 计算摩阻增强因子
    pub fn friction_factor(&self, h: f64) -> f64 {
        let t = self.transition(h);
        1.0 + (self.config.max_friction_factor - 1.0) * (1.0 - t)
    }

    /// 判断是否完全干
    pub fn is_dry(&self, h: f64) -> bool {
        h <= self.config.transition_low()
    }

    /// 判断是否完全湿
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.config.transition_high()
    }

    /// 判断是否在过渡区
    pub fn is_transitional(&self, h: f64) -> bool {
        !self.is_dry(h) && !self.is_wet(h)
    }
}

impl Default for TransitionCalculator {
    fn default() -> Self {
        Self::new(TransitionConfig::default())
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smooth_heaviside_bounds() {
        let h_dry = 0.01;

        // 完全干
        assert_eq!(smooth_heaviside(0.0, h_dry), 0.0);
        assert!(smooth_heaviside(0.001, h_dry) < 0.5);

        // 过渡区
        assert!((smooth_heaviside(h_dry, h_dry) - 0.5).abs() < 0.1);

        // 完全湿
        assert!(smooth_heaviside(0.02, h_dry) > 0.9);
        assert_eq!(smooth_heaviside(0.1, h_dry), 1.0);
    }

    #[test]
    fn test_smooth_heaviside_continuity() {
        let h_dry = 0.01;
        let width = 0.5;

        // 测试连续性
        let mut prev = smooth_heaviside_with_width(0.0, h_dry, width);
        for i in 1..=100 {
            let h = i as f64 * 0.0003;
            let curr = smooth_heaviside_with_width(h, h_dry, width);
            let diff = (curr - prev).abs();
            assert!(diff < 0.1, "Discontinuity at h={}: diff={}", h, diff);
            prev = curr;
        }
    }

    #[test]
    fn test_sinusoidal_transition() {
        let h_dry = 0.01;
        let width = 0.5;

        assert_eq!(sinusoidal_transition(0.0, h_dry, width), 0.0);
        assert_eq!(sinusoidal_transition(0.02, h_dry, width), 1.0);
        assert!((sinusoidal_transition(0.01, h_dry, width) - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_hyperbolic_transition() {
        let h_dry = 0.01;

        // 在中心应该接近 0.5
        assert!((hyperbolic_transition(h_dry, h_dry, 0.5) - 0.5).abs() < 0.01);

        // 远离中心
        assert!(hyperbolic_transition(0.0, h_dry, 0.5) < 0.1);
        assert!(hyperbolic_transition(0.03, h_dry, 0.5) > 0.9);
    }

    #[test]
    fn test_exponential_transition() {
        let h_dry = 0.01;

        assert_eq!(exponential_transition(0.0, h_dry, 3.0), 0.0);
        assert!(exponential_transition(h_dry, h_dry, 3.0) > 0.9);
        assert!(exponential_transition(2.0 * h_dry, h_dry, 3.0) > 0.99);
    }

    #[test]
    fn test_porosity_factor() {
        let h_dry = 0.01;
        let min_porosity = 0.05;

        // 干区
        let phi_dry = porosity_factor(0.0, h_dry, min_porosity);
        assert!((phi_dry - min_porosity).abs() < 0.01);

        // 湿区
        let phi_wet = porosity_factor(0.1, h_dry, min_porosity);
        assert!((phi_wet - 1.0).abs() < 0.01);

        // 过渡区
        let phi_trans = porosity_factor(h_dry, h_dry, min_porosity);
        assert!(phi_trans > min_porosity);
        assert!(phi_trans < 1.0);
    }

    #[test]
    fn test_velocity_damping() {
        let h_dry = 0.01;
        let width = 0.5;
        let max_damping = 0.9;

        // 干区：最大阻尼
        let factor_dry = velocity_damping(0.0, h_dry, width, max_damping);
        assert!((factor_dry - 0.1).abs() < 0.01);

        // 湿区：无阻尼
        let factor_wet = velocity_damping(0.1, h_dry, width, max_damping);
        assert!((factor_wet - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_friction_enhancement() {
        let h_dry = 0.01;
        let max_factor = 10.0;

        // 干区：最大增强
        let factor_dry = friction_enhancement(0.0, h_dry, max_factor);
        assert!((factor_dry - max_factor).abs() < 0.1);

        // 湿区：无增强
        let factor_wet = friction_enhancement(0.1, h_dry, max_factor);
        assert!((factor_wet - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_safe_velocity() {
        let h_dry = 0.01;
        let max_vel = 10.0;

        // 干区
        assert_eq!(safe_velocity(1.0, 0.0, h_dry, max_vel), 0.0);

        // 浅水区
        let u = safe_velocity(0.01, 0.005, h_dry, max_vel); // hu/h = 2
        assert!(u.abs() < 2.0); // 应该被阻尼

        // 湿区
        let u_wet = safe_velocity(0.1, 0.1, h_dry, max_vel); // hu/h = 1
        assert!((u_wet - 1.0).abs() < 0.1); // 接近原始速度
    }

    #[test]
    fn test_safe_froude() {
        let g = 9.81;
        let h_min = 1e-6;

        // 正常计算
        let fr = safe_froude(3.13, 1.0, g, h_min);
        assert!((fr - 1.0).abs() < 0.01); // sqrt(g*1) ≈ 3.13

        // 极小水深
        let fr_small = safe_froude(1.0, 0.0, g, h_min);
        assert!(fr_small.is_finite());
    }

    #[test]
    fn test_transition_calculator() {
        let config = TransitionConfig {
            h_dry: 0.01,
            transition_width: 0.5,
            min_porosity: 0.05,
            max_velocity_damping: 0.9,
            max_friction_factor: 10.0,
            function_type: TransitionFunctionType::Polynomial,
        };
        let calc = TransitionCalculator::new(config);

        // 测试状态判断
        assert!(calc.is_dry(0.001));
        assert!(calc.is_wet(0.02));
        assert!(calc.is_transitional(0.01));

        // 测试各函数
        let t = calc.transition(0.01);
        assert!(t > 0.0 && t < 1.0);

        let p = calc.porosity(0.0);
        assert!((p - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_transition_calculator_presets() {
        let conservative = TransitionCalculator::new(TransitionConfig::conservative());
        let precise = TransitionCalculator::new(TransitionConfig::precise());

        // 保守配置有更宽的过渡
        assert!(conservative.config().transition_width > precise.config().transition_width);

        // 精确配置有更小的 h_dry
        assert!(conservative.config().h_dry > precise.config().h_dry);
    }

    #[test]
    fn test_all_function_types() {
        let h_dry = 0.01;
        let test_h = [0.0, 0.005, 0.01, 0.015, 0.02];

        for func_type in [
            TransitionFunctionType::Polynomial,
            TransitionFunctionType::Sinusoidal,
            TransitionFunctionType::Hyperbolic,
            TransitionFunctionType::Exponential,
        ] {
            let config = TransitionConfig {
                h_dry,
                function_type: func_type,
                ..Default::default()
            };
            let calc = TransitionCalculator::new(config);

            for &h in &test_h {
                let t = calc.transition(h);
                assert!(
                    t >= 0.0 && t <= 1.0,
                    "{:?} at h={}: t={}",
                    func_type,
                    h,
                    t
                );
            }
        }
    }
}
