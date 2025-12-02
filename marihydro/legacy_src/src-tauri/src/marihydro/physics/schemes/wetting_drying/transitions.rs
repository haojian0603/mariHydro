// src-tauri/src/marihydro/physics/schemes/wetting_drying/transitions.rs

//! 干湿过渡函数
//!
//! 提供多种平滑过渡函数，用于干湿边界的数值稳定处理。

use std::f64::consts::PI;

/// 过渡函数类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmoothingType {
    /// 线性过渡
    Linear,
    /// Hermite 多项式 (C1 连续) - 默认
    #[default]
    Hermite,
    /// 余弦过渡 (C∞ 连续)
    Cosine,
    /// 五次多项式 (C2 连续)
    Quintic,
}

/// 过渡函数计算器
#[derive(Debug, Clone, Copy)]
pub struct TransitionFunction {
    /// 过渡类型
    pub smoothing_type: SmoothingType,
    /// 下界（干阈值）
    pub lower: f64,
    /// 上界（湿阈值）
    pub upper: f64,
}

impl TransitionFunction {
    /// 创建过渡函数
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            smoothing_type: SmoothingType::default(),
            lower,
            upper,
        }
    }

    /// 指定过渡类型
    pub fn with_type(mut self, smoothing_type: SmoothingType) -> Self {
        self.smoothing_type = smoothing_type;
        self
    }

    /// 计算过渡因子 [0, 1]
    ///
    /// # 参数
    /// - `x`: 输入值（如水深）
    ///
    /// # 返回
    /// - 0.0: x <= lower
    /// - 1.0: x >= upper
    /// - (0, 1): lower < x < upper
    #[inline]
    pub fn evaluate(&self, x: f64) -> f64 {
        if x <= self.lower {
            0.0
        } else if x >= self.upper {
            1.0
        } else {
            let t = (x - self.lower) / (self.upper - self.lower);
            self.apply(t)
        }
    }

    /// 对归一化参数 t ∈ [0, 1] 应用平滑函数
    #[inline]
    fn apply(&self, t: f64) -> f64 {
        match self.smoothing_type {
            SmoothingType::Linear => t,
            SmoothingType::Hermite => hermite_smooth(t),
            SmoothingType::Cosine => cosine_smooth(t),
            SmoothingType::Quintic => quintic_smooth(t),
        }
    }

    /// 计算过渡函数的导数
    ///
    /// 返回 df/dx 用于某些数值方法
    #[inline]
    pub fn derivative(&self, x: f64) -> f64 {
        if x <= self.lower || x >= self.upper {
            0.0
        } else {
            let range = self.upper - self.lower;
            let t = (x - self.lower) / range;
            self.apply_derivative(t) / range
        }
    }

    /// 对归一化参数 t 计算平滑函数导数
    #[inline]
    fn apply_derivative(&self, t: f64) -> f64 {
        match self.smoothing_type {
            SmoothingType::Linear => 1.0,
            SmoothingType::Hermite => hermite_smooth_derivative(t),
            SmoothingType::Cosine => cosine_smooth_derivative(t),
            SmoothingType::Quintic => quintic_smooth_derivative(t),
        }
    }
}

// ================== 平滑函数实现 ==================

/// 线性插值
/// f(t) = t
#[inline]
pub fn linear_smooth(t: f64) -> f64 {
    t.clamp(0.0, 1.0)
}

/// Hermite 多项式平滑 (C1 连续)
/// f(t) = 3t² - 2t³
///
/// 特点：端点处导数为 0
#[inline]
pub fn hermite_smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Hermite 多项式导数
/// f'(t) = 6t - 6t²
#[inline]
pub fn hermite_smooth_derivative(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    6.0 * t * (1.0 - t)
}

/// 余弦平滑 (C∞ 连续)
/// f(t) = (1 - cos(πt)) / 2
#[inline]
pub fn cosine_smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    (1.0 - (PI * t).cos()) * 0.5
}

/// 余弦平滑导数
/// f'(t) = π/2 * sin(πt)
#[inline]
pub fn cosine_smooth_derivative(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    0.5 * PI * (PI * t).sin()
}

/// 五次多项式平滑 (C2 连续)
/// f(t) = 6t⁵ - 15t⁴ + 10t³
///
/// 特点：端点处一阶和二阶导数都为 0
#[inline]
pub fn quintic_smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    t3 * (6.0 * t2 - 15.0 * t + 10.0)
}

/// 五次多项式导数
/// f'(t) = 30t⁴ - 60t³ + 30t²
#[inline]
pub fn quintic_smooth_derivative(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let t2 = t * t;
    30.0 * t2 * (t2 - 2.0 * t + 1.0)
}

// ================== 复合过渡函数 ==================

/// 双边过渡（用于周期性边界等场景）
///
/// 在 [lower1, upper1] 从 0 过渡到 1，
/// 在 [lower2, upper2] 从 1 过渡到 0
#[derive(Debug, Clone, Copy)]
pub struct DualTransition {
    pub rising: TransitionFunction,
    pub falling: TransitionFunction,
}

impl DualTransition {
    /// 创建双边过渡
    pub fn new(lower1: f64, upper1: f64, lower2: f64, upper2: f64) -> Self {
        Self {
            rising: TransitionFunction::new(lower1, upper1),
            falling: TransitionFunction::new(lower2, upper2),
        }
    }

    /// 计算双边过渡因子
    pub fn evaluate(&self, x: f64) -> f64 {
        let rise = self.rising.evaluate(x);
        let fall = 1.0 - self.falling.evaluate(x);
        rise.min(fall)
    }
}

/// 最小水深过渡函数
///
/// 用于防止除零的安全水深计算
#[derive(Debug, Clone, Copy)]
pub struct MinDepthTransition {
    /// 干燥阈值
    pub h_dry: f64,
    /// 湿润阈值
    pub h_wet: f64,
    /// 最小水深
    pub h_min: f64,
}

impl MinDepthTransition {
    /// 创建
    pub fn new(h_dry: f64, h_wet: f64, h_min: f64) -> Self {
        Self { h_dry, h_wet, h_min }
    }

    /// 计算安全水深
    ///
    /// 在干区返回 h_min，在湿区返回实际水深
    #[inline]
    pub fn safe_depth(&self, h: f64) -> f64 {
        if h <= self.h_dry {
            self.h_min
        } else if h >= self.h_wet {
            h
        } else {
            // 过渡区平滑
            let t = (h - self.h_dry) / (self.h_wet - self.h_dry);
            let factor = hermite_smooth(t);
            self.h_min + (h - self.h_min) * factor
        }
    }

    /// 计算安全的倒数 1/h
    #[inline]
    pub fn safe_reciprocal(&self, h: f64) -> f64 {
        1.0 / self.safe_depth(h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hermite_smooth() {
        assert_eq!(hermite_smooth(0.0), 0.0);
        assert_eq!(hermite_smooth(1.0), 1.0);
        assert!((hermite_smooth(0.5) - 0.5).abs() < 1e-10);

        // 端点导数为 0
        assert!((hermite_smooth_derivative(0.0)).abs() < 1e-10);
        assert!((hermite_smooth_derivative(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_quintic_smooth() {
        assert_eq!(quintic_smooth(0.0), 0.0);
        assert_eq!(quintic_smooth(1.0), 1.0);
        assert!((quintic_smooth(0.5) - 0.5).abs() < 1e-10);

        // 端点导数为 0
        assert!((quintic_smooth_derivative(0.0)).abs() < 1e-10);
        assert!((quintic_smooth_derivative(1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_smooth() {
        assert!((cosine_smooth(0.0)).abs() < 1e-10);
        assert!((cosine_smooth(1.0) - 1.0).abs() < 1e-10);
        assert!((cosine_smooth(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_transition_function() {
        let tf = TransitionFunction::new(0.1, 0.9);

        assert_eq!(tf.evaluate(0.0), 0.0);
        assert_eq!(tf.evaluate(0.1), 0.0);
        assert_eq!(tf.evaluate(0.9), 1.0);
        assert_eq!(tf.evaluate(1.0), 1.0);

        // 单调递增
        let v1 = tf.evaluate(0.3);
        let v2 = tf.evaluate(0.5);
        let v3 = tf.evaluate(0.7);
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_min_depth_transition() {
        let mdt = MinDepthTransition::new(1e-4, 1e-3, 1e-6);

        // 干区返回 h_min
        assert_eq!(mdt.safe_depth(0.0), 1e-6);
        assert_eq!(mdt.safe_depth(1e-4), 1e-6);

        // 湿区返回实际水深
        assert_eq!(mdt.safe_depth(1.0), 1.0);

        // 过渡区在两者之间
        let h_mid = 5e-4;
        let safe = mdt.safe_depth(h_mid);
        assert!(safe > 1e-6);
        assert!(safe < h_mid);
    }
}
