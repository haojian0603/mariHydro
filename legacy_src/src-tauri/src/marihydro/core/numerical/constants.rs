//! 数值阈值常量
//!
//! 定义系统中使用的各种数值阈值和物理常数相关的数值安全边界。

/// 浮点数相等性比较的默认容差
pub const DEFAULT_EPSILON: f64 = 1e-14;

/// 安全除法的最小分母阈值
pub const SAFE_DIV_EPSILON: f64 = 1e-14;

/// 水深干湿判断阈值 (m)
pub const WET_DRY_THRESHOLD: f64 = 1e-6;

/// 最小允许水深 (m)
pub const MIN_WATER_DEPTH: f64 = 1e-8;

/// 最大允许水深 (m) - 用于数值稳定性检查
pub const MAX_WATER_DEPTH: f64 = 1e6;

/// 最小允许流速 (m/s)
pub const MIN_VELOCITY: f64 = 1e-10;

/// 最大允许流速 (m/s) - 用于数值稳定性检查
pub const MAX_VELOCITY: f64 = 100.0;

/// 最小允许面积 (m²)
pub const MIN_AREA: f64 = 1e-12;

/// 最大 CFL 数警告阈值
pub const CFL_WARNING_THRESHOLD: f64 = 0.9;

/// 最大 CFL 数错误阈值
pub const CFL_ERROR_THRESHOLD: f64 = 1.0;

/// 梯度限制器的最小值
pub const LIMITER_MIN: f64 = 0.0;

/// 梯度限制器的最大值
pub const LIMITER_MAX: f64 = 1.0;

/// Venkatakrishnan 限制器的 K 参数默认值
pub const VENKAT_K_DEFAULT: f64 = 0.3;

/// 湍流模型的最小湍动能 (m²/s²)
pub const MIN_TKE: f64 = 1e-10;

/// 湍流模型的最小耗散率 (m²/s³)
pub const MIN_EPSILON: f64 = 1e-10;

/// 最小摩擦系数
pub const MIN_FRICTION_COEF: f64 = 0.001;

/// 最大摩擦系数
pub const MAX_FRICTION_COEF: f64 = 0.5;

/// 最小曼宁系数
pub const MIN_MANNING_N: f64 = 0.01;

/// 最大曼宁系数
pub const MAX_MANNING_N: f64 = 0.1;

/// 矩阵条件数警告阈值
pub const CONDITION_NUMBER_WARNING: f64 = 1e10;

/// 矩阵条件数错误阈值
pub const CONDITION_NUMBER_ERROR: f64 = 1e14;

/// 迭代求解器的默认最大迭代次数
pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// 迭代求解器的默认收敛容差
pub const DEFAULT_CONVERGENCE_TOL: f64 = 1e-8;

/// 时间步长的最小值 (s)
pub const MIN_TIME_STEP: f64 = 1e-10;

/// 时间步长的最大值 (s)
pub const MAX_TIME_STEP: f64 = 3600.0;

/// 泥沙浓度的最大值 (kg/m³)
pub const MAX_SEDIMENT_CONCENTRATION: f64 = 1000.0;

/// 密度变化的最大相对值
pub const MAX_DENSITY_VARIATION: f64 = 0.1;

/// 验证值是否在合理范围内
#[inline]
pub fn is_valid_depth(h: f64) -> bool {
    h.is_finite() && h >= MIN_WATER_DEPTH && h <= MAX_WATER_DEPTH
}

/// 验证流速是否在合理范围内
#[inline]
pub fn is_valid_velocity(v: f64) -> bool {
    v.is_finite() && v.abs() <= MAX_VELOCITY
}

/// 验证时间步长是否在合理范围内
#[inline]
pub fn is_valid_timestep(dt: f64) -> bool {
    dt.is_finite() && dt >= MIN_TIME_STEP && dt <= MAX_TIME_STEP
}

/// 验证曼宁系数是否在合理范围内
#[inline]
pub fn is_valid_manning(n: f64) -> bool {
    n.is_finite() && n >= MIN_MANNING_N && n <= MAX_MANNING_N
}

/// 限制水深到安全范围
#[inline]
pub fn clamp_depth(h: f64) -> f64 {
    if !h.is_finite() {
        MIN_WATER_DEPTH
    } else {
        h.clamp(MIN_WATER_DEPTH, MAX_WATER_DEPTH)
    }
}

/// 限制流速到安全范围
#[inline]
pub fn clamp_velocity(v: f64) -> f64 {
    if !v.is_finite() {
        0.0
    } else {
        v.clamp(-MAX_VELOCITY, MAX_VELOCITY)
    }
}

/// 限制时间步长到安全范围
#[inline]
pub fn clamp_timestep(dt: f64) -> f64 {
    if !dt.is_finite() || dt <= 0.0 {
        MIN_TIME_STEP
    } else {
        dt.clamp(MIN_TIME_STEP, MAX_TIME_STEP)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_depth() {
        assert!(is_valid_depth(1.0));
        assert!(is_valid_depth(0.001));
        assert!(!is_valid_depth(f64::NAN));
        assert!(!is_valid_depth(f64::INFINITY));
        assert!(!is_valid_depth(-1.0));
    }

    #[test]
    fn test_clamp_depth() {
        assert_eq!(clamp_depth(1.0), 1.0);
        assert_eq!(clamp_depth(-1.0), MIN_WATER_DEPTH);
        assert_eq!(clamp_depth(f64::NAN), MIN_WATER_DEPTH);
        assert_eq!(clamp_depth(1e10), MAX_WATER_DEPTH);
    }

    #[test]
    fn test_is_valid_velocity() {
        assert!(is_valid_velocity(10.0));
        assert!(is_valid_velocity(-10.0));
        assert!(!is_valid_velocity(200.0));
        assert!(!is_valid_velocity(f64::NAN));
    }
}
