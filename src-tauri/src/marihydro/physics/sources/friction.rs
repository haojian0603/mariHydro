//! 底摩擦模型
//!
//! 提供Manning、Chezy、植被阻力等多种参数化方案

use crate::marihydro::infra::constants::physics;

/// 半隐式Manning底摩擦（向后兼容接口）
///
/// # 算法
/// 求解隐式方程: u_new = u_old / (1 + dt * C_f * |V|)
/// 其中 C_f = g * n² / h^(1/3)
///
/// # 参数
/// - `u_old`, `v_old`: 旧时刻速度 [m/s]
/// - `h`: 水深 [m]
/// - `n`: Manning糙率系数 [s/m^(1/3)]
/// - `dt`: 时间步长 [s]
/// - `g`: 重力加速度 [m/s²]
/// - `h_min`: 最小水深阈值 [m]
pub fn apply_friction_implicit(
    u_old: f64,
    v_old: f64,
    h: f64,
    n: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) -> (f64, f64) {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let speed = (u_old * u_old + v_old * v_old).sqrt();
    if speed < 1e-10 {
        return (u_old, v_old);
    }

    let cf = compute_manning_coefficient(h, n, g, h_min);
    let denominator = 1.0 + dt * cf * speed;

    (u_old / denominator, v_old / denominator)
}

/// 计算Manning摩擦系数
///
/// C_f = g * n² / h^(1/3)
#[inline]
pub fn compute_manning_coefficient(h: f64, n: f64, g: f64, h_min: f64) -> f64 {
    let h_safe = h.max(h_min);
    g * n.powi(2) / h_safe.powf(1.0 / 3.0)
}

/// 计算Chezy摩擦系数
///
/// C_f = g / C²
///
/// # 参数
/// - `chezy_c`: Chezy系数 [m^(1/2)/s]，典型值20-100
#[inline]
pub fn compute_chezy_coefficient(chezy_c: f64, g: f64) -> f64 {
    g / chezy_c.powi(2)
}

/// 计算植被阻力系数
///
/// C_f = C_d * m * h_submerged / h
///
/// # 参数
/// - `cd`: 植被拖曳系数（无量纲），典型值0.5-2.0
/// - `m`: 植被密度 [stems/m²]
/// - `h_veg`: 植被高度 [m]
/// - `h`: 当前水深 [m]
/// - `h_min`: 最小水深 [m]
#[inline]
pub fn compute_vegetation_coefficient(cd: f64, m: f64, h_veg: f64, h: f64, h_min: f64) -> f64 {
    let h_safe = h.max(h_min);
    let h_submerged = h_safe.min(h_veg);
    cd * m * h_submerged / h_safe
}

/// 半隐式Chezy底摩擦
pub fn apply_friction_chezy(
    u_old: f64,
    v_old: f64,
    h: f64,
    chezy_c: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) -> (f64, f64) {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let speed = (u_old * u_old + v_old * v_old).sqrt();
    if speed < 1e-10 {
        return (u_old, v_old);
    }

    let cf = compute_chezy_coefficient(chezy_c, g);
    let denominator = 1.0 + dt * cf * speed;

    (u_old / denominator, v_old / denominator)
}

/// 半隐式植被阻力
pub fn apply_friction_vegetation(
    u_old: f64,
    v_old: f64,
    h: f64,
    cd: f64,
    m: f64,
    h_veg: f64,
    dt: f64,
    h_min: f64,
) -> (f64, f64) {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let speed = (u_old * u_old + v_old * v_old).sqrt();
    if speed < 1e-10 {
        return (u_old, v_old);
    }

    let cf = compute_vegetation_coefficient(cd, m, h_veg, h, h_min);
    let denominator = 1.0 + dt * cf * speed;

    (u_old / denominator, v_old / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manning_coefficient() {
        let g = physics::STANDARD_GRAVITY;
        let n = 0.025;
        let h = 10.0;

        let cf = compute_manning_coefficient(h, n, g, 0.01);
        let expected = g * n.powi(2) / h.powf(1.0 / 3.0);

        assert!((cf - expected).abs() < 1e-9);
    }

    #[test]
    fn test_chezy_coefficient() {
        let g = physics::STANDARD_GRAVITY;
        let c = 50.0;

        let cf = compute_chezy_coefficient(c, g);
        let expected = g / c.powi(2);

        assert!((cf - expected).abs() < 1e-9);
    }

    #[test]
    fn test_vegetation_submergence() {
        let cd = 1.0;
        let m = 100.0;
        let h_veg = 0.5;

        // 完全淹没
        let cf_deep = compute_vegetation_coefficient(cd, m, h_veg, 2.0, 0.01);
        assert!((cf_deep - 25.0).abs() < 1e-6); // 1.0 * 100 * 0.5 / 2.0

        // 部分淹没
        let cf_shallow = compute_vegetation_coefficient(cd, m, h_veg, 0.3, 0.01);
        assert!((cf_shallow - 100.0).abs() < 1e-6); // 1.0 * 100 * 0.3 / 0.3
    }

    #[test]
    fn test_implicit_stability() {
        let g = physics::STANDARD_GRAVITY;

        // 大时间步长测试
        let (u, v) = apply_friction_implicit(1.0, 1.0, 10.0, 0.025, 100.0, g, 0.01);

        assert!(u > 0.0 && u < 1.0);
        assert!(v > 0.0 && v < 1.0);

        // 速度应按比例衰减
        assert!((u - v).abs() < 1e-10);
    }

    #[test]
    fn test_dry_cell_zero_velocity() {
        let g = physics::STANDARD_GRAVITY;

        let (u, v) = apply_friction_implicit(1.0, 1.0, 0.001, 0.025, 1.0, g, 0.05);

        assert_eq!(u, 0.0);
        assert_eq!(v, 0.0);
    }
}
