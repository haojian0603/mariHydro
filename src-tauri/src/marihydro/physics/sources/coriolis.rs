// src-tauri/src/marihydro/physics/sources/coriolis.rs

use crate::marihydro::infra::constants::tolerances;
use crate::marihydro::types::Velocity2D;
use std::f64::consts::PI;

/// 科氏力精确旋转（保动能）
///
/// 使用辛旋转矩阵精确积分 du/dt = fv, dv/dt = -fu
/// - 北半球 (f > 0): 向右偏转
/// - 南半球 (f < 0): 向左偏转
///
/// # 稳定性
/// 要求 |f·dt| < 0.1（建议），|f·dt| < π（硬性限制）
#[inline(always)]
#[must_use]
pub fn apply_coriolis_exact(u: f64, v: f64, f: f64, dt: f64) -> Velocity2D {
    let theta = f * dt;

    debug_assert!(theta.abs() < PI, "科氏旋转角过大 (θ={:.3} rad)", theta);

    // 小角度泰勒展开优化 (|θ| < 1e-3)
    let (sin_t, cos_t) = if theta.abs() < 1e-3 {
        let t2 = theta * theta;
        (theta * (1.0 - t2 / 6.0), 1.0 - t2 * 0.5)
    } else {
        theta.sin_cos()
    };

    (u * cos_t + v * sin_t, -u * sin_t + v * cos_t)
}

/// 科氏力稳定性检查
#[inline]
pub fn is_stable(f: f64, dt: f64) -> bool {
    (f * dt).abs() < 0.1
}

/// 计算安全时间步长上限
#[inline]
pub fn max_stable_dt(f: f64, safety_factor: f64) -> f64 {
    if f.abs() < tolerances::EPSILON {
        f64::INFINITY
    } else {
        safety_factor * 0.1 / f.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_conservation() {
        let (u0, v0) = (10.0, 5.0);
        let (u1, v1) = apply_coriolis_exact(u0, v0, 1e-4, 3600.0);

        let e0 = 0.5 * (u0 * u0 + v0 * v0);
        let e1 = 0.5 * (u1 * u1 + v1 * v1);

        assert!((e1 - e0).abs() < 1e-12);
    }

    #[test]
    fn test_rotation_direction() {
        // 北半球向东流 → 向南偏转
        let (_, v1) = apply_coriolis_exact(10.0, 0.0, 1e-4, 100.0);
        assert!(v1 < 0.0);

        // 南半球向东流 → 向北偏转
        let (_, v2) = apply_coriolis_exact(10.0, 0.0, -1e-4, 100.0);
        assert!(v2 > 0.0);
    }

    #[test]
    fn test_small_angle_approximation() {
        let (u_approx, v_approx) = apply_coriolis_exact(1.0, 0.0, 1e-4, 1.0);

        let theta: f64 = 1e-4;
        let (sin_exact, cos_exact) = theta.sin_cos();
        let u_exact = cos_exact;
        let v_exact = -sin_exact;

        assert!((u_approx - u_exact).abs() < 1e-15);
        assert!((v_approx - v_exact).abs() < 1e-15);
    }

    #[test]
    fn test_zero_coriolis() {
        let (u, v) = apply_coriolis_exact(10.0, 5.0, 0.0, 100.0);
        assert_eq!((u, v), (10.0, 5.0));
    }

    #[test]
    fn test_stability_check() {
        assert!(is_stable(1e-4, 100.0));
        assert!(!is_stable(1e-4, 2000.0));
    }

    #[test]
    fn test_max_stable_dt() {
        let dt_max = max_stable_dt(1e-4, 0.5);
        assert!((dt_max - 500.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "科氏旋转角过大")]
    #[cfg(debug_assertions)]
    fn test_large_rotation_panic() {
        apply_coriolis_exact(10.0, 5.0, 1e-4, 50000.0);
    }
}
