use crate::marihydro::infra::constants::tolerances;
use std::f64::consts::PI;

pub type Velocity2D = (f64, f64);

#[inline(always)]
#[must_use]
pub fn apply_coriolis_exact(u: f64, v: f64, f: f64, dt: f64) -> Velocity2D {
    let theta = f * dt;

    debug_assert!(theta.abs() < PI, "科氏旋转角过大 (θ={:.3} rad)", theta);

    let (sin_t, cos_t) = if theta.abs() < 1e-3 {
        let t2 = theta * theta;
        (theta * (1.0 - t2 / 6.0), 1.0 - t2 * 0.5)
    } else {
        theta.sin_cos()
    };

    (u * cos_t + v * sin_t, -u * sin_t + v * cos_t)
}

#[inline]
pub fn is_stable(f: f64, dt: f64) -> bool {
    (f * dt).abs() < 0.1
}

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
        let (_, v1) = apply_coriolis_exact(10.0, 0.0, 1e-4, 100.0);
        assert!(v1 < 0.0);

        let (_, v2) = apply_coriolis_exact(10.0, 0.0, -1e-4, 100.0);
        assert!(v2 > 0.0);
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
}
