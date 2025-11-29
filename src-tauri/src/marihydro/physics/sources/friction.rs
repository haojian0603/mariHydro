// src-tauri/src/marihydro/physics/sources/friction.rs

use rayon::prelude::*;

use crate::marihydro::infra::constants::physics;

pub type Velocity2D = (f64, f64);

/// 半隐式 Manning 底摩擦
///
/// 求解隐式方程: u_new = u_old / (1 + dt * C_f * |V|)
#[inline]
pub fn apply_friction_implicit(
    u_old: f64,
    v_old: f64,
    h: f64,
    n: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) -> Velocity2D {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let speed_sq = u_old * u_old + v_old * v_old;
    if speed_sq < 1e-20 {
        return (u_old, v_old);
    }

    let speed = speed_sq.sqrt();
    let cf = compute_manning_coefficient(h, n, g, h_min);
    let denominator = 1.0 + dt * cf * speed;

    debug_assert!(denominator > 0.0, "摩擦衰减系数必须为正");

    (u_old / denominator, v_old / denominator)
}

/// 半隐式 Manning 底摩擦 (守恒形式，直接作用于 hu, hv)
///
/// 这种形式在动量守恒上更精确
#[inline]
pub fn apply_friction_implicit_conservative(
    hu_old: f64,
    hv_old: f64,
    h: f64,
    n: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) -> (f64, f64) {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let h_sq = h * h;
    let speed_sq = (hu_old * hu_old + hv_old * hv_old) / h_sq;

    if speed_sq < 1e-20 {
        return (hu_old, hv_old);
    }

    let speed = speed_sq.sqrt();
    let cf = compute_manning_coefficient(h, n, g, h_min);
    let denominator = 1.0 + dt * cf * speed;

    (hu_old / denominator, hv_old / denominator)
}

/// 计算 Manning 摩擦系数
///
/// C_f = g * n² / h^(1/3)
///
/// 性能优化: 使用 cbrt() 替代 powf(1.0/3.0)，提升 3-5 倍
#[inline]
pub fn compute_manning_coefficient(h: f64, n: f64, g: f64, h_min: f64) -> f64 {
    let h_safe = h.max(h_min);
    g * n * n / h_safe.cbrt()
}

/// 计算 Chezy 摩擦系数
///
/// C_f = g / C²
#[inline]
pub fn compute_chezy_coefficient(chezy_c: f64, g: f64) -> f64 {
    g / (chezy_c * chezy_c)
}

/// 计算植被阻力系数
///
/// C_f = C_d * m * h_submerged / h
#[inline]
pub fn compute_vegetation_coefficient(cd: f64, m: f64, h_veg: f64, h: f64, h_min: f64) -> f64 {
    let h_safe = h.max(h_min);
    let h_submerged = h_safe.min(h_veg);
    cd * m * h_submerged / h_safe
}

/// 半隐式 Chezy 底摩擦
#[inline]
pub fn apply_friction_chezy(
    u_old: f64,
    v_old: f64,
    h: f64,
    chezy_c: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) -> Velocity2D {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let speed_sq = u_old * u_old + v_old * v_old;
    if speed_sq < 1e-20 {
        return (u_old, v_old);
    }

    let speed = speed_sq.sqrt();
    let cf = compute_chezy_coefficient(chezy_c, g);
    let denominator = 1.0 + dt * cf * speed;

    (u_old / denominator, v_old / denominator)
}

/// 半隐式植被阻力
#[inline]
pub fn apply_friction_vegetation(
    u_old: f64,
    v_old: f64,
    h: f64,
    cd: f64,
    m: f64,
    h_veg: f64,
    dt: f64,
    h_min: f64,
) -> Velocity2D {
    if h < h_min * (1.0 + 1e-12) {
        return (0.0, 0.0);
    }

    let speed_sq = u_old * u_old + v_old * v_old;
    if speed_sq < 1e-20 {
        return (u_old, v_old);
    }

    let speed = speed_sq.sqrt();
    let cf = compute_vegetation_coefficient(cd, m, h_veg, h, h_min);
    let denominator = 1.0 + dt * cf * speed;

    (u_old / denominator, v_old / denominator)
}

/// 批量应用 Manning 摩擦 (并行，守恒形式)
///
/// 直接修改 hu, hv 数组，减少内存分配和拷贝
pub fn apply_friction_field(
    hu: &mut [f64],
    hv: &mut [f64],
    h: &[f64],
    manning_n: &[f64],
    dt: f64,
    g: f64,
    h_min: f64,
) {
    debug_assert_eq!(hu.len(), hv.len());
    debug_assert_eq!(hu.len(), h.len());
    debug_assert_eq!(hu.len(), manning_n.len());

    hu.par_iter_mut()
        .zip(hv.par_iter_mut())
        .zip(h.par_iter())
        .zip(manning_n.par_iter())
        .for_each(|(((hu_val, hv_val), &depth), &n)| {
            let (new_hu, new_hv) =
                apply_friction_implicit_conservative(*hu_val, *hv_val, depth, n, dt, g, h_min);
            *hu_val = new_hu;
            *hv_val = new_hv;
        });
}

/// 批量应用 Manning 摩擦 (标量糙率场版本)
///
/// 允许空间变化的 Manning 系数
pub fn apply_friction_field_scalar(
    hu: &mut [f64],
    hv: &mut [f64],
    h: &[f64],
    n: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) {
    debug_assert_eq!(hu.len(), hv.len());
    debug_assert_eq!(hu.len(), h.len());

    hu.par_iter_mut()
        .zip(hv.par_iter_mut())
        .zip(h.par_iter())
        .for_each(|((hu_val, hv_val), &depth)| {
            let (new_hu, new_hv) =
                apply_friction_implicit_conservative(*hu_val, *hv_val, depth, n, dt, g, h_min);
            *hu_val = new_hu;
            *hv_val = new_hv;
        });
}

/// 批量应用 Chezy 摩擦
pub fn apply_friction_field_chezy(
    hu: &mut [f64],
    hv: &mut [f64],
    h: &[f64],
    chezy_c: f64,
    dt: f64,
    g: f64,
    h_min: f64,
) {
    debug_assert_eq!(hu.len(), hv.len());
    debug_assert_eq!(hu.len(), h.len());

    let cf = compute_chezy_coefficient(chezy_c, g);

    hu.par_iter_mut()
        .zip(hv.par_iter_mut())
        .zip(h.par_iter())
        .for_each(|((hu_val, hv_val), &depth)| {
            if depth < h_min * (1.0 + 1e-12) {
                *hu_val = 0.0;
                *hv_val = 0.0;
                return;
            }

            let h_sq = depth * depth;
            let speed_sq = (*hu_val * *hu_val + *hv_val * *hv_val) / h_sq;

            if speed_sq < 1e-20 {
                return;
            }

            let speed = speed_sq.sqrt();
            let denominator = 1.0 + dt * cf * speed;

            *hu_val /= denominator;
            *hv_val /= denominator;
        });
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
        let expected = g * n * n / h.cbrt();

        assert!((cf - expected).abs() < 1e-12);
    }

    #[test]
    fn test_cbrt_vs_powf_correctness() {
        let h = 10.0;
        let result_cbrt = h.cbrt();
        let result_powf = h.powf(1.0 / 3.0);
        assert!((result_cbrt - result_powf).abs() < 1e-14);
    }

    #[test]
    fn test_chezy_coefficient() {
        let g = physics::STANDARD_GRAVITY;
        let c = 50.0;

        let cf = compute_chezy_coefficient(c, g);
        let expected = g / (c * c);

        assert!((cf - expected).abs() < 1e-12);
    }

    #[test]
    fn test_vegetation_submergence() {
        let cd = 1.0;
        let m = 100.0;
        let h_veg = 0.5;

        let cf_deep = compute_vegetation_coefficient(cd, m, h_veg, 2.0, 0.01);
        assert!((cf_deep - 25.0).abs() < 1e-6);

        let cf_shallow = compute_vegetation_coefficient(cd, m, h_veg, 0.3, 0.01);
        assert!((cf_shallow - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_implicit_stability() {
        let g = physics::STANDARD_GRAVITY;

        let (u, v) = apply_friction_implicit(1.0, 1.0, 10.0, 0.025, 100.0, g, 0.01);

        assert!(u > 0.0 && u < 1.0);
        assert!(v > 0.0 && v < 1.0);
        assert!((u - v).abs() < 1e-10);
    }

    #[test]
    fn test_implicit_energy_decay() {
        let g = physics::STANDARD_GRAVITY;
        let (u0, v0) = (1.0, 1.0);

        let (u1, v1) = apply_friction_implicit(u0, v0, 10.0, 0.025, 10.0, g, 0.01);

        let e0 = u0 * u0 + v0 * v0;
        let e1 = u1 * u1 + v1 * v1;

        assert!(e1 < e0, "摩擦应导致能量衰减");
        assert!(e1 > 0.0, "能量应保持正值");
    }

    #[test]
    fn test_dry_cell_zero_velocity() {
        let g = physics::STANDARD_GRAVITY;

        let (u, v) = apply_friction_implicit(1.0, 1.0, 0.001, 0.025, 1.0, g, 0.05);

        assert_eq!(u, 0.0);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn test_conservative_form_equivalence() {
        let g = physics::STANDARD_GRAVITY;
        let h = 2.0;
        let u = 1.0;
        let v = 0.5;
        let hu = h * u;
        let hv = h * v;

        let (u_new, v_new) = apply_friction_implicit(u, v, h, 0.025, 1.0, g, 0.01);
        let (hu_new, hv_new) = apply_friction_implicit_conservative(hu, hv, h, 0.025, 1.0, g, 0.01);

        assert!((hu_new / h - u_new).abs() < 1e-12);
        assert!((hv_new / h - v_new).abs() < 1e-12);
    }

    #[test]
    fn test_batch_processing() {
        let g = physics::STANDARD_GRAVITY;
        let n = 4;

        let mut hu = vec![1.0; n];
        let mut hv = vec![0.5; n];
        let h = vec![2.0; n];
        let manning_n = vec![0.025; n];

        let hu_old = hu.clone();
        let hv_old = hv.clone();

        apply_friction_field(&mut hu, &mut hv, &h, &manning_n, 1.0, g, 0.01);

        for i in 0..n {
            assert!(hu[i] < hu_old[i]);
            assert!(hv[i] < hv_old[i]);
        }
    }
}
