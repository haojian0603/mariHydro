// src-tauri/src/marihydro/physics/schemes/hllc.rs

use crate::marihydro::domain::state::Flux;
use crate::marihydro::physics::flux_calculator::{FluxCalculator, RotatedFlux}; // ✅ 使用公共模块
use glam::DVec2;

#[derive(Debug, Clone, Copy)]
pub struct HllcResult {
    pub flux: Flux,
    pub max_wave_speed: f64,
}

const ENTROPY_FIX_EPSILON: f64 = 1e-2;

/// ✅ HLLC 求解器（使用 FluxCalculator）
#[inline]
pub fn solve_hllc(
    h_l: f64,
    vel_l: DVec2,
    h_r: f64,
    vel_r: DVec2,
    normal: DVec2,
    flux_calc: &FluxCalculator, // ✅ 传入计算器
) -> HllcResult {
    let g = flux_calc.gravity();
    let eps = flux_calc.h_min();

    if flux_calc.is_dry(h_l) && flux_calc.is_dry(h_r) {
        return HllcResult {
            flux: Flux::default(),
            max_wave_speed: 0.0,
        };
    }

    if flux_calc.is_dry(h_l) {
        return solve_dry_left(h_r, vel_r, normal, flux_calc);
    }
    if flux_calc.is_dry(h_r) {
        return solve_dry_right(h_l, vel_l, normal, flux_calc);
    }

    let un_l = vel_l.dot(normal);
    let un_r = vel_r.dot(normal);

    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();

    // ✅ 修正：使用标准 Einfeldt 波速估计（不使用 2.0 因子）
    let s_l = (un_l - c_l).min(un_r - c_r);
    let s_r = (un_l + c_l).max(un_r + c_r);

    let max_speed = s_l.abs().max(s_r.abs());

    // ✅ 使用 FluxCalculator 计算物理通量
    let flux_l = flux_calc.compute_rotated_flux(h_l, vel_l, normal);
    let flux_r = flux_calc.compute_rotated_flux(h_r, vel_r, normal);

    let result_flux = if s_l >= 0.0 {
        flux_l
    } else if s_r <= 0.0 {
        flux_r
    } else {
        hllc_star_region(h_l, un_l, h_r, un_r, s_l, s_r, flux_l, flux_r, g, eps)
    };

    // ✅ 旋转回全局坐标系
    let euler_flux = result_flux.rotate_back(normal);

    HllcResult {
        flux: Flux {
            mass: euler_flux.mass,
            mom_x: euler_flux.momentum_x,
            mom_y: euler_flux.momentum_y,
        },
        max_wave_speed: max_speed,
    }
}

#[inline]
fn hllc_star_region(
    h_l: f64,
    un_l: f64,
    h_r: f64,
    un_r: f64,
    s_l: f64,
    s_r: f64,
    flux_l: RotatedFlux,
    flux_r: RotatedFlux,
    g: f64,
    eps: f64,
) -> RotatedFlux {
    let q_l = h_l * (s_l - un_l);
    let q_r = h_r * (s_r - un_r);
    let denom = q_r - q_l;

    if denom.abs() < eps {
        let inv_ds = 1.0 / (s_r - s_l).max(eps);
        return RotatedFlux {
            mass: (s_r * flux_l.mass - s_l * flux_r.mass + s_l * s_r * (h_r - h_l)) * inv_ds,
            momentum_n: (s_r * flux_l.momentum_n - s_l * flux_r.momentum_n
                + s_l * s_r * (h_r * un_r - h_l * un_l))
                * inv_ds,
            momentum_t: (s_r * flux_l.momentum_t - s_l * flux_r.momentum_t) * inv_ds,
        };
    }

    let s_star = (q_l * un_l - q_r * un_r + flux_r.momentum_n - flux_l.momentum_n) / denom;

    let s_star_fixed = if s_star.abs() < ENTROPY_FIX_EPSILON {
        s_star.signum() * ENTROPY_FIX_EPSILON
    } else {
        s_star
    };

    if s_star_fixed >= 0.0 {
        let h_star = q_l / (s_l - s_star_fixed);
        RotatedFlux {
            mass: flux_l.mass + s_l * (h_star - h_l),
            momentum_n: flux_l.momentum_n + s_l * (h_star * s_star_fixed - h_l * un_l),
            momentum_t: flux_l.momentum_t,
        }
    } else {
        let h_star = q_r / (s_r - s_star_fixed);
        RotatedFlux {
            mass: flux_r.mass + s_r * (h_star - h_r),
            momentum_n: flux_r.momentum_n + s_r * (h_star * s_star_fixed - h_r * un_r),
            momentum_t: flux_r.momentum_t,
        }
    }
}

#[inline]
fn solve_dry_left(h_r: f64, vel_r: DVec2, normal: DVec2, flux_calc: &FluxCalculator) -> HllcResult {
    let g = flux_calc.gravity();
    let un_r = vel_r.dot(normal);
    let c_r = (g * h_r).sqrt();

    let s_head_r = un_r - c_r;
    let s_tail_r = un_r + 2.0 * c_r;

    let result_flux = if s_head_r >= 0.0 {
        flux_calc.compute_rotated_flux(h_r, vel_r, normal)
    } else if s_tail_r <= 0.0 {
        RotatedFlux {
            mass: 0.0,
            momentum_n: 0.0,
            momentum_t: 0.0,
        }
    } else {
        let u_star = 2.0 * (c_r + un_r) / 3.0;
        let c_star = u_star - un_r + c_r;
        let h_star = (c_star * c_star / g).max(flux_calc.h_min());
        let vel_star = DVec2::new(u_star, 0.0); // 简化
        flux_calc.compute_rotated_flux(h_star, vel_star, normal)
    };

    let euler_flux = result_flux.rotate_back(normal);

    HllcResult {
        flux: Flux {
            mass: euler_flux.mass,
            mom_x: euler_flux.momentum_x,
            mom_y: euler_flux.momentum_y,
        },
        max_wave_speed: s_tail_r.abs().max(s_head_r.abs()),
    }
}

#[inline]
fn solve_dry_right(
    h_l: f64,
    vel_l: DVec2,
    normal: DVec2,
    flux_calc: &FluxCalculator,
) -> HllcResult {
    let g = flux_calc.gravity();
    let un_l = vel_l.dot(normal);
    let c_l = (g * h_l).sqrt();

    let s_head_l = un_l + c_l;
    let s_tail_l = un_l - 2.0 * c_l;

    let result_flux = if s_head_l <= 0.0 {
        RotatedFlux {
            mass: 0.0,
            momentum_n: 0.0,
            momentum_t: 0.0,
        }
    } else if s_tail_l >= 0.0 {
        flux_calc.compute_rotated_flux(h_l, vel_l, normal)
    } else {
        let u_star = 2.0 * (c_l - un_l) / 3.0;
        let c_star = c_l + un_l - u_star;
        let h_star = (c_star * c_star / g).max(flux_calc.h_min());
        let vel_star = DVec2::new(u_star, 0.0);
        flux_calc.compute_rotated_flux(h_star, vel_star, normal)
    };

    let euler_flux = result_flux.rotate_back(normal);

    HllcResult {
        flux: Flux {
            mass: euler_flux.mass,
            mom_x: euler_flux.momentum_x,
            mom_y: euler_flux.momentum_y,
        },
        max_wave_speed: s_tail_l.abs().max(s_head_l.abs()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dry_bed() {
        let flux_calc = FluxCalculator::new(9.81, 1e-6);
        let normal = DVec2::new(1.0, 0.0);
        let result = solve_hllc(0.0, DVec2::ZERO, 0.0, DVec2::ZERO, normal, &flux_calc);

        assert_eq!(result.flux.mass, 0.0);
        assert_eq!(result.max_wave_speed, 0.0);
    }

    #[test]
    fn test_still_water() {
        let flux_calc = FluxCalculator::new(9.81, 1e-6);
        let normal = DVec2::new(1.0, 0.0);
        let h = 10.0;
        let vel = DVec2::ZERO;

        let result = solve_hllc(h, vel, h, vel, normal, &flux_calc);

        assert!(result.flux.mass.abs() < 1e-10);
    }
}
