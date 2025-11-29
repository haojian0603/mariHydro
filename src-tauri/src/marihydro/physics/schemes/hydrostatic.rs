//! Audusse 静水重构与 MUSCL 线性重构

use glam::DVec2;

#[derive(Debug, Clone, Copy)]
pub struct HydrostaticFaceState {
    pub h_left: f64,
    pub h_right: f64,
    pub vel_left: DVec2,
    pub vel_right: DVec2,
}

#[derive(Debug, Clone, Copy)]
pub struct BedSlopeSource {
    pub source_x: f64,
    pub source_y: f64,
}

#[inline]
pub fn hydrostatic_reconstruction(
    h_left_cell: f64,
    z_left_cell: f64,
    vel_left: DVec2,
    h_right_cell: f64,
    z_right_cell: f64,
    vel_right: DVec2,
    z_face_left: f64,
    z_face_right: f64,
    eps: f64,
    _g: f64,
) -> HydrostaticFaceState {
    let eta_left = h_left_cell + z_left_cell;
    let eta_right = h_right_cell + z_right_cell;

    let z_face = z_face_left.max(z_face_right);

    let h_star_left = (eta_left - z_face).max(0.0);
    let h_star_right = (eta_right - z_face).max(0.0);

    let vel_l = if h_star_left > eps {
        vel_left
    } else {
        DVec2::ZERO
    };

    let vel_r = if h_star_right > eps {
        vel_right
    } else {
        DVec2::ZERO
    };

    HydrostaticFaceState {
        h_left: h_star_left,
        h_right: h_star_right,
        vel_left: vel_l,
        vel_right: vel_r,
    }
}

#[inline]
pub fn compute_bed_slope_source(
    h_left: f64,
    h_right: f64,
    z_left: f64,
    z_right: f64,
    normal: DVec2,
    length: f64,
    g: f64,
) -> BedSlopeSource {
    let h_avg = 0.5 * (h_left + h_right);
    let z_avg = 0.5 * (z_left + z_right);

    let dz = z_right - z_left;
    let source_magnitude = -g * h_avg * dz * length;

    BedSlopeSource {
        source_x: source_magnitude * normal.x,
        source_y: source_magnitude * normal.y,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ReconstructedState {
    pub h: f64,
    pub hu: f64,
    pub hv: f64,
    pub vel: DVec2,
}

#[inline]
pub fn muscl_reconstruct(
    h_cell: f64,
    hu_cell: f64,
    hv_cell: f64,
    grad_h: DVec2,
    grad_hu: DVec2,
    grad_hv: DVec2,
    delta: DVec2,
    limiter: SlopeLimiter,
    eps: f64,
) -> ReconstructedState {
    let h_grad_limited = limit_gradient(grad_h, limiter);
    let hu_grad_limited = limit_gradient(grad_hu, limiter);
    let hv_grad_limited = limit_gradient(grad_hv, limiter);

    let h_recon = (h_cell + h_grad_limited.dot(delta)).max(0.0);
    let hu_recon = hu_cell + hu_grad_limited.dot(delta);
    let hv_recon = hv_cell + hv_grad_limited.dot(delta);

    let vel = if h_recon > eps {
        DVec2::new(hu_recon / h_recon, hv_recon / h_recon)
    } else {
        DVec2::ZERO
    };

    ReconstructedState {
        h: h_recon,
        hu: hu_recon,
        hv: hv_recon,
        vel,
    }
}

#[inline]
fn limit_gradient(grad: DVec2, limiter: SlopeLimiter) -> DVec2 {
    match limiter {
        SlopeLimiter::None => grad,
        SlopeLimiter::MinMod => {
            let r = 1.0;
            let lim_x = minmod(grad.x, r * grad.x);
            let lim_y = minmod(grad.y, r * grad.y);
            DVec2::new(lim_x, lim_y)
        }
        SlopeLimiter::VanLeer => {
            let mag = grad.length();
            if mag < 1e-12 {
                DVec2::ZERO
            } else {
                grad * (2.0 / (1.0 + mag * mag / (grad.x * grad.x + 1e-12)))
            }
        }
        SlopeLimiter::Superbee => {
            let lim_x = superbee(grad.x, 2.0 * grad.x);
            let lim_y = superbee(grad.y, 2.0 * grad.y);
            DVec2::new(lim_x, lim_y)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlopeLimiter {
    None,
    MinMod,
    VanLeer,
    Superbee,
}

#[inline(always)]
pub fn minmod(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

#[inline(always)]
pub fn van_leer(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        0.0
    } else {
        2.0 * a * b / (a + b)
    }
}

#[inline(always)]
pub fn superbee(a: f64, b: f64) -> f64 {
    if a * b <= 0.0 {
        return 0.0;
    }

    let s = a.signum();
    let aa = a.abs();
    let ab = b.abs();

    s * aa.min(2.0 * ab).max(ab.min(2.0 * aa))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydrostatic_balance() {
        let eta = 10.0;

        let z_left = 5.0;
        let z_right = 3.0;

        let h_left = eta - z_left;
        let h_right = eta - z_right;

        let result = hydrostatic_reconstruction(
            h_left,
            z_left,
            DVec2::ZERO,
            h_right,
            z_right,
            DVec2::ZERO,
            z_left,
            z_right,
            1e-6,
            9.81,
        );

        let eta_left_recon = result.h_left + z_left.max(z_right);
        let eta_right_recon = result.h_right + z_left.max(z_right);

        assert!((eta_left_recon - eta_right_recon).abs() < 1e-10);
    }

    #[test]
    fn test_dry_cell() {
        let result = hydrostatic_reconstruction(
            0.0,
            5.0,
            DVec2::new(1.0, 0.0),
            2.0,
            3.0,
            DVec2::new(0.5, 0.0),
            5.0,
            3.0,
            1e-6,
            9.81,
        );

        assert_eq!(result.vel_left, DVec2::ZERO);
    }

    #[test]
    fn test_limiters() {
        assert_eq!(minmod(1.0, 2.0), 1.0);
        assert_eq!(minmod(-1.0, 2.0), 0.0);

        let vl = van_leer(1.0, 2.0);
        assert!((vl - 4.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_muscl_reconstruction() {
        let state = muscl_reconstruct(
            2.0,
            4.0,
            2.0,
            DVec2::new(0.1, 0.05),
            DVec2::new(0.2, 0.1),
            DVec2::new(0.1, 0.05),
            DVec2::new(1.0, 0.5),
            SlopeLimiter::None,
            1e-6,
        );

        assert!(state.h > 0.0);
        assert!(state.vel.length().is_finite());
    }

    #[test]
    fn test_bed_slope_source() {
        let source = compute_bed_slope_source(1.0, 1.0, 0.0, 1.0, DVec2::X, 1.0, 9.81);

        assert!(source.source_x < 0.0);
    }
}
