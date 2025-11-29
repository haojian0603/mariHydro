// src-tauri/src/marihydro/physics/sources/atmosphere.rs

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;

const MAX_REASONABLE_WIND_SPEED: f64 = 100.0;

#[inline]
pub fn wind_drag_coefficient_lp81(wind_speed_10m: f64) -> f64 {
    let w = wind_speed_10m.abs().min(MAX_REASONABLE_WIND_SPEED);

    if w < 11.0 {
        1.2e-3
    } else if w < 25.0 {
        (0.49 + 0.065 * w) * 1e-3
    } else {
        2.11e-3
    }
}

#[inline]
pub fn wind_drag_coefficient_wu82(wind_speed_10m: f64) -> f64 {
    let w = wind_speed_10m.abs().min(MAX_REASONABLE_WIND_SPEED);
    (0.8 + 0.065 * w) * 1e-3
}

pub fn compute_wind_stress(wind_u: f64, wind_v: f64, air_density: f64) -> (f64, f64) {
    let wind_mag = (wind_u * wind_u + wind_v * wind_v).sqrt();

    if wind_mag < 1e-8 {
        return (0.0, 0.0);
    }

    let cd = wind_drag_coefficient_lp81(wind_mag);
    let tau_mag = air_density * cd * wind_mag;

    (tau_mag * wind_u, tau_mag * wind_v)
}

pub fn compute_wind_acceleration(
    wind_u: f64,
    wind_v: f64,
    h: f64,
    air_density: f64,
    water_density: f64,
) -> (f64, f64) {
    if h < 1e-6 {
        return (0.0, 0.0);
    }

    let (tau_x, tau_y) = compute_wind_stress(wind_u, wind_v, air_density);
    let factor = 1.0 / (water_density * h);

    (tau_x * factor, tau_y * factor)
}

pub fn compute_pressure_gradient(
    pressure: &[f64],
    mesh: &UnstructuredMesh,
    rho: f64,
    acc_x: &mut [f64],
    acc_y: &mut [f64],
) {
    assert_eq!(pressure.len(), mesh.n_cells);
    assert_eq!(acc_x.len(), mesh.n_cells);
    assert_eq!(acc_y.len(), mesh.n_cells);

    let mut grad_p: Vec<DVec2> = vec![DVec2::ZERO; mesh.n_cells];

    for face_idx in 0..mesh.n_faces {
        let owner = mesh.face_owner[face_idx];
        let neighbor = mesh.face_neighbor[face_idx];

        let normal = mesh.face_normal[face_idx];
        let length = mesh.face_length[face_idx];
        let ds = normal * length;

        let p_face = if neighbor != usize::MAX {
            0.5 * (pressure[owner] + pressure[neighbor])
        } else {
            pressure[owner]
        };

        grad_p[owner] += ds * p_face;

        if neighbor != usize::MAX {
            grad_p[neighbor] -= ds * p_face;
        }
    }

    let factor = -1.0 / rho;

    grad_p
        .par_iter()
        .zip(acc_x.par_iter_mut())
        .zip(acc_y.par_iter_mut())
        .zip(&mesh.cell_area)
        .for_each(|(((grad, ax), ay), &area)| {
            let inv_area = 1.0 / area;
            let grad_scaled = *grad * inv_area * factor;
            *ax = grad_scaled.x;
            *ay = grad_scaled.y;
        });
}

pub fn compute_wind_acceleration_field(
    wind_u: &[f64],
    wind_v: &[f64],
    h: &[f64],
    air_density: f64,
    water_density: f64,
    acc_x: &mut [f64],
    acc_y: &mut [f64],
) {
    let n = h.len();
    assert_eq!(wind_u.len(), n);
    assert_eq!(wind_v.len(), n);
    assert_eq!(acc_x.len(), n);
    assert_eq!(acc_y.len(), n);

    acc_x
        .par_iter_mut()
        .zip(acc_y.par_iter_mut())
        .zip(wind_u.par_iter())
        .zip(wind_v.par_iter())
        .zip(h.par_iter())
        .for_each(|((((ax, ay), &wu), &wv), &depth)| {
            let (a_x, a_y) = compute_wind_acceleration(wu, wv, depth, air_density, water_density);
            *ax = a_x;
            *ay = a_y;
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lp81_piecewise() {
        let cd_low = wind_drag_coefficient_lp81(5.0);
        assert!((cd_low - 1.2e-3).abs() < 1e-6);

        let cd_mid = wind_drag_coefficient_lp81(15.0);
        let expected = (0.49 + 0.065 * 15.0) * 1e-3;
        assert!((cd_mid - expected).abs() < 1e-9);

        let cd_high = wind_drag_coefficient_lp81(30.0);
        assert!((cd_high - 2.11e-3).abs() < 1e-6);
    }

    #[test]
    fn test_wu82_continuity() {
        let cd1 = wind_drag_coefficient_wu82(10.0);
        let cd2 = wind_drag_coefficient_wu82(10.1);
        assert!((cd2 - cd1).abs() < 1e-5);
    }

    #[test]
    fn test_wind_stress_direction() {
        let rho_air = 1.225;
        let (tau_x, tau_y) = compute_wind_stress(10.0, 0.0, rho_air);

        assert!(tau_x > 0.0);
        assert!(tau_y.abs() < 1e-12);

        let cd = wind_drag_coefficient_lp81(10.0);
        let expected_tau = rho_air * cd * 10.0 * 10.0;
        assert!((tau_x - expected_tau).abs() < 1e-6);
    }

    #[test]
    fn test_wind_stress_zero_wind() {
        let (tau_x, tau_y) = compute_wind_stress(0.0, 0.0, 1.225);
        assert_eq!(tau_x, 0.0);
        assert_eq!(tau_y, 0.0);
    }

    #[test]
    fn test_wind_acceleration_shallow_water() {
        let rho_air = 1.225;
        let rho_water = 1025.0;

        let (ax_deep, _) = compute_wind_acceleration(10.0, 0.0, 10.0, rho_air, rho_water);
        let (ax_shallow, _) = compute_wind_acceleration(10.0, 0.0, 1.0, rho_air, rho_water);

        assert!(ax_shallow > ax_deep);
        assert!((ax_shallow / ax_deep - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_wind_acceleration_dry_cell() {
        let (ax, ay) = compute_wind_acceleration(10.0, 10.0, 1e-10, 1.225, 1025.0);
        assert_eq!(ax, 0.0);
        assert_eq!(ay, 0.0);
    }
}
