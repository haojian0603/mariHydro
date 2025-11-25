//! 守恒性验证工具

use rayon::prelude::*;

use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::domain::state::State;

pub fn compute_total_mass(state: &State, mesh: &Mesh) -> f64 {
    let (dx, dy) = mesh.transform.resolution();
    let cell_area = dx * dy;

    let h_phys = state.physical_h();

    mesh.active_indices
        .par_iter()
        .map(|(j, i)| {
            let j_phys = j - mesh.ng;
            let i_phys = i - mesh.ng;
            h_phys[[j_phys, i_phys]] * cell_area
        })
        .sum()
}

pub fn compute_total_momentum(state: &State, mesh: &Mesh) -> (f64, f64) {
    let (dx, dy) = mesh.transform.resolution();
    let cell_area = dx * dy;

    let h_phys = state.physical_h();
    let u_phys = state.physical_u();
    let v_phys = state.physical_v();

    let (px, py): (f64, f64) = mesh
        .active_indices
        .par_iter()
        .map(|(j, i)| {
            let j_phys = j - mesh.ng;
            let i_phys = i - mesh.ng;

            let h = h_phys[[j_phys, i_phys]];
            let u = u_phys[[j_phys, i_phys]];
            let v = v_phys[[j_phys, i_phys]];

            (h * u * cell_area, h * v * cell_area)
        })
        .reduce(|| (0.0, 0.0), |(ax, ay), (bx, by)| (ax + bx, ay + by));

    (px, py)
}

pub fn compute_total_energy(state: &State, mesh: &Mesh, g: f64) -> f64 {
    let (dx, dy) = mesh.transform.resolution();
    let cell_area = dx * dy;

    let h_phys = state.physical_h();
    let u_phys = state.physical_u();
    let v_phys = state.physical_v();

    mesh.active_indices
        .par_iter()
        .map(|(j, i)| {
            let j_phys = j - mesh.ng;
            let i_phys = i - mesh.ng;

            let h = h_phys[[j_phys, i_phys]];
            let u = u_phys[[j_phys, i_phys]];
            let v = v_phys[[j_phys, i_phys]];
            let z = mesh.zb[[*j, *i]];

            let kinetic = 0.5 * h * (u * u + v * v);
            let potential = 0.5 * g * (h + z).powi(2);

            (kinetic + potential) * cell_area
        })
        .sum()
}

#[inline]
pub fn relative_error(value: f64, reference: f64) -> f64 {
    if reference.abs() < 1e-12 {
        value.abs()
    } else {
        (value - reference).abs() / reference.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_error() {
        assert_eq!(relative_error(100.0, 100.0), 0.0);
        assert!((relative_error(101.0, 100.0) - 0.01).abs() < 1e-10);
        assert_eq!(relative_error(1e-15, 0.0), 1e-15);
    }
}
