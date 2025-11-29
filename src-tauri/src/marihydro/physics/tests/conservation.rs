// src-tauri/src/marihydro/physics/tests/conservation.rs

use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::domain::state::ConservedState;

pub fn compute_total_mass(state: &ConservedState, mesh: &UnstructuredMesh) -> f64 {
    debug_assert_eq!(state.n_cells, mesh.n_cells);

    state
        .h
        .par_iter()
        .zip(mesh.cell_area.par_iter())
        .map(|(&h, &area)| h * area)
        .sum()
}

pub fn compute_total_momentum(state: &ConservedState, mesh: &UnstructuredMesh) -> (f64, f64) {
    debug_assert_eq!(state.n_cells, mesh.n_cells);

    let (px, py): (f64, f64) = state
        .hu
        .par_iter()
        .zip(state.hv.par_iter())
        .zip(mesh.cell_area.par_iter())
        .map(|((&hu, &hv), &area)| (hu * area, hv * area))
        .reduce(|| (0.0, 0.0), |(ax, ay), (bx, by)| (ax + bx, ay + by));

    (px, py)
}

pub fn compute_total_energy(state: &ConservedState, mesh: &UnstructuredMesh, g: f64) -> f64 {
    debug_assert_eq!(state.n_cells, mesh.n_cells);

    state
        .h
        .par_iter()
        .zip(state.hu.par_iter())
        .zip(state.hv.par_iter())
        .zip(mesh.cell_z_bed.par_iter())
        .zip(mesh.cell_area.par_iter())
        .map(|((((&h, &hu), &hv), &z), &area)| {
            if h < 1e-10 {
                return 0.0;
            }
            let u = hu / h;
            let v = hv / h;
            let eta = h + z;
            let kinetic = 0.5 * h * (u * u + v * v);
            let potential = 0.5 * g * eta * eta;
            (kinetic + potential) * area
        })
        .sum()
}

pub fn compute_total_scalar(state: &ConservedState, mesh: &UnstructuredMesh) -> Option<f64> {
    state.hc.as_ref().map(|hc| {
        hc.par_iter()
            .zip(mesh.cell_area.par_iter())
            .map(|(&hc_val, &area)| hc_val * area)
            .sum()
    })
}

#[inline]
pub fn relative_error(value: f64, reference: f64) -> f64 {
    if reference.abs() < 1e-12 {
        value.abs()
    } else {
        (value - reference).abs() / reference.abs()
    }
}

pub struct ConservationChecker {
    initial_mass: f64,
    initial_momentum: (f64, f64),
    initial_energy: f64,
    gravity: f64,
}

impl ConservationChecker {
    pub fn new(state: &ConservedState, mesh: &UnstructuredMesh, gravity: f64) -> Self {
        Self {
            initial_mass: compute_total_mass(state, mesh),
            initial_momentum: compute_total_momentum(state, mesh),
            initial_energy: compute_total_energy(state, mesh, gravity),
            gravity,
        }
    }

    pub fn check(&self, state: &ConservedState, mesh: &UnstructuredMesh) -> ConservationReport {
        let current_mass = compute_total_mass(state, mesh);
        let current_momentum = compute_total_momentum(state, mesh);
        let current_energy = compute_total_energy(state, mesh, self.gravity);

        ConservationReport {
            mass_error: relative_error(current_mass, self.initial_mass),
            momentum_x_error: relative_error(current_momentum.0, self.initial_momentum.0),
            momentum_y_error: relative_error(current_momentum.1, self.initial_momentum.1),
            energy_error: relative_error(current_energy, self.initial_energy),
            current_mass,
            current_momentum,
            current_energy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConservationReport {
    pub mass_error: f64,
    pub momentum_x_error: f64,
    pub momentum_y_error: f64,
    pub energy_error: f64,
    pub current_mass: f64,
    pub current_momentum: (f64, f64),
    pub current_energy: f64,
}

impl ConservationReport {
    pub fn is_acceptable(&self, tolerance: f64) -> bool {
        self.mass_error < tolerance
            && self.momentum_x_error < tolerance
            && self.momentum_y_error < tolerance
    }

    pub fn max_error(&self) -> f64 {
        self.mass_error
            .max(self.momentum_x_error)
            .max(self.momentum_y_error)
            .max(self.energy_error)
    }
}

impl std::fmt::Display for ConservationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "守恒检查报告:")?;
        writeln!(f, "  质量误差:     {:.2e}", self.mass_error)?;
        writeln!(f, "  动量X误差:    {:.2e}", self.momentum_x_error)?;
        writeln!(f, "  动量Y误差:    {:.2e}", self.momentum_y_error)?;
        writeln!(f, "  能量误差:     {:.2e}", self.energy_error)?;
        writeln!(f, "  当前质量:     {:.6e} kg", self.current_mass)?;
        writeln!(
            f,
            "  当前动量:     ({:.6e}, {:.6e}) kg·m/s",
            self.current_momentum.0, self.current_momentum.1
        )?;
        writeln!(f, "  当前能量:     {:.6e} J", self.current_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mesh(n: usize) -> UnstructuredMesh {
        let mut mesh = UnstructuredMesh::new();
        mesh.n_cells = n;
        mesh.cell_area = vec![100.0; n];
        mesh.cell_z_bed = vec![0.0; n];
        mesh
    }

    #[test]
    fn test_total_mass() {
        let mesh = create_test_mesh(10);
        let mut state = ConservedState::new(10);
        state.h.iter_mut().for_each(|h| *h = 1.0);

        let mass = compute_total_mass(&state, &mesh);
        assert!((mass - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_total_momentum() {
        let mesh = create_test_mesh(10);
        let mut state = ConservedState::new(10);
        state.hu.iter_mut().for_each(|hu| *hu = 1.0);
        state.hv.iter_mut().for_each(|hv| *hv = 2.0);

        let (px, py) = compute_total_momentum(&state, &mesh);
        assert!((px - 1000.0).abs() < 1e-10);
        assert!((py - 2000.0).abs() < 1e-10);
    }

    #[test]
    fn test_relative_error() {
        assert_eq!(relative_error(100.0, 100.0), 0.0);
        assert!((relative_error(101.0, 100.0) - 0.01).abs() < 1e-10);
        assert_eq!(relative_error(1e-15, 0.0), 1e-15);
    }

    #[test]
    fn test_conservation_checker() {
        let mesh = create_test_mesh(10);
        let mut state = ConservedState::new(10);
        state.h.iter_mut().for_each(|h| *h = 1.0);

        let checker = ConservationChecker::new(&state, &mesh, 9.81);
        let report = checker.check(&state, &mesh);

        assert!(report.mass_error < 1e-12);
        assert!(report.is_acceptable(1e-6));
    }
}
