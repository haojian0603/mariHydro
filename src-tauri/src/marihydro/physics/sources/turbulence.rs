// src-tauri/src/marihydro/physics/sources/turbulence.rs

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::domain::state::ConservedState;
use crate::marihydro::infra::error::MhResult;
use crate::marihydro::physics::numerics::{
    compute_gradient_green_gauss, compute_vector_gradient_green_gauss, VectorGradient,
};

/// Smagorinsky 亚格子尺度模型
#[derive(Debug, Clone, Copy)]
pub struct SmagorinskyModel {
    /// Smagorinsky 常数，典型值 0.1-0.2
    pub cs: f64,
    /// 最小涡粘系数 [m²/s]
    pub nu_min: f64,
    /// 最大涡粘系数 [m²/s]
    pub nu_max: f64,
}

impl Default for SmagorinskyModel {
    fn default() -> Self {
        Self {
            cs: 0.15,
            nu_min: 0.0,
            nu_max: 1000.0,
        }
    }
}

impl SmagorinskyModel {
    pub fn new(cs: f64) -> Self {
        Self {
            cs,
            ..Default::default()
        }
    }

    pub fn with_limits(mut self, nu_min: f64, nu_max: f64) -> Self {
        self.nu_min = nu_min;
        self.nu_max = nu_max;
        self
    }

    /// 计算涡粘系数场
    ///
    /// ν_t = (C_s * Δ)² * |S|
    pub fn compute_eddy_viscosity(
        &self,
        state: &ConservedState,
        mesh: &UnstructuredMesh,
        h_min: f64,
    ) -> MhResult<Vec<f64>> {
        let n_cells = mesh.n_cells;

        let velocities: Vec<DVec2> = (0..n_cells).map(|i| state.velocity(i, h_min)).collect();

        let grad = compute_vector_gradient_green_gauss(&velocities, mesh);

        let nu_t: Vec<f64> = (0..n_cells)
            .into_par_iter()
            .map(|i| {
                let area = mesh.cell_area[i];
                if area < 1e-14 {
                    return self.nu_min;
                }

                let delta = area.sqrt();
                let cs_delta_sq = (self.cs * delta).powi(2);
                let s_mag = grad.strain_rate_magnitude(i);

                let nu = cs_delta_sq * s_mag;
                nu.clamp(self.nu_min, self.nu_max)
            })
            .collect();

        Ok(nu_t)
    }

    /// 计算有效粘性系数 (分子粘性 + 涡粘性)
    pub fn compute_effective_viscosity(
        &self,
        state: &ConservedState,
        mesh: &UnstructuredMesh,
        nu_molecular: f64,
        h_min: f64,
    ) -> MhResult<Vec<f64>> {
        let nu_t = self.compute_eddy_viscosity(state, mesh, h_min)?;
        Ok(nu_t.into_iter().map(|nut| nu_molecular + nut).collect())
    }
}

/// 计算标量场梯度 (便捷函数)
pub fn compute_gradient_field(field: &[f64], mesh: &UnstructuredMesh) -> Vec<DVec2> {
    compute_gradient_green_gauss(field, mesh)
}

/// 计算涡量场 ω = ∂v/∂x - ∂u/∂y
pub fn compute_vorticity(state: &ConservedState, mesh: &UnstructuredMesh, h_min: f64) -> Vec<f64> {
    let n_cells = mesh.n_cells;

    let velocities: Vec<DVec2> = (0..n_cells).map(|i| state.velocity(i, h_min)).collect();

    let grad = compute_vector_gradient_green_gauss(&velocities, mesh);

    (0..n_cells).map(|i| grad.vorticity(i)).collect()
}

/// 计算湍流动能估计 (用于诊断)
///
/// k ≈ C_k * (C_s * Δ)² * |S|²
pub fn compute_turbulent_kinetic_energy(
    state: &ConservedState,
    mesh: &UnstructuredMesh,
    h_min: f64,
    cs: f64,
) -> Vec<f64> {
    const CK: f64 = 0.094;

    let n_cells = mesh.n_cells;

    let velocities: Vec<DVec2> = (0..n_cells).map(|i| state.velocity(i, h_min)).collect();

    let grad = compute_vector_gradient_green_gauss(&velocities, mesh);

    (0..n_cells)
        .into_par_iter()
        .map(|i| {
            let area = mesh.cell_area[i];
            if area < 1e-14 {
                return 0.0;
            }

            let delta = area.sqrt();
            let s_mag = grad.strain_rate_magnitude(i);

            CK * (cs * delta).powi(2) * s_mag.powi(2)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cs_default_range() {
        let model = SmagorinskyModel::default();
        assert!(model.cs > 0.0 && model.cs < 0.5);
    }

    #[test]
    fn test_custom_cs() {
        let model = SmagorinskyModel::new(0.12);
        assert_eq!(model.cs, 0.12);
    }

    #[test]
    fn test_with_limits() {
        let model = SmagorinskyModel::new(0.15).with_limits(0.01, 100.0);
        assert_eq!(model.nu_min, 0.01);
        assert_eq!(model.nu_max, 100.0);
    }
}
