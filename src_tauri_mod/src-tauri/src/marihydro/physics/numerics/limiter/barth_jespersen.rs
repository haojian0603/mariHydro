// src-tauri/src/marihydro/physics/numerics/limiter/barth_jespersen.rs

//! Barth-Jespersen 限制器 - 解决 PERF-002

use super::{GradientLimiter, ScalarGradientStorage};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::CellIndex;
use glam::DVec2;
use rayon::prelude::*;

pub struct BarthJespersenLimiter {
    parallel: bool,
}

impl Default for BarthJespersenLimiter {
    fn default() -> Self {
        Self { parallel: true }
    }
}

impl BarthJespersenLimiter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_parallel(mut self, p: bool) -> Self {
        self.parallel = p;
        self
    }

    fn compute_cell<M: MeshAccess>(
        &self,
        i: usize,
        field: &[f64],
        grad: &ScalarGradientStorage,
        mesh: &M,
    ) -> f64 {
        let cell = CellIndex(i);
        let phi_c = field[i];
        let center = mesh.cell_centroid(cell);
        let g = grad.get(i);

        let (mut phi_max, mut phi_min) = (phi_c, phi_c);
        for &nb in mesh.cell_neighbors(cell) {
            if nb.is_valid() {
                let phi_n = field[nb.0];
                phi_max = phi_max.max(phi_n);
                phi_min = phi_min.min(phi_n);
            }
        }

        let mut alpha: f64 = 1.0;
        for &face in mesh.cell_faces(cell) {
            let fc = mesh.face_centroid(face);
            let r = fc - center;
            let delta = g.dot(r);
            if delta.abs() > 1e-14 {
                let ratio = if delta > 0.0 {
                    (phi_max - phi_c) / delta
                } else {
                    (phi_min - phi_c) / delta
                };
                alpha = alpha.min(ratio.clamp(0.0, 1.0));
            }
        }
        alpha
    }
}

impl GradientLimiter for BarthJespersenLimiter {
    fn limit<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &M,
    ) -> MhResult<()> {
        let n = mesh.n_cells();
        let mut lim = vec![1.0; n];
        self.compute_limiters(field, gradient, mesh, &mut lim)?;
        gradient.apply_limiter(&lim);
        Ok(())
    }

    fn compute_limiters<M: MeshAccess>(
        &self,
        field: &[f64],
        gradient: &ScalarGradientStorage,
        mesh: &M,
        output: &mut [f64],
    ) -> MhResult<()> {
        for i in 0..mesh.n_cells() {
            output[i] = self.compute_cell(i, field, gradient, mesh);
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "Barth-Jespersen"
    }
}

pub mod optimized {
    use super::*;
    use crate::marihydro::domain::mesh::UnstructuredMesh;

    pub fn limit_parallel(
        field: &[f64],
        gradient: &mut ScalarGradientStorage,
        mesh: &UnstructuredMesh,
    ) -> MhResult<()> {
        let lim: Vec<f64> = (0..mesh.n_cells())
            .into_par_iter()
            .map(|i| {
                let cell = CellIndex(i);
                let phi_c = field[i];
                let center = mesh.cell_centroid(cell);
                let g = gradient.get(i);
                let (mut pmax, mut pmin) = (phi_c, phi_c);
                for &nb in mesh.cell_neighbors(cell) {
                    if nb.is_valid() {
                        let pn = field[nb.0];
                        pmax = pmax.max(pn);
                        pmin = pmin.min(pn);
                    }
                }
                let mut a: f64 = 1.0;
                for &face in mesh.cell_faces(cell) {
                    let r = mesh.face_centroid(face) - center;
                    let d = g.dot(r);
                    if d.abs() > 1e-14 {
                        let ratio = if d > 0.0 {
                            (pmax - phi_c) / d
                        } else {
                            (pmin - phi_c) / d
                        };
                        a = a.min(ratio.clamp(0.0, 1.0));
                    }
                }
                a
            })
            .collect();
        gradient.apply_limiter(&lim);
        Ok(())
    }
}
