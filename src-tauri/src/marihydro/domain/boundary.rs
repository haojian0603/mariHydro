use rayon::prelude::*;
use std::collections::HashSet;

use crate::marihydro::domain::mesh::indices::{CellId, FaceId};
use crate::marihydro::domain::mesh::unstructured::{BoundaryKind, UnstructuredMesh};
use crate::marihydro::domain::state::ConservedState;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::ProjectManifest;

#[derive(Debug, Clone, Default)]
pub struct ExternalForcing {
    pub eta: f64,
    pub u: f64,
    pub v: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryFaceInfo {
    pub face_id: FaceId,
    pub cell_id: CellId,
    pub normal: (f64, f64),
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryParams {
    pub gravity: f64,
    pub h_min: f64,
    pub sqrt_g: f64,
}

pub trait BoundaryFluxCalculator: Send + Sync {
    fn compute_flux(
        &self,
        h_interior: f64,
        u_interior: f64,
        v_interior: f64,
        z_interior: f64,
        external: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> (f64, f64, f64);
}

struct WallBoundary;
impl BoundaryFluxCalculator for WallBoundary {
    fn compute_flux(
        &self,
        h_interior: f64,
        _u: f64,
        _v: f64,
        _z: f64,
        _ext: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> (f64, f64, f64) {
        let p = 0.5 * params.gravity * h_interior * h_interior;
        (0.0, p * normal.0, p * normal.1)
    }
}

struct OpenFlowBoundary;
impl BoundaryFluxCalculator for OpenFlowBoundary {
    fn compute_flux(
        &self,
        h: f64,
        u: f64,
        v: f64,
        _z: f64,
        _ext: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> (f64, f64, f64) {
        let qn = h * (u * normal.0 + v * normal.1);
        let p = 0.5 * params.gravity * h * h;
        (qn, qn * u + p * normal.0, qn * v + p * normal.1)
    }
}

struct FlatherBoundary;
impl BoundaryFluxCalculator for FlatherBoundary {
    fn compute_flux(
        &self,
        h_int: f64,
        u_int: f64,
        v_int: f64,
        z_int: f64,
        external: &ExternalForcing,
        normal: (f64, f64),
        params: &BoundaryParams,
    ) -> (f64, f64, f64) {
        let h_safe = h_int.max(params.h_min);
        let c = params.sqrt_g * h_safe.sqrt();

        let un_int = u_int * normal.0 + v_int * normal.1;
        let eta_int = h_int + z_int;

        let un_star = external.u + (c / h_safe) * (eta_int - external.eta);

        let qn = h_int * un_star;
        let p = 0.5 * params.gravity * h_int * h_int;

        (qn, qn * u_int + p * normal.0, qn * v_int + p * normal.1)
    }
}

pub trait BoundaryDataProvider: Sync + Send {
    fn get_forcing(&self, face_id: FaceId, t: f64) -> MhResult<ExternalForcing>;
}

pub struct BoundaryManager {
    walls: Vec<BoundaryFaceInfo>,
    open_flows: Vec<BoundaryFaceInfo>,
    flathers: Vec<BoundaryFaceInfo>,

    params: BoundaryParams,
    impl_wall: WallBoundary,
    impl_open: OpenFlowBoundary,
    impl_flather: FlatherBoundary,
}

impl BoundaryManager {
    pub fn from_manifest(manifest: &ProjectManifest) -> Self {
        Self {
            walls: Vec::new(),
            open_flows: Vec::new(),
            flathers: Vec::new(),
            params: BoundaryParams {
                gravity: manifest.physics.gravity,
                h_min: manifest.physics.h_min,
                sqrt_g: manifest.physics.gravity.sqrt(),
            },
            impl_wall: WallBoundary,
            impl_open: OpenFlowBoundary,
            impl_flather: FlatherBoundary,
        }
    }

    pub fn register_unstructured_mesh(&mut self, mesh: &UnstructuredMesh) {
        for face_idx in mesh.boundary_faces() {
            let bc_idx = mesh.boundary_index(face_idx);
            let kind = mesh.bc_kind[bc_idx];

            let info = BoundaryFaceInfo {
                face_id: FaceId::new(face_idx),
                cell_id: CellId::new(mesh.face_owner[face_idx]),
                normal: (mesh.face_normal[face_idx].x, mesh.face_normal[face_idx].y),
            };

            match kind {
                BoundaryKind::Wall | BoundaryKind::Symmetry => self.walls.push(info),
                BoundaryKind::Outflow | BoundaryKind::OpenSea => self.open_flows.push(info),
                BoundaryKind::RiverInflow => self.flathers.push(info),
            }
        }

        log::info!(
            "边界注册完成: {} 固壁, {} 开边界, {} 潮汐/河流",
            self.walls.len(),
            self.open_flows.len(),
            self.flathers.len()
        );
    }

    pub fn apply_boundary_fluxes(
        &self,
        mesh: &UnstructuredMesh,
        state: &ConservedState,
        face_fluxes: &mut [(f64, f64, f64)],
        provider: &impl BoundaryDataProvider,
        time: f64,
    ) -> MhResult<()> {
        let flather_results: Vec<(FaceId, (f64, f64, f64))> = self
            .flathers
            .par_iter()
            .map(|face| -> MhResult<(FaceId, (f64, f64, f64))> {
                let cell_idx = face.cell_id.idx();
                let (h, u, v) = state.primitive(cell_idx, self.params.h_min);
                let z = mesh.cell_z_bed[cell_idx];
                let forcing = provider.get_forcing(face.face_id, time)?;

                let flux =
                    self.impl_flather
                        .compute_flux(h, u, v, z, &forcing, face.normal, &self.params);
                Ok((face.face_id, flux))
            })
            .collect::<MhResult<Vec<_>>>()?;

        for (face_id, flux) in flather_results {
            face_fluxes[face_id.idx()] = flux;
        }

        for face in &self.walls {
            let cell_idx = face.cell_id.idx();
            let (h, u, v) = state.primitive(cell_idx, self.params.h_min);
            let z = mesh.cell_z_bed[cell_idx];

            let flux = self.impl_wall.compute_flux(
                h,
                u,
                v,
                z,
                &Default::default(),
                face.normal,
                &self.params,
            );
            face_fluxes[face.face_id.idx()] = flux;
        }

        for face in &self.open_flows {
            let cell_idx = face.cell_id.idx();
            let (h, u, v) = state.primitive(cell_idx, self.params.h_min);
            let z = mesh.cell_z_bed[cell_idx];

            let flux = self.impl_open.compute_flux(
                h,
                u,
                v,
                z,
                &Default::default(),
                face.normal,
                &self.params,
            );
            face_fluxes[face.face_id.idx()] = flux;
        }

        Ok(())
    }

    pub fn validate(&self) -> MhResult<()> {
        let mut seen = HashSet::new();
        for face in self.all_faces() {
            let (nx, ny) = face.normal;
            let mag_sq = nx * nx + ny * ny;
            if (mag_sq - 1.0).abs() > 1e-10 {
                return Err(MhError::BoundaryCondition {
                    message: format!("法向未单位化: ({}, {})", nx, ny),
                });
            }
            if !seen.insert(face.face_id) {
                return Err(MhError::BoundaryCondition {
                    message: format!("重复的边界面: {}", face.face_id),
                });
            }
        }
        Ok(())
    }

    fn all_faces(&self) -> impl Iterator<Item = &BoundaryFaceInfo> {
        self.walls
            .iter()
            .chain(&self.open_flows)
            .chain(&self.flathers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wall_boundary() {
        let wall = WallBoundary;
        let params = BoundaryParams {
            gravity: 9.81,
            h_min: 0.01,
            sqrt_g: 9.81f64.sqrt(),
        };

        let (mass, mom_x, _) = wall.compute_flux(
            1.0,
            10.0,
            0.0,
            0.0,
            &ExternalForcing::default(),
            (-1.0, 0.0),
            &params,
        );

        assert_eq!(mass, 0.0);
        assert!(mom_x < 0.0);
    }
}
