// src-tauri/src/marihydro/forcing/manager.rs

use super::context::ForcingContext;
use super::tide::TideProvider;
use super::wind::WindProvider;
use crate::marihydro::domain::boundary::{BoundaryDataProvider, ExternalForcing};
use crate::marihydro::domain::mesh::indices::FaceId;
use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::infra::error::MhResult;
use crate::marihydro::infra::manifest::ProjectManifest;
use chrono::{DateTime, Utc};

pub struct ForcingManager {
    context: ForcingContext,
    wind_provider: Option<WindProvider>,
    tide_level: f64,
}

impl ForcingManager {
    pub fn init(manifest: &ProjectManifest, mesh: &UnstructuredMesh) -> MhResult<Self> {
        let context = ForcingContext::new(mesh.n_cells, manifest.physics.eddy_viscosity, 101325.0);

        let wind_src = manifest
            .sources
            .iter()
            .find(|s| s.mappings.iter().any(|m| m.target_var == "wind_u"));

        let wind_provider = if let Some(src) = wind_src {
            Some(WindProvider::init(src, mesh, manifest)?)
        } else {
            None
        };

        Ok(Self {
            context,
            wind_provider,
            tide_level: 0.0,
        })
    }

    pub fn update(&mut self, time: DateTime<Utc>, _mesh: &UnstructuredMesh) -> MhResult<()> {
        if let Some(wp) = &mut self.wind_provider {
            wp.get_wind_at(time, &mut self.context.wind_u, &mut self.context.wind_v)?;
        }
        self.context.reset_sources();
        Ok(())
    }

    pub fn set_tide_level(&mut self, level: f64) {
        self.tide_level = level;
    }

    pub fn get_context(&self) -> &ForcingContext {
        &self.context
    }

    pub fn get_context_mut(&mut self) -> &mut ForcingContext {
        &mut self.context
    }
}

impl BoundaryDataProvider for ForcingManager {
    fn get_forcing(&self, _face_id: FaceId, _t: f64) -> MhResult<ExternalForcing> {
        Ok(ExternalForcing {
            eta: self.tide_level,
            u: 0.0,
            v: 0.0,
        })
    }
}
