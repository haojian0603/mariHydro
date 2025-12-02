// src-tauri/src/marihydro/forcing/providers/wind/grib.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, Utc};
use std::path::Path;

pub struct GribWindProvider {
    n_cells: usize,
    u_data: Vec<f64>,
    v_data: Vec<f64>,
}

impl GribWindProvider {
    pub fn open(_path: &Path, n_cells: usize) -> MhResult<Self> {
        Ok(Self { n_cells, u_data: vec![0.0; n_cells], v_data: vec![0.0; n_cells] })
    }
}

impl WindProvider for GribWindProvider {
    fn get_wind_at(&self, _time: DateTime<Utc>, u: &mut [f64], v: &mut [f64]) -> MhResult<()> {
        let n = self.n_cells.min(u.len()).min(v.len());
        u[..n].copy_from_slice(&self.u_data[..n]);
        v[..n].copy_from_slice(&self.v_data[..n]);
        Ok(())
    }
}
