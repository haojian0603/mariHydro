// src-tauri/src/marihydro/forcing/providers/wind/netcdf.rs
use super::provider::{WindFrame, WindProviderConfig};
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, Utc};
use std::path::Path;

pub struct NetCdfWindProvider {
    config: WindProviderConfig,
    n_cells: usize,
    time_axis: Vec<DateTime<Utc>>,
    current_frame: WindFrame,
    next_frame: WindFrame,
    current_idx: usize,
}

impl NetCdfWindProvider {
    pub fn open(_path: &Path, n_cells: usize) -> MhResult<Self> {
        let config = WindProviderConfig::default();
        let now = Utc::now();
        Ok(Self {
            config, n_cells,
            time_axis: vec![now],
            current_frame: WindFrame::new(now, n_cells),
            next_frame: WindFrame::new(now, n_cells),
            current_idx: 0,
        })
    }

    fn interpolate(&self, time: DateTime<Utc>, out_u: &mut [f64], out_v: &mut [f64]) {
        let t0 = self.current_frame.time.timestamp() as f64;
        let t1 = self.next_frame.time.timestamp() as f64;
        let t = time.timestamp() as f64;
        let dt = t1 - t0;
        let alpha = if dt.abs() > 1e-6 { ((t - t0) / dt).clamp(0.0, 1.0) } else { 0.0 };
        for i in 0..self.n_cells.min(out_u.len()) {
            out_u[i] = self.current_frame.u[i] * (1.0 - alpha) + self.next_frame.u[i] * alpha;
            out_v[i] = self.current_frame.v[i] * (1.0 - alpha) + self.next_frame.v[i] * alpha;
        }
    }
}

impl WindProvider for NetCdfWindProvider {
    fn get_wind_at(&self, time: DateTime<Utc>, u: &mut [f64], v: &mut [f64]) -> MhResult<()> {
        self.interpolate(time, u, v); Ok(())
    }
}
