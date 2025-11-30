// src-tauri/src/marihydro/forcing/providers/wind/provider.rs
use crate::marihydro::core::error::MhResult;
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone)]
pub struct WindProviderConfig {
    pub name: String,
    pub file_path: Option<String>,
    pub u_var: String,
    pub v_var: String,
    pub scale_factor: f64,
}

impl Default for WindProviderConfig {
    fn default() -> Self {
        Self { name: "wind".into(), file_path: None, u_var: "u10".into(), v_var: "v10".into(), scale_factor: 1.0 }
    }
}

#[derive(Debug, Clone)]
pub struct WindFrame {
    pub time: DateTime<Utc>,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

impl WindFrame {
    pub fn new(time: DateTime<Utc>, n_cells: usize) -> Self {
        Self { time, u: vec![0.0; n_cells], v: vec![0.0; n_cells] }
    }
    pub fn reset(&mut self) { self.u.fill(0.0); self.v.fill(0.0); }
}

pub struct UniformWindProvider { u: f64, v: f64 }
impl UniformWindProvider {
    pub fn new(u: f64, v: f64) -> Self { Self { u, v } }
}
impl WindProvider for UniformWindProvider {
    fn get_wind_at(&self, _time: DateTime<Utc>, u: &mut [f64], v: &mut [f64]) -> MhResult<()> {
        u.fill(self.u); v.fill(self.v); Ok(())
    }
}
