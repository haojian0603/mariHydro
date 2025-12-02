// src-tauri/src/marihydro/forcing/context.rs
use crate::marihydro::core::types::CellIndex;

#[derive(Debug, Clone)]
pub struct ActiveRiverSource {
    pub cell_id: CellIndex,
    pub flow_rate: f64,
    pub name: String,
}

impl ActiveRiverSource {
    pub fn new(cell_id: CellIndex, flow_rate: f64, name: String) -> Self {
        Self { cell_id, flow_rate, name }
    }
}

#[derive(Debug, Clone)]
pub struct ForcingContext {
    pub n_cells: usize,
    pub wind_u: Vec<f64>,
    pub wind_v: Vec<f64>,
    pub pressure_anomaly: Vec<f64>,
    pub river_sources: Vec<ActiveRiverSource>,
    pub pressure_ref: f64,
    pub viscosity: f64,
    pub current_time: f64,
}

impl ForcingContext {
    pub fn new(n_cells: usize, viscosity: f64, pressure_ref: f64) -> Self {
        Self {
            n_cells, wind_u: vec![0.0; n_cells], wind_v: vec![0.0; n_cells],
            pressure_anomaly: vec![0.0; n_cells], river_sources: Vec::new(),
            pressure_ref, viscosity, current_time: 0.0,
        }
    }

    pub fn reset_sources(&mut self) { self.river_sources.clear(); }
    pub fn update_time(&mut self, t: f64) { self.current_time = t; }
    pub fn add_river(&mut self, source: ActiveRiverSource) { self.river_sources.push(source); }

    #[inline]
    pub fn wind_magnitude(&self, cell_id: CellIndex) -> f64 {
        let idx = cell_id.get();
        (self.wind_u[idx].powi(2) + self.wind_v[idx].powi(2)).sqrt()
    }

    pub fn set_uniform_wind(&mut self, u: f64, v: f64) {
        self.wind_u.fill(u);
        self.wind_v.fill(v);
    }

    pub fn set_uniform_pressure(&mut self, p: f64) {
        self.pressure_anomaly.fill(p - self.pressure_ref);
    }

    pub fn validate(&self) -> Result<(), String> {
        const MAX_WIND: f64 = 100.0;
        let max = self.wind_u.iter().chain(self.wind_v.iter()).map(|v| v.abs()).fold(0.0, f64::max);
        if max > MAX_WIND {
            return Err(format!("Wind speed abnormal: {:.1} m/s > {:.1}", max, MAX_WIND));
        }
        Ok(())
    }
}
