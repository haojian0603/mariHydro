use crate::marihydro::domain::mesh::indices::CellId;
use crate::marihydro::infra::manifest::RiverSource as ManifestRiverSource;

#[derive(Debug, Clone)]
pub struct ActiveRiverSource {
    pub cell_id: CellId,
    pub flow_rate: f64,
    pub name: String,
}

impl ActiveRiverSource {
    pub fn new(cell_id: CellId, flow_rate: f64, name: String) -> Self {
        Self {
            cell_id,
            flow_rate,
            name,
        }
    }

    pub fn from_manifest(manifest_source: &ManifestRiverSource, cell_id: CellId) -> Self {
        Self {
            cell_id,
            flow_rate: manifest_source.constant_discharge,
            name: manifest_source.name.clone(),
        }
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
            n_cells,
            wind_u: vec![0.0; n_cells],
            wind_v: vec![0.0; n_cells],
            pressure_anomaly: vec![0.0; n_cells],
            river_sources: Vec::new(),
            pressure_ref,
            viscosity,
            current_time: 0.0,
        }
    }

    pub fn reset_sources(&mut self) {
        self.river_sources.clear();
    }

    pub fn update_time(&mut self, t: f64) {
        self.current_time = t;
    }

    pub fn add_river(&mut self, source: ActiveRiverSource) {
        self.river_sources.push(source);
    }

    #[inline]
    pub fn wind_magnitude(&self, cell_id: CellId) -> f64 {
        let idx = cell_id.idx();
        let u = self.wind_u[idx];
        let v = self.wind_v[idx];
        (u * u + v * v).sqrt()
    }

    pub fn validate(&self) -> Result<(), String> {
        const MAX_WIND_SPEED: f64 = 100.0;

        let max_wind = self
            .wind_u
            .iter()
            .chain(self.wind_v.iter())
            .map(|&v| v.abs())
            .fold(0.0, f64::max);

        if max_wind > MAX_WIND_SPEED {
            return Err(format!(
                "风速异常: {:.1} m/s 超过上限 {:.1}",
                max_wind, MAX_WIND_SPEED
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forcing_context_creation() {
        let ctx = ForcingContext::new(100, 1.0, 101325.0);
        assert_eq!(ctx.wind_u.len(), 100);
        assert_eq!(ctx.river_sources.len(), 0);
    }

    #[test]
    fn test_wind_magnitude() {
        let mut ctx = ForcingContext::new(100, 1.0, 101325.0);
        ctx.wind_u[50] = 3.0;
        ctx.wind_v[50] = 4.0;

        let mag = ctx.wind_magnitude(CellId::new(50));
        assert!((mag - 5.0).abs() < 1e-10);
    }
}
