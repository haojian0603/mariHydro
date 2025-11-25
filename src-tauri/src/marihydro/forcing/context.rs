// src-tauri/src/marihydro/forcing/context.rs

use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverSource {
    pub idx_1d: usize,
    pub flow_rate: f64,

    #[cfg(feature = "river_momentum")]
    pub velocity_u: f64,

    #[cfg(feature = "river_momentum")]
    pub velocity_v: f64,
}

impl RiverSource {
    pub fn new(idx_1d: usize, flow_rate: f64) -> Self {
        Self {
            idx_1d,
            flow_rate,
            #[cfg(feature = "river_momentum")]
            velocity_u: 0.0,
            #[cfg(feature = "river_momentum")]
            velocity_v: 0.0,
        }
    }

    #[cfg(feature = "river_momentum")]
    pub fn with_velocity(idx_1d: usize, flow_rate: f64, velocity_u: f64, velocity_v: f64) -> Self {
        Self {
            idx_1d,
            flow_rate,
            velocity_u,
            velocity_v,
        }
    }

    #[cfg(feature = "river_momentum")]
    pub fn from_polar(idx_1d: usize, flow_rate: f64, speed: f64, angle_deg: f64) -> Self {
        use std::f64::consts::PI;
        let angle_rad = angle_deg * PI / 180.0;
        Self {
            idx_1d,
            flow_rate,
            velocity_u: speed * angle_rad.cos(),
            velocity_v: speed * angle_rad.sin(),
        }
    }

    #[inline]
    pub fn to_2d(&self, stride: usize) -> (usize, usize) {
        (self.idx_1d / stride, self.idx_1d % stride)
    }
}

#[derive(Debug, Clone)]
pub struct ForcingContext {
    pub wind_u: Array2<f64>,
    pub wind_v: Array2<f64>,
    pub pressure_anomaly: Array2<f64>,
    pub river_discharge: Vec<RiverSource>,
    pub pressure_ref: f64,
    pub viscosity: f64,
    pub current_time: f64,
}

impl ForcingContext {
    pub fn new(nx: usize, ny: usize, ng: usize, viscosity: f64, pressure_ref: f64) -> Self {
        let shape = (ny + 2 * ng, nx + 2 * ng);

        Self {
            wind_u: Array2::zeros(shape),
            wind_v: Array2::zeros(shape),
            pressure_anomaly: Array2::zeros(shape),
            river_discharge: Vec::new(),
            pressure_ref,
            viscosity,
            current_time: 0.0,
        }
    }

    pub fn reset_sources(&mut self) {
        self.river_discharge.clear();
    }

    pub fn update_time(&mut self, t: f64) {
        self.current_time = t;
    }

    pub fn add_river(&mut self, idx_1d: usize, flow_rate: f64) {
        self.river_discharge
            .push(RiverSource::new(idx_1d, flow_rate));
    }

    #[cfg(feature = "river_momentum")]
    pub fn add_river_with_velocity(
        &mut self,
        idx_1d: usize,
        flow_rate: f64,
        velocity_u: f64,
        velocity_v: f64,
    ) {
        self.river_discharge.push(RiverSource::with_velocity(
            idx_1d, flow_rate, velocity_u, velocity_v,
        ));
    }

    #[inline]
    pub fn wind_magnitude(&self, j: usize, i: usize) -> f64 {
        let u = self.wind_u[[j, i]];
        let v = self.wind_v[[j, i]];
        (u * u + v * v).sqrt()
    }

    pub fn validate(&self) -> Result<(), String> {
        use crate::marihydro::infra::constants::validation;

        let max_wind = self
            .wind_u
            .iter()
            .chain(self.wind_v.iter())
            .map(|&v| v.abs())
            .fold(0.0, f64::max);

        if max_wind > validation::MAX_REASONABLE_WIND_SPEED {
            return Err(format!(
                "风速异常: {:.1} m/s 超过上限 {:.1}",
                max_wind,
                validation::MAX_REASONABLE_WIND_SPEED
            ));
        }

        let max_pressure_anomaly = self
            .pressure_anomaly
            .iter()
            .map(|&v| v.abs())
            .fold(0.0, f64::max);

        const MAX_PRESSURE_ANOMALY: f64 = 10000.0;
        if max_pressure_anomaly > MAX_PRESSURE_ANOMALY {
            return Err(format!("气压异常值过大: {:.0} Pa", max_pressure_anomaly));
        }

        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        let elem_size = std::mem::size_of::<f64>();
        let arrays_mem =
            (self.wind_u.len() + self.wind_v.len() + self.pressure_anomaly.len()) * elem_size;
        let vec_mem = self.river_discharge.capacity() * std::mem::size_of::<RiverSource>();
        arrays_mem + vec_mem + std::mem::size_of::<Self>()
    }

    pub fn new_empty() -> Self {
        Self {
            wind_u: Array2::zeros((0, 0)),
            wind_v: Array2::zeros((0, 0)),
            pressure_anomaly: Array2::zeros((0, 0)),
            river_discharge: Vec::new(),
            pressure_ref: 101325.0,
            viscosity: 0.0,
            current_time: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forcing_context_creation() {
        let ctx = ForcingContext::new(100, 100, 2, 1.0, 101325.0);
        assert_eq!(ctx.wind_u.dim(), (104, 104));
        assert_eq!(ctx.river_discharge.len(), 0);
        assert_eq!(ctx.current_time, 0.0);
    }

    #[test]
    fn test_add_river_source() {
        let mut ctx = ForcingContext::new(10, 10, 2, 1.0, 101325.0);
        ctx.add_river(500, 10.0);
        ctx.add_river(600, -5.0);

        assert_eq!(ctx.river_discharge.len(), 2);
        assert_eq!(ctx.river_discharge[0].flow_rate, 10.0);
        assert_eq!(ctx.river_discharge[1].flow_rate, -5.0);
    }

    #[test]
    fn test_river_source_to_2d() {
        let source = RiverSource::new(45, 10.0);
        let (j, i) = source.to_2d(10);
        assert_eq!(j, 4);
        assert_eq!(i, 5);
    }

    #[test]
    fn test_reset_sources() {
        let mut ctx = ForcingContext::new(10, 10, 2, 1.0, 101325.0);
        ctx.add_river(100, 10.0);
        ctx.reset_sources();
        assert!(ctx.river_discharge.is_empty());
    }

    #[test]
    fn test_wind_magnitude() {
        let mut ctx = ForcingContext::new(10, 10, 2, 1.0, 101325.0);
        ctx.wind_u[[5, 5]] = 3.0;
        ctx.wind_v[[5, 5]] = 4.0;
        assert_eq!(ctx.wind_magnitude(5, 5), 5.0);
    }

    #[test]
    fn test_validate_normal() {
        let ctx = ForcingContext::new(10, 10, 2, 1.0, 101325.0);
        assert!(ctx.validate().is_ok());
    }

    #[test]
    fn test_validate_extreme_wind() {
        let mut ctx = ForcingContext::new(10, 10, 2, 1.0, 101325.0);
        ctx.wind_u[[5, 5]] = 200.0;
        assert!(ctx.validate().is_err());
    }

    #[test]
    fn test_memory_usage() {
        let ctx = ForcingContext::new(100, 100, 2, 1.0, 101325.0);
        let mem = ctx.memory_usage();
        let expected_min = 104 * 104 * 3 * 8;
        assert!(mem >= expected_min);
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_river_with_velocity() {
        let source = RiverSource::with_velocity(100, 10.0, 2.0, 1.5);
        assert_eq!(source.velocity_u, 2.0);
        assert_eq!(source.velocity_v, 1.5);
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_river_from_polar() {
        let source = RiverSource::from_polar(100, 10.0, 5.0, 0.0);
        assert!((source.velocity_u - 5.0).abs() < 1e-10);
        assert!(source.velocity_v.abs() < 1e-10);

        let source = RiverSource::from_polar(100, 10.0, 5.0, 90.0);
        assert!(source.velocity_u.abs() < 1e-10);
        assert!((source.velocity_v - 5.0).abs() < 1e-10);
    }

    #[test]
    #[cfg(feature = "river_momentum")]
    fn test_add_river_with_velocity() {
        let mut ctx = ForcingContext::new(10, 10, 2, 1.0, 101325.0);
        ctx.add_river_with_velocity(100, 10.0, 2.0, 1.5);

        assert_eq!(ctx.river_discharge.len(), 1);
        assert_eq!(ctx.river_discharge[0].velocity_u, 2.0);
        assert_eq!(ctx.river_discharge[0].velocity_v, 1.5);
    }
}
