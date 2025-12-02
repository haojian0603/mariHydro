// src-tauri/src/marihydro/forcing/manager.rs
use super::context::{ActiveRiverSource, ForcingContext};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::CellIndex;
use chrono::{DateTime, Utc};

pub trait WindProvider: Send + Sync {
    fn get_wind_at(&self, time: DateTime<Utc>, u: &mut [f64], v: &mut [f64]) -> MhResult<()>;
}

pub trait TideProvider: Send + Sync {
    fn get_level(&self, time: DateTime<Utc>, tag: u32) -> f64;
    fn get_velocity(&self, time: DateTime<Utc>, tag: u32) -> (f64, f64) { let _ = (time, tag); (0.0, 0.0) }
}

pub trait RiverProvider: Send + Sync {
    fn get_rivers_at(&self, time: DateTime<Utc>) -> Vec<ActiveRiverSource>;
}

pub struct ForcingManager {
    context: ForcingContext,
    wind_provider: Option<Box<dyn WindProvider>>,
    tide_provider: Option<Box<dyn TideProvider>>,
    river_provider: Option<Box<dyn RiverProvider>>,
    tide_level: f64,
    current_time: DateTime<Utc>,
}

impl ForcingManager {
    pub fn new(n_cells: usize, viscosity: f64) -> Self {
        Self {
            context: ForcingContext::new(n_cells, viscosity, 101325.0),
            wind_provider: None, tide_provider: None, river_provider: None,
            tide_level: 0.0, current_time: Utc::now(),
        }
    }

    pub fn with_wind(mut self, provider: Box<dyn WindProvider>) -> Self {
        self.wind_provider = Some(provider); self
    }
    pub fn with_tide(mut self, provider: Box<dyn TideProvider>) -> Self {
        self.tide_provider = Some(provider); self
    }
    pub fn with_rivers(mut self, provider: Box<dyn RiverProvider>) -> Self {
        self.river_provider = Some(provider); self
    }

    pub fn update(&mut self, time: DateTime<Utc>) -> MhResult<()> {
        self.current_time = time;
        self.context.update_time(time.timestamp() as f64);
        if let Some(wp) = &self.wind_provider {
            wp.get_wind_at(time, &mut self.context.wind_u, &mut self.context.wind_v)?;
        }
        self.context.reset_sources();
        if let Some(rp) = &self.river_provider {
            for src in rp.get_rivers_at(time) { self.context.add_river(src); }
        }
        Ok(())
    }

    pub fn set_tide_level(&mut self, level: f64) { self.tide_level = level; }
    pub fn get_tide_level(&self, tag: u32) -> f64 {
        self.tide_provider.as_ref().map(|p| p.get_level(self.current_time, tag)).unwrap_or(self.tide_level)
    }
    pub fn context(&self) -> &ForcingContext { &self.context }
    pub fn context_mut(&mut self) -> &mut ForcingContext { &mut self.context }
}

pub struct ConstantWindProvider { u: f64, v: f64 }
impl ConstantWindProvider { pub fn new(u: f64, v: f64) -> Self { Self { u, v } } }
impl WindProvider for ConstantWindProvider {
    fn get_wind_at(&self, _time: DateTime<Utc>, u: &mut [f64], v: &mut [f64]) -> MhResult<()> {
        u.fill(self.u); v.fill(self.v); Ok(())
    }
}

pub struct ConstantTideProvider { level: f64 }
impl ConstantTideProvider { pub fn new(level: f64) -> Self { Self { level } } }
impl TideProvider for ConstantTideProvider {
    fn get_level(&self, _time: DateTime<Utc>, _tag: u32) -> f64 { self.level }
}

pub struct HarmonicTideProvider {
    mean: f64, amplitude: f64, period_hours: f64, phase_deg: f64, start: DateTime<Utc>,
}
impl HarmonicTideProvider {
    pub fn new(mean: f64, amplitude: f64, period_hours: f64, phase_deg: f64, start: DateTime<Utc>) -> Self {
        Self { mean, amplitude, period_hours, phase_deg, start }
    }
}
impl TideProvider for HarmonicTideProvider {
    fn get_level(&self, time: DateTime<Utc>, _tag: u32) -> f64 {
        let hours = (time - self.start).num_seconds() as f64 / 3600.0;
        let omega = if self.period_hours.abs() > 1e-6 { 2.0 * std::f64::consts::PI / self.period_hours } else { 0.0 };
        self.mean + self.amplitude * (omega * hours - self.phase_deg.to_radians()).cos()
    }
}
