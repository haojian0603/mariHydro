// src-tauri/src/marihydro/forcing/providers/river/provider.rs
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::CellIndex;
use crate::marihydro::forcing::context::ActiveRiverSource;
use crate::marihydro::forcing::manager::RiverProvider;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiverTimeSeries {
    pub times: Vec<f64>,
    pub flow_rates: Vec<f64>,
}

impl RiverTimeSeries {
    pub fn constant(flow_rate: f64) -> Self { Self { times: vec![0.0], flow_rates: vec![flow_rate] } }

    pub fn new(times: Vec<f64>, flow_rates: Vec<f64>) -> MhResult<Self> {
        if times.len() != flow_rates.len() { return Err(MhError::InvalidInput("Length mismatch".into())); }
        if times.is_empty() { return Err(MhError::InvalidInput("Empty series".into())); }
        for i in 0..times.len().saturating_sub(1) {
            if times[i] >= times[i + 1] { return Err(MhError::InvalidInput("Not monotonic".into())); }
        }
        Ok(Self { times, flow_rates })
    }

    pub fn interpolate(&self, t: f64) -> f64 {
        if self.times.len() == 1 { return self.flow_rates[0]; }
        if t <= self.times[0] { return self.flow_rates[0]; }
        if t >= *self.times.last().unwrap() { return *self.flow_rates.last().unwrap(); }
        for i in 0..self.times.len() - 1 {
            if t >= self.times[i] && t < self.times[i + 1] {
                let alpha = (t - self.times[i]) / (self.times[i + 1] - self.times[i]);
                return self.flow_rates[i] + alpha * (self.flow_rates[i + 1] - self.flow_rates[i]);
            }
        }
        *self.flow_rates.last().unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct RiverConfig {
    pub name: String,
    pub cell_id: CellIndex,
    pub time_series: RiverTimeSeries,
    pub start_time: DateTime<Utc>,
}

impl RiverConfig {
    pub fn constant(name: &str, cell_id: CellIndex, flow_rate: f64, start: DateTime<Utc>) -> Self {
        Self { name: name.into(), cell_id, time_series: RiverTimeSeries::constant(flow_rate), start_time: start }
    }
}

pub struct TimeSeriesRiverProvider {
    rivers: Vec<RiverConfig>,
}

impl TimeSeriesRiverProvider {
    pub fn new() -> Self { Self { rivers: Vec::new() } }
    pub fn add_river(&mut self, config: RiverConfig) { self.rivers.push(config); }
    pub fn with_river(mut self, config: RiverConfig) -> Self { self.add_river(config); self }
}

impl Default for TimeSeriesRiverProvider {
    fn default() -> Self { Self::new() }
}

impl RiverProvider for TimeSeriesRiverProvider {
    fn get_rivers_at(&self, time: DateTime<Utc>) -> Vec<ActiveRiverSource> {
        self.rivers.iter().map(|r| {
            let elapsed = (time - r.start_time).num_seconds() as f64;
            let flow = r.time_series.interpolate(elapsed);
            ActiveRiverSource::new(r.cell_id, flow, r.name.clone())
        }).collect()
    }
}

pub struct ConstantRiverProvider {
    rivers: Vec<(CellIndex, f64, String)>,
}

impl ConstantRiverProvider {
    pub fn new() -> Self { Self { rivers: Vec::new() } }
    pub fn add(&mut self, cell_id: CellIndex, flow_rate: f64, name: &str) {
        self.rivers.push((cell_id, flow_rate, name.into()));
    }
}

impl Default for ConstantRiverProvider {
    fn default() -> Self { Self::new() }
}

impl RiverProvider for ConstantRiverProvider {
    fn get_rivers_at(&self, _time: DateTime<Utc>) -> Vec<ActiveRiverSource> {
        self.rivers.iter().map(|(cell_id, flow, name)| ActiveRiverSource::new(*cell_id, *flow, name.clone())).collect()
    }
}
