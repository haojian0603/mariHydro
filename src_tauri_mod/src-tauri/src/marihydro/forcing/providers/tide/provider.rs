// src-tauri/src/marihydro/forcing/providers/tide/provider.rs
use crate::marihydro::forcing::manager::TideProvider;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::f64::consts::PI;

pub struct ConstantTide { level: f64 }
impl ConstantTide { pub fn new(level: f64) -> Self { Self { level } } }
impl TideProvider for ConstantTide {
    fn get_level(&self, _time: DateTime<Utc>, _tag: u32) -> f64 { self.level }
}

#[derive(Debug, Clone)]
pub struct TidalConstituent {
    pub name: String,
    pub amplitude: f64,
    pub phase_deg: f64,
    pub speed_deg_per_hour: f64,
}

impl TidalConstituent {
    pub fn m2(amplitude: f64, phase_deg: f64) -> Self {
        Self { name: "M2".into(), amplitude, phase_deg, speed_deg_per_hour: 28.984104 }
    }
    pub fn s2(amplitude: f64, phase_deg: f64) -> Self {
        Self { name: "S2".into(), amplitude, phase_deg, speed_deg_per_hour: 30.0 }
    }
    pub fn k1(amplitude: f64, phase_deg: f64) -> Self {
        Self { name: "K1".into(), amplitude, phase_deg, speed_deg_per_hour: 15.041069 }
    }
    pub fn o1(amplitude: f64, phase_deg: f64) -> Self {
        Self { name: "O1".into(), amplitude, phase_deg, speed_deg_per_hour: 13.943036 }
    }
}

pub struct HarmonicTide {
    mean_level: f64,
    constituents: Vec<TidalConstituent>,
    start_time: DateTime<Utc>,
}

impl HarmonicTide {
    pub fn new(mean_level: f64, start_time: DateTime<Utc>) -> Self {
        Self { mean_level, constituents: Vec::new(), start_time }
    }
    pub fn with_constituent(mut self, c: TidalConstituent) -> Self {
        self.constituents.push(c); self
    }
    pub fn simple(mean: f64, amplitude: f64, period_hours: f64, phase_deg: f64, start: DateTime<Utc>) -> Self {
        let speed = if period_hours.abs() > 1e-6 { 360.0 / period_hours } else { 0.0 };
        Self {
            mean_level: mean, start_time: start,
            constituents: vec![TidalConstituent { name: "simple".into(), amplitude, phase_deg, speed_deg_per_hour: speed }],
        }
    }
}

impl TideProvider for HarmonicTide {
    fn get_level(&self, time: DateTime<Utc>, _tag: u32) -> f64 {
        let hours = (time - self.start_time).num_seconds() as f64 / 3600.0;
        let mut level = self.mean_level;
        for c in &self.constituents {
            let phase = (c.speed_deg_per_hour * hours - c.phase_deg) * PI / 180.0;
            level += c.amplitude * phase.cos();
        }
        level
    }
}

pub struct TaggedTide {
    providers: HashMap<u32, Box<dyn TideProvider>>,
    default_level: f64,
}

impl TaggedTide {
    pub fn new(default_level: f64) -> Self { Self { providers: HashMap::new(), default_level } }
    pub fn with(mut self, tag: u32, p: Box<dyn TideProvider>) -> Self { self.providers.insert(tag, p); self }
}

impl TideProvider for TaggedTide {
    fn get_level(&self, time: DateTime<Utc>, tag: u32) -> f64 {
        self.providers.get(&tag).map(|p| p.get_level(time, tag)).unwrap_or(self.default_level)
    }
    fn get_velocity(&self, time: DateTime<Utc>, tag: u32) -> (f64, f64) {
        self.providers.get(&tag).map(|p| p.get_velocity(time, tag)).unwrap_or((0.0, 0.0))
    }
}
