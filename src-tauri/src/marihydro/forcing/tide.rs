// src-tauri/src/marihydro/forcing/tide.rs

use crate::marihydro::domain::boundary::{BoundaryDataProvider, ExternalForcing};
use crate::marihydro::domain::mesh::indices::FaceId;
use crate::marihydro::infra::error::MhResult;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::f64::consts::PI;

pub trait TideProvider: Send + Sync {
    fn get_level(&self, time: DateTime<Utc>, tag: u32) -> f64;
    fn get_velocity(&self, time: DateTime<Utc>, tag: u32) -> (f64, f64) {
        let _ = (time, tag);
        (0.0, 0.0)
    }
}

pub struct ConstantTide {
    pub level: f64,
}

impl TideProvider for ConstantTide {
    fn get_level(&self, _time: DateTime<Utc>, _tag: u32) -> f64 {
        self.level
    }
}

pub struct HarmonicTide {
    pub mean_level: f64,
    pub amplitude: f64,
    pub period_hours: f64,
    pub phase_deg: f64,
    pub start_time: DateTime<Utc>,
}

impl TideProvider for HarmonicTide {
    fn get_level(&self, time: DateTime<Utc>, _tag: u32) -> f64 {
        let elapsed_hours = (time - self.start_time).num_seconds() as f64 / 3600.0;
        let omega = if self.period_hours.abs() > 1e-6 {
            2.0 * PI / self.period_hours
        } else {
            0.0
        };
        let phase_rad = self.phase_deg.to_radians();
        self.mean_level + self.amplitude * (omega * elapsed_hours - phase_rad).cos()
    }
}

pub struct TagBasedTide {
    providers: HashMap<u32, Box<dyn TideProvider>>,
    default_level: f64,
}

impl TagBasedTide {
    pub fn new(default_level: f64) -> Self {
        Self {
            providers: HashMap::new(),
            default_level,
        }
    }

    pub fn add_provider(&mut self, tag: u32, provider: Box<dyn TideProvider>) {
        self.providers.insert(tag, provider);
    }

    pub fn with_provider(mut self, tag: u32, provider: Box<dyn TideProvider>) -> Self {
        self.add_provider(tag, provider);
        self
    }
}

impl TideProvider for TagBasedTide {
    fn get_level(&self, time: DateTime<Utc>, tag: u32) -> f64 {
        self.providers
            .get(&tag)
            .map(|p| p.get_level(time, tag))
            .unwrap_or(self.default_level)
    }

    fn get_velocity(&self, time: DateTime<Utc>, tag: u32) -> (f64, f64) {
        self.providers
            .get(&tag)
            .map(|p| p.get_velocity(time, tag))
            .unwrap_or((0.0, 0.0))
    }
}

pub struct TideBoundaryAdapter<T: TideProvider> {
    provider: T,
    face_tags: Vec<u32>,
    current_time: DateTime<Utc>,
}

impl<T: TideProvider> TideBoundaryAdapter<T> {
    pub fn new(provider: T, n_boundary_faces: usize) -> Self {
        Self {
            provider,
            face_tags: vec![0; n_boundary_faces],
            current_time: Utc::now(),
        }
    }

    pub fn set_face_tag(&mut self, bc_idx: usize, tag: u32) {
        if bc_idx < self.face_tags.len() {
            self.face_tags[bc_idx] = tag;
        }
    }

    pub fn update_time(&mut self, time: DateTime<Utc>) {
        self.current_time = time;
    }

    pub fn get_level_for_face(&self, bc_idx: usize) -> f64 {
        let tag = self.face_tags.get(bc_idx).copied().unwrap_or(0);
        self.provider.get_level(self.current_time, tag)
    }
}

impl<T: TideProvider + Send + Sync> BoundaryDataProvider for TideBoundaryAdapter<T> {
    fn get_forcing(&self, face_id: FaceId, _t: f64) -> MhResult<ExternalForcing> {
        let bc_idx = face_id.idx();
        let tag = self.face_tags.get(bc_idx).copied().unwrap_or(0);
        let eta = self.provider.get_level(self.current_time, tag);
        let (u, v) = self.provider.get_velocity(self.current_time, tag);
        Ok(ExternalForcing { eta, u, v })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_tide() {
        let tide = ConstantTide { level: 1.5 };
        let time = Utc::now();
        assert_eq!(tide.get_level(time, 0), 1.5);
        assert_eq!(tide.get_level(time, 99), 1.5);
    }

    #[test]
    fn test_harmonic_tide() {
        let start = Utc::now();
        let tide = HarmonicTide {
            mean_level: 0.0,
            amplitude: 1.0,
            period_hours: 12.42,
            phase_deg: 0.0,
            start_time: start,
        };
        let level_at_start = tide.get_level(start, 0);
        assert!((level_at_start - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tag_based_tide() {
        let mut tide = TagBasedTide::new(0.0);
        tide.add_provider(1, Box::new(ConstantTide { level: 2.0 }));
        tide.add_provider(2, Box::new(ConstantTide { level: 3.0 }));

        let time = Utc::now();
        assert_eq!(tide.get_level(time, 0), 0.0);
        assert_eq!(tide.get_level(time, 1), 2.0);
        assert_eq!(tide.get_level(time, 2), 3.0);
    }
}
