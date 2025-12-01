// src-tauri/src/marihydro/infra/config.rs
use serde::{Deserialize, Serialize};
use super::time::TimezoneConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project_name: String,
    pub version: String,
    pub timezone: TimezoneConfig,
    pub start_time_iso: String,
    pub duration_seconds: f64,
    pub output_interval: f64,
    pub default_roughness: f64,
    pub min_depth: f64,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            project_name: "Untitled".into(), version: "2.0".into(),
            timezone: TimezoneConfig::Utc, start_time_iso: "2024-01-01T00:00:00Z".into(),
            duration_seconds: 86400.0, output_interval: 3600.0, default_roughness: 0.025, min_depth: 0.05,
        }
    }
}

impl ProjectConfig {
    pub fn new(name: &str) -> Self { Self { project_name: name.into(), ..Default::default() } }
    pub fn with_duration(mut self, s: f64) -> Self { self.duration_seconds = s; self }
    pub fn with_output_interval(mut self, s: f64) -> Self { self.output_interval = s; self }
    pub fn validate(&self) -> Result<(), String> {
        if self.duration_seconds <= 0.0 { return Err("Duration must be positive".to_string()); }
        if self.output_interval <= 0.0 { return Err("Output interval must be positive".to_string()); }
        if self.min_depth < 0.0 { return Err("Min depth cannot be negative".to_string()); }
        Ok(())
    }
}
