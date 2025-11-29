// src-tauri/src/marihydro/infra/config.rs

use crate::marihydro::geo::crs::CrsStrategy;
use crate::marihydro::infra::time::TimezoneConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    pub project_name: String,
    pub version: String,
    pub crs_strategy: CrsStrategy,
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
            project_name: "Untitled_Project".into(),
            version: "2.0".into(),
            crs_strategy: CrsStrategy::FromFirstFile,
            timezone: TimezoneConfig::Local,
            start_time_iso: "2024-01-01T00:00:00Z".into(),
            duration_seconds: 86400.0,
            output_interval: 3600.0,
            default_roughness: 0.025,
            min_depth: 0.05,
        }
    }
}

impl ProjectConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            project_name: name.into(),
            ..Default::default()
        }
    }

    pub fn with_duration(mut self, seconds: f64) -> Self {
        self.duration_seconds = seconds;
        self
    }

    pub fn with_output_interval(mut self, seconds: f64) -> Self {
        self.output_interval = seconds;
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.duration_seconds <= 0.0 {
            return Err("模拟时长必须为正数".into());
        }
        if self.output_interval <= 0.0 {
            return Err("输出间隔必须为正数".into());
        }
        if self.min_depth < 0.0 {
            return Err("最小水深不能为负数".into());
        }
        if self.default_roughness < 0.0 || self.default_roughness > 1.0 {
            return Err("曼宁糙率系数应在 [0, 1] 范围内".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProjectConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = ProjectConfig::new("Test")
            .with_duration(3600.0)
            .with_output_interval(60.0);
        assert_eq!(config.duration_seconds, 3600.0);
        assert_eq!(config.output_interval, 60.0);
    }

    #[test]
    fn test_validation_failure() {
        let mut config = ProjectConfig::default();
        config.duration_seconds = -100.0;
        assert!(config.validate().is_err());
    }
}
