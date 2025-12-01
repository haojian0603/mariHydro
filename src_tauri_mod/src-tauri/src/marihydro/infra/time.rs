// src-tauri/src/marihydro/infra/time.rs
use crate::marihydro::core::error::{MhError, MhResult};
use chrono::{DateTime, FixedOffset, Local, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimezoneConfig { Utc, Local, Fixed(i32), Named(String) }

impl Default for TimezoneConfig { fn default() -> Self { Self::Utc } }

#[derive(Debug, Clone)]
pub struct TimeManager {
    base_utc: DateTime<Utc>,
    elapsed_seconds: f64,
    display_offset: Option<FixedOffset>,
    mode: TimezoneConfig,
}

impl TimeManager {
    pub fn new(start_iso: &str, tz_config: TimezoneConfig) -> MhResult<Self> {
        let base_utc = DateTime::parse_from_rfc3339(start_iso)
            .map_err(|e| MhError::config(format!("Time format error: {}", e)))?
            .with_timezone(&Utc);
        let offset = match &tz_config {
            TimezoneConfig::Fixed(h) => {
                let secs = h * 3600;
                if secs.abs() > 86400 { return Err(MhError::config(format!("Offset {} hours out of range", h))); }
                FixedOffset::east_opt(secs)
            }
            _ => None,
        };
        Ok(Self { base_utc, elapsed_seconds: 0.0, display_offset: offset, mode: tz_config })
    }

    pub fn advance(&mut self, dt: f64) { self.elapsed_seconds += dt; }
    pub fn elapsed_seconds(&self) -> f64 { self.elapsed_seconds }
    pub fn current_utc(&self) -> DateTime<Utc> {
        self.base_utc + chrono::Duration::milliseconds((self.elapsed_seconds * 1000.0) as i64)
    }
    pub fn current_display_str(&self) -> String {
        let utc = self.current_utc();
        match &self.mode {
            TimezoneConfig::Utc => utc.to_rfc3339(),
            TimezoneConfig::Local => DateTime::<Local>::from(utc).to_rfc3339(),
            TimezoneConfig::Fixed(_) => self.display_offset.map(|o| utc.with_timezone(&o).to_rfc3339()).unwrap_or_else(|| utc.to_rfc3339()),
            TimezoneConfig::Named(_) => utc.to_rfc3339(),
        }
    }
    pub fn reset(&mut self) { self.elapsed_seconds = 0.0; }
}
