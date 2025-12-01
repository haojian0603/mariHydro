// src-tauri/src/marihydro/workflow/job.rs
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JobStatus { Pending, Running, Paused, Completed, Failed, Cancelled }

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self { Self::Pending => "PENDING", Self::Running => "RUNNING", Self::Paused => "PAUSED", Self::Completed => "COMPLETED", Self::Failed => "FAILED", Self::Cancelled => "CANCELLED" }
    }
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() { "PENDING" => Self::Pending, "RUNNING" => Self::Running, "PAUSED" => Self::Paused, "COMPLETED" => Self::Completed, "FAILED" => Self::Failed, "CANCELLED" => Self::Cancelled, _ => Self::Pending }
    }
    pub fn is_terminal(&self) -> bool { matches!(self, Self::Completed | Self::Failed | Self::Cancelled) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationJob {
    pub id: String,
    pub project_id: String,
    pub status: JobStatus,
    pub progress: f64,
    pub message: Option<String>,
    pub result_path: Option<String>,
    pub parameter_overrides: Option<HashMap<String, f64>>,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl SimulationJob {
    pub fn new(id: String, project_id: String) -> Self {
        Self {
            id, project_id, status: JobStatus::Pending, progress: 0.0,
            message: Some("Pending".to_string()), result_path: None, parameter_overrides: None,
            created_at: Utc::now(), started_at: None, finished_at: None,
        }
    }
    pub fn with_overrides(mut self, overrides: HashMap<String, f64>) -> Self { self.parameter_overrides = Some(overrides); self }
    pub fn start(&mut self) { self.status = JobStatus::Running; self.started_at = Some(Utc::now()); self.message = Some("Running".to_string()); }
    pub fn update_progress(&mut self, progress: f64, message: Option<&str>) { self.progress = progress.clamp(0.0, 100.0); if let Some(m) = message { self.message = Some(m.to_string()); } }
    pub fn complete(&mut self, result_path: &str) { self.status = JobStatus::Completed; self.progress = 100.0; self.result_path = Some(result_path.to_string()); self.finished_at = Some(Utc::now()); self.message = Some("Completed".to_string()); }
    pub fn fail(&mut self, error: &str) { self.status = JobStatus::Failed; self.message = Some(error.to_string()); self.finished_at = Some(Utc::now()); }
    pub fn cancel(&mut self) { self.status = JobStatus::Cancelled; self.message = Some("Cancelled".to_string()); self.finished_at = Some(Utc::now()); }
}
