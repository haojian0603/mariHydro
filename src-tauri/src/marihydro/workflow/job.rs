// src-tauri/src/marihydro/workflow/job.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum JobStatus {
    Pending,   // 排队中
    Running,   // 计算中
    Paused,    // 暂停 (预留)
    Completed, // 成功
    Failed,    // 崩溃/报错
}

impl ToString for JobStatus {
    fn to_string(&self) -> String {
        match self {
            JobStatus::Pending => "PENDING",
            JobStatus::Running => "RUNNING",
            JobStatus::Paused => "PAUSED",
            JobStatus::Completed => "COMPLETED",
            JobStatus::Failed => "FAILED",
        }
        .to_string()
    }
}

impl From<&str> for JobStatus {
    fn from(s: &str) -> Self {
        match s {
            "RUNNING" => JobStatus::Running,
            "PAUSED" => JobStatus::Paused,
            "COMPLETED" => JobStatus::Completed,
            "FAILED" => JobStatus::Failed,
            _ => JobStatus::Pending,
        }
    }
}

/// 模拟任务实体
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct SimulationJob {
    pub id: String,         // Uuid string
    pub project_id: String, // Uuid string

    pub status: String, // 存储为字符串，使用时转换

    /// 参数覆盖配置 (JSON String)
    /// 允许在不修改 Project Manifest 的情况下微调参数
    pub parameter_overrides: Option<String>,

    pub progress: f64, // 0.0 - 100.0
    pub message: Option<String>,
    pub result_path: Option<String>,

    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
}

impl SimulationJob {
    /// 获取解析后的参数覆盖表
    pub fn get_overrides(&self) -> HashMap<String, f64> {
        if let Some(json) = &self.parameter_overrides {
            serde_json::from_str(json).unwrap_or_default()
        } else {
            HashMap::new()
        }
    }

    /// 获取枚举类型的状态
    pub fn get_status_enum(&self) -> JobStatus {
        JobStatus::from(self.status.as_str())
    }
}
