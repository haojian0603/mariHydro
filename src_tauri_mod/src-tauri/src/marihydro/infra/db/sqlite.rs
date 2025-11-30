// src-tauri/src/marihydro/infra/db/sqlite.rs
use crate::marihydro::core::error::{MhError, MhResult};
use rusqlite::{Connection, params};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus { Pending, Running, Completed, Failed, Cancelled }

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self { Self::Pending => "PENDING", Self::Running => "RUNNING", Self::Completed => "COMPLETED", Self::Failed => "FAILED", Self::Cancelled => "CANCELLED" }
    }
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() { "PENDING" => Self::Pending, "RUNNING" => Self::Running, "COMPLETED" => Self::Completed, "FAILED" => Self::Failed, "CANCELLED" => Self::Cancelled, _ => Self::Pending }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectRecord {
    pub id: String,
    pub name: String,
    pub manifest_json: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub id: String,
    pub project_id: String,
    pub status: JobStatus,
    pub progress: f64,
    pub message: Option<String>,
    pub result_path: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
}

pub struct Database {
    conn: Arc<Mutex<Connection>>,
}

impl Database {
    pub fn open(path: &str) -> MhResult<Self> {
        let db_path = Path::new(path);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| MhError::Io(e.to_string()))?;
        }
        let conn = Connection::open(path).map_err(|e| MhError::Io(e.to_string()))?;
        let db = Self { conn: Arc::new(Mutex::new(conn)) };
        db.init_schema()?;
        Ok(db)
    }

    pub fn in_memory() -> MhResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| MhError::Io(e.to_string()))?;
        let db = Self { conn: Arc::new(Mutex::new(conn)) };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> MhResult<()> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        conn.execute_batch(include_str!("schema.sql")).map_err(|e| MhError::Io(e.to_string()))?;
        Ok(())
    }

    pub fn insert_project(&self, id: &str, name: &str, manifest_json: &str) -> MhResult<()> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        conn.execute("INSERT INTO projects (id, name, manifest_json) VALUES (?1, ?2, ?3)", params![id, name, manifest_json]).map_err(|e| MhError::Io(e.to_string()))?;
        Ok(())
    }

    pub fn get_project(&self, id: &str) -> MhResult<Option<ProjectRecord>> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        let mut stmt = conn.prepare("SELECT id, name, manifest_json, created_at, updated_at FROM projects WHERE id = ?1").map_err(|e| MhError::Io(e.to_string()))?;
        let mut rows = stmt.query(params![id]).map_err(|e| MhError::Io(e.to_string()))?;
        if let Some(row) = rows.next().map_err(|e| MhError::Io(e.to_string()))? {
            Ok(Some(ProjectRecord { id: row.get(0).unwrap(), name: row.get(1).unwrap(), manifest_json: row.get(2).unwrap(), created_at: row.get(3).unwrap(), updated_at: row.get(4).unwrap() }))
        } else { Ok(None) }
    }

    pub fn list_projects(&self) -> MhResult<Vec<ProjectRecord>> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        let mut stmt = conn.prepare("SELECT id, name, manifest_json, created_at, updated_at FROM projects ORDER BY created_at DESC").map_err(|e| MhError::Io(e.to_string()))?;
        let iter = stmt.query_map([], |row| Ok(ProjectRecord { id: row.get(0)?, name: row.get(1)?, manifest_json: row.get(2)?, created_at: row.get(3)?, updated_at: row.get(4)? })).map_err(|e| MhError::Io(e.to_string()))?;
        let mut result = Vec::new();
        for r in iter { result.push(r.map_err(|e| MhError::Io(e.to_string()))?); }
        Ok(result)
    }

    pub fn insert_job(&self, id: &str, project_id: &str) -> MhResult<()> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        conn.execute("INSERT INTO jobs (id, project_id, status, progress) VALUES (?1, ?2, 'PENDING', 0.0)", params![id, project_id]).map_err(|e| MhError::Io(e.to_string()))?;
        Ok(())
    }

    pub fn update_job_status(&self, id: &str, status: JobStatus, progress: f64, message: Option<&str>) -> MhResult<()> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        conn.execute("UPDATE jobs SET status = ?2, progress = ?3, message = ?4 WHERE id = ?1", params![id, status.as_str(), progress, message]).map_err(|e| MhError::Io(e.to_string()))?;
        Ok(())
    }

    pub fn get_job(&self, id: &str) -> MhResult<Option<JobRecord>> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        let mut stmt = conn.prepare("SELECT id, project_id, status, progress, message, result_path, created_at, started_at, finished_at FROM jobs WHERE id = ?1").map_err(|e| MhError::Io(e.to_string()))?;
        let mut rows = stmt.query(params![id]).map_err(|e| MhError::Io(e.to_string()))?;
        if let Some(row) = rows.next().map_err(|e| MhError::Io(e.to_string()))? {
            Ok(Some(JobRecord {
                id: row.get(0).unwrap(), project_id: row.get(1).unwrap(),
                status: JobStatus::from_str(&row.get::<_, String>(2).unwrap()),
                progress: row.get(3).unwrap(), message: row.get(4).ok(), result_path: row.get(5).ok(),
                created_at: row.get(6).unwrap(), started_at: row.get(7).ok(), finished_at: row.get(8).ok(),
            }))
        } else { Ok(None) }
    }

    pub fn list_jobs(&self, project_id: &str) -> MhResult<Vec<JobRecord>> {
        let conn = self.conn.lock().map_err(|_| MhError::Config("Lock failed".into()))?;
        let mut stmt = conn.prepare("SELECT id, project_id, status, progress, message, result_path, created_at, started_at, finished_at FROM jobs WHERE project_id = ?1 ORDER BY created_at DESC").map_err(|e| MhError::Io(e.to_string()))?;
        let iter = stmt.query_map(params![project_id], |row| Ok(JobRecord {
            id: row.get(0)?, project_id: row.get(1)?,
            status: JobStatus::from_str(&row.get::<_, String>(2)?),
            progress: row.get(3)?, message: row.get(4).ok(), result_path: row.get(5).ok(),
            created_at: row.get(6)?, started_at: row.get(7).ok(), finished_at: row.get(8).ok(),
        })).map_err(|e| MhError::Io(e.to_string()))?;
        let mut result = Vec::new();
        for r in iter { result.push(r.map_err(|e| MhError::Io(e.to_string()))?); }
        Ok(result)
    }
}
