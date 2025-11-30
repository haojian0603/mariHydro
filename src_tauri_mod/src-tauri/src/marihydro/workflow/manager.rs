// src-tauri/src/marihydro/workflow/manager.rs
use super::job::{JobStatus, SimulationJob};
use crate::marihydro::core::error::{MhError, MhResult};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Default)]
struct InMemoryStore {
    jobs: HashMap<String, SimulationJob>,
    projects: HashMap<String, String>,
}

pub struct WorkflowManager {
    store: Arc<Mutex<InMemoryStore>>,
}

impl WorkflowManager {
    pub fn new() -> Self { Self { store: Arc::new(Mutex::new(InMemoryStore::default())) } }

    pub fn save_project(&self, id: &str, manifest_json: &str) -> MhResult<()> {
        self.store.lock().projects.insert(id.into(), manifest_json.into());
        Ok(())
    }

    pub fn load_project(&self, id: &str) -> MhResult<String> {
        self.store.lock().projects.get(id).cloned().ok_or_else(|| MhError::Config(format!("Project {} not found", id)))
    }

    pub fn list_projects(&self) -> Vec<String> { self.store.lock().projects.keys().cloned().collect() }

    pub fn delete_project(&self, id: &str) -> MhResult<()> {
        let mut store = self.store.lock();
        store.projects.remove(id);
        store.jobs.retain(|_, j| j.project_id != id);
        Ok(())
    }

    pub fn submit_job(&self, project_id: &str, overrides: Option<HashMap<String, f64>>) -> MhResult<String> {
        let store = self.store.lock();
        if !store.projects.contains_key(project_id) { return Err(MhError::Config(format!("Project {} not found", project_id))); }
        drop(store);
        let job_id = Uuid::new_v4().to_string();
        let mut job = SimulationJob::new(job_id.clone(), project_id.into());
        if let Some(o) = overrides { job = job.with_overrides(o); }
        self.store.lock().jobs.insert(job_id.clone(), job);
        Ok(job_id)
    }

    pub fn get_job(&self, job_id: &str) -> MhResult<SimulationJob> {
        self.store.lock().jobs.get(job_id).cloned().ok_or_else(|| MhError::Config(format!("Job {} not found", job_id)))
    }

    pub fn update_job_progress(&self, job_id: &str, progress: f64, message: Option<&str>) -> MhResult<()> {
        let mut store = self.store.lock();
        if let Some(job) = store.jobs.get_mut(job_id) {
            if job.status == JobStatus::Pending { job.start(); }
            job.update_progress(progress, message);
            Ok(())
        } else { Err(MhError::Config(format!("Job {} not found", job_id))) }
    }

    pub fn complete_job(&self, job_id: &str, result_path: &str) -> MhResult<()> {
        let mut store = self.store.lock();
        if let Some(job) = store.jobs.get_mut(job_id) { job.complete(result_path); Ok(()) }
        else { Err(MhError::Config(format!("Job {} not found", job_id))) }
    }

    pub fn fail_job(&self, job_id: &str, error: &str) -> MhResult<()> {
        let mut store = self.store.lock();
        if let Some(job) = store.jobs.get_mut(job_id) { job.fail(error); Ok(()) }
        else { Err(MhError::Config(format!("Job {} not found", job_id))) }
    }

    pub fn cancel_job(&self, job_id: &str) -> MhResult<()> {
        let mut store = self.store.lock();
        if let Some(job) = store.jobs.get_mut(job_id) {
            if !job.status.is_terminal() { job.cancel(); }
            Ok(())
        } else { Err(MhError::Config(format!("Job {} not found", job_id))) }
    }

    pub fn list_jobs(&self, project_id: &str) -> Vec<SimulationJob> {
        self.store.lock().jobs.values().filter(|j| j.project_id == project_id).cloned().collect()
    }

    pub fn list_pending_jobs(&self) -> Vec<SimulationJob> {
        self.store.lock().jobs.values().filter(|j| j.status == JobStatus::Pending).cloned().collect()
    }
}

impl Default for WorkflowManager { fn default() -> Self { Self::new() } }
