// src-tauri/src/marihydro/workflow/runner.rs
use super::manager::WorkflowManager;
use crate::marihydro::core::error::{MhError, MhResult};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

pub trait SimulationExecutor: Send + Sync {
    fn run(&self, job_id: &str, project_json: &str, on_progress: Box<dyn Fn(f64, &str) + Send>) -> MhResult<String>;
}

pub struct SimulationRunner {
    manager: Arc<WorkflowManager>,
    running: Arc<AtomicBool>,
    poll_interval: Duration,
}

impl SimulationRunner {
    pub fn new(manager: Arc<WorkflowManager>) -> Self {
        Self { manager, running: Arc::new(AtomicBool::new(false)), poll_interval: Duration::from_millis(500) }
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self { self.poll_interval = interval; self }

    pub fn start<E: SimulationExecutor + 'static>(&self, executor: E) {
        if self.running.swap(true, Ordering::SeqCst) { return; }
        let manager = Arc::clone(&self.manager);
        let running = Arc::clone(&self.running);
        let interval = self.poll_interval;
        let executor = Arc::new(executor);
        thread::spawn(move || {
            while running.load(Ordering::SeqCst) {
                let pending = manager.list_pending_jobs();
                for job in pending {
                    if !running.load(Ordering::SeqCst) { break; }
                    let job_id = job.id.clone();
                    let project_id = job.project_id.clone();
                    let mgr = Arc::clone(&manager);
                    let exec = Arc::clone(&executor);
                    let project_json = match mgr.load_project(&project_id) { Ok(p) => p, Err(e) => { let _ = mgr.fail_job(&job_id, &e.to_string()); continue; } };
                    let _ = mgr.update_job_progress(&job_id, 0.0, Some("Starting"));
                    let mgr_cb = Arc::clone(&mgr);
                    let jid = job_id.clone();
                    let on_progress: Box<dyn Fn(f64, &str) + Send> = Box::new(move |p, m| { let _ = mgr_cb.update_job_progress(&jid, p, Some(m)); });
                    match exec.run(&job_id, &project_json, on_progress) {
                        Ok(result_path) => { let _ = mgr.complete_job(&job_id, &result_path); }
                        Err(e) => { let _ = mgr.fail_job(&job_id, &e.to_string()); }
                    }
                }
                thread::sleep(interval);
            }
        });
    }

    pub fn stop(&self) { self.running.store(false, Ordering::SeqCst); }
    pub fn is_running(&self) -> bool { self.running.load(Ordering::SeqCst) }
}

pub struct MockExecutor;

impl SimulationExecutor for MockExecutor {
    fn run(&self, _job_id: &str, _project_json: &str, on_progress: Box<dyn Fn(f64, &str) + Send>) -> MhResult<String> {
        for i in 0..10 {
            on_progress((i + 1) as f64 * 10.0, &format!("Step {}/10", i + 1));
            thread::sleep(Duration::from_millis(100));
        }
        Ok("./output/result.vtu".into())
    }
}
