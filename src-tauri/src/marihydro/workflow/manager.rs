// src-tauri/src/marihydro/workflow/manager.rs

use log::{error, info, warn};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

use super::job::{JobStatus, SimulationJob};
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::ProjectManifest;
use chrono::Utc;

/// 内存数据库的内部结构
#[derive(Default, Clone)]
pub struct MockDb {
    jobs: HashMap<Uuid, SimulationJob>,
    projects: HashMap<Uuid, ProjectManifest>,
}

/// 工作流管理器 (Service Layer)
/// 负责所有业务逻辑，现在基于内存存储
#[derive(Clone)]
pub struct WorkflowManager {
    db: Arc<Mutex<MockDb>>,
}

impl WorkflowManager {
    pub fn new() -> Self {
        Self {
            db: Arc::new(Mutex::new(MockDb::default())),
        }
    }

    // ========================================================================
    // Project Operations (工程管理)
    // ========================================================================

    pub async fn save_project(&self, manifest: &ProjectManifest) -> MhResult<()> {
        let name = manifest
            .meta
            .get("name")
            .map(|s| s.as_str())
            .unwrap_or("Untitled");
        info!("正在保存工程: {} ({})", name, manifest.id);

        let mut db = self.db.lock();
        db.projects.insert(manifest.id, manifest.clone());
        Ok(())
    }

    pub async fn load_project(&self, project_id: Uuid) -> MhResult<ProjectManifest> {
        let db = self.db.lock();
        db.projects
            .get(&project_id)
            .cloned()
            .ok_or_else(|| MhError::Config(format!("未找到工程 ID: {}", project_id)))
    }

    pub async fn list_projects(&self) -> MhResult<Vec<(Uuid, String, String)>> {
        let db = self.db.lock();
        let mut projects: Vec<_> = db
            .projects
            .values()
            .map(|p| {
                let name = p
                    .meta
                    .get("name")
                    .map(|s| s.as_str())
                    .unwrap_or("Untitled")
                    .to_string();
                // Mock updated_at with current time for simplicity
                (p.id, name, Utc::now().to_rfc3339())
            })
            .collect();

        // Sort by name for consistent ordering, as we don't store updated_at
        projects.sort_by(|a, b| b.2.cmp(&a.2));
        Ok(projects)
    }

    pub async fn delete_project(&self, project_id: Uuid) -> MhResult<()> {
        info!("正在删除工程: {}", project_id);
        let mut db = self.db.lock();

        if db.projects.remove(&project_id).is_some() {
            // Cascade delete jobs associated with the project
            db.jobs.retain(|_, job| job.project_id != project_id);
            Ok(())
        } else {
            warn!("尝试删除不存在的工程: {}", project_id);
            Err(MhError::Config("工程不存在".into()))
        }
    }

    // ========================================================================
    // Job Operations (任务管理)
    // ========================================================================

    pub async fn submit_job(
        &self,
        project_id: Uuid,
        overrides: Option<HashMap<String, f64>>,
    ) -> MhResult<Uuid> {
        let mut db = self.db.lock();

        if !db.projects.contains_key(&project_id) {
            return Err(MhError::Config(format!(
                "无法提交任务：工程 {} 不存在",
                project_id
            )));
        }

        let job_id = Uuid::new_v4();
        let new_job = SimulationJob {
            id: job_id,
            project_id,
            status: JobStatus::Pending,
            progress: 0.0,
            message: Some("任务已提交".to_string()),
            parameter_overrides: overrides.map(|o| serde_json::to_value(o).unwrap()),
            result_path: None,
            created_at: Utc::now(),
            started_at: None,
            finished_at: None,
        };

        info!("提交新任务: JobID={}, ProjectID={}", job_id, project_id);
        db.jobs.insert(job_id, new_job);

        Ok(job_id)
    }

    pub async fn get_job(&self, job_id: Uuid) -> MhResult<SimulationJob> {
        let db = self.db.lock();
        db.jobs
            .get(&job_id)
            .cloned()
            .ok_or_else(|| MhError::Workflow(format!("任务不存在: {}", job_id)))
    }

    pub async fn update_job_progress(
        &self,
        job_id: Uuid,
        progress: f64,
        message: Option<&str>,
    ) -> MhResult<()> {
        let mut db = self.db.lock();
        if let Some(job) = db.jobs.get_mut(&job_id) {
            if job.status == JobStatus::Pending {
                job.status = JobStatus::Running;
                job.started_at = Some(Utc::now());
            }
            job.progress = progress;
            if let Some(msg) = message {
                job.message = Some(msg.to_string());
            }
            Ok(())
        } else {
            Err(MhError::Workflow(format!(
                "更新进度失败：任务 {} 不存在",
                job_id
            )))
        }
    }

    pub async fn complete_job(&self, job_id: Uuid, result_path: &str) -> MhResult<()> {
        info!("任务完成: {}", job_id);
        let mut db = self.db.lock();
        if let Some(job) = db.jobs.get_mut(&job_id) {
            job.status = JobStatus::Completed;
            job.progress = 100.0;
            job.result_path = Some(result_path.to_string());
            job.finished_at = Some(Utc::now());
            Ok(())
        } else {
            Err(MhError::Workflow(format!(
                "完成任务失败：任务 {} 不存在",
                job_id
            )))
        }
    }

    pub async fn fail_job(&self, job_id: Uuid, error_msg: &str) -> MhResult<()> {
        error!("任务失败: {} - {}", job_id, error_msg);
        let mut db = self.db.lock();
        if let Some(job) = db.jobs.get_mut(&job_id) {
            job.status = JobStatus::Failed;
            job.message = Some(error_msg.to_string());
            job.finished_at = Some(Utc::now());
            Ok(())
        } else {
            Err(MhError::Workflow(format!(
                "标记任务失败错误：任务 {} 不存在",
                job_id
            )))
        }
    }

    pub async fn list_jobs(
        &self,
        project_id: Uuid,
        limit: Option<i32>,
    ) -> MhResult<Vec<SimulationJob>> {
        let db = self.db.lock();
        let mut jobs: Vec<_> = db
            .jobs
            .values()
            .filter(|job| job.project_id == project_id)
            .cloned()
            .collect();

        jobs.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        if let Some(limit_val) = limit {
            jobs.truncate(limit_val as usize);
        }

        Ok(jobs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marihydro::infra::manifest::ProjectManifest;
    use tokio::task;

    fn create_test_manager() -> WorkflowManager {
        WorkflowManager::new()
    }

    #[tokio::test]
    async fn test_mock_db_crud_operations() {
        let manager = create_test_manager();
        let project_id = Uuid::new_v4();

        let mut manifest = ProjectManifest::default();
        manifest.id = project_id;
        manager.save_project(&manifest).await.unwrap();

        // Create: 提交任务
        let mut overrides = HashMap::new();
        overrides.insert("physics.gravity".to_string(), 9.81);

        let job_id = manager
            .submit_job(project_id, Some(overrides))
            .await
            .unwrap();
        assert!(!job_id.is_nil());

        // Read: 查询任务
        let job = manager.get_job(job_id).await.unwrap();
        assert_eq!(job.project_id, project_id);
        assert_eq!(job.status, JobStatus::Pending);

        // Update: 修改状态
        manager
            .update_job_progress(job_id, 50.0, Some("Running"))
            .await
            .unwrap();
        let updated = manager.get_job(job_id).await.unwrap();
        assert_eq!(updated.status, JobStatus::Running);
        assert_eq!(updated.progress, 50.0);

        // Delete: 删除工程应级联删除任务
        manager.delete_project(project_id).await.unwrap();
        assert!(manager.get_job(job_id).await.is_err());
    }

    #[tokio::test]
    async fn test_concurrent_job_creation() {
        let manager = Arc::new(create_test_manager());
        let project_id = Uuid::new_v4();

        let mut manifest = ProjectManifest::default();
        manifest.id = project_id;
        manager.save_project(&manifest).await.unwrap();

        let mut handles = vec![];
        for i in 0..10 {
            let mgr_clone = Arc::clone(&manager);
            handles.push(task::spawn(async move {
                let mut overrides = HashMap::new();
                overrides.insert("run_id".to_string(), i as f64);
                mgr_clone.submit_job(project_id, Some(overrides)).await
            }));
        }

        let results = futures::future::join_all(handles).await;
        assert_eq!(results.len(), 10);

        for result in results {
            assert!(result.is_ok());
            assert!(result.unwrap().is_ok());
        }

        let jobs = manager.list_jobs(project_id, None).await.unwrap();
        assert_eq!(jobs.len(), 10);
    }

    #[tokio::test]
    async fn test_job_progress_updates() {
        let manager = create_test_manager();
        let project_id = Uuid::new_v4();
        let mut manifest = ProjectManifest::default();
        manifest.id = project_id;
        manager.save_project(&manifest).await.unwrap();

        let job_id = manager.submit_job(project_id, None).await.unwrap();

        for progress in [10.0, 25.0, 50.0, 75.0] {
            manager
                .update_job_progress(job_id, progress, Some(&format!("Step {}", progress)))
                .await
                .unwrap();

            let job = manager.get_job(job_id).await.unwrap();
            assert_eq!(job.progress, progress);
            assert_eq!(job.status, JobStatus::Running);
        }

        manager
            .complete_job(job_id, "/path/to/results")
            .await
            .unwrap();
        let final_job = manager.get_job(job_id).await.unwrap();
        assert_eq!(final_job.progress, 100.0);
        assert_eq!(final_job.status, JobStatus::Completed);
    }
}
