//! 内存存储实现
//!
//! 基于 `parking_lot::RwLock` 和 `HashMap` 的并发安全内存存储。
//! 适用于测试和短期运行场景。

use super::traits::WorkflowStorage;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::workflow::job::{JobStatus, SimulationJob};
use parking_lot::RwLock;
use std::collections::HashMap;

/// 并发安全的内存存储
///
/// 使用 `RwLock` 实现读写锁，允许多个读取者同时访问。
///
/// # 特性
///
/// - 线程安全：所有操作都是原子的
/// - 高性能：读操作无锁竞争
/// - 零持久化：程序退出后数据丢失
///
/// # 使用场景
///
/// - 单元测试
/// - 开发调试
/// - 短期模拟任务
pub struct MemoryStorage {
    projects: RwLock<HashMap<String, String>>,
    jobs: RwLock<HashMap<String, SimulationJob>>,
}

impl MemoryStorage {
    /// 创建新的内存存储
    pub fn new() -> Self {
        Self {
            projects: RwLock::new(HashMap::new()),
            jobs: RwLock::new(HashMap::new()),
        }
    }

    /// 获取项目数量
    pub fn project_count(&self) -> usize {
        self.projects.read().len()
    }

    /// 获取作业数量
    pub fn job_count(&self) -> usize {
        self.jobs.read().len()
    }

    /// 清空所有数据
    pub fn clear(&self) {
        self.projects.write().clear();
        self.jobs.write().clear();
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkflowStorage for MemoryStorage {
    fn save_project(&self, id: &str, data: &str) -> MhResult<()> {
        self.projects.write().insert(id.to_string(), data.to_string());
        Ok(())
    }

    fn load_project(&self, id: &str) -> MhResult<Option<String>> {
        Ok(self.projects.read().get(id).cloned())
    }

    fn delete_project(&self, id: &str) -> MhResult<bool> {
        let removed = self.projects.write().remove(id).is_some();
        if removed {
            // 同时删除关联的作业
            self.jobs.write().retain(|_, j| j.project_id != id);
        }
        Ok(removed)
    }

    fn list_project_ids(&self) -> MhResult<Vec<String>> {
        Ok(self.projects.read().keys().cloned().collect())
    }

    fn save_job(&self, job: &SimulationJob) -> MhResult<()> {
        self.jobs.write().insert(job.id.clone(), job.clone());
        Ok(())
    }

    fn get_job(&self, id: &str) -> MhResult<Option<SimulationJob>> {
        Ok(self.jobs.read().get(id).cloned())
    }

    fn update_job_status(
        &self,
        id: &str,
        status: JobStatus,
        message: Option<&str>,
    ) -> MhResult<bool> {
        let mut jobs = self.jobs.write();
        if let Some(job) = jobs.get_mut(id) {
            job.status = status;
            if let Some(msg) = message {
                job.message = Some(msg.to_string());
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn update_job_progress(
        &self,
        id: &str,
        progress: f64,
        message: Option<&str>,
    ) -> MhResult<bool> {
        let mut jobs = self.jobs.write();
        if let Some(job) = jobs.get_mut(id) {
            job.progress = progress.clamp(0.0, 100.0);
            if let Some(msg) = message {
                job.message = Some(msg.to_string());
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn list_jobs(&self, project_id: &str) -> MhResult<Vec<SimulationJob>> {
        Ok(self
            .jobs
            .read()
            .values()
            .filter(|j| j.project_id == project_id)
            .cloned()
            .collect())
    }

    fn list_jobs_by_status(&self, status: JobStatus) -> MhResult<Vec<SimulationJob>> {
        Ok(self
            .jobs
            .read()
            .values()
            .filter(|j| j.status == status)
            .cloned()
            .collect())
    }

    fn delete_job(&self, id: &str) -> MhResult<bool> {
        Ok(self.jobs.write().remove(id).is_some())
    }

    fn delete_jobs_by_project(&self, project_id: &str) -> MhResult<usize> {
        let mut jobs = self.jobs.write();
        let before = jobs.len();
        jobs.retain(|_, j| j.project_id != project_id);
        Ok(before - jobs.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_crud() {
        let storage = MemoryStorage::new();

        // Create
        storage.save_project("p1", "data1").unwrap();
        assert_eq!(storage.project_count(), 1);

        // Read
        assert_eq!(storage.load_project("p1").unwrap(), Some("data1".into()));
        assert_eq!(storage.load_project("p2").unwrap(), None);

        // Update
        storage.save_project("p1", "data1_updated").unwrap();
        assert_eq!(
            storage.load_project("p1").unwrap(),
            Some("data1_updated".into())
        );

        // Delete
        assert!(storage.delete_project("p1").unwrap());
        assert!(!storage.delete_project("p1").unwrap());
        assert_eq!(storage.project_count(), 0);
    }

    #[test]
    fn test_job_crud() {
        let storage = MemoryStorage::new();

        // 先创建项目
        storage.save_project("p1", "{}").unwrap();

        // 创建作业
        let job = SimulationJob::new("j1".into(), "p1".into());
        storage.save_job(&job).unwrap();
        assert_eq!(storage.job_count(), 1);

        // 读取
        let loaded = storage.get_job("j1").unwrap().unwrap();
        assert_eq!(loaded.id, "j1");
        assert_eq!(loaded.status, JobStatus::Pending);

        // 更新状态
        storage
            .update_job_status("j1", JobStatus::Running, Some("Started"))
            .unwrap();
        let updated = storage.get_job("j1").unwrap().unwrap();
        assert_eq!(updated.status, JobStatus::Running);
        assert_eq!(updated.message, Some("Started".into()));

        // 更新进度
        storage.update_job_progress("j1", 50.0, None).unwrap();
        let updated = storage.get_job("j1").unwrap().unwrap();
        assert!((updated.progress - 50.0).abs() < 0.01);

        // 删除
        assert!(storage.delete_job("j1").unwrap());
        assert!(!storage.delete_job("j1").unwrap());
    }

    #[test]
    fn test_list_jobs_by_status() {
        let storage = MemoryStorage::new();
        storage.save_project("p1", "{}").unwrap();

        let job1 = SimulationJob::new("j1".into(), "p1".into());
        let mut job2 = SimulationJob::new("j2".into(), "p1".into());
        job2.status = JobStatus::Running;

        storage.save_job(&job1).unwrap();
        storage.save_job(&job2).unwrap();

        let pending = storage.list_jobs_by_status(JobStatus::Pending).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, "j1");

        let running = storage.list_jobs_by_status(JobStatus::Running).unwrap();
        assert_eq!(running.len(), 1);
        assert_eq!(running[0].id, "j2");
    }

    #[test]
    fn test_delete_project_cascades() {
        let storage = MemoryStorage::new();
        storage.save_project("p1", "{}").unwrap();

        let job = SimulationJob::new("j1".into(), "p1".into());
        storage.save_job(&job).unwrap();

        assert_eq!(storage.job_count(), 1);

        // 删除项目应该级联删除作业
        storage.delete_project("p1").unwrap();
        assert_eq!(storage.job_count(), 0);
    }
}
