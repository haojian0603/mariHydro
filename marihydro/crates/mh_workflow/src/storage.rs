// crates/mh_workflow/src/storage.rs

//! 存储后端模块
//!
//! 提供任务持久化存储的抽象和实现。

use crate::job::{JobId, SimulationJob};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// 存储错误
#[derive(Debug, Error)]
pub enum StorageError {
    /// IO错误
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// 序列化错误
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// 任务不存在
    #[error("Job not found: {0}")]
    NotFound(JobId),

    /// 存储已满
    #[error("Storage is full")]
    Full,

    /// 其他错误
    #[error("{0}")]
    Other(String),
}

/// 存储后端trait
pub trait Storage: Send + Sync {
    /// 保存任务
    fn save_job(&self, job: &SimulationJob) -> Result<(), StorageError>;

    /// 加载任务
    fn load_job(&self, id: JobId) -> Result<Option<SimulationJob>, StorageError>;

    /// 删除任务
    fn delete_job(&self, id: JobId) -> Result<(), StorageError>;

    /// 列出所有任务
    fn list_jobs(&self) -> Result<Vec<SimulationJob>, StorageError>;

    /// 检查任务是否存在
    fn contains(&self, id: JobId) -> Result<bool, StorageError> {
        Ok(self.load_job(id)?.is_some())
    }

    /// 获取任务数量
    fn count(&self) -> Result<usize, StorageError> {
        Ok(self.list_jobs()?.len())
    }

    /// 清空所有任务
    fn clear(&self) -> Result<(), StorageError>;
}

/// 内存存储
#[derive(Debug, Default)]
pub struct MemoryStorage {
    jobs: RwLock<HashMap<JobId, SimulationJob>>,
    max_capacity: Option<usize>,
}

impl MemoryStorage {
    /// 创建新的内存存储
    pub fn new() -> Self {
        Self {
            jobs: RwLock::new(HashMap::new()),
            max_capacity: None,
        }
    }

    /// 创建带容量限制的内存存储
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            jobs: RwLock::new(HashMap::with_capacity(max_capacity)),
            max_capacity: Some(max_capacity),
        }
    }

    /// 获取当前任务数量
    pub fn len(&self) -> usize {
        self.jobs.read().len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.jobs.read().is_empty()
    }
}

impl Storage for MemoryStorage {
    fn save_job(&self, job: &SimulationJob) -> Result<(), StorageError> {
        let mut jobs = self.jobs.write();

        // 检查容量
        if let Some(max) = self.max_capacity {
            if jobs.len() >= max && !jobs.contains_key(&job.id) {
                return Err(StorageError::Full);
            }
        }

        jobs.insert(job.id, job.clone());
        Ok(())
    }

    fn load_job(&self, id: JobId) -> Result<Option<SimulationJob>, StorageError> {
        Ok(self.jobs.read().get(&id).cloned())
    }

    fn delete_job(&self, id: JobId) -> Result<(), StorageError> {
        self.jobs.write().remove(&id);
        Ok(())
    }

    fn list_jobs(&self) -> Result<Vec<SimulationJob>, StorageError> {
        Ok(self.jobs.read().values().cloned().collect())
    }

    fn count(&self) -> Result<usize, StorageError> {
        Ok(self.jobs.read().len())
    }

    fn clear(&self) -> Result<(), StorageError> {
        self.jobs.write().clear();
        Ok(())
    }
}

/// 文件存储
#[derive(Debug)]
pub struct FileStorage {
    /// 存储目录
    dir: PathBuf,
    /// 内存缓存
    cache: RwLock<HashMap<JobId, SimulationJob>>,
    /// 是否启用缓存
    use_cache: bool,
}

impl FileStorage {
    /// 创建新的文件存储
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self, StorageError> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;

        let storage = Self {
            dir,
            cache: RwLock::new(HashMap::new()),
            use_cache: true,
        };

        // 加载现有任务到缓存
        storage.load_all_to_cache()?;

        Ok(storage)
    }

    /// 禁用缓存
    pub fn without_cache(mut self) -> Self {
        self.use_cache = false;
        self.cache.write().clear();
        self
    }

    /// 获取任务文件路径
    fn job_path(&self, id: JobId) -> PathBuf {
        self.dir.join(format!("{}.json", id))
    }

    /// 加载所有任务到缓存
    fn load_all_to_cache(&self) -> Result<(), StorageError> {
        if !self.use_cache {
            return Ok(());
        }

        let mut cache = self.cache.write();
        cache.clear();

        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(job) = self.load_from_file(&path) {
                    cache.insert(job.id, job);
                }
            }
        }

        Ok(())
    }

    /// 从文件加载任务
    fn load_from_file(&self, path: &Path) -> Result<SimulationJob, StorageError> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(|e| StorageError::Serialization(e.to_string()))
    }

    /// 保存任务到文件
    fn save_to_file(&self, job: &SimulationJob) -> Result<(), StorageError> {
        let path = self.job_path(job.id);
        let json = serde_json::to_string_pretty(job)
            .map_err(|e| StorageError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// 获取存储目录
    pub fn directory(&self) -> &Path {
        &self.dir
    }
}

impl Storage for FileStorage {
    fn save_job(&self, job: &SimulationJob) -> Result<(), StorageError> {
        // 保存到文件
        self.save_to_file(job)?;

        // 更新缓存
        if self.use_cache {
            self.cache.write().insert(job.id, job.clone());
        }

        Ok(())
    }

    fn load_job(&self, id: JobId) -> Result<Option<SimulationJob>, StorageError> {
        // 先查缓存
        if self.use_cache {
            if let Some(job) = self.cache.read().get(&id).cloned() {
                return Ok(Some(job));
            }
        }

        // 再查文件
        let path = self.job_path(id);
        if !path.exists() {
            return Ok(None);
        }

        let job = self.load_from_file(&path)?;

        // 更新缓存
        if self.use_cache {
            self.cache.write().insert(id, job.clone());
        }

        Ok(Some(job))
    }

    fn delete_job(&self, id: JobId) -> Result<(), StorageError> {
        // 删除文件
        let path = self.job_path(id);
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        // 从缓存移除
        if self.use_cache {
            self.cache.write().remove(&id);
        }

        Ok(())
    }

    fn list_jobs(&self) -> Result<Vec<SimulationJob>, StorageError> {
        if self.use_cache {
            return Ok(self.cache.read().values().cloned().collect());
        }

        // 无缓存时从文件加载
        let mut jobs = Vec::new();
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Ok(job) = self.load_from_file(&path) {
                    jobs.push(job);
                }
            }
        }

        Ok(jobs)
    }

    fn count(&self) -> Result<usize, StorageError> {
        if self.use_cache {
            return Ok(self.cache.read().len());
        }

        let mut count = 0;
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.path().extension().is_some_and(|ext| ext == "json") {
                count += 1;
            }
        }
        Ok(count)
    }

    fn clear(&self) -> Result<(), StorageError> {
        // 删除所有文件
        for entry in std::fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "json") {
                std::fs::remove_file(path)?;
            }
        }

        // 清空缓存
        if self.use_cache {
            self.cache.write().clear();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::SimulationConfig;

    fn create_test_job(name: &str) -> SimulationJob {
        let config = SimulationConfig::new("test.mhp");
        SimulationJob::new(name, config)
    }

    #[test]
    fn test_memory_storage() {
        let storage = MemoryStorage::new();

        let job1 = create_test_job("Job 1");
        let job2 = create_test_job("Job 2");

        storage.save_job(&job1).unwrap();
        storage.save_job(&job2).unwrap();

        assert_eq!(storage.count().unwrap(), 2);

        let loaded = storage.load_job(job1.id).unwrap().unwrap();
        assert_eq!(loaded.name, "Job 1");

        storage.delete_job(job1.id).unwrap();
        assert_eq!(storage.count().unwrap(), 1);

        storage.clear().unwrap();
        assert_eq!(storage.count().unwrap(), 0);
    }

    #[test]
    fn test_memory_storage_capacity() {
        let storage = MemoryStorage::with_capacity(2);

        let job1 = create_test_job("Job 1");
        let job2 = create_test_job("Job 2");
        let job3 = create_test_job("Job 3");

        storage.save_job(&job1).unwrap();
        storage.save_job(&job2).unwrap();

        // 超出容量应该失败
        let result = storage.save_job(&job3);
        assert!(matches!(result, Err(StorageError::Full)));

        // 更新已有任务应该成功
        storage.save_job(&job1).unwrap();
    }

    #[test]
    fn test_file_storage() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage = FileStorage::new(temp_dir.path()).unwrap();

        let job = create_test_job("Test Job");
        storage.save_job(&job).unwrap();

        assert!(storage.contains(job.id).unwrap());

        let loaded = storage.load_job(job.id).unwrap().unwrap();
        assert_eq!(loaded.name, "Test Job");

        // 检查文件存在
        let job_path = temp_dir.path().join(format!("{}.json", job.id));
        assert!(job_path.exists());

        storage.delete_job(job.id).unwrap();
        assert!(!job_path.exists());
    }
}
