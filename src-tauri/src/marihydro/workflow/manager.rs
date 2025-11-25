// src-tauri/src/marihydro/workflow/manager.rs

use log::{error, info, warn};
use sqlx::sqlite::SqlitePool;
use std::collections::HashMap;
use uuid::Uuid;

use super::job::{JobStatus, SimulationJob};
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::ProjectManifest;

/// 工作流管理器 (Service Layer)
/// 负责所有与持久化存储交互的高级业务逻辑
pub struct WorkflowManager {
    pool: SqlitePool,
}

impl WorkflowManager {
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    // ========================================================================
    // Project Operations (工程管理)
    // ========================================================================

    /// 保存工程 (Upsert: 新建或更新)
    pub async fn save_project(&self, manifest: &ProjectManifest) -> MhResult<()> {
        let json = serde_json::to_string(manifest)
            .map_err(|e| MhError::Serialization(format!("ProjectManifest 序列化失败: {}", e)))?;

        let name = manifest
            .meta
            .get("name")
            .map(|s| s.as_str())
            .unwrap_or("Untitled");

        info!("正在保存工程: {} ({})", name, manifest.id);

        // 使用原生 Uuid 类型绑定
        sqlx::query(
            r#"
            INSERT INTO projects (id, name, manifest_json, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                manifest_json = excluded.manifest_json,
                updated_at = CURRENT_TIMESTAMP
            "#,
        )
        .bind(manifest.id) // sqlx 自动处理 Uuid -> String/Blob
        .bind(name)
        .bind(json)
        .execute(&self.pool)
        .await
        .map_err(|e| MhError::Database(format!("保存工程失败: {}", e)))?;

        Ok(())
    }

    /// 加载工程蓝图
    pub async fn load_project(&self, project_id: Uuid) -> MhResult<ProjectManifest> {
        let row: (String,) = sqlx::query_as("SELECT manifest_json FROM projects WHERE id = ?")
            .bind(project_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| MhError::Database(format!("查询工程失败: {}", e)))?
            .ok_or_else(|| MhError::Config(format!("未找到工程 ID: {}", project_id)))?;

        let manifest: ProjectManifest = serde_json::from_str(&row.0)
            .map_err(|e| MhError::Serialization(format!("工程文件损坏/版本不兼容: {}", e)))?;

        Ok(manifest)
    }

    /// 列出所有工程摘要 (用于首页列表)
    /// 返回: Vec<(id, name, updated_at)>
    pub async fn list_projects(&self) -> MhResult<Vec<(Uuid, String, String)>> {
        let rows: Vec<(Uuid, String, String)> =
            sqlx::query_as("SELECT id, name, updated_at FROM projects ORDER BY updated_at DESC")
                .fetch_all(&self.pool)
                .await
                .map_err(|e| MhError::Database(format!("获取工程列表失败: {}", e)))?;

        Ok(rows)
    }

    /// 删除工程 (及其关联的所有任务)
    /// 注意：SQLite 需要开启外键约束 (PRAGMA foreign_keys = ON) 才能级联删除 jobs
    pub async fn delete_project(&self, project_id: Uuid) -> MhResult<()> {
        info!("正在删除工程: {}", project_id);

        // 手动级联删除 Jobs (为了保险，防止数据库未配置级联)
        sqlx::query("DELETE FROM jobs WHERE project_id = ?")
            .bind(project_id)
            .execute(&self.pool)
            .await
            .map_err(|e| MhError::Database(format!("级联删除任务失败: {}", e)))?;

        let result = sqlx::query("DELETE FROM projects WHERE id = ?")
            .bind(project_id)
            .execute(&self.pool)
            .await
            .map_err(|e| MhError::Database(format!("删除工程失败: {}", e)))?;

        if result.rows_affected() == 0 {
            warn!("尝试删除不存在的工程: {}", project_id);
            return Err(MhError::Config("工程不存在".into()));
        }

        Ok(())
    }

    // ========================================================================
    // Job Operations (任务管理)
    // ========================================================================

    /// 提交一个新任务
    pub async fn submit_job(
        &self,
        project_id: Uuid,
        overrides: Option<HashMap<String, f64>>,
    ) -> MhResult<Uuid> {
        // 1. 校验工程是否存在
        let project_exists: (i64,) = sqlx::query_as("SELECT count(*) FROM projects WHERE id = ?")
            .bind(project_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| MhError::Database(format!("校验工程存在性失败: {}", e)))?;

        if project_exists.0 == 0 {
            return Err(MhError::Config(format!(
                "无法提交任务：工程 {} 不存在",
                project_id
            )));
        }

        // 2. 准备数据
        let job_id = Uuid::new_v4();
        let overrides_json = match overrides {
            Some(map) => Some(serde_json::to_string(&map).unwrap()),
            None => None,
        };

        info!("提交新任务: JobID={}, ProjectID={}", job_id, project_id);

        // 3. 插入数据库
        sqlx::query(
            r#"
            INSERT INTO jobs (id, project_id, status, parameter_overrides, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            "#,
        )
        .bind(job_id)
        .bind(project_id)
        .bind(JobStatus::Pending.to_string())
        .bind(overrides_json)
        .execute(&self.pool)
        .await
        .map_err(|e| MhError::Database(format!("提交任务失败: {}", e)))?;

        Ok(job_id)
    }

    /// 获取单个任务详情
    pub async fn get_job(&self, job_id: Uuid) -> MhResult<SimulationJob> {
        let job = sqlx::query_as::<_, SimulationJob>("SELECT * FROM jobs WHERE id = ?")
            .bind(job_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| MhError::Database(e.to_string()))?
            .ok_or_else(|| MhError::Workflow(format!("任务不存在: {}", job_id)))?;

        Ok(job)
    }

    /// 更新任务进度 (高频调用)
    pub async fn update_job_progress(
        &self,
        job_id: Uuid,
        progress: f64,
        message: Option<&str>,
    ) -> MhResult<()> {
        // 状态转换为 "Running" 如果之前是 Pending
        sqlx::query(
            r#"
            UPDATE jobs
            SET progress = ?, message = ?, status = CASE WHEN status = 'PENDING' THEN 'RUNNING' ELSE status END, started_at = COALESCE(started_at, CURRENT_TIMESTAMP)
            WHERE id = ?
            "#
        )
        .bind(progress)
        .bind(message)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| MhError::Database(format!("更新进度失败: {}", e)))?;

        Ok(())
    }

    /// 标记任务完成
    pub async fn complete_job(&self, job_id: Uuid, result_path: &str) -> MhResult<()> {
        info!("任务完成: {}", job_id);
        sqlx::query(
            "UPDATE jobs SET status = ?, progress = 100.0, result_path = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?"
        )
        .bind(JobStatus::Completed.to_string())
        .bind(result_path)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| MhError::Database(format!("标记任务完成失败: {}", e)))?;
        Ok(())
    }

    /// 标记任务失败
    pub async fn fail_job(&self, job_id: Uuid, error_msg: &str) -> MhResult<()> {
        error!("任务失败: {} - {}", job_id, error_msg);
        sqlx::query(
            "UPDATE jobs SET status = ?, message = ?, finished_at = CURRENT_TIMESTAMP WHERE id = ?",
        )
        .bind(JobStatus::Failed.to_string())
        .bind(error_msg)
        .bind(job_id)
        .execute(&self.pool)
        .await
        .map_err(|e| MhError::Database(format!("标记任务失败错误: {}", e)))?;
        Ok(())
    }

    /// 获取项目下的任务列表 (分页)
    /// limit: 限制返回数量 (默认 50)
    pub async fn list_jobs(
        &self,
        project_id: Uuid,
        limit: Option<i32>,
    ) -> MhResult<Vec<SimulationJob>> {
        let limit_val = limit.unwrap_or(50);

        let jobs = sqlx::query_as::<_, SimulationJob>(
            "SELECT * FROM jobs WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
        )
        .bind(project_id)
        .bind(limit_val)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| MhError::Database(format!("获取任务列表失败: {}", e)))?;

        Ok(jobs)
    }
}
