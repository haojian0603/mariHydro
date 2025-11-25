// src-tauri/src/marihydro/infra/db.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePool, SqlitePoolOptions};
use std::path::Path;
use std::str::FromStr;

/// 初始化嵌入式数据库
/// path: 数据库文件路径 (例如 "app_data/history.db")
pub async fn init_sqlite(path: &str) -> MhResult<SqlitePool> {
    // 1. 确保文件存在 (SqliteConnectOptions create_if_missing 不一定创建目录)
    let db_path = Path::new(path);
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).map_err(MhError::Io)?;
    }

    // 2. 连接配置
    let options = SqliteConnectOptions::from_str(&format!("sqlite://{}", path))
        .map_err(|e| MhError::Config(format!("DB连接串错误: {}", e)))?
        .create_if_missing(true);

    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(options)
        .await
        .map_err(|e| {
            MhError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        })?;

    // 3. 自动建表 (Schema Migration)
    // 这是一个简化的迁移逻辑，生产环境建议使用 sqlx::migrate!
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            manifest_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            status TEXT NOT NULL, -- 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'

            -- 参数覆盖 (JSON): 用于敏感性分析
            -- 例如: { "physics.bottom_friction_coeff": 0.03 }
            parameter_overrides TEXT,

            progress REAL DEFAULT 0.0,
            message TEXT,          -- 当前状态消息或报错信息
            result_path TEXT,      -- 结果输出目录

            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME,
            finished_at DATETIME,

            FOREIGN KEY(project_id) REFERENCES projects(id)
        );
        "#,
    )
    .execute(&pool)
    .await
    .map_err(|e| {
        MhError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("建表失败: {}", e),
        ))
    })?;

    Ok(pool)
}
