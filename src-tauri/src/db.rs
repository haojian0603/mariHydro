// src-tauri/src/db.rs

use sqlx::postgres::PgPoolOptions;
use sqlx::{Pool, Postgres};

// 定义一个别名方便使用
pub type DbPool = Pool<Postgres>;

/// 初始化数据库连接池
/// database_url 格式: "postgres://user:password@host/database_name"
pub async fn init_db(database_url: &str) -> Result<DbPool, sqlx::Error> {
    println!("正在连接数据库: {}", database_url);

    let pool = PgPoolOptions::new()
        .max_connections(5) // 对于桌面应用，不需要太大的连接池
        .connect(database_url)
        .await?;

    // (可选) 运行数据库迁移 (Migrations)
    // 这可以自动在数据库中创建 simulations, terrains 等表
    // sqlx::migrate!("./migrations").run(&pool).await?;

    Ok(pool)
}
