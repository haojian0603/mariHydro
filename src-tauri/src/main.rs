// src-tauri/src/main.rs

// 引入库定义
use marihydro_lib::commands;
// use marihydro_lib::db; // 暂时注释，数据库未配置前避免连接错误
use tauri::Manager;

#[tokio::main]
async fn main() {
    println!("正在启动 mariHydro 桌面后端...");

    // 1. 数据库初始化 (预留)
    // 如果你已经安装了 PostgreSQL 并且想启用数据库功能，请取消注释以下代码
    // 并且确保在 .env 文件或环境变量中设置了 DATABASE_URL
    /*
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:password@localhost/marihydro".to_string());

    let pool = match db::init_db(&db_url).await {
        Ok(p) => {
            println!("数据库连接成功: {}", db_url);
            p
        },
        Err(e) => {
            eprintln!("(警告) 数据库连接失败: {}。部分功能将不可用。", e);
            // 这里可以决定是 panic 还是继续运行 (无数据库模式)
            // 为了演示方便，我们这里不 panic，但实际代码中最好处理这个 pool
            // 因为后面的 commands 可能需要依赖 pool
            panic!("数据库必须连接才能启动 (开发阶段限制)");
        }
    };
    */

    // 2. 构建并运行 Tauri 应用
    tauri::Builder::default()
        // 注册所有前端可调用的 API 命令
        .invoke_handler(tauri::generate_handler![
            commands::get_default_config,
            commands::run_simulation,
            commands::save_simulation_record // 预留接口
        ])
        // 状态管理 (依赖注入)
        // .manage(pool) // 将数据库连接池注入到 Tauri 状态中，供 commands 使用
        .setup(|app| {
            // 应用启动后的初始化钩子
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").unwrap();
                window.open_devtools(); // 开发模式下自动打开控制台
            }
            println!("mariHydro 准备就绪，等待前端指令。");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("mariHydro 运行过程中发生严重错误");
}
