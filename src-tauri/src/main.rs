// src-tauri/src/main.rs
// 引入库定义
use marihydro_lib::commands;
use tauri::Manager;

#[tokio::main]
async fn main() {
    println!("正在启动 mariHydro 桌面后端...");

    // 构建并运行 Tauri 应用
    tauri::Builder::default()
        // 注册所有前端可调用的 API 命令
        .invoke_handler(tauri::generate_handler![
            commands::get_default_config,
            commands::run_simulation,
            commands::save_simulation_record // 预留接口
        ])
        // 状态管理 (依赖注入)
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
