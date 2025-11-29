// src-tauri/src/main.rs

use marihydro_lib::init_logging;

fn main() {
    // ✅ 初始化日志系统
    init_logging(Some("info"));

    log::info!("mariHydro Desktop 启动中...");

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            marihydro_lib::commands::get_default_config,
            marihydro_lib::commands::run_simulation,
            marihydro_lib::commands::save_simulation_record,
        ])
        .run(tauri::generate_context!())
        .expect("启动 Tauri 应用失败");
}
