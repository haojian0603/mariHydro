// src-tauri/src/main.rs
use marihydro_lib::marihydro::infra::init_logging;

fn main() {
    init_logging(Some("info"));
    log::info!("MariHydro Desktop starting...");
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            marihydro_lib::commands::get_default_config,
            marihydro_lib::commands::run_simulation,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to start Tauri app");
}
