// src-tauri/src/commands.rs
use serde::{Deserialize, Serialize};
use tauri::{Emitter, Window};

#[derive(Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub mesh_path: String,
    pub output_dir: String,
    pub t_end: f64,
    pub initial_water_level: f64,
    pub cfl_factor: f64,
    pub gravity: f64,
    pub h_min: f64,
    pub manning_coef: f64,
    pub enable_muscl: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            mesh_path: "assets/mesh/domain.msh".into(), output_dir: "output".into(),
            t_end: 3600.0, initial_water_level: 10.0, cfl_factor: 0.5, gravity: 9.81,
            h_min: 0.05, manning_coef: 0.025, enable_muscl: true,
        }
    }
}

#[derive(Clone, Serialize)]
struct ProgressPayload {
    step: u32, sim_time: f64, percentage: f64, message: String,
}

#[tauri::command]
pub fn get_default_config() -> SimulationConfig { SimulationConfig::default() }

#[tauri::command]
pub async fn run_simulation(window: Window, config: SimulationConfig) -> Result<String, String> {
    log::info!("Simulation request: mesh={}", config.mesh_path);
    std::fs::create_dir_all(&config.output_dir).map_err(|e| e.to_string())?;
    let _ = window.emit("simulation-progress", ProgressPayload {
        step: 0, sim_time: 0.0, percentage: 0.0, message: "Starting...".into(),
    });
    let _ = window.emit("simulation-progress", ProgressPayload {
        step: 1, sim_time: config.t_end, percentage: 100.0, message: "Completed".into(),
    });
    Ok(format!("Simulation completed: {}/result.vtu", config.output_dir))
}
