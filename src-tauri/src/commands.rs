// src-tauri/src/commands.rs

use tauri::{Emitter, Window};
use tokio::task;

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::domain::state::ConservedState;
use crate::marihydro::infra::error::MhResult;
use crate::marihydro::io::exporters::vtu::VtuExporter;
use crate::marihydro::io::loaders::gmsh::GmshLoader;
use crate::marihydro::physics::engine::UnstructuredSolver;

#[derive(Clone, serde::Serialize)]
struct ProgressPayload {
    step: u32,
    sim_time: f64,
    percentage: f64,
    message: String,
}

#[derive(Clone, serde::Deserialize)]
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
            mesh_path: "assets/mesh/domain.msh".into(),
            output_dir: "output".into(),
            t_end: 3600.0,
            initial_water_level: 10.0,
            cfl_factor: 0.5,
            gravity: 9.81,
            h_min: 0.05,
            manning_coef: 0.025,
            enable_muscl: true,
        }
    }
}

#[tauri::command]
pub fn get_default_config() -> SimulationConfig {
    SimulationConfig::default()
}

#[tauri::command]
pub async fn run_simulation(window: Window, config: SimulationConfig) -> Result<String, String> {
    log::info!("收到模拟请求: mesh={}", config.mesh_path);

    let result = task::spawn_blocking(move || -> MhResult<String> {
        std::fs::create_dir_all(&config.output_dir)
            .map_err(|e| crate::marihydro::infra::error::MhError::io("创建输出目录", e))?;

        log::info!("正在加载网格: {}", config.mesh_path);
        let loader = GmshLoader::new();
        let mesh = loader.load(std::path::Path::new(&config.mesh_path))?;

        log::info!(
            "网格加载成功: {} 个单元, {} 个面, {} 个节点",
            mesh.n_cells,
            mesh.n_faces,
            mesh.n_nodes
        );

        // 验证拓扑
        mesh.validate_topology()?;
        log::info!("网格拓扑验证通过");

        // 初始化状态
        log::info!("初始化状态...");
        let state = ConservedState::cold_start(
            mesh.n_cells,
            config.initial_water_level,
            &mesh.cell_center_z, // ✅ 使用单元中心高程
        )?;

        let mut next_state = state.clone_structure();
        next_state.h.copy_from_slice(&state.h);
        next_state.hu.copy_from_slice(&state.hu);
        next_state.hv.copy_from_slice(&state.hv);

        // 初始化求解器
        log::info!("初始化求解器...");
        let mut solver = UnstructuredSolver::new(mesh, config.gravity, config.h_min);
        solver.manning_coef = config.manning_coef;
        if config.enable_muscl {
            solver.enable_muscl();
        }

        log::info!("开始时间积分 (t_end={:.1}s)...", config.t_end);
        let mut current_time = 0.0;
        let mut step = 0u32;
        let start = std::time::Instant::now();

        let mut current_state = state;
        let output_interval = 100;
        let mut output_count = 0;

        while current_time < config.t_end {
            let dt_cfl = solver.compute_cfl_dt(&current_state, config.cfl_factor);
            let dt = dt_cfl.min(config.t_end - current_time);

            solver.step(&current_state, &mut next_state, dt)?;

            std::mem::swap(&mut current_state, &mut next_state);

            current_time += dt;
            step += 1;

            // 发送进度
            if step % 20 == 0 {
                let payload = ProgressPayload {
                    step,
                    sim_time: current_time,
                    percentage: (current_time / config.t_end) * 100.0,
                    message: format!("Step {}: t={:.2}s, dt={:.4}s", step, current_time, dt),
                };

                if let Err(e) = window.emit("sim-progress", payload) {
                    log::warn!("发送进度失败: {}", e);
                }
            }

            // 导出结果
            if step % output_interval == 0 {
                let vtu_path = format!("{}/result_{:06}.vtu", config.output_dir, output_count);
                VtuExporter::export(&vtu_path, &solver.mesh, &current_state, current_time)?;
                log::debug!("导出 VTU: {}", vtu_path);
                output_count += 1;
            }
        }

        // 导出最终结果
        let final_path = format!("{}/result_final.vtu", config.output_dir);
        VtuExporter::export(&final_path, &solver.mesh, &current_state, current_time)?;

        let elapsed = start.elapsed().as_secs_f64();
        log::info!(
            "模拟完成: {} 步, {:.2}s 耗时, 平均 {:.2} 步/秒",
            step,
            elapsed,
            step as f64 / elapsed
        );

        Ok(format!(
            "模拟成功完成！\n步数: {}\n耗时: {:.2}s\n输出: {}",
            step, elapsed, config.output_dir
        ))
    })
    .await;

    match result {
        Ok(Ok(msg)) => Ok(msg),
        Ok(Err(e)) => Err(format!("模拟失败: {}", e)),
        Err(e) => Err(format!("模拟线程崩溃: {:?}", e)),
    }
}

#[tauri::command]
pub async fn save_simulation_record(name: String, description: String) -> Result<String, String> {
    log::info!("保存记录: {} - {}", name, description);
    // TODO: 在阶段2连接数据库
    Ok("记录已保存 (Mock)".to_string())
}
