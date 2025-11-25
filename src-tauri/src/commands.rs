// src-tauri/src/commands.rs

use crate::simulation::config::{Config, SlopeLimiterType};
use crate::simulation::grid::Grid;
use crate::simulation::solver::Solver;
use tauri::{Emitter, State, Window};
// use crate::db::DbPool; // 如果启用数据库，请取消注释
use std::time::Instant;
use tokio::task; // 用于将繁重的计算任务移出异步主线程

// 定义发回给前端的进度事件数据结构
#[derive(Clone, serde::Serialize)]
struct ProgressPayload {
    step: u32,
    sim_time: f64,
    percentage: f64,
    message: String,
}

/// API: 获取默认配置模板
/// 前端调用: invoke('get_default_config')
#[tauri::command]
pub fn get_default_config() -> Config {
    Config::default_template()
}

/// API: 运行模拟
/// 前端调用: invoke('run_simulation', { config: ... })
/// 这是一个异步命令，Tauri 会自动在后台处理，不会冻结 UI
#[tauri::command]
pub async fn run_simulation(window: Window, config: Config) -> Result<String, String> {
    println!(
        "(Backend) 收到模拟请求，配置概要: Grid {}x{}",
        config.nx, config.ny
    );

    // 使用 spawn_blocking 将 CPU 密集型任务移入专用线程
    // 否则长时间的 while 循环会阻塞 Tokio 的异步运行时，导致事件发送延迟
    let result = task::spawn_blocking(move || {
        // --- 1. 初始化阶段 ---
        // Grid::new_from_config 会处理文件加载 (IO)
        let mut grid = Grid::new_from_config(&config);

        // (临时) 初始化水深地形 - 未来这将由 load_terrain 替代
        grid.init_dam_break(10.0, 10.0);

        let mut solver = Solver::new(
            config.nx,
            config.ny,
            config.ng,
            config.cfl_number,
            SlopeLimiterType::VanLeer,
        );

        // 模拟参数 (应从 Config 中获取，这里为了演示暂时硬编码)
        let t_end = 3600.0;
        let mut current_time = 0.0;
        let mut step = 0;
        let start_time = Instant::now();

        // --- 2. 时间循环阶段 ---
        println!("(Backend) 开始计算循环...");
        while current_time < t_end {
            // 2a. 计算动态时间步长
            let (dt_calc, _, _) = solver.compute_dt(&grid);
            let dt = dt_calc.min(t_end - current_time).max(1e-6);

            // 2b. 执行核心物理步进 (Strang Split)
            // 参数 true/false 控制是否同化，这里暂时关闭
            solver.step_strang_split(&mut grid, dt, false);

            current_time += dt;
            step += 1;

            // 2c. 发送进度事件 (每 20 步一次，避免通信过于频繁)
            if step % 20 == 0 {
                let payload = ProgressPayload {
                    step,
                    sim_time: current_time,
                    percentage: (current_time / t_end) * 100.0,
                    message: format!("正在计算 Step {} (T={:.1}s)", step, current_time),
                };
                // 注意：在 spawn_blocking 中，我们仍然持有 window 的克隆
                // emit 是线程安全的
                if let Err(e) = window.emit("sim-progress", payload) {
                    eprintln!("发送进度事件失败: {}", e);
                }
            }
        }

        // --- 3. 结束阶段 ---
        let duration = start_time.elapsed().as_secs_f64();
        format!("模拟成功完成！总耗时 {:.2}s", duration)
    })
    .await; // 等待阻塞任务结束

    // 处理线程可能的 Panic 或结果
    match result {
        Ok(msg) => Ok(msg),
        Err(e) => Err(format!("模拟线程崩溃: {:?}", e)),
    }
}

/// API: 保存实验记录 (预留)
#[tauri::command]
pub async fn save_simulation_record(
    // pool: State<'_, DbPool>, // 需要在 main.rs 中 manage(pool)
    name: String,
    description: String,
) -> Result<String, String> {
    // 示例逻辑
    println!("保存记录: {} - {}", name, description);
    // sqlx::query!...
    Ok("记录已保存".to_string())
}
