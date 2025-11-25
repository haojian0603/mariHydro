//! 模拟配置模块
//! 变更：添加了 Serialize, Deserialize，使得前端 React 可以直接传 JSON 进来

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum SlopeLimiterType {
    FirstOrder,
    Minmod,
    VanLeer,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    // --- 基础网格 ---
    pub nx: usize,
    pub ny: usize,
    pub ng: usize,
    pub dx: f64,
    pub dy: f64,

    // --- 物理开关与参数 ---
    pub gravity: f64,
    pub cfl_number: f64,
    pub h_min: f64,

    // (新) 是否启用 AI 代理模型加速 (预留接口)
    pub use_ai_proxy: bool,

    // --- 外部强迫 (支持非均匀场的文件路径) ---
    // 如果是 None，则使用下面的 uniform 值；如果是 Some(path)，则读取文件
    pub wind_file_path: Option<String>,
    pub uniform_wind_speed: f64,
    pub uniform_wind_direction: f64,

    pub manning_file_path: Option<String>,
    pub uniform_manning_n: f64,

    // --- 边界条件 (预留) ---
    // pub open_boundary_file: Option<String>,

    // --- IO ---
    pub output_dir: String,
    // ... 其他参数保持不变，确保加上 Serialize/Deserialize
}

impl Config {
    // 提供给前端的一个默认模板
    pub fn default_template() -> Self {
        Self {
            nx: 200,
            ny: 200,
            ng: 2,
            dx: 500.0,
            dy: 500.0,
            gravity: 9.81,
            cfl_number: 0.4,
            h_min: 1e-6,
            use_ai_proxy: false,
            wind_file_path: None,
            uniform_wind_speed: 10.0,
            uniform_wind_direction: 270.0,
            manning_file_path: None,
            uniform_manning_n: 0.02,
            output_dir: "./output".to_string(),
        }
    }
}
