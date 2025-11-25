//! 网格管理模块
//! 变更：初始化逻辑现在会检查 Config 中的文件路径，决定是使用均匀值还是加载文件

use crate::simulation::config::Config;
use crate::simulation::io; // 引入增强的 IO 模块
use ndarray::Array2;

pub struct Grid {
    // ... 字段保持不变 ...
    pub nx: usize,
    pub ny: usize,
    pub ng: usize,
    pub h: Array2<f64>,
    pub n: Array2<f64>, // 曼宁系数
    pub wind_u: Array2<f64>,
    pub wind_v: Array2<f64>,
    // ...
}

impl Grid {
    pub fn new_from_config(config: &Config) -> Self {
        let (nx, ny, ng) = (config.nx, config.ny, config.ng);
        let shape = (nx + 2 * ng, ny + 2 * ng);

        // 1. 基础分配
        let mut grid = Self {
            nx,
            ny,
            ng,
            h: Array2::zeros(shape),
            n: Array2::zeros(shape), // 先全零
            wind_u: Array2::zeros(shape),
            wind_v: Array2::zeros(shape),
            // ... 其他字段初始化 ...
        };

        // 2. 初始化曼宁糙率 (均匀 vs 非均匀)
        if let Some(path) = &config.manning_file_path {
            match io::load_manning_field(path, &grid) {
                Ok(field) => grid.n = field,
                Err(e) => println!("(错误) 加载曼宁文件失败，回退到默认值: {}", e),
            }
        } else {
            grid.n.fill(config.uniform_manning_n);
        }

        // 3. 初始化风场 (均匀 vs 非均匀)
        if let Some(path) = &config.wind_file_path {
            // 调用 io::load_wind_field ...
        } else {
            // 计算均匀风分量
            let rad = config.uniform_wind_direction.to_radians();
            let u = -config.uniform_wind_speed * rad.sin();
            let v = -config.uniform_wind_speed * rad.cos();
            grid.wind_u.fill(u);
            grid.wind_v.fill(v);
        }

        grid
    }

    // ... 原有的 set_state, get_state 等方法 ...
}
