// marihydro\apps\mh_cli\src\commands\run.rs

//! 运行模拟命令
//!
//! 执行浅水方程模拟。
//!
//! # 架构说明
//!
//! 本模块属于 Layer 5: Application，遵循零泛型原则：
//! - 使用 `SolverConfig` 配置求解器
//! - 通过 `SolverBuilder` 构建 `Box<dyn DynSolver>`
//! - 精度通过 `Precision` 枚举选择，无需泛型参数

use anyhow::{Context, Result};
use clap::Args;
use mh_config::Precision;
use mh_physics::builder::{SolverBuilder, SolverConfig};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

/// 运行模拟参数
#[derive(Args)]
pub struct RunArgs {
    /// 配置文件路径
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// 网格文件路径
    #[arg(short, long)]
    pub mesh: Option<PathBuf>,

    /// 输出目录
    #[arg(short, long, default_value = "output")]
    pub output: PathBuf,

    /// 模拟结束时间 [秒]
    #[arg(short = 't', long, default_value = "100.0")]
    pub end_time: f64,

    /// 时间步长 [秒]
    #[arg(long, default_value = "0.01")]
    pub dt: f64,

    /// 输出间隔 [秒]
    #[arg(long, default_value = "1.0")]
    pub output_interval: f64,

    /// 使用 f32 精度
    #[arg(long)]
    pub f32: bool,

    /// 网格单元数量（简化模式）
    #[arg(long, default_value = "1000")]
    pub cells: usize,

    /// 初始水深 [m]
    #[arg(long, default_value = "1.0")]
    pub initial_depth: f64,

    /// 溃坝模拟模式
    #[arg(long)]
    pub dambreak: bool,
}

/// 执行运行命令
pub fn execute(args: RunArgs) -> Result<()> {
    info!("=== MariHydro 模拟启动 ===");

    // 确定精度（Layer 5 唯一接触精度系统的地方）
    let precision = if args.f32 {
        Precision::F32
    } else {
        Precision::F64
    };
    info!("使用精度: {:?}", precision);

    // 构建配置（无泛型语法）
    let mut config = SolverConfig::default();
    config.precision = precision;
    config.cfl = 0.5;
    config.gravity = 9.81;
    config.h_dry = 1e-4;
    config.h_min = 1e-6;

    info!("配置: CFL={}, 重力={} m/s²", config.cfl, config.gravity);

    // 构建求解器（完全运行时，零泛型语法）
    let n_cells = args.cells;
    
    let builder = if args.dambreak {
        // 溃坝初始条件：左半边水深高，右半边水深低
        let mut h = vec![0.1; n_cells];
        for i in 0..(n_cells / 2) {
            h[i] = args.initial_depth;
        }
        
        info!("溃坝模拟: {} 单元, 上游水深={} m", n_cells, args.initial_depth);
        
        SolverBuilder::new(config)
            .with_cells(n_cells)
            .with_initial_depth(h)
            .with_initial_velocity(vec![0.0; n_cells], vec![0.0; n_cells])
            .with_bathymetry(vec![0.0; n_cells])
    } else {
        info!("静水模拟: {} 单元, 水深={} m", n_cells, args.initial_depth);
        
        SolverBuilder::new(config)
            .with_still_water(args.initial_depth, n_cells)
    };

    let mut solver = builder.build()
        .context("构建求解器失败")?;

    info!("求解器: {}, 精度: {:?}", solver.name(), solver.precision());
    info!("网格: {} 单元, {} 面", solver.n_cells(), solver.n_faces());

    // 创建输出目录
    std::fs::create_dir_all(&args.output)?;

    // 运行模拟循环（完全运行时）
    let start = Instant::now();
    let mut last_output_time = 0.0;
    let mut output_count = 0;

    info!("开始模拟: 结束时间={} s, 时间步长={} s", args.end_time, args.dt);

    while solver.time() < args.end_time {
        // 执行时间步进
        let result = solver.step(args.dt);

        if !result.success {
            warn!("时间步失败: {:?}", result.error_message);
            break;
        }

        // 检查输出时机
        if solver.time() - last_output_time >= args.output_interval {
            output_count += 1;
            let state = solver.export_state();
            
            // 计算统计
            let h_max: f64 = state.h.iter().cloned().fold(0.0_f64, f64::max);
            let h_min: f64 = state.h.iter().cloned().filter(|&h| h > 1e-6).fold(f64::MAX, f64::min);
            let v_max: f64 = state.velocity_magnitude().into_iter().fold(0.0_f64, f64::max);

            info!(
                "t={:.2} s: h_max={:.4} m, h_min={:.4} m, v_max={:.4} m/s, CFL={:.3}",
                solver.time(), h_max, h_min, v_max, result.max_cfl
            );

            last_output_time = solver.time();
        }
    }

    let elapsed = start.elapsed();
    let stats = solver.stats();

    info!("=== 模拟完成 ===");
    info!("总步数: {}", stats.total_steps);
    info!("计算时间: {:.2} s", elapsed.as_secs_f64());
    info!("平均步耗时: {:.3} ms", stats.avg_step_time * 1000.0);
    info!("输出文件数: {}", output_count);

    Ok(())
}
