// marihydro\apps\mh_cli\src\commands\info.rs

//! 信息显示命令
//!
//! 显示系统、真实配置默认值和精度信息。

use anyhow::Result;
use clap::Args;
use mh_config::SolverConfig;
use std::path::PathBuf;
use tracing::info;

/// 信息显示参数
#[derive(Args)]
pub struct InfoArgs {
    /// 配置文件路径
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// 显示系统信息
    #[arg(long)]
    pub system: bool,

    /// 显示默认配置
    #[arg(long)]
    pub defaults: bool,
}

/// 执行信息命令
pub fn execute(args: InfoArgs) -> Result<()> {
    info!("=== MariHydro 信息 ===");

    if args.system {
        print_system_info();
    }

    if args.defaults {
        print_default_config();
    }

    if args.config.is_none() && !args.system && !args.defaults {
        print_system_info();
        println!();
        print_default_config();
    }

    Ok(())
}

fn print_system_info() {
    println!("=== 系统信息 ===");
    println!("MariHydro CLI 版本: {}", env!("CARGO_PKG_VERSION"));
    println!("Rust 版本: {}", rustc_version());
    println!("目标平台: {}", std::env::consts::ARCH);
    println!("操作系统: {}", std::env::consts::OS);

    println!("\n可用精度:");
    println!("  - f32 (单精度): ✓");
    println!("  - f64 (双精度): ✓");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("\nCPU 特性: AVX2 可用");
        }
        if is_x86_feature_detected!("fma") {
            println!("CPU 特性: FMA 可用");
        }
    }
}

fn print_default_config() {
    println!("=== 默认配置 ===");

    let config = SolverConfig::default();

    println!("精度: {:?}", config.precision);
    println!("重力加速度: {} m/s²", config.physics.gravity);
    println!("CFL 数: {}", config.physics.cfl);
    println!("干单元阈值: {} m", config.physics.h_dry);
    println!("最小水深: {} m", config.physics.h_min);
    println!("黎曼求解器: {:?}", config.numerical.riemann_solver);
    println!("时间积分: {:?}", config.numerical.time_integration);
    println!("输出间隔: {} s", config.output.interval);
    println!("最大模拟时间: {} s", config.max_time);

    println!("\n容差设置 (f64):");
    let tol = mh_runtime::Tolerance::<f64>::default();
    println!("  h_min: {}", tol.h_min);
    println!("  h_dry: {}", tol.h_dry);
    println!("  velocity_cap: {}", tol.velocity_cap);

    println!("\n容差设置 (f32):");
    let tol32 = mh_runtime::Tolerance::<f32>::default();
    println!("  h_min: {}", tol32.h_min);
    println!("  h_dry: {}", tol32.h_dry);
    println!("  velocity_cap: {}", tol32.velocity_cap);
}

fn rustc_version() -> &'static str {
    "stable (编译时确定)"
}
