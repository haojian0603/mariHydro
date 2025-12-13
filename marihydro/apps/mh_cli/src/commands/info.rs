// marihydro\apps\mh_cli\src\commands\info.rs

//! 信息显示命令
//!
//! 显示系统、配置和网格信息。

use anyhow::Result;
use clap::Args;
use mh_physics::builder::SolverConfig;
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
        // 默认显示所有信息
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
    
    // 检查可用精度
    println!("\n可用精度:");
    println!("  - f32 (单精度): ✓");
    println!("  - f64 (双精度): ✓");
    
    // 检查 CPU 特性
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
    println!("CFL 数: {}", config.cfl);
    println!("重力加速度: {} m/s²", config.gravity);
    println!("干单元阈值: {} m", config.h_dry);
    println!("最小水深: {} m", config.h_min);
    
    println!("\n容差设置 (f64):");
    let tol = mh_core::Tolerance::<f64>::default();
    println!("  h_min: {}", tol.h_min);
    println!("  h_dry: {}", tol.h_dry);
    println!("  velocity_cap: {}", tol.velocity_cap);
    
    println!("\n容差设置 (f32):");
    let tol32 = mh_core::Tolerance::<f32>::default();
    println!("  h_min: {}", tol32.h_min);
    println!("  h_dry: {}", tol32.h_dry);
    println!("  velocity_cap: {}", tol32.velocity_cap);
}

fn rustc_version() -> &'static str {
    // 在编译时获取 rustc 版本较复杂，这里简化处理
    "stable (编译时确定)"
}
