// marihydro\apps\mh_cli\src/main.rs

//! MariHydro 命令行界面
//!
//! 提供浅水方程模拟的命令行工具。
//!
//! # 架构层级
//!
//! 本模块属于 **Layer 5: Application**，遵循以下原则：
//! - 零泛型语法：仅使用 `SolverConfig` 和 `Box<dyn DynSolver>`
//! - 通过 `Precision` 枚举选择精度，无需指定类型参数

mod commands;

use clap::{Parser, Subcommand};
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// MariHydro 浅水方程求解器命令行工具
#[derive(Parser)]
#[command(name = "mh_cli")]
#[command(author = "MariHydro Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "MariHydro shallow water equation solver", long_about = None)]
struct Cli {
    /// 日志级别 (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 运行模拟
    Run(commands::run::RunArgs),
    /// 显示信息
    Info(commands::info::InfoArgs),
    /// 验证配置
    Validate(commands::validate::ValidateArgs),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // 初始化日志
    let level = match cli.log_level.to_lowercase().as_str() {
        "trace" => Level::TRACE,
        "debug" => Level::DEBUG,
        "info" => Level::INFO,
        "warn" => Level::WARN,
        "error" => Level::ERROR,
        _ => Level::INFO,
    };

    let subscriber = FmtSubscriber::builder()
        .with_max_level(level)
        .with_target(false)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // 执行命令
    match cli.command {
        Commands::Run(args) => commands::run::execute(args),
        Commands::Info(args) => commands::info::execute(args),
        Commands::Validate(args) => commands::validate::execute(args),
    }
}

