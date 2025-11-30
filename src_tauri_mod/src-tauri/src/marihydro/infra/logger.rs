// src-tauri/src/marihydro/infra/logger.rs
use chrono::Local;
use log::LevelFilter;
use serde::Serialize;
use std::io::Write;

#[derive(Debug, Clone, Serialize)]
pub struct FrontendLogEntry {
    pub level: String,
    pub message: String,
    pub target: String,
    pub timestamp: String,
}

impl FrontendLogEntry {
    pub fn new(level: &str, message: &str, target: &str) -> Self {
        Self { level: level.to_uppercase(), message: message.into(), target: target.into(), timestamp: Local::now().format("%H:%M:%S").to_string() }
    }
    pub fn info(msg: &str) -> Self { Self::new("INFO", msg, "System") }
    pub fn warn(msg: &str) -> Self { Self::new("WARN", msg, "System") }
    pub fn error(msg: &str) -> Self { Self::new("ERROR", msg, "System") }
}

pub fn init_logging(level: Option<&str>) {
    let log_level = level.and_then(|l| l.parse::<LevelFilter>().ok())
        .or_else(|| std::env::var("RUST_LOG").ok().and_then(|v| v.parse().ok()))
        .unwrap_or(LevelFilter::Info);
    env_logger::Builder::new()
        .filter_level(log_level)
        .format(|buf, record| {
            writeln!(buf, "[{} {:5}] {}", Local::now().format("%H:%M:%S"), record.level(), record.args())
        })
        .init();
    log::info!("Logger initialized (level: {})", log_level);
}
