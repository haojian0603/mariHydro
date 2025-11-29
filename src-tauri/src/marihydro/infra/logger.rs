// src-tauri/src/marihydro/infra/logger.rs

use chrono::Local;
use env_logger::{Builder, Target};
use log::{LevelFilter, Record};
use serde::Serialize;
use std::io::Write;

/// 前端日志条目（用于 Tauri 事件发送）
#[derive(Debug, Clone, Serialize)]
pub struct FrontendLogEntry {
    pub level: String,
    pub message: String,
    pub target: String,
    pub file: String,
    pub line: u32,
    pub timestamp: String,
}

impl FrontendLogEntry {
    /// 从 log::Record 构建结构化日志
    pub fn from_record(record: &Record) -> Self {
        Self {
            level: record.level().to_string(),
            message: format!("{}", record.args()),
            target: record.target().to_string(),
            file: record.file().unwrap_or("unknown").to_string(),
            line: record.line().unwrap_or(0),
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
        }
    }

    /// 简易构造器
    pub fn new_simple(level: &str, msg: &str) -> Self {
        Self {
            level: level.to_uppercase(),
            message: msg.to_string(),
            target: "System".to_string(),
            file: "".to_string(),
            line: 0,
            timestamp: Local::now().format("%H:%M:%S").to_string(),
        }
    }
}

/// 初始化日志系统
///
/// # 参数
/// - `level`: 日志级别字符串（如 "info", "debug", "trace"），None 则使用环境变量
///
/// # 示例
///
/// ```rust
/// use marihydro_lib::marihydro::infra::logger::init_logging;
///
/// init_logging(Some("debug"));
/// ```
pub fn init_logging(level: Option<&str>) {
    let log_level = level
        .and_then(|l| l.parse::<LevelFilter>().ok())
        .or_else(|| {
            std::env::var("RUST_LOG")
                .ok()
                .and_then(|v| v.parse::<LevelFilter>().ok())
        })
        .unwrap_or(LevelFilter::Info);

    Builder::new()
        .filter_level(log_level)
        .target(Target::Stdout)
        .format(|buf, record| {
            writeln!(
                buf,
                "[{} {:5} {}:{}] {}",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                record.level(),
                record
                    .file()
                    .unwrap_or("unknown")
                    .rsplit('/')
                    .next()
                    .unwrap_or("unknown"),
                record.line().unwrap_or(0),
                record.args()
            )
        })
        .init();

    log::info!(
        "日志系统初始化完成 (Level: {})",
        log_level.to_string().to_uppercase()
    );
}

/// 带上下文的日志宏
#[macro_export]
macro_rules! log_with_context {
    (info, $context:expr, $($arg:tt)*) => {
        log::info!("[{}] {}", $context, format!($($arg)*))
    };
    (warn, $context:expr, $($arg:tt)*) => {
        log::warn!("[{}] {}", $context, format!($($arg)*))
    };
    (error, $context:expr, $($arg:tt)*) => {
        log::error!("[{}] {}", $context, format!($($arg)*))
    };
    (debug, $context:expr, $($arg:tt)*) => {
        log::debug!("[{}] {}", $context, format!($($arg)*))
    };
    (trace, $context:expr, $($arg:tt)*) => {
        log::trace!("[{}] {}", $context, format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontend_log_entry() {
        let entry = FrontendLogEntry::new_simple("info", "测试消息");
        assert_eq!(entry.level, "INFO");
        assert_eq!(entry.message, "测试消息");
    }
}
