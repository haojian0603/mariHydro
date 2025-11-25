use chrono::Local;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct FrontendLogEntry {
    pub level: String,     // INFO, ERROR
    pub message: String,   // 具体消息
    pub target: String,    // 模块路径 marihydro::infra::...
    pub file: String,      // 源代码文件
    pub line: u32,         // 代码行号
    pub timestamp: String, // 本地时间字符串
}

impl FrontendLogEntry {
    /// 从 log::Record 构建结构化日志
    pub fn from_record(record: &log::Record) -> Self {
        Self {
            level: record.level().to_string(),
            message: format!("{}", record.args()),
            target: record.target().to_string(),
            file: record.file().unwrap_or("unknown").to_string(),
            line: record.line().unwrap_or(0),
            timestamp: Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
        }
    }

    /// 简易构造器 (用于非 log crate 场景)
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
