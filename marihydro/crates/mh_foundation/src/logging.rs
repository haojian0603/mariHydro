// crates/mh_foundation/src/logging.rs

//! 结构化日志系统
//!
//! 基于 tracing 的统一日志系统，支持：
//! - 环境变量控制的日志级别
//! - 结构化日志字段
//! - 性能指标集成
//! - 多输出目标（控制台、文件、JSON）
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_foundation::logging::{init_logging, LogConfig};
//!
//! let config = LogConfig::default();
//! let _guard = init_logging(&config)?;
//!
//! tracing::info!(time_step = 0.001, "开始模拟");
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::io::Write;
use serde_json;

/// 日志配置
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// 日志级别（环境变量 RUST_LOG 覆盖）
    pub level: LogLevel,
    /// 输出目标
    pub output: LogOutput,
    /// 是否包含时间戳
    pub timestamps: bool,
    /// 是否包含目标（模块路径）
    pub targets: bool,
    /// 是否包含文件位置
    pub file_locations: bool,
    /// 是否使用 ANSI 颜色
    pub ansi_colors: bool,
    /// 是否使用紧凑格式
    pub compact: bool,
    /// JSON 格式输出
    pub json: bool,
    /// 日志文件路径（如果输出到文件）
    pub log_file: Option<PathBuf>,
    /// 模块级别覆盖
    pub module_levels: Vec<(String, LogLevel)>,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            output: LogOutput::Stderr,
            timestamps: true,
            targets: true,
            file_locations: false,
            ansi_colors: true,
            compact: false,
            json: false,
            log_file: None,
            module_levels: Vec::new(),
        }
    }
}

impl LogConfig {
    /// 创建开发环境配置
    pub fn development() -> Self {
        Self {
            level: LogLevel::Debug,
            file_locations: true,
            compact: false,
            ..Default::default()
        }
    }

    /// 创建生产环境配置
    pub fn production() -> Self {
        Self {
            level: LogLevel::Info,
            ansi_colors: false,
            timestamps: true,
            compact: true,
            ..Default::default()
        }
    }

    /// 创建性能分析配置
    pub fn profiling() -> Self {
        Self {
            level: LogLevel::Trace,
            timestamps: true,
            compact: true,
            module_levels: vec![
                ("mh_physics".to_string(), LogLevel::Trace),
                ("mh_runtime".to_string(), LogLevel::Debug),
            ],
            ..Default::default()
        }
    }

    /// 添加模块级别覆盖
    pub fn with_module_level(mut self, module: impl Into<String>, level: LogLevel) -> Self {
        self.module_levels.push((module.into(), level));
        self
    }

    /// 设置日志文件
    pub fn with_log_file(mut self, path: impl Into<PathBuf>) -> Self {
        self.log_file = Some(path.into());
        self.output = LogOutput::File;
        self
    }

    /// 启用 JSON 格式
    pub fn with_json(mut self) -> Self {
        self.json = true;
        self
    }
}

/// 日志级别
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// 跟踪级别（最详细）
    Trace,
    /// 调试级别
    Debug,
    /// 信息级别
    Info,
    /// 警告级别
    Warn,
    /// 错误级别
    Error,
    /// 关闭日志
    Off,
}

impl LogLevel {
    /// 转换为 tracing 级别字符串
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Off => "off",
        }
    }
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// 日志输出目标
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogOutput {
    /// 标准输出
    Stdout,
    /// 标准错误
    Stderr,
    /// 文件
    File,
    /// 标准错误 + 文件
    Both,
}

/// 构建 EnvFilter 字符串
fn build_filter_string(config: &LogConfig) -> String {
    let mut parts = vec![config.level.as_str().to_string()];
    
    for (module, level) in &config.module_levels {
        parts.push(format!("{}={}", module, level));
    }
    
    parts.join(",")
}

/// 日志守卫（RAII）
///
/// 当守卫离开作用域时，日志系统会被正确关闭
pub struct LogGuard {
    _guard: Option<LogGuardInner>,
}

enum LogGuardInner {
    /// 文件写入守卫
    FileWriter(std::sync::Arc<std::sync::Mutex<std::fs::File>>),
    /// 无特殊守卫
    None,
}

static LOG_CONFIG: OnceLock<LogConfig> = OnceLock::new();
static LOG_FILE: OnceLock<Arc<Mutex<std::fs::File>>> = OnceLock::new();

fn store_log_config(config: &LogConfig) {
    let _ = LOG_CONFIG.set(config.clone());
}

fn current_log_config() -> Option<&'static LogConfig> {
    LOG_CONFIG.get()
}

fn store_log_file(file: std::fs::File) -> Arc<Mutex<std::fs::File>> {
    let handle = Arc::new(Mutex::new(file));
    let _ = LOG_FILE.set(handle.clone());
    handle
}

fn current_log_file() -> Option<&'static Arc<Mutex<std::fs::File>>> {
    LOG_FILE.get()
}

pub(crate) fn level_from_ident(level: &str) -> LogLevel {
    match level {
        "trace" => LogLevel::Trace,
        "debug" => LogLevel::Debug,
        "info" => LogLevel::Info,
        "warn" => LogLevel::Warn,
        "error" => LogLevel::Error,
        _ => LogLevel::Info,
    }
}

pub(crate) fn emit_log(level: LogLevel, fields: Vec<(String, String)>, msg: &str) {
    let config = current_log_config().cloned().unwrap_or_default();
    if level < config.level || config.level == LogLevel::Off {
        return;
    }

    let line = if config.json {
        let mut json = String::new();
        json.push_str("{\"level\":\"");
        json.push_str(level.as_str());
        json.push_str("\",\"message\":");
        json.push_str(&serde_json::to_string(msg).unwrap_or_else(|_| "\"\"".into()));
        json.push_str(",\"fields\":{");
        for (i, (k, v)) in fields.iter().enumerate() {
            if i > 0 {
                json.push(',');
            }
            json.push_str(&serde_json::to_string(k).unwrap_or_else(|_| "\"\"".into()));
            json.push(':');
            json.push_str(&serde_json::to_string(v).unwrap_or_else(|_| "\"\"".into()));
        }
        json.push_str("}}\n");
        json
    } else {
        let mut s = String::new();
        s.push('[');
        s.push_str(level.as_str());
        s.push_str("] ");
        s.push_str(msg);
        if !fields.is_empty() {
            s.push(' ');
            s.push_str(&format!("{:?}", fields));
        }
        s.push('\n');
        s
    };

    match config.output {
        LogOutput::Stdout => {
            let _ = std::io::stdout().write_all(line.as_bytes());
        }
        LogOutput::Stderr => {
            let _ = std::io::stderr().write_all(line.as_bytes());
        }
        LogOutput::File => {
            if let Some(file) = current_log_file() {
                if let Ok(mut f) = file.lock() {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
        LogOutput::Both => {
            let _ = std::io::stderr().write_all(line.as_bytes());
            if let Some(file) = current_log_file() {
                if let Ok(mut f) = file.lock() {
                    let _ = f.write_all(line.as_bytes());
                }
            }
        }
    }
}

impl Drop for LogGuard {
    fn drop(&mut self) {
        // 确保所有日志被刷新
        if let Some(LogGuardInner::FileWriter(file)) = &self._guard {
            if let Ok(mut f) = file.lock() {
                let _ = std::io::Write::flush(&mut *f);
            }
        }
    }
}

/// 初始化日志系统
///
/// # 参数
///
/// * `config` - 日志配置
///
/// # 返回
///
/// 返回日志守卫，必须保持到程序结束
///
/// # 示例
///
/// ```ignore
/// let _guard = init_logging(&LogConfig::default())?;
/// ```
pub fn init_logging(config: &LogConfig) -> Result<LogGuard, String> {
    let _filter = std::env::var("RUST_LOG").unwrap_or_else(|_| build_filter_string(config));
    store_log_config(config);

    let guard = match config.output {
        LogOutput::File | LogOutput::Both => {
            let path = config
                .log_file
                .as_ref()
                .ok_or_else(|| "日志输出为文件但未设置 log_file".to_string())?;
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|e| format!("无法打开日志文件: {}", e))?;
            let handle = store_log_file(file);
            LogGuard {
                _guard: Some(LogGuardInner::FileWriter(handle)),
            }
        }
        _ => LogGuard {
            _guard: Some(LogGuardInner::None),
        },
    };

    Ok(guard)
}

/// 日志宏（结构化日志）
///
/// 用于记录带有结构化字段的日志
#[macro_export]
macro_rules! log_event {
    ($level:ident, $($field:ident = $value:expr),* ; $msg:expr) => {
        {
            let _fields = vec![
                $(
                    (stringify!($field).to_string(), format!("{:?}", $value)),
                )*
            ];
            let _level = $crate::logging::level_from_ident(stringify!($level));
            $crate::logging::emit_log(_level, _fields, $msg);
        }
    };
}

/// 性能追踪 span 宏
#[macro_export]
macro_rules! perf_span {
    ($name:expr) => {
        $crate::logging::PerfSpan::new($name)
    };
}

/// 性能追踪 span
pub struct PerfSpan {
    name: &'static str,
    start: std::time::Instant,
}

impl PerfSpan {
    /// 创建新的性能 span
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }

    /// 获取经过的时间
    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }
}

impl Drop for PerfSpan {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if elapsed.as_micros() > 100 {
            // 只记录 > 100μs 的操作
            eprintln!(
                "[PERF] {} completed in {:.3}ms",
                self.name,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert!(config.timestamps);
    }

    #[test]
    fn test_log_config_development() {
        let config = LogConfig::development();
        assert_eq!(config.level, LogLevel::Debug);
        assert!(config.file_locations);
    }

    #[test]
    fn test_log_config_production() {
        let config = LogConfig::production();
        assert_eq!(config.level, LogLevel::Info);
        assert!(!config.ansi_colors);
    }

    #[test]
    fn test_filter_string() {
        let config = LogConfig::default()
            .with_module_level("mh_physics", LogLevel::Debug);
        
        let filter = build_filter_string(&config);
        assert!(filter.contains("info"));
        assert!(filter.contains("mh_physics=debug"));
    }

    #[test]
    fn test_perf_span() {
        let span = PerfSpan::new("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(span.elapsed().as_micros() > 0);
    }

    #[test]
    fn test_init_logging() {
        let config = LogConfig::default();
        let result = init_logging(&config);
        assert!(result.is_ok());
    }
}
