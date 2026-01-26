// crates/mh_foundation/src/error.rs

//! 统一错误类型（工业级错误处理）
//!
//! 提供 MariHydro 系统的核心错误类型，支持：
//! - 错误链与上下文
//! - 分类与恢复策略
//! - 结构化日志集成
//!
//! # 设计原则
//!
//! 1. **单一错误类型**：整个系统使用 `HydroError`
//! 2. **错误链保留**：通过 `source` 追踪原因
//! 3. **分类明确**：数值/配置/IO/并行/边界
//! 4. **可恢复性标记**：支持自动恢复策略

use std::error::Error;
use std::fmt;
use crate::error::MhError;

// ============================================================================
// 错误分类
// ============================================================================

/// 错误分类
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// 数值错误（发散、NaN、溢出）
    Numerical,
    /// 配置错误（无效参数、缺失字段）
    Configuration,
    /// IO 错误（文件读写、网络）
    Io,
    /// 并行错误（线程、同步）
    Parallel,
    /// 边界错误（边界条件、强迫数据）
    Boundary,
    /// 网格错误（拓扑、索引）
    Mesh,
    /// 内存错误（分配失败、越界）
    Memory,
    /// 后端错误（GPU、设备）
    Backend,
    /// 插值错误（数据不足、外推）
    Interpolation,
    /// 内部错误（断言失败、不变量违反）
    Internal,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numerical => write!(f, "数值"),
            Self::Configuration => write!(f, "配置"),
            Self::Io => write!(f, "IO"),
            Self::Parallel => write!(f, "并行"),
            Self::Boundary => write!(f, "边界"),
            Self::Mesh => write!(f, "网格"),
            Self::Memory => write!(f, "内存"),
            Self::Backend => write!(f, "后端"),
            Self::Interpolation => write!(f, "插值"),
            Self::Internal => write!(f, "内部"),
        }
    }
}

/// 恢复策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryStrategy {
    /// 不可恢复，必须终止
    Fatal,
    /// 可重试（减小时间步、降阶）
    Retry,
    /// 可跳过（警告后继续）
    Skip,
    /// 可回退（使用默认值）
    Fallback,
}

// ============================================================================
// 核心错误类型
// ============================================================================

/// MariHydro 统一错误类型
///
/// 所有子系统的错误都可以转换为此类型
#[derive(Debug)]
pub struct HydroError {
    /// 错误分类
    category: ErrorCategory,
    /// 错误消息
    message: String,
    /// 原因（错误链）
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
    /// 恢复策略
    recovery: RecoveryStrategy,
    /// 上下文信息
    context: Vec<(String, String)>,
}

impl HydroError {
    /// 创建新错误
    pub fn new(category: ErrorCategory, message: impl Into<String>) -> Self {
        Self {
            category,
            message: message.into(),
            source: None,
            recovery: RecoveryStrategy::Fatal,
            context: Vec::new(),
        }
    }

    /// 设置原因
    pub fn with_source(mut self, source: impl Error + Send + Sync + 'static) -> Self {
        self.source = Some(Box::new(source));
        self
    }

    /// 设置恢复策略
    pub fn with_recovery(mut self, recovery: RecoveryStrategy) -> Self {
        self.recovery = recovery;
        self
    }

    /// 添加上下文
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    /// 获取错误分类
    pub fn category(&self) -> ErrorCategory {
        self.category
    }

    /// 获取恢复策略
    pub fn recovery(&self) -> RecoveryStrategy {
        self.recovery
    }

    /// 是否可恢复
    pub fn is_recoverable(&self) -> bool {
        !matches!(self.recovery, RecoveryStrategy::Fatal)
    }

    /// 是否为致命错误
    pub fn is_fatal(&self) -> bool {
        matches!(self.recovery, RecoveryStrategy::Fatal)
    }

    /// 获取上下文
    pub fn context(&self) -> &[(String, String)] {
        &self.context
    }

    // === 便捷构造函数 ===

    /// 数值发散错误
    pub fn numerical_divergence(msg: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Numerical, msg)
            .with_recovery(RecoveryStrategy::Retry)
    }

    /// NaN 检测错误
    pub fn nan_detected(location: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Numerical, format!("检测到 NaN: {}", location.into()))
            .with_recovery(RecoveryStrategy::Retry)
    }

    /// CFL 违反错误
    pub fn cfl_violation(cfl: f64, max_cfl: f64) -> Self {
        Self::new(
            ErrorCategory::Numerical,
            format!("CFL 违反: {} > {}", cfl, max_cfl),
        )
        .with_recovery(RecoveryStrategy::Retry)
        .with_context("cfl", format!("{:.4}", cfl))
        .with_context("max_cfl", format!("{:.4}", max_cfl))
    }

    /// 配置无效参数
    pub fn invalid_parameter(name: &str, value: f64, reason: &str) -> Self {
        Self::new(
            ErrorCategory::Configuration,
            format!("无效参数 '{}': {} ({})", name, value, reason),
        )
        .with_context("parameter", name.to_string())
        .with_context("value", format!("{}", value))
    }

    /// 配置缺失字段
    pub fn missing_field(field: &str) -> Self {
        Self::new(
            ErrorCategory::Configuration,
            format!("缺失配置字段: {}", field),
        )
    }

    /// IO 文件未找到
    pub fn file_not_found(path: impl AsRef<std::path::Path>) -> Self {
        Self::new(
            ErrorCategory::Io,
            format!("文件未找到: {}", path.as_ref().display()),
        )
    }

    /// IO 读取失败
    pub fn read_failed(path: impl AsRef<std::path::Path>, source: std::io::Error) -> Self {
        Self::new(
            ErrorCategory::Io,
            format!("读取失败: {}", path.as_ref().display()),
        )
        .with_source(source)
    }

    /// 边界数据缺失
    pub fn missing_boundary_data(boundary: &str, time: f64) -> Self {
        Self::new(
            ErrorCategory::Boundary,
            format!("边界 '{}' 在时刻 {:.2}s 缺少数据", boundary, time),
        )
        .with_recovery(RecoveryStrategy::Fallback)
    }

    /// 网格索引越界
    pub fn index_out_of_bounds(index_type: &str, index: usize, max: usize) -> Self {
        Self::new(
            ErrorCategory::Mesh,
            format!("{} 索引 {} 越界 (max={})", index_type, index, max),
        )
    }

    /// 内存分配失败
    pub fn allocation_failed(size: usize) -> Self {
        Self::new(
            ErrorCategory::Memory,
            format!("内存分配失败: {} 字节", size),
        )
    }

    /// 后端不支持
    pub fn backend_unsupported(feature: &str) -> Self {
        Self::new(
            ErrorCategory::Backend,
            format!("后端不支持: {}", feature),
        )
    }

    /// 插值失败
    pub fn interpolation_failed(reason: &str) -> Self {
        Self::new(ErrorCategory::Interpolation, format!("插值失败: {}", reason))
            .with_recovery(RecoveryStrategy::Fallback)
    }

    /// 内部断言失败
    pub fn internal_assertion(msg: impl Into<String>) -> Self {
        Self::new(ErrorCategory::Internal, msg)
    }

    /// 未实现
    pub fn not_implemented(feature: &str) -> Self {
        Self::new(
            ErrorCategory::Internal,
            format!("未实现: {}", feature),
        )
    }
}

impl From<MhError> for HydroError {
    fn from(err: MhError) -> Self {
        HydroError::new(ErrorCategory::Internal, err.to_string())
    }
}

impl fmt::Display for HydroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.category, self.message)?;
        
        if !self.context.is_empty() {
            write!(f, " (")?;
            for (i, (k, v)) in self.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}={}", k, v)?;
            }
            write!(f, ")")?;
        }
        
        Ok(())
    }
}

impl Error for HydroError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn Error + 'static))
    }
}

// ============================================================================
// 结果类型
// ============================================================================

/// MariHydro 结果类型
pub type HydroResult<T> = Result<T, HydroError>;

// ============================================================================
// 错误转换
// ============================================================================

impl From<std::io::Error> for HydroError {
    fn from(err: std::io::Error) -> Self {
        Self::new(ErrorCategory::Io, err.to_string())
            .with_source(err)
    }
}

impl From<std::num::ParseFloatError> for HydroError {
    fn from(err: std::num::ParseFloatError) -> Self {
        Self::new(ErrorCategory::Configuration, format!("浮点数解析失败: {}", err))
    }
}

impl From<std::num::ParseIntError> for HydroError {
    fn from(err: std::num::ParseIntError) -> Self {
        Self::new(ErrorCategory::Configuration, format!("整数解析失败: {}", err))
    }
}

// ============================================================================
// 扩展 trait
// ============================================================================

/// 为 Result 添加上下文的扩展 trait
pub trait ResultExt<T> {
    /// 添加错误上下文
    fn context(self, key: &str, value: impl Into<String>) -> HydroResult<T>;
    
    /// 添加错误消息上下文
    fn with_message(self, msg: impl Into<String>) -> HydroResult<T>;
}

impl<T, E: Error + Send + Sync + 'static> ResultExt<T> for Result<T, E> {
    fn context(self, key: &str, value: impl Into<String>) -> HydroResult<T> {
        self.map_err(|e| {
            HydroError::new(ErrorCategory::Internal, e.to_string())
                .with_source(e)
                .with_context(key.to_string(), value)
        })
    }
    
    fn with_message(self, msg: impl Into<String>) -> HydroResult<T> {
        self.map_err(|e| {
            HydroError::new(ErrorCategory::Internal, msg)
                .with_source(e)
        })
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = HydroError::new(ErrorCategory::Numerical, "测试错误");
        assert_eq!(err.category(), ErrorCategory::Numerical);
        assert!(err.is_fatal());
    }

    #[test]
    fn test_error_with_context() {
        let err = HydroError::cfl_violation(1.5, 1.0);
        assert!(!err.context().is_empty());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_error_display() {
        let err = HydroError::invalid_parameter("dt", -0.001, "必须为正值");
        let msg = format!("{}", err);
        assert!(msg.contains("无效参数"));
        assert!(msg.contains("dt"));
    }

    #[test]
    fn test_error_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "文件不存在");
        let err = HydroError::new(ErrorCategory::Io, "读取配置失败")
            .with_source(io_err);
        
        assert!(err.source().is_some());
    }

    #[test]
    fn test_recovery_strategy() {
        let fatal = HydroError::internal_assertion("断言失败");
        assert!(!fatal.is_recoverable());

        let retry = HydroError::cfl_violation(2.0, 1.0);
        assert!(retry.is_recoverable());
    }
}
