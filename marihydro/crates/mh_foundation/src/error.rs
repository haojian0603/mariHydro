// marihydro\crates\mh_foundation\src/error.rs

//! 错误处理模块，定义统一错误类型
//!
//! 提供 `MhError` 枚举和 `MhResult` 类型别名，用于整个项目的错误处理。
//!
//! # 设计原则
//!
//! 1. **层次化**: 基础层只定义核心错误，物理相关错误在 mh_physics 中定义
//! 2. **易用性**: 提供便捷的构造方法
//! 3. **可追溯**: 支持错误链
//!
//! # 示例
//!
//! ```
//! use mh_foundation::error::{MhError, MhResult};
//!
//! fn read_config() -> MhResult<()> {
//!     Err(MhError::config("配置文件格式错误"))
//! }
//! ```

use std::path::PathBuf;
use thiserror::Error;

/// 统一结果类型
pub type MhResult<T> = Result<T, MhError>;

/// MariHydro 错误类型
///
/// 核心错误类型，用于整个项目。物理计算相关的错误应在 `mh_physics` 中扩展。
#[derive(Error, Debug)]
pub enum MhError {
    // ========================================================================
    // IO 相关错误
    // ========================================================================
    
    /// IO 错误
    #[error("IO错误: {message}")]
    Io {
        /// 描述性错误信息
        message: String,
        #[source]
        /// 可选的底层 IO 错误
        source: Option<std::io::Error>,
    },

    /// 文件不存在
    #[error("文件不存在: {path}")]
    FileNotFound {
        /// 未找到的路径
        path: PathBuf,
    },

    /// 不支持的文件格式
    #[error("不支持的文件格式: {format} (支持的格式: {supported:?})")]
    UnsupportedFormat {
        /// 输入文件格式
        format: String,
        /// 支持的格式列表
        supported: Vec<String>,
    },

    /// 文件解析错误
    #[error("文件解析错误: {file} 第{line}行: {message}")]
    ParseError {
        /// 文件路径
        file: PathBuf,
        /// 行号
        line: usize,
        /// 错误信息
        message: String,
    },

    /// 无效输入
    #[error("无效的输入数据: {message}")]
    InvalidInput {
        /// 说明无效原因
        message: String,
    },

    /// 数据超出范围
    #[error("数据超出范围: {field}={value}, 期望范围=[{min}, {max}]")]
    OutOfRange {
        /// 字段名
        field: &'static str,
        /// 实际值
        value: f64,
        /// 最小允许值
        min: f64,
        /// 最大允许值
        max: f64,
    },

    /// 数组大小不匹配
    #[error("数组大小不匹配: {name} 期望{expected}, 实际{actual}")]
    SizeMismatch {
        /// 数据名称
        name: &'static str,
        /// 期望大小
        expected: usize,
        /// 实际大小
        actual: usize,
    },

    /// 索引越界
    #[error("索引越界: {index_type} 索引 {index} 超出范围 0..{len}")]
    IndexOutOfBounds {
        /// 索引类别描述
        index_type: &'static str,
        /// 访问的索引
        index: usize,
        /// 上界（长度）
        len: usize,
    },

    /// 无效索引（代际不匹配）
    #[error("无效索引: 元素已被删除或索引过期")]
    InvalidIndex,

    /// 无效网格拓扑
    #[error("无效的网格拓扑: {message}")]
    InvalidMesh {
        /// 具体错误信息
        message: String,
    },

    /// 配置错误
    #[error("配置错误: {message}")]
    Config {
        /// 具体错误信息
        message: String,
    },

    /// 缺少配置项
    #[error("缺少必需的配置项: {key}")]
    MissingConfig {
        /// 配置键名
        key: String,
    },

    /// 配置值无效
    #[error("配置值无效: {key}={value}, 原因: {reason}")]
    InvalidConfig {
        /// 配置键名
        key: String,
        /// 配置值        
        value: String,
        /// 无效原因说明
        reason: String,
    },

    /// 序列化错误
    #[error("序列化错误: {message}")]
    Serialization {
        /// 序列化失败原因
        message: String,
    },

    /// 投影错误
    #[error("投影错误: {0}")]
    Projection(String),

    /// 坐标系错误
    #[error("坐标系错误: {0}")]
    Crs(String),

    /// 锁获取失败
    #[error("锁获取失败: {resource}")]
    LockError {
        /// 失败的资源名
        resource: String,
    },

    /// 通道发送失败
    #[error("通道发送失败")]
    ChannelSendError,

    /// 任务取消
    #[error("任务取消")]
    TaskCancelled,

    /// 验证失败
    #[error("验证失败: {0}")]
    Validation(String),

    /// 内部错误
    #[error("内部错误: {message}")]
    Internal {
        /// 内部错误描述
        message: String,
    },

    /// 运行时错误
    #[error("运行时错误: {0}")]
    Runtime(String),

    /// 功能未实现
    #[error("功能未实现: {feature}")]
    NotImplemented {
        /// 未实现的功能描述
        feature: String,
    },

    /// 资源未找到
    #[error("资源未找到: {resource}")]
    NotFound {
        /// 资源名称
        resource: String,
    },
}

// ========================================================================
// 便捷构造方法
// ========================================================================

impl MhError {
    /// 从IO错误创建
    pub fn io(message: impl Into<String>) -> Self {
        Self::Io {
            message: message.into(),
            source: None,
        }
    }

    /// 从IO错误创建（带源）
    pub fn io_with_source(message: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            message: message.into(),
            source: Some(source),
        }
    }

    /// 文件不存在
    pub fn file_not_found(path: impl Into<PathBuf>) -> Self {
        Self::FileNotFound { path: path.into() }
    }

    /// 不支持的格式
    pub fn unsupported_format(format: impl Into<String>, supported: Vec<String>) -> Self {
        Self::UnsupportedFormat {
            format: format.into(),
            supported,
        }
    }

    /// 解析错误
    pub fn parse(file: impl Into<PathBuf>, line: usize, message: impl Into<String>) -> Self {
        Self::ParseError {
            file: file.into(),
            line,
            message: message.into(),
        }
    }

    /// 无效输入
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// 数据超出范围
    pub fn out_of_range(field: &'static str, value: f64, min: f64, max: f64) -> Self {
        Self::OutOfRange {
            field,
            value,
            min,
            max,
        }
    }

    /// 数组大小不匹配
    pub fn size_mismatch(name: &'static str, expected: usize, actual: usize) -> Self {
        Self::SizeMismatch {
            name,
            expected,
            actual,
        }
    }

    /// 索引越界
    pub fn index_out_of_bounds(index_type: &'static str, index: usize, len: usize) -> Self {
        Self::IndexOutOfBounds {
            index_type,
            index,
            len,
        }
    }

    /// 无效网格
    pub fn invalid_mesh(message: impl Into<String>) -> Self {
        Self::InvalidMesh {
            message: message.into(),
        }
    }

    /// 配置错误
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// 缺少配置
    pub fn missing_config(key: impl Into<String>) -> Self {
        Self::MissingConfig { key: key.into() }
    }

    /// 配置值无效
    pub fn invalid_config(
        key: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidConfig {
            key: key.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// 序列化错误
    pub fn serialization(message: impl Into<String>) -> Self {
        Self::Serialization {
            message: message.into(),
        }
    }

    /// 投影错误
    pub fn projection(message: impl Into<String>) -> Self {
        Self::Projection(message.into())
    }

    /// 坐标系错误
    pub fn crs(message: impl Into<String>) -> Self {
        Self::Crs(message.into())
    }

    /// 锁错误
    pub fn lock_error(resource: impl Into<String>) -> Self {
        Self::LockError {
            resource: resource.into(),
        }
    }

    /// 验证失败
    pub fn validation(message: impl Into<String>) -> Self {
        Self::Validation(message.into())
    }

    /// 内部错误
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// 运行时错误
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::Runtime(message.into())
    }

    /// 功能未实现
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }

    /// 资源未找到
    pub fn not_found(resource: impl Into<String>) -> Self {
        Self::NotFound {
            resource: resource.into(),
        }
    }
}

// ========================================================================
// 验证辅助方法
// ========================================================================

impl MhError {
    /// 检查数组大小是否匹配
    #[inline]
    pub fn check_size(name: &'static str, expected: usize, actual: usize) -> MhResult<()> {
        if expected != actual {
            Err(Self::size_mismatch(name, expected, actual))
        } else {
            Ok(())
        }
    }

    /// 检查值是否在范围内
    #[inline]
    pub fn check_range(field: &'static str, value: f64, min: f64, max: f64) -> MhResult<()> {
        if value < min || value > max {
            Err(Self::out_of_range(field, value, min, max))
        } else {
            Ok(())
        }
    }

    /// 检查索引是否在范围内
    #[inline]
    pub fn check_index(index_type: &'static str, index: usize, len: usize) -> MhResult<()> {
        if index >= len {
            Err(Self::index_out_of_bounds(index_type, index, len))
        } else {
            Ok(())
        }
    }
}

// ========================================================================
// 标准库错误转换
// ========================================================================

impl From<std::io::Error> for MhError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            source: Some(err),
        }
    }
}

impl<T> From<std::sync::PoisonError<T>> for MhError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Self::LockError {
            resource: "mutex".into(),
        }
    }
}

impl<T> From<std::sync::mpsc::SendError<T>> for MhError {
    fn from(_: std::sync::mpsc::SendError<T>) -> Self {
        Self::ChannelSendError
    }
}

// ========================================================================
// 测试
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MhError::config("测试配置错误");
        assert!(err.to_string().contains("配置错误"));
    }

    #[test]
    fn test_io_error() {
        let err = MhError::io("读取失败");
        assert!(err.to_string().contains("IO错误"));
    }

    #[test]
    fn test_file_not_found() {
        let err = MhError::file_not_found("/path/to/file");
        assert!(err.to_string().contains("/path/to/file"));
    }

    #[test]
    fn test_index_out_of_bounds() {
        let err = MhError::index_out_of_bounds("Cell", 10, 5);
        assert!(err.to_string().contains("Cell"));
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_check_size() {
        assert!(MhError::check_size("test", 10, 10).is_ok());
        assert!(MhError::check_size("test", 10, 5).is_err());
    }

    #[test]
    fn test_check_range() {
        assert!(MhError::check_range("value", 5.0, 0.0, 10.0).is_ok());
        assert!(MhError::check_range("value", -1.0, 0.0, 10.0).is_err());
        assert!(MhError::check_range("value", 11.0, 0.0, 10.0).is_err());
    }

    #[test]
    fn test_check_index() {
        assert!(MhError::check_index("Cell", 5, 10).is_ok());
        assert!(MhError::check_index("Cell", 10, 10).is_err());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let mh_err: MhError = io_err.into();
        assert!(matches!(mh_err, MhError::Io { .. }));
    }

    #[test]
    fn test_ensure_macro() {
        fn check(value: i32) -> MhResult<()> {
            ensure!(value > 0, MhError::invalid_input("value must be positive"));
            Ok(())
        }

        assert!(check(1).is_ok());
        assert!(check(-1).is_err());
    }

    #[test]
    fn test_require_macro() {
        fn get_value(opt: Option<i32>) -> MhResult<i32> {
            let v = require!(opt, MhError::not_found("value"));
            Ok(v)
        }

        assert_eq!(get_value(Some(42)).unwrap(), 42);
        assert!(get_value(None).is_err());
    }
}