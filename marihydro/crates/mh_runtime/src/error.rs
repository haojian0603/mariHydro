// crates/mh_runtime/src/error.rs

//! 运行时错误类型
//!
//! 定义 Runtime 层的错误类型，包括计算错误、缓冲区错误等。

use std::fmt;

/// 运行时错误
#[derive(Debug)]
pub enum RuntimeError {
    /// 索引越界
    IndexOutOfBounds {
        /// 索引类型名称
        index_type: &'static str,
        /// 索引值
        index: usize,
        /// 容量
        len: usize,
    },
    /// 缓冲区大小不匹配
    BufferSizeMismatch {
        /// 期望大小
        expected: usize,
        /// 实际大小
        actual: usize,
    },
    /// 无效操作
    InvalidOperation {
        /// 操作描述
        operation: String,
        /// 原因
        reason: String,
    },
    /// 数值错误（NaN/Inf）
    NumericalError {
        /// 错误描述
        message: String,
    },
    /// 配置转换错误
    ConfigConversionError {
        /// 字段名
        field: String,
        /// 原始值
        value: f64,
    },
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds { index_type, index, len } => {
                write!(f, "{}({}) 越界，长度为 {}", index_type, index, len)
            }
            Self::BufferSizeMismatch { expected, actual } => {
                write!(f, "缓冲区大小不匹配: 期望 {}, 实际 {}", expected, actual)
            }
            Self::InvalidOperation { operation, reason } => {
                write!(f, "无效操作 '{}': {}", operation, reason)
            }
            Self::NumericalError { message } => {
                write!(f, "数值错误: {}", message)
            }
            Self::ConfigConversionError { field, value } => {
                write!(f, "配置转换失败: {} = {} 无法转换", field, value)
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

/// 运行时结果类型
pub type RuntimeResult<T> = Result<T, RuntimeError>;
