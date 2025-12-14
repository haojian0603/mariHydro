// crates/mh_runtime/src/error.rs
//! 运行时错误类型
//!
//! 包含数值计算、索引验证、Backend 操作等运行时相关的错误。
//! 与 Foundation 层的 MhError 不同，此错误类型面向 Layer 3 引擎层。

use thiserror::Error;
use mh_foundation::MhError;

/// 运行时结果类型
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// 运行时错误（面向 Layer 3 引擎层）
#[derive(Error, Debug)]
pub enum RuntimeError {
    /// 数值超出范围
    #[error("数值超出范围: {value} 不在 [{min}, {max}] 范围内")]
    OutOfRange {
        /// 超出范围的数值
        value: f64,
        /// 数值范围最小值
        min: f64,
        /// 数值范围最大值
        max: f64,
    },

    /// 无效索引（代际不匹配）
    #[error("无效索引: 元素已被删除或索引过期")]
    InvalidIndex,

    /// 数值计算错误
    #[error("数值计算错误: {message}")]
    NumericalError {
        /// 错误描述信息
        message: String,
    },

    /// 非有限值（NaN 或 Inf）
    #[error("非有限值: {value}")]
    NonFinite {
        /// 非有限数值
        value: f64,
    },

    /// Backend 操作错误
    #[error("Backend 错误: {message}")]
    BackendError {
        /// 错误描述信息
        message: String,
    },

    /// 缓冲区操作错误
    #[error("缓冲区错误: {message}")]
    BufferError {
        /// 错误描述信息
        message: String,
    },

    /// 从 Foundation 层错误转换
    #[error("基础层错误: {0}")]
    Foundation(#[from] MhError),
}

// =======================================================================
// 便捷构造方法
// =======================================================================

impl RuntimeError {
    /// 创建数值范围错误
    pub fn out_of_range(value: impl Into<f64>, min: impl Into<f64>, max: impl Into<f64>) -> Self {
        Self::OutOfRange {
            value: value.into(),
            min: min.into(),
            max: max.into(),
        }
    }

    /// 创建数值计算错误
    pub fn numerical(message: impl Into<String>) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// 创建 Backend 错误
    pub fn backend(message: impl Into<String>) -> Self {
        Self::BackendError {
            message: message.into(),
        }
    }

    /// 创建缓冲区错误
    pub fn buffer(message: impl Into<String>) -> Self {
        Self::BufferError {
            message: message.into(),
        }
    }
}

// =======================================================================
// 测试
// =======================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_out_of_range_error() {
        let err = RuntimeError::out_of_range(10.0, 0.0, 5.0);
        assert!(matches!(err, RuntimeError::OutOfRange { .. }));
        let msg = format!("{}", err);
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_numerical_error() {
        let err = RuntimeError::numerical("division by zero");
        assert!(matches!(err, RuntimeError::NumericalError { .. }));
        assert!(format!("{}", err).contains("division by zero"));
    }

    #[test]
    fn test_backend_error() {
        let err = RuntimeError::backend("CUDA unavailable");
        assert!(matches!(err, RuntimeError::BackendError { .. }));
        assert!(format!("{}", err).contains("CUDA"));
    }

    #[test]
    fn test_from_foundation_error() {
        let foundation_err = MhError::internal("io error");
        let runtime_err: RuntimeError = foundation_err.into();
        assert!(matches!(runtime_err, RuntimeError::Foundation { .. }));
    }
}