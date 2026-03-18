// crates/mh_io/src/error.rs
//! IO 错误类型定义
//!
//! 提供 IO 模块的统一错误枚举，支持通过 thiserror 自动转换底层错误。
//! 所有错误最终可转换为 MhError 以实现跨层错误传递。

use thiserror::Error;
use mh_foundation::MhError;

/// IO 模块结果类型别名
pub type IoResult<T> = Result<T, IoError>;

/// IO 错误枚举
#[derive(Error, Debug)]
pub enum IoError {
    /// 驱动加载失败
    #[error("驱动加载失败: {driver_type}, {name}, {reason}")]
    DriverLoadFailed {
        driver_type: &'static str,
        name: String,
        reason: String,
    },

    /// 文件格式识别失败
    #[error("无法识别文件格式: {path}")]
    UnknownFormat { path: String },

    /// 时间序列数据不连续
    #[error("时间序列数据不连续: 期望间隔 {expected_interval}s, 实际间隔 {actual_interval}s, 在 {timestamp}")]
    TimeSeriesDiscontinuity {
        expected_interval: f64,
        actual_interval: f64,
        timestamp: String,
    },

    /// 投影信息缺失
    #[error("投影信息缺失: 文件 {file}")]
    MissingProjection { file: String },

    /// 数据类型不匹配
    #[error("数据类型不匹配: 期望 {expected}, 实际 {actual}, 变量 {variable}")]
    DataTypeMismatch {
        expected: String,
        actual: String,
        variable: String,
    },

    /// 检查点损坏
    #[error("检查点损坏: {checkpoint}, 原因: {reason}")]
    CheckpointCorruption {
        checkpoint: String,
        reason: String,
    },

    /// 管道处理失败
    #[error("管道处理失败: 阶段 {stage}, {message}")]
    PipelineFailed {
        stage: String,
        message: String,
    },

    /// 解析错误
    #[error("文件解析错误: {file}:{line} - {message}")]
    ParseError {
        file: String,
        line: usize,
        message: String,
    },

    /// 基础层错误转换
    #[error("基础层错误: {0}")]
    Foundation(#[from] MhError),
}

impl From<IoError> for MhError {
    fn from(err: IoError) -> Self {
        match err {
            IoError::DriverLoadFailed { driver_type, name, reason } => {
                MhError::internal(format!("驱动加载失败 [{driver_type}, {name}]: {reason}"))
            }
            IoError::UnknownFormat { path } => {
                MhError::invalid_input(format!("无法识别文件格式: {path}"))
            }
            IoError::TimeSeriesDiscontinuity { expected_interval, actual_interval, timestamp } => {
                MhError::invalid_input(format!(
                    "时间序列数据不连续 (期望间隔 {expected_interval}s, 实际间隔 {actual_interval}s, 时间 {timestamp})"
                ))
            }
            IoError::MissingProjection { file } => {
                MhError::invalid_input(format!("投影信息缺失: {file}"))
            }
            IoError::DataTypeMismatch { expected, actual, variable } => {
                MhError::invalid_input(format!(
                    "数据类型不匹配 (变量 {variable}: 期望 {expected}, 实际 {actual})"
                ))
            }
            IoError::CheckpointCorruption { checkpoint, reason } => {
                MhError::internal(format!("检查点损坏 [{checkpoint}]: {reason}"))
            }
            IoError::PipelineFailed { stage, message } => {
                MhError::internal(format!("管道处理失败 [{stage}]: {message}"))
            }
            IoError::ParseError { file, line, message } => {
                MhError::invalid_input(format!("文件解析错误 [{file}:{line}]: {message}"))
            }
            IoError::Foundation(mh_err) => mh_err,
        }
    }
}

impl From<crate::pipeline::PipelineError> for IoError {
    fn from(err: crate::pipeline::PipelineError) -> Self {
        match err {
            crate::pipeline::PipelineError::Io(e) => {
                IoError::PipelineFailed {
                    stage: "pipeline".to_string(),
                    message: format!("IO 错误: {}", e),
                }
            }
            crate::pipeline::PipelineError::Serialization(msg) => {
                IoError::PipelineFailed {
                    stage: "serialization".to_string(),
                    message: format!("序列化失败: {}", msg),
                }
            }
            crate::pipeline::PipelineError::Timeout(dur) => {
                IoError::PipelineFailed {
                    stage: "timeout".to_string(),
                    message: format!("操作超时: {:?}", dur),
                }
            }
        }
    }
}