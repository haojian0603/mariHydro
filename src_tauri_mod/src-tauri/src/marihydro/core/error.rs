// src-tauri/src/marihydro/core/error.rs

use std::path::PathBuf;
use thiserror::Error;

/// 统一结果类型
pub type MhResult<T> = Result<T, MhError>;

/// MariHydro错误类型（完整分类）
#[derive(Error, Debug)]
pub enum MhError {
    // ============================================================
    // IO相关错误
    // ============================================================
    #[error("IO错误: {message}")]
    Io {
        message: String,
        #[source]
        source: Option<std::io::Error>,
    },

    #[error("文件不存在: {path}")]
    FileNotFound { path: PathBuf },

    #[error("不支持的文件格式: {format} (支持的格式: {supported:?})")]
    UnsupportedFormat {
        format: String,
        supported: Vec<String>,
    },

    #[error("文件解析错误: {file} 第{line}行: {message}")]
    ParseError {
        file: PathBuf,
        line: usize,
        message: String,
    },

    // ============================================================
    // 数据验证错误
    // ============================================================
    #[error("无效的输入数据: {message}")]
    InvalidInput { message: String },

    #[error("数据超出范围: {field}={value}, 期望范围=[{min}, {max}]")]
    OutOfRange {
        field: &'static str,
        value: f64,
        min: f64,
        max: f64,
    },

    #[error("时间超出数据覆盖范围: 请求时间={requested}s, 数据范围=[{start}s, {end}s]")]
    TimeOutOfRange {
        requested: f64,
        start: f64,
        end: f64,
    },

    #[error("数组大小不匹配: {name} 期望{expected}, 实际{actual}")]
    SizeMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },

    // ============================================================
    // 数值计算错误
    // ============================================================
    #[error("数值计算失败: {message}")]
    Numerical { message: String },

    #[error("检测到非有限数值: {field}={value} 在单元{cell_id}")]
    NonFinite {
        field: &'static str,
        value: f64,
        cell_id: usize,
    },

    #[error("检测到负水深: h={value} 在单元{cell_id}")]
    NegativeDepth { value: f64, cell_id: usize },

    #[error("求解器不收敛: {reason} (迭代{iterations}次后)")]
    NonConvergence { reason: String, iterations: usize },

    #[error("CFL条件违反: 计算dt={computed_dt}, 最小允许dt={min_dt}")]
    CflViolation { computed_dt: f64, min_dt: f64 },

    #[error("数值不稳定 (t={time:.4}s): {message}")]
    NumericalInstability {
        message: String,
        time: f64,
        location: Option<(f64, f64)>,
    },

    // ============================================================
    // 网格相关错误
    // ============================================================
    #[error("无效的网格拓扑: {message}")]
    InvalidMesh { message: String },

    #[error("单元索引越界: {index} >= {n_cells}")]
    CellIndexOutOfBounds { index: usize, n_cells: usize },

    #[error("面索引越界: {index} >= {n_faces}")]
    FaceIndexOutOfBounds { index: usize, n_faces: usize },

    #[error("边界ID未找到: {boundary_id}")]
    BoundaryNotFound { boundary_id: String },

    #[error("边界条件错误: {message}")]
    BoundaryCondition { message: String },

    // ============================================================
    // 配置相关错误
    // ============================================================
    #[error("配置错误: {message}")]
    Config { message: String },

    #[error("缺少必需的配置项: {key}")]
    MissingConfig { key: String },

    #[error("配置值无效: {key}={value}, 原因: {reason}")]
    InvalidConfig {
        key: String,
        value: String,
        reason: String,
    },

    // ============================================================
    // 序列化/数据库错误
    // ============================================================
    #[error("序列化错误: {message}")]
    Serialization { message: String },

    #[error("数据库错误: {message}")]
    Database { message: String },

    #[error("数据加载失败 ({file}): {message}")]
    DataLoad { file: String, message: String },

    // ============================================================
    // 地理投影错误
    // ============================================================
    #[error("投影错误: {0}")]
    Projection(String),

    #[error("时区错误: {0}")]
    Timezone(String),

    // ============================================================
    // 并发相关错误
    // ============================================================
    #[error("锁获取失败: {resource}")]
    LockError { resource: String },

    #[error("通道发送失败")]
    ChannelSendError,

    #[error("任务取消")]
    TaskCancelled,

    // ============================================================
    // 工作流错误
    // ============================================================
    #[error("工作流错误: {0}")]
    Workflow(String),

    #[error("验证失败: {0}")]
    Validation(String),

    // ============================================================
    // 内部/系统错误
    // ============================================================
    #[error("内部错误: {message}")]
    Internal { message: String },

    #[error("运行时错误: {0}")]
    Runtime(String),

    #[error("功能未实现: {feature}")]
    NotImplemented { feature: String },
}

// ============================================================
// 便捷构造方法
// ============================================================

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

    /// 数组大小不匹配
    pub fn size_mismatch(name: &'static str, expected: usize, actual: usize) -> Self {
        Self::SizeMismatch {
            name,
            expected,
            actual,
        }
    }

    /// 非有限值
    pub fn non_finite(field: &'static str, value: f64, cell_id: usize) -> Self {
        Self::NonFinite {
            field,
            value,
            cell_id,
        }
    }

    /// 负水深
    pub fn negative_depth(value: f64, cell_id: usize) -> Self {
        Self::NegativeDepth { value, cell_id }
    }

    /// 解析错误
    pub fn parse(file: impl Into<PathBuf>, line: usize, message: impl Into<String>) -> Self {
        Self::ParseError {
            file: file.into(),
            line,
            message: message.into(),
        }
    }

    /// 配置错误
    pub fn config(message: impl Into<String>) -> Self {
        Self::Config {
            message: message.into(),
        }
    }

    /// 无效网格
    pub fn invalid_mesh(message: impl Into<String>) -> Self {
        Self::InvalidMesh {
            message: message.into(),
        }
    }

    /// 数值不稳定
    pub fn numerical_instability(message: impl Into<String>, time: f64) -> Self {
        Self::NumericalInstability {
            message: message.into(),
            time,
            location: None,
        }
    }

    /// 数值不稳定（带位置）
    pub fn numerical_instability_at(message: impl Into<String>, time: f64, x: f64, y: f64) -> Self {
        Self::NumericalInstability {
            message: message.into(),
            time,
            location: Some((x, y)),
        }
    }

    /// 内部错误
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }
}

// ============================================================
// 验证辅助函数
// ============================================================

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

    /// 检查值是否有限
    #[inline]
    pub fn check_finite(field: &'static str, value: f64, cell_id: usize) -> MhResult<()> {
        if !value.is_finite() {
            Err(Self::non_finite(field, value, cell_id))
        } else {
            Ok(())
        }
    }

    /// 检查水深是否非负
    #[inline]
    pub fn check_non_negative_depth(value: f64, cell_id: usize) -> MhResult<()> {
        if value < 0.0 {
            Err(Self::negative_depth(value, cell_id))
        } else {
            Ok(())
        }
    }

    /// 检查值是否在范围内
    #[inline]
    pub fn check_range(field: &'static str, value: f64, min: f64, max: f64) -> MhResult<()> {
        if value < min || value > max {
            Err(Self::OutOfRange {
                field,
                value,
                min,
                max,
            })
        } else {
            Ok(())
        }
    }
}

// ============================================================
// 标准库错误转换
// ============================================================

impl From<std::io::Error> for MhError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
            source: Some(err),
        }
    }
}

impl From<serde_json::Error> for MhError {
    fn from(err: serde_json::Error) -> Self {
        Self::Serialization {
            message: err.to_string(),
        }
    }
}

impl From<rusqlite::Error> for MhError {
    fn from(err: rusqlite::Error) -> Self {
        Self::Database {
            message: err.to_string(),
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

#[cfg(feature = "netcdf")]
impl From<netcdf::error::Error> for MhError {
    fn from(err: netcdf::error::Error) -> Self {
        Self::DataLoad {
            file: "NetCDF文件".into(),
            message: err.to_string(),
        }
    }
}

// ============================================================
// 验证宏
// ============================================================

/// 验证条件，失败时返回错误
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $err:expr) => {
        if !$cond {
            return Err($err);
        }
    };
}

/// 验证Option，None时返回错误
#[macro_export]
macro_rules! require {
    ($opt:expr, $err:expr) => {
        match $opt {
            Some(v) => v,
            None => return Err($err),
        }
    };
}

/// 验证数组大小
#[macro_export]
macro_rules! check_size {
    ($name:literal, $expected:expr, $actual:expr) => {
        $crate::marihydro::core::MhError::check_size($name, $expected, $actual)?
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MhError::config("测试配置错误");
        assert!(err.to_string().contains("配置错误"));
    }

    #[test]
    fn test_numerical_instability() {
        let err = MhError::numerical_instability("水深异常", 10.5);
        let msg = err.to_string();
        assert!(msg.contains("10.5"));
        assert!(msg.contains("水深异常"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let mh_err: MhError = io_err.into();
        assert!(matches!(mh_err, MhError::Io { .. }));
    }

    #[test]
    fn test_check_size() {
        assert!(MhError::check_size("test", 10, 10).is_ok());
        assert!(MhError::check_size("test", 10, 5).is_err());
    }

    #[test]
    fn test_check_finite() {
        assert!(MhError::check_finite("h", 1.0, 0).is_ok());
        assert!(MhError::check_finite("h", f64::NAN, 0).is_err());
        assert!(MhError::check_finite("h", f64::INFINITY, 0).is_err());
    }

    #[test]
    fn test_check_range() {
        assert!(MhError::check_range("value", 5.0, 0.0, 10.0).is_ok());
        assert!(MhError::check_range("value", -1.0, 0.0, 10.0).is_err());
        assert!(MhError::check_range("value", 11.0, 0.0, 10.0).is_err());
    }
}
