use std::path::PathBuf;
use thiserror::Error;

pub type MhResult<T> = Result<T, MhError>;

#[derive(Debug, Error)]
pub enum MhError {
    #[error("IO 错误 ({context}): {source}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    #[error("配置错误: {0}")]
    Config(String),

    #[error("数据加载失败 ({file}): {message}")]
    DataLoad { file: String, message: String },

    #[error("投影错误: {0}")]
    Projection(String),

    #[error("时区错误: {0}")]
    Timezone(String),

    #[error("网格错误: {message}")]
    InvalidMesh { message: String },

    #[error("边界条件错误: {message}")]
    BoundaryCondition { message: String },

    #[error("数值不稳定 (t={time:.4}s): {message}")]
    NumericalInstability {
        message: String,
        time: f64,
        location: Option<(f64, f64)>,
    },

    #[error("验证失败: {0}")]
    Validation(String),

    #[error("未实现: {0}")]
    NotImplemented(String),

    #[error("工作流错误: {0}")]
    Workflow(String),

    #[error("输入参数错误: {0}")]
    InvalidInput(String),

    #[error("运行时错误: {0}")]
    Runtime(String),

    #[error("内部错误: {0}")]
    InternalError(String),

    #[error("NetCDF 错误: {0}")]
    NetCdf(#[from] netcdf::error::Error),
}

impl MhError {
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }

    pub fn io_not_found(path: &str) -> Self {
        Self::Io {
            context: format!("文件不存在: {}", path),
            source: std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("文件不存在: {}", path),
            ),
        }
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn invalid_mesh(msg: impl Into<String>) -> Self {
        Self::InvalidMesh {
            message: msg.into(),
        }
    }

    pub fn numerical_instability(msg: impl Into<String>, time: f64) -> Self {
        Self::NumericalInstability {
            message: msg.into(),
            time,
            location: None,
        }
    }

    pub fn numerical_instability_at(msg: impl Into<String>, time: f64, x: f64, y: f64) -> Self {
        Self::NumericalInstability {
            message: msg.into(),
            time,
            location: Some((x, y)),
        }
    }
}

impl From<std::io::Error> for MhError {
    fn from(e: std::io::Error) -> Self {
        Self::Io {
            context: "IO 操作".into(),
            source: e,
        }
    }
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
}
