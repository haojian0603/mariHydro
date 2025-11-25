// src-tauri/src/marihydro/infra/error.rs

use serde::{Serialize, Serializer};
use std::path::PathBuf;
use thiserror::Error;

pub type MhResult<T> = Result<T, MhError>;

#[derive(Error, Debug)]
pub enum MhError {
    // ========================================================================
    // 1. 基础设施层
    // ========================================================================
    #[error("[System] IO错误: {context}")]
    Io {
        context: String,
        #[source]
        source: std::io::Error,
    },

    #[error("[System] 数据库错误: {0}")]
    Database(String),

    #[error("[System] 序列化失败: {0}")]
    Serialization(String),

    #[error("[Config] 配置无效: {0}")]
    Config(String),

    #[error("[Config] 配置字段无效: {field} - {message}")]
    ConfigError { field: String, message: String },

    #[error("[Config] 时区错误: {0}")]
    Timezone(String),

    #[error("[Input] 用户输入无效: {0}")]
    InvalidInput(String),

    // ========================================================================
    // 2. 数据层
    // ========================================================================
    #[error("[Data] 数据源加载失败: {file} - {message}")]
    DataLoad { file: String, message: String },

    #[error("[Data] NetCDF操作失败: {0}")]
    NetCdf(String),

    #[error("[Geo] 坐标投影变换失败: {0}")]
    Projection(String),

    #[error("[Geo] 不支持的格式或特性: {0}")]
    Unsupported(String),

    // ========================================================================
    // 3. 领域与物理层
    // ========================================================================
    #[error("[Mesh] 网格无效: {message}")]
    InvalidMesh { message: String },

    #[error("[Config] 配置无效: {field} - {message}")]
    InvalidConfig { field: String, message: String },

    #[error("[Physics] 数值计算不稳定性: {message} (T={time:.2}s)")]
    NumericalInstability {
        message: String,
        time: f64,
        location: Option<(usize, usize)>,
    },

    #[error("[Workflow] 任务执行错误: {0}")]
    Workflow(String),

    // ========================================================================
    // 4. 内部错误
    // ========================================================================
    #[error("[Internal] 内部错误: {0}")]
    InternalError(String),

    #[error("[Runtime] 运行时错误: {0}")]
    Runtime(String),

    #[error("[Unknown] 未知错误: {0}")]
    Unknown(String),
}

// ==================== 序列化实现 ====================

impl Serialize for MhError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("MhError", 3)?;

        let code = match self {
            MhError::Io { .. } => "ERR_IO",
            MhError::Database(_) => "ERR_DB",
            MhError::Serialization(_) => "ERR_SER",
            MhError::Config(_) => "ERR_CONFIG",
            MhError::ConfigError { .. } => "ERR_CONFIG",
            MhError::Timezone(_) => "ERR_CONFIG_TIMEZONE",
            MhError::InvalidInput(_) => "ERR_INPUT_INVALID",
            MhError::DataLoad { .. } => "ERR_DATA_LOAD",
            MhError::NetCdf(_) => "ERR_NETCDF",
            MhError::Projection(_) => "ERR_PROJ",
            MhError::Unsupported(_) => "ERR_UNSUPPORTED",
            MhError::InvalidMesh { .. } => "ERR_MESH",
            MhError::InvalidConfig { .. } => "ERR_CONFIG",
            MhError::NumericalInstability { .. } => "ERR_PHYSICS_NAN",
            MhError::Workflow(_) => "ERR_WORKFLOW",
            MhError::InternalError(_) => "ERR_INTERNAL",
            MhError::Runtime(_) => "ERR_RUNTIME",
            MhError::Unknown(_) => "ERR_UNKNOWN",
        };

        state.serialize_field("code", code)?;
        state.serialize_field("message", &self.to_string())?;
        state.serialize_field("details", &format!("{:?}", self))?;

        state.end()
    }
}

// ==================== 外部错误转换 ====================

impl From<proj::ProjError> for MhError {
    fn from(err: proj::ProjError) -> Self {
        MhError::Projection(err.to_string())
    }
}

impl From<serde_json::Error> for MhError {
    fn from(err: serde_json::Error) -> Self {
        MhError::Serialization(err.to_string())
    }
}

#[cfg(feature = "netcdf")]
impl From<netcdf::error::Error> for MhError {
    fn from(err: netcdf::error::Error) -> Self {
        MhError::NetCdf(err.to_string())
    }
}

impl From<Box<bincode::ErrorKind>> for MhError {
    fn from(err: Box<bincode::ErrorKind>) -> Self {
        MhError::Serialization(err.to_string())
    }
}

// ==================== 上下文扩展 ====================

pub trait MhContext<T, E> {
    fn context<C>(self, context: C) -> Result<T, MhError>
    where
        C: std::fmt::Display + Send + Sync + 'static;

    fn with_file_context<P>(self, path: P) -> Result<T, MhError>
    where
        P: Into<PathBuf>;
}

impl<T> MhContext<T, std::io::Error> for Result<T, std::io::Error> {
    fn context<C>(self, context: C) -> Result<T, MhError>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        self.map_err(|e| MhError::Io {
            context: context.to_string(),
            source: e,
        })
    }

    fn with_file_context<P>(self, path: P) -> Result<T, MhError>
    where
        P: Into<PathBuf>,
    {
        let path_str = path.into().to_string_lossy().to_string();
        self.map_err(|e| MhError::DataLoad {
            file: path_str.clone(),
            message: format!("IO Error: {}", e),
        })
    }
}

#[cfg(feature = "netcdf")]
impl<T> MhContext<T, netcdf::error::Error> for Result<T, netcdf::error::Error> {
    fn context<C>(self, context: C) -> Result<T, MhError>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        self.map_err(|e| MhError::NetCdf(format!("{}: {}", context, e)))
    }

    fn with_file_context<P>(self, path: P) -> Result<T, MhError>
    where
        P: Into<PathBuf>,
    {
        let path_str = path.into().to_string_lossy().to_string();
        self.map_err(|e| MhError::DataLoad {
            file: path_str,
            message: format!("NetCDF Error: {}", e),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_serialization() {
        let err = MhError::InvalidMesh {
            message: "测试错误".into(),
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains("ERR_MESH"));
        assert!(json.contains("测试错误"));
    }

    #[test]
    fn test_error_context() {
        let result: Result<(), std::io::Error> =
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "test"));
        let converted = result.context("加载文件失败");
        assert!(matches!(converted, Err(MhError::Io { .. })));
    }
}
