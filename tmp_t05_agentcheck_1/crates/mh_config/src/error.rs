// crates/mh_config/src/error.rs

//! 配置层错误类型

use mh_foundation::MhError;

/// 配置错误
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// IO 错误
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
    
    /// 解析错误
    #[error("解析错误: {0}")]
    Parse(String),
    
    /// 无效值
    #[error("无效值 '{key}': {value} - {reason}")]
    InvalidValue {
        /// 配置键
        key: String,
        /// 配置值
        value: String,
        /// 原因
        reason: String,
    },
    
    /// 缺失配置
    #[error("缺失配置: {0}")]
    Missing(String),
    
    /// 构建错误
    #[error("构建错误: {0}")]
    Build(String),
}

impl From<ConfigError> for MhError {
    fn from(err: ConfigError) -> Self {
        match err {
            ConfigError::Io(source) => MhError::io_with_source("配置读取失败", source),
            ConfigError::Parse(message) => MhError::invalid_input(format!("配置解析失败: {message}")),
            ConfigError::InvalidValue { key, value, reason } => MhError::invalid_input(format!(
                "配置无效 [{key}={value}]: {reason}"
            )),
            ConfigError::Missing(key) => MhError::invalid_input(format!("缺失配置项: {key}")),
            ConfigError::Build(message) => MhError::internal(format!("配置构建失败: {message}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error_display() {
        let err = ConfigError::InvalidValue {
            key: "cfl".to_string(),
            value: "-1".to_string(),
            reason: "必须为正".to_string(),
        };
        assert!(err.to_string().contains("cfl"));
    }
}
