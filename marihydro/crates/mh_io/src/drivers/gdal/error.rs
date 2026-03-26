// crates/mh_io/src/drivers/gdal/error.rs

//! GDAL 错误类型

use std::fmt;

/// GDAL 错误
#[derive(Debug)]
pub enum GdalError {
    /// 文件不存在
    FileNotFound(String),
    /// 打开失败
    OpenFailed(String),
    /// 波段不存在
    BandNotFound(usize),
    /// 读取失败
    ReadFailed(String),
    /// 投影错误
    ProjectionError(String),
    /// GDAL CLI 工具不可用
    NotAvailable { tool: String, detail: String },
    /// 其他错误
    Other(String),
}

impl fmt::Display for GdalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GdalError::FileNotFound(path) => write!(f, "文件不存在: {}", path),
            GdalError::OpenFailed(msg) => write!(f, "打开 GDAL 数据集失败: {}", msg),
            GdalError::BandNotFound(idx) => write!(f, "波段不存在: {}", idx),
            GdalError::ReadFailed(msg) => write!(f, "读取 GDAL 数据失败: {}", msg),
            GdalError::ProjectionError(msg) => write!(f, "投影解析失败: {}", msg),
            GdalError::NotAvailable { tool, detail } => {
                write!(f, "当前环境未接入 GDAL 运行时工具 {tool}: {detail}")
            }
            GdalError::Other(msg) => write!(f, "GDAL 错误: {}", msg),
        }
    }
}

impl std::error::Error for GdalError {}

#[cfg(test)]
mod tests {
    use super::GdalError;

    #[test]
    fn test_not_available_message_preserves_tool_identity() {
        let err = GdalError::NotAvailable {
            tool: "gdalinfo-custom".to_string(),
            detail: "系统找不到指定文件".to_string(),
        };

        let message = err.to_string();
        assert!(message.contains("gdalinfo-custom"));
        assert!(message.contains("系统找不到指定文件"));
    }
}

#[cfg(feature = "gdal")]
impl From<gdal::errors::GdalError> for GdalError {
    fn from(e: gdal::errors::GdalError) -> Self {
        GdalError::Other(e.to_string())
    }
}
