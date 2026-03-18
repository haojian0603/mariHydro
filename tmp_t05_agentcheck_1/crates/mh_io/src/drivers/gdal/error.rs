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
    /// GDAL 不可用
    NotAvailable,
    /// 其他错误
    Other(String),
}

impl fmt::Display for GdalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GdalError::FileNotFound(path) => write!(f, "File not found: {}", path),
            GdalError::OpenFailed(msg) => write!(f, "Failed to open dataset: {}", msg),
            GdalError::BandNotFound(idx) => write!(f, "Band {} not found", idx),
            GdalError::ReadFailed(msg) => write!(f, "Failed to read data: {}", msg),
            GdalError::ProjectionError(msg) => write!(f, "Projection error: {}", msg),
            GdalError::NotAvailable => write!(f, "GDAL is not available"),
            GdalError::Other(msg) => write!(f, "GDAL error: {}", msg),
        }
    }
}

impl std::error::Error for GdalError {}

#[cfg(feature = "gdal")]
impl From<gdal::errors::GdalError> for GdalError {
    fn from(e: gdal::errors::GdalError) -> Self {
        GdalError::Other(e.to_string())
    }
}
