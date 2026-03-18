// crates/mh_io/src/drivers/netcdf/error.rs

//! NetCDF 错误类型

use std::fmt;

/// NetCDF 错误
#[derive(Debug)]
pub enum NetCdfError {
    /// 文件不存在
    FileNotFound(String),
    /// 打开失败
    OpenFailed(String),
    /// 维度不存在
    DimensionNotFound(String),
    /// 变量不存在
    VariableNotFound(String),
    /// 读取失败
    ReadFailed(String),
    /// 属性不存在
    AttributeNotFound(String),
    /// 时间解析失败
    TimeParseError(String),
    /// NetCDF 不可用
    NotAvailable,
    /// 其他错误
    Other(String),
}

impl fmt::Display for NetCdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetCdfError::FileNotFound(path) => write!(f, "File not found: {}", path),
            NetCdfError::OpenFailed(msg) => write!(f, "Failed to open file: {}", msg),
            NetCdfError::DimensionNotFound(name) => write!(f, "Dimension not found: {}", name),
            NetCdfError::VariableNotFound(name) => write!(f, "Variable not found: {}", name),
            NetCdfError::ReadFailed(msg) => write!(f, "Failed to read data: {}", msg),
            NetCdfError::AttributeNotFound(name) => write!(f, "Attribute not found: {}", name),
            NetCdfError::TimeParseError(msg) => write!(f, "Failed to parse time: {}", msg),
            NetCdfError::NotAvailable => write!(f, "NetCDF is not available"),
            NetCdfError::Other(msg) => write!(f, "NetCDF error: {}", msg),
        }
    }
}

impl std::error::Error for NetCdfError {}

#[cfg(feature = "netcdf")]
impl From<netcdf::error::Error> for NetCdfError {
    fn from(e: netcdf::error::Error) -> Self {
        NetCdfError::Other(e.to_string())
    }
}
