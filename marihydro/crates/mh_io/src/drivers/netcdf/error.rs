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
    /// NetCDF CLI 工具不可用
    NotAvailable { tool: String, detail: String },
    /// 数据布局不受支持
    UnsupportedLayout(String),
    /// 其他错误
    Other(String),
}

impl fmt::Display for NetCdfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetCdfError::FileNotFound(path) => write!(f, "文件不存在 {}", path),
            NetCdfError::OpenFailed(msg) => write!(f, "打开 NetCDF 文件失败: {}", msg),
            NetCdfError::DimensionNotFound(name) => write!(f, "维度不存在 {}", name),
            NetCdfError::VariableNotFound(name) => write!(f, "变量不存在 {}", name),
            NetCdfError::ReadFailed(msg) => write!(f, "读取 NetCDF 数据失败: {}", msg),
            NetCdfError::AttributeNotFound(name) => write!(f, "属性不存在: {}", name),
            NetCdfError::TimeParseError(msg) => write!(f, "时间解析失败: {}", msg),
            NetCdfError::NotAvailable { tool, detail } => {
                write!(f, "当前环境未接入 NetCDF 运行时工具 {tool}: {detail}")
            }
            NetCdfError::UnsupportedLayout(msg) => write!(f, "不支持的 NetCDF 布局: {}", msg),
            NetCdfError::Other(msg) => write!(f, "NetCDF 错误: {}", msg),
        }
    }
}

impl std::error::Error for NetCdfError {}

#[cfg(test)]
mod tests {
    use super::NetCdfError;

    #[test]
    fn test_not_available_message_preserves_tool_identity() {
        let err = NetCdfError::NotAvailable {
            tool: "ncdump-custom".to_string(),
            detail: "os error 2".to_string(),
        };

        let message = err.to_string();
        assert!(message.contains("ncdump-custom"));
        assert!(message.contains("os error 2"));
    }
}

#[cfg(feature = "netcdf")]
impl From<netcdf::error::Error> for NetCdfError {
    fn from(e: netcdf::error::Error) -> Self {
        NetCdfError::Other(e.to_string())
    }
}
