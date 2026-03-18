// crates/mh_mesh/src/error.rs
//! 网格处理错误类型
//! 
//! 包含网格拓扑、格式、质量等错误定义
//! 所有错误可转换为 `mh_runtime::RuntimeError` 向上传播

use thiserror::Error;
use mh_foundation::MhError;
use mh_runtime::RuntimeError;

/// 网格模块结果类型
pub type MeshResult<T> = Result<T, MeshError>;

/// 网格错误枚举
#[derive(Error, Debug)]
pub enum MeshError {
    /// 拓扑错误
    #[error("拓扑错误: {operation} 失败, {details}")]
    InvalidTopology {
        operation: &'static str,
        details: String,
    },

    /// 网格格式错误
    #[error("网格格式错误: {format}, {file}, 行 {line}: {message}")]
    MeshFormatError {
        format: &'static str,
        file: String,
        line: usize,
        message: String,
    },

    /// 半边结构损坏
    #[error("半边结构损坏: {check}, 元素 {element_id}, {message}")]
    HalfEdgeCorruption {
        check: &'static str,
        element_id: usize,
        message: String,
    },

    /// 网格质量过低
    #[error("网格质量过低: {metric} = {value:.3}, 阈值 {threshold:.3}, 单元 {cell_id}")]
    QualityTooLow {
        metric: &'static str,
        value: f64,
        threshold: f64,
        cell_id: usize,
    },

    /// 元素不匹配
    #[error("元素不匹配: 操作需要 {required} 个单元, 提供 {provided}")]
    ElementCountMismatch {
        required: usize,
        provided: usize,
        context: String,
    },

    /// 聚合运行时错误
    #[error("运行时错误: {0}")]
    Runtime(#[from] RuntimeError),
}

/// 转换到 Runtime 层错误（Layer 2）
impl From<MeshError> for RuntimeError {
    fn from(err: MeshError) -> Self {
        match err {
            MeshError::InvalidTopology { operation, details } => {
                RuntimeError::numerical(format!("网格拓扑错误 [{}]: {}", operation, details))
            }
            MeshError::MeshFormatError { format, file, line, message } => {
                RuntimeError::backend(format!("网格格式错误 [{} {}:{}]: {}", format, file, line, message))
            }
            MeshError::HalfEdgeCorruption { check, element_id, message } => {
                RuntimeError::internal(format!("半边结构损坏 [{}, 元素 {}]: {}", check, element_id, message))
            }
            MeshError::QualityTooLow { metric, value, threshold, cell_id } => {
                RuntimeError::validation(format!("网格质量过低 [单元 {}, {}={:.3}, 阈值={:.3}]", cell_id, metric, value, threshold))
            }
            MeshError::ElementCountMismatch { required, provided, context: _  } => {
                RuntimeError::size_mismatch("mesh_elements", required, provided)
            }
            MeshError::Runtime(runtime_err) => runtime_err,
        }
    }
}

/// 转换到 Foundation 层错误（Layer 1）
impl From<MeshError> for MhError {
    fn from(err: MeshError) -> Self {
        let runtime_err: RuntimeError = err.into();
        runtime_err.into()
    }
}

/// 便捷构造函数（避免直接使用 RuntimeError 变体）
impl MeshError {
    pub fn invalid_topology(operation: &'static str, details: impl Into<String>) -> Self {
        Self::InvalidTopology {
            operation,
            details: details.into(),
        }
    }

    pub fn mesh_format_error(format: &'static str, file: impl Into<String>, line: usize, message: impl Into<String>) -> Self {
        Self::MeshFormatError {
            format,
            file: file.into(),
            line,
            message: message.into(),
        }
    }

    pub fn halfedge_corruption(check: &'static str, element_id: usize, message: impl Into<String>) -> Self {
        Self::HalfEdgeCorruption {
            check,
            element_id,
            message: message.into(),
        }
    }

    pub fn quality_too_low(metric: &'static str, value: f64, threshold: f64, cell_id: usize) -> Self {
        Self::QualityTooLow {
            metric,
            value,
            threshold,
            cell_id,
        }
    }

    pub fn element_count_mismatch(required: usize, provided: usize, context: impl Into<String>) -> Self {
        Self::ElementCountMismatch {
            required,
            provided,
            context: context.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion_to_runtime() {
        let mesh_err = MeshError::invalid_topology("merge_cells", "duplicate vertex");
        let runtime_err: RuntimeError = mesh_err.into();
        assert!(matches!(runtime_err, RuntimeError::NumericalError { .. }));
    }

    #[test]
    fn test_error_chain_to_foundation() {
        let mesh_err = MeshError::invalid_topology("validate", "non-manifold");
        let foundation_err: MhError = mesh_err.into();
        assert!(foundation_err.to_string().contains("网格拓扑错误"));
    }
}