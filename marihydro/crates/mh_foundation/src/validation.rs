// marihydro\crates\mh_foundation\src/validation.rs

//! 运行时验证工具
//!
//! 提供验证报告和错误/警告类型，用于数据验证。
//!
//! # 示例
//!
//! ```
//! use mh_foundation::validation::{ValidationReport, ValidationError};
//!
//! let some_value = -1.0f64;
//! let mut report = ValidationReport::new();
//! if some_value < 0.0 {
//!     report.add_error(ValidationError::OutOfRange {
//!         field: "value",
//!         cell_id: 0,
//!         value: some_value,
//!         min: 0.0,
//!         max: f64::MAX,
//!     });
//! }
//!
//! if report.has_errors() {
//!     // 处理错误
//! }
//! ```

use std::fmt;

/// 验证报告
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// 错误列表
    pub errors: Vec<ValidationError>,
    /// 警告列表
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationReport {
    /// 创建空的验证报告
    pub fn new() -> Self {
        Self::default()
    }

    /// 添加错误
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// 添加警告
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// 是否有错误
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// 是否有警告
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// 错误数量
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    /// 警告数量
    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }

    /// 是否通过（无错误）
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// 合并另一个报告
    pub fn merge(&mut self, other: ValidationReport) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }

    /// 清空报告
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "验证报告:")?;
        writeln!(f, "  错误: {} 个", self.error_count())?;
        writeln!(f, "  警告: {} 个", self.warning_count())?;

        if self.has_errors() {
            writeln!(f, "\n错误详情:")?;
            for (i, err) in self.errors.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, err)?;
            }
        }

        if self.has_warnings() {
            writeln!(f, "\n警告详情:")?;
            for (i, warn) in self.warnings.iter().enumerate() {
                writeln!(f, "  {}. {}", i + 1, warn)?;
            }
        }

        Ok(())
    }
}

/// 验证错误类型
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// 非有限值
    NonFinite {
        /// 字段名称
        field: &'static str,
        /// 所在单元 ID
        cell_id: usize,
        /// 非有限的数值
        value: f64,
    },
    /// 数据超出范围
    OutOfRange {
        /// 字段名称
        field: &'static str,
        /// 所在单元 ID
        cell_id: usize,
        /// 实际值
        value: f64,
        /// 下界
        min: f64,
        /// 上界
        max: f64,
    },
    /// 拓扑错误
    TopologyError {
        /// 错误描述
        message: String,
        /// 可选的元素 ID
        element_id: Option<usize>,
    },
    /// 一致性错误
    ConsistencyError {
        /// 错误描述
        message: String,
    },
    /// 自定义错误
    Custom {
        /// 自定义消息
        message: String,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite {
                field,
                cell_id,
                value,
            } => {
                write!(f, "单元{}: 字段{}={} (非有限值)", cell_id, field, value)
            }
            Self::OutOfRange {
                field,
                cell_id,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "单元{}: 字段{}={} 超出范围[{}, {}]",
                    cell_id, field, value, min, max
                )
            }
            Self::TopologyError { message, element_id } => {
                if let Some(id) = element_id {
                    write!(f, "元素{}: 拓扑错误: {}", id, message)
                } else {
                    write!(f, "拓扑错误: {}", message)
                }
            }
            Self::ConsistencyError { message } => {
                write!(f, "一致性错误: {}", message)
            }
            Self::Custom { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// 验证警告类型
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    /// 高数值
    HighValue {
        /// 字段名称
        field: &'static str,
        /// 单元 ID
        cell_id: usize,
        /// 实际值
        value: f64,
        /// 阈值
        threshold: f64,
    },
    /// 低数值
    LowValue {
        /// 字段名称
        field: &'static str,
        /// 单元 ID
        cell_id: usize,
        /// 实际值
        value: f64,
        /// 阈值
        threshold: f64,
    },
    /// 质量警告
    QualityWarning {
        /// 警告描述
        message: String,
        /// 可选的元素 ID
        element_id: Option<usize>,
    },
    /// 自定义警告
    Custom {
        /// 自定义消息
        message: String,
    },
}

impl fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HighValue {
                field,
                cell_id,
                value,
                threshold,
            } => {
                write!(
                    f,
                    "单元{}: 字段{}={} 超过阈值{}",
                    cell_id, field, value, threshold
                )
            }
            Self::LowValue {
                field,
                cell_id,
                value,
                threshold,
            } => {
                write!(
                    f,
                    "单元{}: 字段{}={} 低于阈值{}",
                    cell_id, field, value, threshold
                )
            }
            Self::QualityWarning { message, element_id } => {
                if let Some(id) = element_id {
                    write!(f, "元素{}: 质量警告: {}", id, message)
                } else {
                    write!(f, "质量警告: {}", message)
                }
            }
            Self::Custom { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

// ============================================================================
// 验证辅助函数
// ============================================================================

/// 检查值是否有限
pub fn check_finite(
    report: &mut ValidationReport,
    field: &'static str,
    cell_id: usize,
    value: f64,
) -> bool {
    if !value.is_finite() {
        report.add_error(ValidationError::NonFinite {
            field,
            cell_id,
            value,
        });
        false
    } else {
        true
    }
}

/// 检查值是否在范围内
pub fn check_range(
    report: &mut ValidationReport,
    field: &'static str,
    cell_id: usize,
    value: f64,
    min: f64,
    max: f64,
) -> bool {
    if value < min || value > max {
        report.add_error(ValidationError::OutOfRange {
            field,
            cell_id,
            value,
            min,
            max,
        });
        false
    } else {
        true
    }
}

/// 检查值是否超过阈值并添加警告
pub fn warn_if_high(
    report: &mut ValidationReport,
    field: &'static str,
    cell_id: usize,
    value: f64,
    threshold: f64,
) -> bool {
    if value > threshold {
        report.add_warning(ValidationWarning::HighValue {
            field,
            cell_id,
            value,
            threshold,
        });
        true
    } else {
        false
    }
}

/// 检查值是否低于阈值并添加警告
pub fn warn_if_low(
    report: &mut ValidationReport,
    field: &'static str,
    cell_id: usize,
    value: f64,
    threshold: f64,
) -> bool {
    if value < threshold {
        report.add_warning(ValidationWarning::LowValue {
            field,
            cell_id,
            value,
            threshold,
        });
        true
    } else {
        false
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_report_new() {
        let report = ValidationReport::new();
        assert!(!report.has_errors());
        assert!(!report.has_warnings());
        assert!(report.is_valid());
    }

    #[test]
    fn test_validation_report_add_error() {
        let mut report = ValidationReport::new();
        report.add_error(ValidationError::Custom {
            message: "test error".into(),
        });

        assert!(report.has_errors());
        assert_eq!(report.error_count(), 1);
        assert!(!report.is_valid());
    }

    #[test]
    fn test_validation_report_add_warning() {
        let mut report = ValidationReport::new();
        report.add_warning(ValidationWarning::Custom {
            message: "test warning".into(),
        });

        assert!(report.has_warnings());
        assert_eq!(report.warning_count(), 1);
        // 警告不影响有效性
        assert!(report.is_valid());
    }

    #[test]
    fn test_validation_report_merge() {
        let mut report1 = ValidationReport::new();
        report1.add_error(ValidationError::Custom {
            message: "error 1".into(),
        });

        let mut report2 = ValidationReport::new();
        report2.add_error(ValidationError::Custom {
            message: "error 2".into(),
        });
        report2.add_warning(ValidationWarning::Custom {
            message: "warning 1".into(),
        });

        report1.merge(report2);
        assert_eq!(report1.error_count(), 2);
        assert_eq!(report1.warning_count(), 1);
    }

    #[test]
    fn test_validation_report_clear() {
        let mut report = ValidationReport::new();
        report.add_error(ValidationError::Custom {
            message: "error".into(),
        });
        report.add_warning(ValidationWarning::Custom {
            message: "warning".into(),
        });

        report.clear();
        assert!(!report.has_errors());
        assert!(!report.has_warnings());
    }

    #[test]
    fn test_check_finite() {
        let mut report = ValidationReport::new();

        assert!(check_finite(&mut report, "h", 0, 1.0));
        assert!(!report.has_errors());

        assert!(!check_finite(&mut report, "h", 0, f64::NAN));
        assert!(report.has_errors());
    }

    #[test]
    fn test_check_range() {
        let mut report = ValidationReport::new();

        assert!(check_range(&mut report, "h", 0, 5.0, 0.0, 10.0));
        assert!(!report.has_errors());

        assert!(!check_range(&mut report, "h", 0, -1.0, 0.0, 10.0));
        assert!(report.has_errors());
    }

    #[test]
    fn test_warn_if_high() {
        let mut report = ValidationReport::new();

        assert!(!warn_if_high(&mut report, "speed", 0, 5.0, 10.0));
        assert!(!report.has_warnings());

        assert!(warn_if_high(&mut report, "speed", 0, 15.0, 10.0));
        assert!(report.has_warnings());
    }

    #[test]
    fn test_warn_if_low() {
        let mut report = ValidationReport::new();

        assert!(!warn_if_low(&mut report, "depth", 0, 5.0, 1.0));
        assert!(!report.has_warnings());

        assert!(warn_if_low(&mut report, "depth", 0, 0.5, 1.0));
        assert!(report.has_warnings());
    }

    #[test]
    fn test_error_display() {
        let err = ValidationError::NonFinite {
            field: "h",
            cell_id: 42,
            value: f64::NAN,
        };
        let s = format!("{}", err);
        assert!(s.contains("42"));
        assert!(s.contains("h"));
    }

    #[test]
    fn test_warning_display() {
        let warn = ValidationWarning::HighValue {
            field: "speed",
            cell_id: 10,
            value: 100.0,
            threshold: 50.0,
        };
        let s = format!("{}", warn);
        assert!(s.contains("10"));
        assert!(s.contains("speed"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_report_display() {
        let mut report = ValidationReport::new();
        report.add_error(ValidationError::Custom {
            message: "test error".into(),
        });
        report.add_warning(ValidationWarning::Custom {
            message: "test warning".into(),
        });

        let s = format!("{}", report);
        assert!(s.contains("错误: 1 个"));
        assert!(s.contains("警告: 1 个"));
    }
}
