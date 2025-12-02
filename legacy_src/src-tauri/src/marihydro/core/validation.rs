// src-tauri/src/marihydro/core/validation.rs

//! 运行时验证工具

use super::{MhError, MhResult};

/// 验证报告
#[derive(Debug, Default)]
pub struct ValidationReport {
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    pub fn error_count(&self) -> usize {
        self.errors.len()
    }

    pub fn warning_count(&self) -> usize {
        self.warnings.len()
    }
}

#[derive(Debug)]
pub enum ValidationError {
    NonFinite {
        field: &'static str,
        cell_id: usize,
        value: f64,
    },
    NegativeDepth {
        cell_id: usize,
        value: f64,
    },
    OutOfRange {
        field: &'static str,
        cell_id: usize,
        value: f64,
        min: f64,
        max: f64,
    },
}

#[derive(Debug)]
pub enum ValidationWarning {
    HighVelocity {
        cell_id: usize,
        speed: f64,
        limit: f64,
    },
    SmallDepth {
        cell_id: usize,
        depth: f64,
        threshold: f64,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite {
                field,
                cell_id,
                value,
            } => {
                write!(f, "单元{}: 字段{}={} (非有限)", cell_id, field, value)
            }
            Self::NegativeDepth { cell_id, value } => {
                write!(f, "单元{}: 负水深h={}", cell_id, value)
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
                    "单元{}: 字段{}={}超出范围[{}, {}]",
                    cell_id, field, value, min, max
                )
            }
        }
    }
}

impl std::fmt::Display for ValidationWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighVelocity {
                cell_id,
                speed,
                limit,
            } => {
                write!(f, "单元{}: 速度{}超过限制{}", cell_id, speed, limit)
            }
            Self::SmallDepth {
                cell_id,
                depth,
                threshold,
            } => {
                write!(f, "单元{}: 水深{}接近阈值{}", cell_id, depth, threshold)
            }
        }
    }
}
