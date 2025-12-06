// crates/mh_physics/src/gpu/validator.rs

//! GPU 数据验证器
//!
//! 验证 GPU 计算结果的正确性和数值稳定性。
//!
//! # 功能
//!
//! - NaN/Inf 检测
//! - 守恒定律验证
//! - GPU vs CPU 结果对比
//! - 数值精度分析
//! - 异常值检测
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::gpu::validator::{GpuValidator, ValidationConfig};
//!
//! let validator = GpuValidator::new(ValidationConfig::default());
//!
//! // 验证 GPU 计算结果
//! let result = validator.validate_state(&gpu_state);
//! if !result.is_valid {
//!     eprintln!("验证失败: {:?}", result.errors);
//! }
//! ```

use std::collections::HashMap;

// ============================================================================
// 验证配置
// ============================================================================

/// 验证配置
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// 是否检查 NaN
    pub check_nan: bool,

    /// 是否检查 Inf
    pub check_inf: bool,

    /// 是否检查负水深
    pub check_negative_depth: bool,

    /// 是否检查负浓度
    pub check_negative_concentration: bool,

    /// 是否验证质量守恒
    pub check_mass_conservation: bool,

    /// 是否验证动量守恒
    pub check_momentum_conservation: bool,

    /// 质量守恒相对容差
    pub mass_tolerance: f64,

    /// 动量守恒相对容差
    pub momentum_tolerance: f64,

    /// 最大允许速度 (m/s)
    pub max_velocity: f64,

    /// 最大允许水深 (m)
    pub max_depth: f64,

    /// 最小允许水深 (m)（小于此值视为干）
    pub min_depth: f64,

    /// 异常值检测阈值（标准差的倍数）
    pub outlier_threshold: f64,

    /// 是否启用 GPU vs CPU 对比
    pub enable_cpu_comparison: bool,

    /// GPU vs CPU 对比的相对容差
    pub comparison_tolerance: f64,

    /// 采样率（0-1，用于大规模验证）
    pub sampling_rate: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            check_negative_depth: true,
            check_negative_concentration: true,
            check_mass_conservation: true,
            check_momentum_conservation: false, // 动量守恒检查较耗时
            mass_tolerance: 1e-6,
            momentum_tolerance: 1e-4,
            max_velocity: 100.0,   // 100 m/s
            max_depth: 1000.0,     // 1000 m
            min_depth: 1e-6,       // 1 μm
            outlier_threshold: 5.0, // 5 倍标准差
            enable_cpu_comparison: false,
            comparison_tolerance: 1e-5,
            sampling_rate: 1.0, // 默认全量检查
        }
    }
}

impl ValidationConfig {
    /// 严格验证配置
    pub fn strict() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            check_negative_depth: true,
            check_negative_concentration: true,
            check_mass_conservation: true,
            check_momentum_conservation: true,
            mass_tolerance: 1e-10,
            momentum_tolerance: 1e-8,
            max_velocity: 50.0,
            max_depth: 500.0,
            min_depth: 1e-8,
            outlier_threshold: 3.0,
            enable_cpu_comparison: true,
            comparison_tolerance: 1e-8,
            sampling_rate: 1.0,
        }
    }

    /// 快速验证配置
    pub fn fast() -> Self {
        Self {
            check_nan: true,
            check_inf: true,
            check_negative_depth: true,
            check_negative_concentration: false,
            check_mass_conservation: false,
            check_momentum_conservation: false,
            mass_tolerance: 1e-4,
            momentum_tolerance: 1e-3,
            max_velocity: 200.0,
            max_depth: 2000.0,
            min_depth: 1e-4,
            outlier_threshold: 10.0,
            enable_cpu_comparison: false,
            comparison_tolerance: 1e-4,
            sampling_rate: 0.01, // 1% 采样
        }
    }

    /// 禁用所有验证
    pub fn disabled() -> Self {
        Self {
            check_nan: false,
            check_inf: false,
            check_negative_depth: false,
            check_negative_concentration: false,
            check_mass_conservation: false,
            check_momentum_conservation: false,
            sampling_rate: 0.0,
            ..Default::default()
        }
    }
}

// ============================================================================
// 验证错误
// ============================================================================

/// 验证错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationErrorType {
    /// 发现 NaN 值
    NanDetected,
    /// 发现 Inf 值
    InfDetected,
    /// 负水深
    NegativeDepth,
    /// 负浓度
    NegativeConcentration,
    /// 速度超限
    VelocityExceeded,
    /// 水深超限
    DepthExceeded,
    /// 质量不守恒
    MassNotConserved,
    /// 动量不守恒
    MomentumNotConserved,
    /// 检测到异常值
    OutlierDetected,
    /// GPU vs CPU 结果不一致
    CpuGpuMismatch,
}

impl std::fmt::Display for ValidationErrorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationErrorType::NanDetected => write!(f, "NaN 检测"),
            ValidationErrorType::InfDetected => write!(f, "Inf 检测"),
            ValidationErrorType::NegativeDepth => write!(f, "负水深"),
            ValidationErrorType::NegativeConcentration => write!(f, "负浓度"),
            ValidationErrorType::VelocityExceeded => write!(f, "速度超限"),
            ValidationErrorType::DepthExceeded => write!(f, "水深超限"),
            ValidationErrorType::MassNotConserved => write!(f, "质量不守恒"),
            ValidationErrorType::MomentumNotConserved => write!(f, "动量不守恒"),
            ValidationErrorType::OutlierDetected => write!(f, "异常值"),
            ValidationErrorType::CpuGpuMismatch => write!(f, "GPU/CPU 不一致"),
        }
    }
}

/// 验证错误
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// 错误类型
    pub error_type: ValidationErrorType,
    /// 字段名称
    pub field: String,
    /// 受影响的单元索引
    pub cell_indices: Vec<usize>,
    /// 错误值（最多显示前几个）
    pub values: Vec<f64>,
    /// 错误消息
    pub message: String,
    /// 严重程度 [0, 1]
    pub severity: f64,
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}] {}: {} (影响 {} 个单元)",
            self.error_type,
            self.field,
            self.message,
            self.cell_indices.len()
        )
    }
}

// ============================================================================
// 验证结果
// ============================================================================

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// 是否通过验证
    pub is_valid: bool,
    /// 检测到的错误
    pub errors: Vec<ValidationError>,
    /// 警告（不影响有效性）
    pub warnings: Vec<ValidationError>,
    /// 统计信息
    pub stats: ValidationStats,
    /// 验证耗时（微秒）
    pub elapsed_us: u64,
}

impl ValidationResult {
    /// 创建有效结果
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            stats: ValidationStats::default(),
            elapsed_us: 0,
        }
    }

    /// 添加错误
    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }

    /// 添加警告
    pub fn add_warning(&mut self, warning: ValidationError) {
        self.warnings.push(warning);
    }

    /// 合并另一个结果
    pub fn merge(&mut self, other: ValidationResult) {
        self.is_valid = self.is_valid && other.is_valid;
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
        self.elapsed_us += other.elapsed_us;
    }
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_valid {
            write!(f, "✓ 验证通过")?;
        } else {
            write!(f, "✗ 验证失败 ({} 个错误)", self.errors.len())?;
        }

        if !self.warnings.is_empty() {
            write!(f, " ({} 个警告)", self.warnings.len())?;
        }

        writeln!(f)?;

        for error in &self.errors {
            writeln!(f, "  错误: {}", error)?;
        }

        for warning in &self.warnings {
            writeln!(f, "  警告: {}", warning)?;
        }

        Ok(())
    }
}

/// 验证统计
#[derive(Debug, Clone, Default)]
pub struct ValidationStats {
    /// 检查的单元数
    pub cells_checked: usize,
    /// 检查的字段数
    pub fields_checked: usize,
    /// NaN 数量
    pub nan_count: usize,
    /// Inf 数量
    pub inf_count: usize,
    /// 负水深数量
    pub negative_depth_count: usize,
    /// 异常值数量
    pub outlier_count: usize,
    /// 质量变化率
    pub mass_change_rate: f64,
    /// 动量变化率
    pub momentum_change_rate: f64,
    /// 最大速度
    pub max_velocity_found: f64,
    /// 最大水深
    pub max_depth_found: f64,
}

// ============================================================================
// GPU 数据验证器
// ============================================================================

/// GPU 数据验证器
///
/// 验证 GPU 计算结果的正确性。
pub struct GpuValidator {
    /// 配置
    config: ValidationConfig,

    /// 上一步的质量
    prev_total_mass: Option<f64>,

    /// 上一步的动量
    prev_total_momentum: Option<(f64, f64)>,
}

impl GpuValidator {
    /// 创建新的验证器
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            prev_total_mass: None,
            prev_total_momentum: None,
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &ValidationConfig {
        &self.config
    }

    /// 更新配置
    pub fn set_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }

    // =========================================================================
    // 基础验证
    // =========================================================================

    /// 验证单个数组是否包含 NaN/Inf
    pub fn validate_array(&self, name: &str, data: &[f64]) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::valid();

        if !self.config.check_nan && !self.config.check_inf {
            return result;
        }

        let mut nan_indices = Vec::new();
        let mut inf_indices = Vec::new();
        let mut nan_values = Vec::new();
        let mut inf_values = Vec::new();

        // 采样检查
        let step = if self.config.sampling_rate >= 1.0 {
            1
        } else {
            (1.0 / self.config.sampling_rate).ceil() as usize
        };

        for (i, &value) in data.iter().enumerate().step_by(step) {
            if self.config.check_nan && value.is_nan() {
                nan_indices.push(i);
                if nan_values.len() < 10 {
                    nan_values.push(value);
                }
            } else if self.config.check_inf && value.is_infinite() {
                inf_indices.push(i);
                if inf_values.len() < 10 {
                    inf_values.push(value);
                }
            }
        }

        if !nan_indices.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::NanDetected,
                field: name.to_string(),
                cell_indices: nan_indices.clone(),
                values: nan_values,
                message: format!("发现 {} 个 NaN 值", nan_indices.len()),
                severity: 1.0,
            });
        }

        if !inf_indices.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::InfDetected,
                field: name.to_string(),
                cell_indices: inf_indices.clone(),
                values: inf_values,
                message: format!("发现 {} 个 Inf 值", inf_indices.len()),
                severity: 1.0,
            });
        }

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result.stats.cells_checked = data.len();
        result.stats.fields_checked = 1;
        result.stats.nan_count = nan_indices.len();
        result.stats.inf_count = inf_indices.len();

        result
    }

    /// 验证水深数组
    pub fn validate_depth(&self, h: &[f64]) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = self.validate_array("h (水深)", h);

        if !self.config.check_negative_depth {
            return result;
        }

        let mut negative_indices = Vec::new();
        let mut negative_values = Vec::new();
        let mut max_depth = 0.0f64;

        let step = if self.config.sampling_rate >= 1.0 {
            1
        } else {
            (1.0 / self.config.sampling_rate).ceil() as usize
        };

        for (i, &value) in h.iter().enumerate().step_by(step) {
            if value < 0.0 {
                negative_indices.push(i);
                if negative_values.len() < 10 {
                    negative_values.push(value);
                }
            }
            max_depth = max_depth.max(value);
        }

        result.stats.max_depth_found = max_depth;
        result.stats.negative_depth_count = negative_indices.len();

        if !negative_indices.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::NegativeDepth,
                field: "h (水深)".to_string(),
                cell_indices: negative_indices.clone(),
                values: negative_values,
                message: format!("发现 {} 个负水深", negative_indices.len()),
                severity: 0.9,
            });
        }

        if max_depth > self.config.max_depth {
            result.add_warning(ValidationError {
                error_type: ValidationErrorType::DepthExceeded,
                field: "h (水深)".to_string(),
                cell_indices: vec![],
                values: vec![max_depth],
                message: format!(
                    "最大水深 {:.2}m 超过阈值 {:.2}m",
                    max_depth, self.config.max_depth
                ),
                severity: 0.5,
            });
        }

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    /// 验证速度数组
    pub fn validate_velocity(&self, u: &[f64], v: &[f64]) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = self.validate_array("u (x速度)", u);
        result.merge(self.validate_array("v (y速度)", v));

        let mut exceed_indices = Vec::new();
        let mut exceed_values = Vec::new();
        let mut max_vel = 0.0f64;

        let step = if self.config.sampling_rate >= 1.0 {
            1
        } else {
            (1.0 / self.config.sampling_rate).ceil() as usize
        };

        for (i, (&ui, &vi)) in u.iter().zip(v.iter()).enumerate().step_by(step) {
            let vel_mag = (ui * ui + vi * vi).sqrt();
            max_vel = max_vel.max(vel_mag);

            if vel_mag > self.config.max_velocity {
                exceed_indices.push(i);
                if exceed_values.len() < 10 {
                    exceed_values.push(vel_mag);
                }
            }
        }

        result.stats.max_velocity_found = max_vel;

        if !exceed_indices.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::VelocityExceeded,
                field: "velocity".to_string(),
                cell_indices: exceed_indices.clone(),
                values: exceed_values,
                message: format!(
                    "{} 个单元速度超过 {:.1} m/s",
                    exceed_indices.len(),
                    self.config.max_velocity
                ),
                severity: 0.8,
            });
        }

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    /// 验证浓度数组
    pub fn validate_concentration(&self, name: &str, c: &[f64]) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = self.validate_array(name, c);

        if !self.config.check_negative_concentration {
            return result;
        }

        let mut negative_indices = Vec::new();
        let mut negative_values = Vec::new();

        let step = if self.config.sampling_rate >= 1.0 {
            1
        } else {
            (1.0 / self.config.sampling_rate).ceil() as usize
        };

        for (i, &value) in c.iter().enumerate().step_by(step) {
            if value < 0.0 {
                negative_indices.push(i);
                if negative_values.len() < 10 {
                    negative_values.push(value);
                }
            }
        }

        if !negative_indices.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::NegativeConcentration,
                field: name.to_string(),
                cell_indices: negative_indices.clone(),
                values: negative_values,
                message: format!("发现 {} 个负浓度值", negative_indices.len()),
                severity: 0.7,
            });
        }

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    // =========================================================================
    // 守恒验证
    // =========================================================================

    /// 验证质量守恒
    ///
    /// # 参数
    ///
    /// * `h` - 水深数组
    /// * `cell_area` - 单元面积数组
    /// * `inflow` - 入流质量
    /// * `outflow` - 出流质量
    pub fn validate_mass_conservation(
        &mut self,
        h: &[f64],
        cell_area: &[f64],
        inflow: f64,
        outflow: f64,
    ) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::valid();

        if !self.config.check_mass_conservation {
            return result;
        }

        // 计算当前总质量
        let current_mass: f64 = h
            .iter()
            .zip(cell_area.iter())
            .map(|(&hi, &ai)| hi * ai)
            .sum();

        if let Some(prev_mass) = self.prev_total_mass {
            // 预期质量 = 上一步质量 + 入流 - 出流
            let expected_mass = prev_mass + inflow - outflow;
            let mass_error = (current_mass - expected_mass).abs();
            let relative_error = if expected_mass.abs() > 1e-10 {
                mass_error / expected_mass.abs()
            } else {
                mass_error
            };

            result.stats.mass_change_rate = relative_error;

            if relative_error > self.config.mass_tolerance {
                result.add_error(ValidationError {
                    error_type: ValidationErrorType::MassNotConserved,
                    field: "总质量".to_string(),
                    cell_indices: vec![],
                    values: vec![prev_mass, current_mass, expected_mass],
                    message: format!(
                        "质量变化 {:.6} ({:.2e}%)，超过容差 {:.2e}",
                        mass_error,
                        relative_error * 100.0,
                        self.config.mass_tolerance
                    ),
                    severity: 0.9,
                });
            }
        }

        // 更新上一步质量
        self.prev_total_mass = Some(current_mass);

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    /// 验证动量守恒
    pub fn validate_momentum_conservation(
        &mut self,
        hu: &[f64],
        hv: &[f64],
        cell_area: &[f64],
    ) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::valid();

        if !self.config.check_momentum_conservation {
            return result;
        }

        // 计算当前总动量
        let current_momentum_x: f64 = hu
            .iter()
            .zip(cell_area.iter())
            .map(|(&hui, &ai)| hui * ai)
            .sum();

        let current_momentum_y: f64 = hv
            .iter()
            .zip(cell_area.iter())
            .map(|(&hvi, &ai)| hvi * ai)
            .sum();

        if let Some((prev_mx, prev_my)) = self.prev_total_momentum {
            let change_x = (current_momentum_x - prev_mx).abs();
            let change_y = (current_momentum_y - prev_my).abs();
            let total_change = (change_x * change_x + change_y * change_y).sqrt();
            let total_prev = (prev_mx * prev_mx + prev_my * prev_my).sqrt();

            let relative_error = if total_prev > 1e-10 {
                total_change / total_prev
            } else {
                total_change
            };

            result.stats.momentum_change_rate = relative_error;

            if relative_error > self.config.momentum_tolerance {
                result.add_warning(ValidationError {
                    error_type: ValidationErrorType::MomentumNotConserved,
                    field: "总动量".to_string(),
                    cell_indices: vec![],
                    values: vec![prev_mx, prev_my, current_momentum_x, current_momentum_y],
                    message: format!(
                        "动量变化 {:.2e} ({:.2e}%)",
                        total_change,
                        relative_error * 100.0
                    ),
                    severity: 0.6,
                });
            }
        }

        self.prev_total_momentum = Some((current_momentum_x, current_momentum_y));

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    // =========================================================================
    // 异常值检测
    // =========================================================================

    /// 检测异常值
    pub fn detect_outliers(&self, name: &str, data: &[f64]) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::valid();

        if data.len() < 10 {
            return result;
        }

        // 计算均值和标准差
        let n = data.len() as f64;
        let mean: f64 = data.iter().sum::<f64>() / n;
        let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return result;
        }

        let threshold = self.config.outlier_threshold * std_dev;

        let mut outlier_indices = Vec::new();
        let mut outlier_values = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            if (value - mean).abs() > threshold {
                outlier_indices.push(i);
                if outlier_values.len() < 10 {
                    outlier_values.push(value);
                }
            }
        }

        result.stats.outlier_count = outlier_indices.len();

        // 如果异常值超过 1%，发出警告
        if outlier_indices.len() as f64 / n > 0.01 {
            result.add_warning(ValidationError {
                error_type: ValidationErrorType::OutlierDetected,
                field: name.to_string(),
                cell_indices: outlier_indices.clone(),
                values: outlier_values,
                message: format!(
                    "发现 {} 个异常值 (>{:.1}σ)，均值={:.4}, σ={:.4}",
                    outlier_indices.len(),
                    self.config.outlier_threshold,
                    mean,
                    std_dev
                ),
                severity: 0.4,
            });
        }

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    // =========================================================================
    // GPU vs CPU 对比
    // =========================================================================

    /// 对比 GPU 和 CPU 计算结果
    pub fn compare_gpu_cpu(&self, name: &str, gpu_data: &[f64], cpu_data: &[f64]) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::valid();

        if !self.config.enable_cpu_comparison {
            return result;
        }

        if gpu_data.len() != cpu_data.len() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::CpuGpuMismatch,
                field: name.to_string(),
                cell_indices: vec![],
                values: vec![gpu_data.len() as f64, cpu_data.len() as f64],
                message: format!(
                    "数组长度不匹配: GPU={}, CPU={}",
                    gpu_data.len(),
                    cpu_data.len()
                ),
                severity: 1.0,
            });
            return result;
        }

        let mut mismatch_indices = Vec::new();
        let mut mismatch_values = Vec::new();
        let mut max_rel_error = 0.0f64;

        for (i, (&gpu_val, &cpu_val)) in gpu_data.iter().zip(cpu_data.iter()).enumerate() {
            let abs_error = (gpu_val - cpu_val).abs();
            let rel_error = if cpu_val.abs() > 1e-10 {
                abs_error / cpu_val.abs()
            } else {
                abs_error
            };

            max_rel_error = max_rel_error.max(rel_error);

            if rel_error > self.config.comparison_tolerance {
                mismatch_indices.push(i);
                if mismatch_values.len() < 10 {
                    mismatch_values.push(rel_error);
                }
            }
        }

        if !mismatch_indices.is_empty() {
            result.add_error(ValidationError {
                error_type: ValidationErrorType::CpuGpuMismatch,
                field: name.to_string(),
                cell_indices: mismatch_indices.clone(),
                values: mismatch_values,
                message: format!(
                    "{} 个值不匹配，最大相对误差 {:.2e}",
                    mismatch_indices.len(),
                    max_rel_error
                ),
                severity: 0.8,
            });
        }

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    // =========================================================================
    // 综合验证
    // =========================================================================

    /// 验证完整状态
    ///
    /// 验证水深、速度等所有状态变量。
    pub fn validate_state(
        &mut self,
        h: &[f64],
        hu: &[f64],
        hv: &[f64],
        cell_area: &[f64],
    ) -> ValidationResult {
        let start = std::time::Instant::now();
        let mut result = ValidationResult::valid();

        // 验证水深
        result.merge(self.validate_depth(h));

        // 验证动量
        result.merge(self.validate_array("hu", hu));
        result.merge(self.validate_array("hv", hv));

        // 计算速度并验证
        let mut u = vec![0.0; h.len()];
        let mut v = vec![0.0; h.len()];

        for i in 0..h.len() {
            if h[i] > self.config.min_depth {
                u[i] = hu[i] / h[i];
                v[i] = hv[i] / h[i];
            }
        }

        result.merge(self.validate_velocity(&u, &v));

        // 验证守恒
        result.merge(self.validate_mass_conservation(h, cell_area, 0.0, 0.0));
        result.merge(self.validate_momentum_conservation(hu, hv, cell_area));

        // 异常值检测
        result.merge(self.detect_outliers("h", h));

        result.elapsed_us = start.elapsed().as_micros() as u64;
        result
    }

    /// 重置守恒追踪状态
    pub fn reset(&mut self) {
        self.prev_total_mass = None;
        self.prev_total_momentum = None;
    }

    /// 生成验证报告
    pub fn generate_report(&self, result: &ValidationResult) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════╗\n");
        report.push_str("║                    GPU 数据验证报告                          ║\n");
        report.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

        // 总体状态
        if result.is_valid {
            report.push_str("【状态】 ✓ 验证通过\n\n");
        } else {
            report.push_str(&format!(
                "【状态】 ✗ 验证失败 ({} 个错误)\n\n",
                result.errors.len()
            ));
        }

        // 统计信息
        report.push_str("【统计】\n");
        report.push_str(&format!(
            "  检查单元数: {}\n",
            result.stats.cells_checked
        ));
        report.push_str(&format!(
            "  NaN 数量: {}\n",
            result.stats.nan_count
        ));
        report.push_str(&format!(
            "  Inf 数量: {}\n",
            result.stats.inf_count
        ));
        report.push_str(&format!(
            "  负水深数量: {}\n",
            result.stats.negative_depth_count
        ));
        report.push_str(&format!(
            "  最大速度: {:.2} m/s\n",
            result.stats.max_velocity_found
        ));
        report.push_str(&format!(
            "  最大水深: {:.2} m\n",
            result.stats.max_depth_found
        ));
        report.push_str(&format!(
            "  验证耗时: {:.2} ms\n\n",
            result.elapsed_us as f64 / 1000.0
        ));

        // 错误详情
        if !result.errors.is_empty() {
            report.push_str("【错误详情】\n");
            for (i, error) in result.errors.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. [{}] {}: {}\n",
                    i + 1,
                    error.error_type,
                    error.field,
                    error.message
                ));
                if !error.values.is_empty() {
                    report.push_str(&format!(
                        "     示例值: {:?}\n",
                        &error.values[..error.values.len().min(5)]
                    ));
                }
            }
            report.push('\n');
        }

        // 警告详情
        if !result.warnings.is_empty() {
            report.push_str("【警告】\n");
            for warning in &result.warnings {
                report.push_str(&format!(
                    "  ⚠ [{}] {}: {}\n",
                    warning.error_type, warning.field, warning.message
                ));
            }
        }

        report
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nan_detection() {
        let validator = GpuValidator::new(ValidationConfig::default());
        let data = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];

        let result = validator.validate_array("test", &data);

        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].error_type, ValidationErrorType::NanDetected);
    }

    #[test]
    fn test_inf_detection() {
        let validator = GpuValidator::new(ValidationConfig::default());
        let data = vec![1.0, f64::INFINITY, 3.0, f64::NEG_INFINITY, 5.0];

        let result = validator.validate_array("test", &data);

        assert!(!result.is_valid);
        assert_eq!(result.stats.inf_count, 2);
    }

    #[test]
    fn test_negative_depth() {
        let validator = GpuValidator::new(ValidationConfig::default());
        let h = vec![1.0, 2.0, -0.5, 3.0, -1.0];

        let result = validator.validate_depth(&h);

        assert!(!result.is_valid);
        assert_eq!(result.stats.negative_depth_count, 2);
    }

    #[test]
    fn test_velocity_exceeded() {
        let mut config = ValidationConfig::default();
        config.max_velocity = 10.0;
        let validator = GpuValidator::new(config);

        let u = vec![5.0, 8.0, 0.0];
        let v = vec![0.0, 8.0, 0.0]; // 第二个点速度 = sqrt(64+64) ≈ 11.3

        let result = validator.validate_velocity(&u, &v);

        assert!(!result.is_valid);
    }

    #[test]
    fn test_mass_conservation() {
        let mut validator = GpuValidator::new(ValidationConfig::default());
        let cell_area = vec![1.0, 1.0, 1.0];

        // 第一步：初始化
        let h1 = vec![1.0, 2.0, 3.0];
        let result1 = validator.validate_mass_conservation(&h1, &cell_area, 0.0, 0.0);
        assert!(result1.is_valid);

        // 第二步：质量守恒
        let h2 = vec![1.5, 2.5, 2.0]; // 总质量仍为 6.0
        let result2 = validator.validate_mass_conservation(&h2, &cell_area, 0.0, 0.0);
        assert!(result2.is_valid);

        // 第三步：质量不守恒
        let h3 = vec![1.0, 1.0, 1.0]; // 总质量 3.0，减少了 3.0
        let result3 = validator.validate_mass_conservation(&h3, &cell_area, 0.0, 0.0);
        assert!(!result3.is_valid);
    }

    #[test]
    fn test_outlier_detection() {
        let validator = GpuValidator::new(ValidationConfig::default());
        let mut data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        data[50] = 1000.0; // 明显的异常值

        let result = validator.detect_outliers("test", &data);

        assert!(result.stats.outlier_count > 0);
    }

    #[test]
    fn test_cpu_gpu_comparison() {
        let mut config = ValidationConfig::default();
        config.enable_cpu_comparison = true;
        config.comparison_tolerance = 1e-6;
        let validator = GpuValidator::new(config);

        let gpu_data = vec![1.0, 2.0, 3.0, 4.0];
        let cpu_data = vec![1.0, 2.0, 3.0001, 4.0];

        let result = validator.compare_gpu_cpu("test", &gpu_data, &cpu_data);

        assert!(!result.is_valid); // 3.0 vs 3.0001 误差超过 1e-6
    }

    #[test]
    fn test_valid_state() {
        let mut validator = GpuValidator::new(ValidationConfig::default());

        let h = vec![1.0, 2.0, 3.0];
        let hu = vec![0.5, 1.0, 1.5];
        let hv = vec![0.1, 0.2, 0.3];
        let cell_area = vec![1.0, 1.0, 1.0];

        let result = validator.validate_state(&h, &hu, &hv, &cell_area);

        assert!(result.is_valid);
    }

    #[test]
    fn test_disabled_validation() {
        let validator = GpuValidator::new(ValidationConfig::disabled());
        let data = vec![f64::NAN, f64::INFINITY, -1.0];

        let result = validator.validate_array("test", &data);

        assert!(result.is_valid); // 禁用时不检查
    }

    #[test]
    fn test_sampling() {
        let mut config = ValidationConfig::default();
        config.sampling_rate = 0.1; // 10% 采样
        let validator = GpuValidator::new(config);

        let data = vec![1.0; 1000];
        let result = validator.validate_array("test", &data);

        // 采样检查，应该检查约 100 个
        assert!(result.stats.cells_checked <= 1000);
    }

    #[test]
    fn test_report_generation() {
        let mut validator = GpuValidator::new(ValidationConfig::default());
        let h = vec![1.0, -0.5, f64::NAN];
        let hu = vec![0.0; 3];
        let hv = vec![0.0; 3];
        let cell_area = vec![1.0; 3];

        let result = validator.validate_state(&h, &hu, &hv, &cell_area);
        let report = validator.generate_report(&result);

        assert!(report.contains("验证报告"));
        assert!(report.contains("验证失败"));
    }
}
