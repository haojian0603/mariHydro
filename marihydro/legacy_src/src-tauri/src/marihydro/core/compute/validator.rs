//! GPU验证工具
//!
//! 提供GPU计算结果验证和精度比较功能

use std::fmt;

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// 是否通过
    pub passed: bool,
    /// 最大绝对误差
    pub max_abs_error: f64,
    /// 平均绝对误差
    pub avg_abs_error: f64,
    /// 最大相对误差
    pub max_rel_error: f64,
    /// 平均相对误差
    pub avg_rel_error: f64,
    /// L2范数误差
    pub l2_error: f64,
    /// 不通过的元素数量
    pub failed_count: usize,
    /// 总元素数量
    pub total_count: usize,
    /// 字段名称
    pub field_name: String,
}

impl ValidationResult {
    /// 检查是否在容差范围内
    pub fn is_within_tolerance(&self, abs_tol: f64, rel_tol: f64) -> bool {
        self.max_abs_error <= abs_tol || self.max_rel_error <= rel_tol
    }
    
    /// 通过率
    pub fn pass_rate(&self) -> f64 {
        if self.total_count == 0 {
            1.0
        } else {
            (self.total_count - self.failed_count) as f64 / self.total_count as f64
        }
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} | max_abs={:.2e} avg_abs={:.2e} max_rel={:.2e}% L2={:.2e} pass_rate={:.1}%",
            self.field_name,
            if self.passed { "✓" } else { "✗" },
            self.max_abs_error,
            self.avg_abs_error,
            self.max_rel_error * 100.0,
            self.l2_error,
            self.pass_rate() * 100.0
        )
    }
}

/// 验证器
pub struct GpuValidator {
    /// 绝对容差
    abs_tolerance: f64,
    /// 相对容差
    rel_tolerance: f64,
    /// 忽略小值的阈值
    small_value_threshold: f64,
}

impl GpuValidator {
    /// 创建验证器
    pub fn new(abs_tolerance: f64, rel_tolerance: f64) -> Self {
        Self {
            abs_tolerance,
            rel_tolerance,
            small_value_threshold: 1e-10,
        }
    }
    
    /// 设置小值阈值
    pub fn with_small_value_threshold(mut self, threshold: f64) -> Self {
        self.small_value_threshold = threshold;
        self
    }
    
    /// 比较两个标量数组
    pub fn compare_scalars(&self, name: &str, cpu: &[f64], gpu: &[f32]) -> ValidationResult {
        let n = cpu.len().min(gpu.len());
        if n == 0 {
            return ValidationResult {
                passed: true,
                max_abs_error: 0.0,
                avg_abs_error: 0.0,
                max_rel_error: 0.0,
                avg_rel_error: 0.0,
                l2_error: 0.0,
                failed_count: 0,
                total_count: 0,
                field_name: name.to_string(),
            };
        }
        
        let mut max_abs = 0.0_f64;
        let mut sum_abs = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut sum_rel = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut failed = 0_usize;
        let mut rel_count = 0_usize;
        
        for i in 0..n {
            let cpu_val = cpu[i];
            let gpu_val = gpu[i] as f64;
            let abs_err = (cpu_val - gpu_val).abs();
            
            max_abs = max_abs.max(abs_err);
            sum_abs += abs_err;
            sum_sq += abs_err * abs_err;
            
            // 相对误差（仅对较大值计算）
            if cpu_val.abs() > self.small_value_threshold {
                let rel_err = abs_err / cpu_val.abs();
                max_rel = max_rel.max(rel_err);
                sum_rel += rel_err;
                rel_count += 1;
            }
            
            // 检查是否通过
            let passes = abs_err <= self.abs_tolerance 
                || (cpu_val.abs() > self.small_value_threshold 
                    && abs_err / cpu_val.abs() <= self.rel_tolerance);
            if !passes {
                failed += 1;
            }
        }
        
        let avg_abs = sum_abs / n as f64;
        let avg_rel = if rel_count > 0 { sum_rel / rel_count as f64 } else { 0.0 };
        let l2 = (sum_sq / n as f64).sqrt();
        let passed = failed == 0 || (failed as f64 / n as f64) < 0.001; // 允许0.1%的失败
        
        ValidationResult {
            passed,
            max_abs_error: max_abs,
            avg_abs_error: avg_abs,
            max_rel_error: max_rel,
            avg_rel_error: avg_rel,
            l2_error: l2,
            failed_count: failed,
            total_count: n,
            field_name: name.to_string(),
        }
    }
    
    /// 比较f32数组
    pub fn compare_f32(&self, name: &str, reference: &[f32], test: &[f32]) -> ValidationResult {
        let n = reference.len().min(test.len());
        if n == 0 {
            return ValidationResult {
                passed: true,
                max_abs_error: 0.0,
                avg_abs_error: 0.0,
                max_rel_error: 0.0,
                avg_rel_error: 0.0,
                l2_error: 0.0,
                failed_count: 0,
                total_count: 0,
                field_name: name.to_string(),
            };
        }
        
        let ref_f64: Vec<f64> = reference.iter().map(|&x| x as f64).collect();
        self.compare_scalars(name, &ref_f64, test)
    }
    
    /// 验证守恒性
    pub fn check_conservation(&self, initial: &[f64], final_val: &[f64]) -> ConservationResult {
        let initial_sum: f64 = initial.iter().sum();
        let final_sum: f64 = final_val.iter().sum();
        let change = (final_sum - initial_sum).abs();
        let relative_change = if initial_sum.abs() > 1e-10 {
            change / initial_sum.abs()
        } else {
            0.0
        };
        
        ConservationResult {
            initial_sum,
            final_sum,
            absolute_change: change,
            relative_change,
            is_conserved: relative_change < 1e-10,
        }
    }
    
    /// 检查NaN和Inf
    pub fn check_valid_numbers(&self, name: &str, data: &[f32]) -> NumberValidityResult {
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut first_nan_idx = None;
        let mut first_inf_idx = None;
        
        for (i, &val) in data.iter().enumerate() {
            if val.is_nan() {
                nan_count += 1;
                if first_nan_idx.is_none() {
                    first_nan_idx = Some(i);
                }
            } else if val.is_infinite() {
                inf_count += 1;
                if first_inf_idx.is_none() {
                    first_inf_idx = Some(i);
                }
            }
        }
        
        NumberValidityResult {
            field_name: name.to_string(),
            total_count: data.len(),
            nan_count,
            inf_count,
            is_valid: nan_count == 0 && inf_count == 0,
            first_nan_idx,
            first_inf_idx,
        }
    }
    
    /// 检查物理约束
    pub fn check_physical_constraints(&self, h: &[f32], hu: &[f32], hv: &[f32]) -> PhysicalConstraintResult {
        let mut negative_h_count = 0;
        let mut extreme_velocity_count = 0;
        let max_velocity = 100.0_f32; // 合理的最大速度限制
        
        for i in 0..h.len() {
            if h[i] < -1e-6 {
                negative_h_count += 1;
            }
            
            if h[i] > 1e-6 {
                let u = hu[i] / h[i];
                let v = hv[i] / h[i];
                let vel_mag = (u * u + v * v).sqrt();
                if vel_mag > max_velocity {
                    extreme_velocity_count += 1;
                }
            }
        }
        
        PhysicalConstraintResult {
            total_cells: h.len(),
            negative_h_count,
            extreme_velocity_count,
            is_valid: negative_h_count == 0 && extreme_velocity_count == 0,
        }
    }
}

impl Default for GpuValidator {
    fn default() -> Self {
        Self::new(1e-5, 1e-4)
    }
}

/// 守恒性检查结果
#[derive(Debug, Clone)]
pub struct ConservationResult {
    pub initial_sum: f64,
    pub final_sum: f64,
    pub absolute_change: f64,
    pub relative_change: f64,
    pub is_conserved: bool,
}

impl fmt::Display for ConservationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "守恒性: {} | 初始={:.6e} 最终={:.6e} 变化={:.2e} ({:.2e}%)",
            if self.is_conserved { "✓" } else { "✗" },
            self.initial_sum,
            self.final_sum,
            self.absolute_change,
            self.relative_change * 100.0
        )
    }
}

/// 数值有效性检查结果
#[derive(Debug, Clone)]
pub struct NumberValidityResult {
    pub field_name: String,
    pub total_count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub is_valid: bool,
    pub first_nan_idx: Option<usize>,
    pub first_inf_idx: Option<usize>,
}

impl fmt::Display for NumberValidityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid {
            write!(f, "{}: ✓ 所有值有效", self.field_name)
        } else {
            write!(
                f,
                "{}: ✗ NaN={} Inf={} (共{}个值)",
                self.field_name, self.nan_count, self.inf_count, self.total_count
            )
        }
    }
}

/// 物理约束检查结果
#[derive(Debug, Clone)]
pub struct PhysicalConstraintResult {
    pub total_cells: usize,
    pub negative_h_count: usize,
    pub extreme_velocity_count: usize,
    pub is_valid: bool,
}

impl fmt::Display for PhysicalConstraintResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid {
            write!(f, "物理约束: ✓")
        } else {
            write!(
                f,
                "物理约束: ✗ 负水深={} 极端速度={}",
                self.negative_h_count, self.extreme_velocity_count
            )
        }
    }
}

/// 综合验证报告
#[derive(Debug)]
pub struct ValidationReport {
    /// 字段验证结果
    pub field_results: Vec<ValidationResult>,
    /// 守恒性结果
    pub conservation: Option<ConservationResult>,
    /// 数值有效性结果
    pub validity: Vec<NumberValidityResult>,
    /// 物理约束结果
    pub physical: Option<PhysicalConstraintResult>,
    /// 总体是否通过
    pub overall_passed: bool,
}

impl ValidationReport {
    /// 创建新报告
    pub fn new() -> Self {
        Self {
            field_results: Vec::new(),
            conservation: None,
            validity: Vec::new(),
            physical: None,
            overall_passed: true,
        }
    }
    
    /// 添加字段验证结果
    pub fn add_field(&mut self, result: ValidationResult) {
        if !result.passed {
            self.overall_passed = false;
        }
        self.field_results.push(result);
    }
    
    /// 设置守恒性结果
    pub fn set_conservation(&mut self, result: ConservationResult) {
        if !result.is_conserved {
            self.overall_passed = false;
        }
        self.conservation = Some(result);
    }
    
    /// 添加有效性结果
    pub fn add_validity(&mut self, result: NumberValidityResult) {
        if !result.is_valid {
            self.overall_passed = false;
        }
        self.validity.push(result);
    }
    
    /// 设置物理约束结果
    pub fn set_physical(&mut self, result: PhysicalConstraintResult) {
        if !result.is_valid {
            self.overall_passed = false;
        }
        self.physical = Some(result);
    }
    
    /// 生成报告字符串
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        report.push_str("=== GPU验证报告 ===\n\n");
        
        report.push_str(&format!("总体结果: {}\n\n", 
            if self.overall_passed { "✓ 通过" } else { "✗ 失败" }));
        
        if !self.field_results.is_empty() {
            report.push_str("字段精度验证:\n");
            for result in &self.field_results {
                report.push_str(&format!("  {}\n", result));
            }
            report.push_str("\n");
        }
        
        if !self.validity.is_empty() {
            report.push_str("数值有效性:\n");
            for result in &self.validity {
                report.push_str(&format!("  {}\n", result));
            }
            report.push_str("\n");
        }
        
        if let Some(ref phys) = self.physical {
            report.push_str(&format!("{}\n\n", phys));
        }
        
        if let Some(ref cons) = self.conservation {
            report.push_str(&format!("{}\n", cons));
        }
        
        report
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validator_identical() {
        let validator = GpuValidator::new(1e-6, 1e-5);
        let cpu: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        
        let result = validator.compare_scalars("test", &cpu, &gpu);
        assert!(result.passed);
        assert!(result.max_abs_error < 1e-6);
    }
    
    #[test]
    fn test_validator_with_error() {
        let validator = GpuValidator::new(1e-3, 1e-2);
        let cpu: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let gpu: Vec<f32> = vec![1.001, 2.002, 3.003, 4.004];
        
        let result = validator.compare_scalars("test", &cpu, &gpu);
        assert!(result.passed);
        assert!(result.max_abs_error < 0.005);
    }
    
    #[test]
    fn test_check_nan() {
        let validator = GpuValidator::default();
        let data = vec![1.0, 2.0, f32::NAN, 4.0];
        
        let result = validator.check_valid_numbers("test", &data);
        assert!(!result.is_valid);
        assert_eq!(result.nan_count, 1);
        assert_eq!(result.first_nan_idx, Some(2));
    }
    
    #[test]
    fn test_conservation() {
        let validator = GpuValidator::default();
        let initial = vec![100.0, 200.0, 300.0];
        let final_val = vec![100.0, 200.0, 300.0];
        
        let result = validator.check_conservation(&initial, &final_val);
        assert!(result.is_conserved);
    }
    
    #[test]
    fn test_physical_constraints() {
        let validator = GpuValidator::default();
        let h = vec![1.0, 2.0, 0.5];
        let hu = vec![1.0, 2.0, 0.5];
        let hv = vec![0.5, 1.0, 0.25];
        
        let result = validator.check_physical_constraints(&h, &hu, &hv);
        assert!(result.is_valid);
    }
}
