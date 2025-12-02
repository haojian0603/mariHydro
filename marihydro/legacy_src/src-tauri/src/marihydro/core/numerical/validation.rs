//! 数值验证工具
//!
//! 提供数组和矩阵的数值有效性验证功能。

/// 数值验证结果
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// 是否有效
    pub is_valid: bool,
    /// NaN 计数
    pub nan_count: usize,
    /// Inf 计数
    pub inf_count: usize,
    /// 负数计数（对于必须非负的量）
    pub negative_count: usize,
    /// 问题值的索引列表（最多记录前 10 个）
    pub problem_indices: Vec<usize>,
}

impl ValidationResult {
    /// 创建有效结果
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            ..Default::default()
        }
    }

    /// 创建无效结果
    pub fn invalid() -> Self {
        Self {
            is_valid: false,
            ..Default::default()
        }
    }

    /// 合并两个验证结果
    pub fn merge(&mut self, other: &ValidationResult) {
        self.is_valid = self.is_valid && other.is_valid;
        self.nan_count += other.nan_count;
        self.inf_count += other.inf_count;
        self.negative_count += other.negative_count;
        // 限制问题索引数量
        if self.problem_indices.len() < 10 {
            let remaining = 10 - self.problem_indices.len();
            self.problem_indices
                .extend(other.problem_indices.iter().take(remaining));
        }
    }

    /// 获取问题描述
    pub fn describe(&self) -> String {
        if self.is_valid {
            "数值验证通过".to_string()
        } else {
            format!(
                "数值验证失败: {} NaN, {} Inf, {} 负数",
                self.nan_count, self.inf_count, self.negative_count
            )
        }
    }
}

/// 验证数组中的数值有效性
///
/// # Arguments
/// * `arr` - 要验证的数组
/// * `allow_negative` - 是否允许负数
///
/// # Returns
/// 验证结果
pub fn validate_array(arr: &[f64], allow_negative: bool) -> ValidationResult {
    let mut result = ValidationResult {
        is_valid: true,
        nan_count: 0,
        inf_count: 0,
        negative_count: 0,
        problem_indices: Vec::new(),
    };

    for (i, &v) in arr.iter().enumerate() {
        let mut has_problem = false;

        if v.is_nan() {
            result.nan_count += 1;
            result.is_valid = false;
            has_problem = true;
        } else if v.is_infinite() {
            result.inf_count += 1;
            result.is_valid = false;
            has_problem = true;
        } else if !allow_negative && v < 0.0 {
            result.negative_count += 1;
            result.is_valid = false;
            has_problem = true;
        }

        if has_problem && result.problem_indices.len() < 10 {
            result.problem_indices.push(i);
        }
    }

    result
}

/// 验证数组中所有值都是有限的
#[inline]
pub fn all_finite(arr: &[f64]) -> bool {
    arr.iter().all(|v| v.is_finite())
}

/// 验证数组中所有值都是非负有限的
#[inline]
pub fn all_non_negative_finite(arr: &[f64]) -> bool {
    arr.iter().all(|v| v.is_finite() && *v >= 0.0)
}

/// 验证数组中所有值都是正有限的
#[inline]
pub fn all_positive_finite(arr: &[f64]) -> bool {
    arr.iter().all(|v| v.is_finite() && *v > 0.0)
}

/// 检验 2x2 矩阵条件数
///
/// # Returns
/// 如果矩阵奇异返回 None，否则返回条件数估计
pub fn check_matrix_condition_2x2(a: f64, b: f64, c: f64, d: f64) -> Option<f64> {
    let det = a * d - b * c;
    let max_elem = a.abs().max(b.abs()).max(c.abs()).max(d.abs());

    // 相对阈值
    let threshold = max_elem * max_elem * 1e-12;
    if det.abs() < threshold.max(1e-15) {
        return None;
    }

    // 返回条件数估计 (无穷范数)
    let norm_a = (a.abs() + b.abs()).max(c.abs() + d.abs());
    let inv_a = d / det;
    let inv_b = -b / det;
    let inv_c = -c / det;
    let inv_d = a / det;
    let norm_inv = (inv_a.abs() + inv_b.abs()).max(inv_c.abs() + inv_d.abs());

    Some(norm_a * norm_inv)
}

/// 安全除法：除零返回默认值
#[inline]
pub fn safe_div(num: f64, den: f64, default: f64) -> f64 {
    if den.abs() < 1e-14 {
        default
    } else {
        let result = num / den;
        if result.is_finite() {
            result
        } else {
            default
        }
    }
}

/// 安全除法：除零返回零
#[inline]
pub fn safe_div_zero(num: f64, den: f64) -> f64 {
    safe_div(num, den, 0.0)
}

/// 安全开方：负数返回零
#[inline]
pub fn safe_sqrt(value: f64) -> f64 {
    if value >= 0.0 {
        value.sqrt()
    } else {
        0.0
    }
}

/// 安全除法宏
#[macro_export]
macro_rules! safe_div {
    ($num:expr, $den:expr) => {
        $crate::core::numerical::validation::safe_div($num, $den, 0.0)
    };
    ($num:expr, $den:expr, $default:expr) => {
        $crate::core::numerical::validation::safe_div($num, $den, $default)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_array_valid() {
        let arr = vec![1.0, 2.0, 3.0, 4.0];
        let result = validate_array(&arr, true);
        assert!(result.is_valid);
        assert_eq!(result.nan_count, 0);
        assert_eq!(result.inf_count, 0);
    }

    #[test]
    fn test_validate_array_with_nan() {
        let arr = vec![1.0, f64::NAN, 3.0];
        let result = validate_array(&arr, true);
        assert!(!result.is_valid);
        assert_eq!(result.nan_count, 1);
        assert_eq!(result.problem_indices, vec![1]);
    }

    #[test]
    fn test_validate_array_with_inf() {
        let arr = vec![1.0, f64::INFINITY, f64::NEG_INFINITY];
        let result = validate_array(&arr, true);
        assert!(!result.is_valid);
        assert_eq!(result.inf_count, 2);
    }

    #[test]
    fn test_validate_array_no_negative() {
        let arr = vec![1.0, -2.0, 3.0];
        let result = validate_array(&arr, false);
        assert!(!result.is_valid);
        assert_eq!(result.negative_count, 1);
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(safe_div(10.0, 2.0, 0.0), 5.0);
        assert_eq!(safe_div(10.0, 0.0, 999.0), 999.0);
        assert_eq!(safe_div(10.0, 1e-20, 0.0), 0.0); // 接近零
    }

    #[test]
    fn test_matrix_condition() {
        // 良态矩阵
        let cond = check_matrix_condition_2x2(1.0, 0.0, 0.0, 1.0);
        assert!(cond.is_some());
        assert!((cond.unwrap() - 1.0).abs() < 1e-10);

        // 奇异矩阵
        let cond = check_matrix_condition_2x2(1.0, 2.0, 2.0, 4.0);
        assert!(cond.is_none());
    }
}
