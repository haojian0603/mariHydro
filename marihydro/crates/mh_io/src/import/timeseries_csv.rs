// crates/mh_io/src/import/timeseries_csv.rs

//! CSV 时序数据导入
//!
//! 提供从 CSV 文件加载时间序列数据的功能，支持：
//! - 灵活的列配置
//! - 多种分隔符
//! - 错误行跳过
//! - 数据验证
//!
//! # 使用示例
//!
//! ```ignore
//! use std::path::Path;
//! use mh_io::import::timeseries_csv::{load_timeseries, CsvConfig};
//!
//! let config = CsvConfig {
//!     has_header: true,
//!     time_column: 0,
//!     value_column: 1,
//!     delimiter: ',',
//!     skip_invalid: true,
//! };
//!
//! let (times, values) = load_timeseries(Path::new("data.csv"), &config)?;
//! ```

use std::path::Path;
use mh_foundation::error::{MhError, MhResult};

/// CSV 加载配置
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// 是否有表头行
    pub has_header: bool,
    /// 时间列索引（从 0 开始）
    pub time_column: usize,
    /// 值列索引（从 0 开始）
    pub value_column: usize,
    /// 分隔符
    pub delimiter: char,
    /// 是否跳过无效行
    pub skip_invalid: bool,
    /// 时间单位转换因子（例如：小时->秒 = 3600）
    pub time_scale: f64,
    /// 值单位转换因子
    pub value_scale: f64,
    /// 注释行前缀（以此开头的行将被跳过）
    pub comment_prefix: Option<char>,
}

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            has_header: true,
            time_column: 0,
            value_column: 1,
            delimiter: ',',
            skip_invalid: true,
            time_scale: 1.0,
            value_scale: 1.0,
            comment_prefix: Some('#'),
        }
    }
}

impl CsvConfig {
    /// 创建不带表头的配置
    pub fn no_header() -> Self {
        Self {
            has_header: false,
            ..Default::default()
        }
    }

    /// 创建制表符分隔的配置
    pub fn tab_separated() -> Self {
        Self {
            delimiter: '\t',
            ..Default::default()
        }
    }

    /// 创建分号分隔的配置
    pub fn semicolon_separated() -> Self {
        Self {
            delimiter: ';',
            ..Default::default()
        }
    }

    /// 设置列索引
    pub fn with_columns(mut self, time_col: usize, value_col: usize) -> Self {
        self.time_column = time_col;
        self.value_column = value_col;
        self
    }

    /// 设置时间缩放因子
    pub fn with_time_scale(mut self, scale: f64) -> Self {
        self.time_scale = scale;
        self
    }

    /// 设置值缩放因子
    pub fn with_value_scale(mut self, scale: f64) -> Self {
        self.value_scale = scale;
        self
    }
}

/// 从 CSV 文件加载时间序列
///
/// # 参数
///
/// - `path`: CSV 文件路径
/// - `config`: CSV 配置
///
/// # 返回
///
/// 成功时返回 (times, values) 元组
///
/// # 错误
///
/// - 文件读取失败
/// - 无有效数据
/// - 严格模式下遇到无效行
pub fn load_timeseries(path: &Path, config: &CsvConfig) -> MhResult<(Vec<f64>, Vec<f64>)> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        MhError::Io {
            message: format!("Failed to read {}: {}", path.display(), e),
            source: Some(e),
        }
    })?;

    parse_csv_content(&content, config, Some(path))
}

/// 从字符串解析 CSV 时间序列
///
/// # 参数
///
/// - `content`: CSV 内容字符串
/// - `config`: CSV 配置
///
/// # 返回
///
/// 成功时返回 (times, values) 元组
pub fn parse_csv_string(content: &str, config: &CsvConfig) -> MhResult<(Vec<f64>, Vec<f64>)> {
    parse_csv_content(content, config, None)
}

/// 内部 CSV 解析函数
fn parse_csv_content(
    content: &str,
    config: &CsvConfig,
    path: Option<&Path>,
) -> MhResult<(Vec<f64>, Vec<f64>)> {
    let mut times = Vec::new();
    let mut values = Vec::new();
    let mut errors = Vec::new();
    let mut skipped_lines = 0;

    let path_str = path
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|| "<string>".to_string());

    for (line_num, line) in content.lines().enumerate() {
        // 跳过表头
        if config.has_header && line_num == 0 {
            continue;
        }

        // 跳过空行
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // 跳过注释行
        if let Some(prefix) = config.comment_prefix {
            if trimmed.starts_with(prefix) {
                continue;
            }
        }

        // 分割列
        let parts: Vec<&str> = line.split(config.delimiter).collect();

        let max_col = config.time_column.max(config.value_column);
        if parts.len() <= max_col {
            if !config.skip_invalid {
                return Err(MhError::ParseError {
                    file: path.map(|p| p.to_path_buf()).unwrap_or_default(),
                    line: line_num + 1,
                    message: format!(
                        "Insufficient columns: expected at least {}, got {}",
                        max_col + 1,
                        parts.len()
                    ),
                });
            }
            errors.push(line_num + 1);
            skipped_lines += 1;
            continue;
        }

        // 解析时间和值
        let time_str = parts[config.time_column].trim();
        let value_str = parts[config.value_column].trim();

        match (time_str.parse::<f64>(), value_str.parse::<f64>()) {
            (Ok(t), Ok(v)) if t.is_finite() && v.is_finite() => {
                times.push(t * config.time_scale);
                values.push(v * config.value_scale);
            }
            _ => {
                if !config.skip_invalid {
                    return Err(MhError::ParseError {
                        file: path.map(|p| p.to_path_buf()).unwrap_or_default(),
                        line: line_num + 1,
                        message: format!(
                            "Failed to parse time='{}' or value='{}'",
                            time_str, value_str
                        ),
                    });
                }
                errors.push(line_num + 1);
                skipped_lines += 1;
            }
        }
    }

    // 记录跳过的行
    if !errors.is_empty() {
        let preview: Vec<_> = errors.iter().take(5).collect();
        eprintln!(
            "WARNING: {}: Skipped {} invalid lines (first few: {:?}{})",
            path_str,
            skipped_lines,
            preview,
            if errors.len() > 5 { "..." } else { "" }
        );
    }

    // 检查是否有有效数据
    if times.is_empty() {
        return Err(MhError::InvalidInput {
            message: format!("{}: No valid data found", path_str),
        });
    }

    // 验证时间单调性
    let mut sorted_indices: Vec<_> = (0..times.len()).collect();
    sorted_indices.sort_by(|&a, &b| times[a].partial_cmp(&times[b]).unwrap());

    let needs_sort = sorted_indices.iter().enumerate().any(|(i, &j)| i != j);
    if needs_sort {
        eprintln!("INFO: {}: Sorting {} data points by time", path_str, times.len());
        let sorted_times: Vec<_> = sorted_indices.iter().map(|&i| times[i]).collect();
        let sorted_values: Vec<_> = sorted_indices.iter().map(|&i| values[i]).collect();
        times = sorted_times;
        values = sorted_values;
    }

    // 检查时间严格递增
    for i in 1..times.len() {
        if times[i] <= times[i - 1] {
            eprintln!(
                "WARNING: {}: Duplicate time {} at index {}, removing duplicate",
                path_str,
                times[i],
                i
            );
        }
    }

    // 移除重复时间点（保留最后一个）
    let mut dedup_times = Vec::with_capacity(times.len());
    let mut dedup_values = Vec::with_capacity(values.len());
    
    for i in 0..times.len() {
        let is_last = i == times.len() - 1;
        let is_unique = is_last || times[i] < times[i + 1];
        
        if is_unique {
            dedup_times.push(times[i]);
            dedup_values.push(values[i]);
        }
    }

    if dedup_times.len() < times.len() {
        eprintln!(
            "WARNING: {}: Removed {} duplicate time points",
            path_str,
            times.len() - dedup_times.len()
        );
    }

    if dedup_times.is_empty() {
        return Err(MhError::InvalidInput {
            message: format!("{}: No unique time points found", path_str),
        });
    }

    Ok((dedup_times, dedup_values))
}

/// 加载多列 CSV 为多个时间序列
///
/// 第一列为时间，其余列为不同的值序列
pub fn load_multi_column_timeseries(
    path: &Path,
    config: &CsvConfig,
) -> MhResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        MhError::Io {
            message: format!("Failed to read {}: {}", path.display(), e),
            source: Some(e),
        }
    })?;

    let mut times = Vec::new();
    let mut all_values: Vec<Vec<f64>> = Vec::new();
    let mut n_cols = 0;

    for (line_num, line) in content.lines().enumerate() {
        if config.has_header && line_num == 0 {
            continue;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(prefix) = config.comment_prefix {
            if trimmed.starts_with(prefix) {
                continue;
            }
        }

        let parts: Vec<&str> = line.split(config.delimiter).collect();
        
        if parts.len() < 2 {
            if config.skip_invalid {
                continue;
            }
            return Err(MhError::ParseError {
                file: path.to_path_buf(),
                line: line_num + 1,
                message: "Need at least 2 columns".into(),
            });
        }

        // 初始化列数
        if n_cols == 0 {
            n_cols = parts.len() - 1;
            all_values = vec![Vec::new(); n_cols];
        }

        // 解析时间
        let t: f64 = match parts[0].trim().parse::<f64>() {
            Ok(v) if v.is_finite() => v * config.time_scale,
            _ => {
                if config.skip_invalid {
                    continue;
                }
                return Err(MhError::ParseError {
                    file: path.to_path_buf(),
                    line: line_num + 1,
                    message: format!("Failed to parse time: {}", parts[0]),
                });
            }
        };

        // 解析各列值
        let mut row_values = Vec::with_capacity(n_cols);
        let mut valid = true;
        
        for i in 1..parts.len().min(n_cols + 1) {
            match parts[i].trim().parse::<f64>() {
                Ok(v) if v.is_finite() => row_values.push(v * config.value_scale),
                _ => {
                    valid = false;
                    break;
                }
            }
        }

        if valid && row_values.len() == n_cols {
            times.push(t);
            for (i, v) in row_values.into_iter().enumerate() {
                all_values[i].push(v);
            }
        } else if !config.skip_invalid {
            return Err(MhError::ParseError {
                file: path.to_path_buf(),
                line: line_num + 1,
                message: "Failed to parse values".into(),
            });
        }
    }

    if times.is_empty() {
        return Err(MhError::InvalidInput {
            message: format!("{}: No valid data found", path.display()),
        });
    }

    Ok((times, all_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_csv() {
        let content = "time,value\n0.0,1.0\n1.0,2.0\n2.0,3.0";
        let config = CsvConfig::default();

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_no_header() {
        let content = "0.0,1.0\n1.0,2.0";
        let config = CsvConfig::no_header();

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 1.0]);
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_parse_tab_separated() {
        let content = "time\tvalue\n0.0\t1.0\n1.0\t2.0";
        let config = CsvConfig::tab_separated();

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 1.0]);
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_parse_with_comments() {
        let content = "# Header comment\ntime,value\n# Data comment\n0.0,1.0\n1.0,2.0";
        let config = CsvConfig::default();

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 1.0]);
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_skip_invalid_lines() {
        let content = "time,value\n0.0,1.0\ninvalid,line\n2.0,3.0";
        let config = CsvConfig::default();

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 2.0]);
        assert_eq!(values, vec![1.0, 3.0]);
    }

    #[test]
    fn test_time_scaling() {
        let content = "time,value\n0.0,1.0\n1.0,2.0";  // 时间单位：小时
        let config = CsvConfig::default().with_time_scale(3600.0);  // 转换为秒

        let (times, _) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 3600.0]);
    }

    #[test]
    fn test_unsorted_times() {
        let content = "time,value\n2.0,3.0\n0.0,1.0\n1.0,2.0";
        let config = CsvConfig::no_header();

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 1.0, 2.0]);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_empty_file_error() {
        let content = "time,value\n";
        let config = CsvConfig::default();

        let result = parse_csv_string(content, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_columns() {
        let content = "idx,time,value,extra\n0,0.0,1.0,x\n1,1.0,2.0,y";
        let config = CsvConfig::default().with_columns(1, 2);

        let (times, values) = parse_csv_string(content, &config).unwrap();

        assert_eq!(times, vec![0.0, 1.0]);
        assert_eq!(values, vec![1.0, 2.0]);
    }
}
