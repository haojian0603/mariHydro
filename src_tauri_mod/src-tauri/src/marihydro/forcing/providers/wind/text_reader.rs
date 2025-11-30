// src-tauri/src/marihydro/forcing/providers/wind/text_reader.rs
//! 文本格式风场读取器
//! 支持空格/制表符分隔的简单文本格式

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, Utc, TimeZone, NaiveDateTime};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// 文本格式风场配置
#[derive(Debug, Clone)]
pub struct TextWindConfig {
    /// 时间列索引（从0开始）
    pub time_column: usize,
    /// U分量列索引
    pub u_column: usize,
    /// V分量列索引
    pub v_column: usize,
    /// X坐标列索引（可选，用于空间变化风场）
    pub x_column: Option<usize>,
    /// Y坐标列索引（可选）
    pub y_column: Option<usize>,
    /// 时间格式类型
    pub time_format: TimeFormat,
    /// 分隔符
    pub delimiter: char,
    /// 跳过的起始行数（头部注释等）
    pub skip_lines: usize,
    /// 注释前缀
    pub comment_prefix: char,
    /// 缩放因子
    pub scale_factor: f64,
}

impl Default for TextWindConfig {
    fn default() -> Self {
        Self {
            time_column: 0,
            u_column: 1,
            v_column: 2,
            x_column: None,
            y_column: None,
            time_format: TimeFormat::Seconds,
            delimiter: ' ',
            skip_lines: 0,
            comment_prefix: '#',
            scale_factor: 1.0,
        }
    }
}

/// 时间格式类型
#[derive(Debug, Clone, Copy)]
pub enum TimeFormat {
    /// 自起始时间的秒数
    Seconds,
    /// 自起始时间的小时数
    Hours,
    /// ISO 8601 格式
    Iso8601,
    /// 自定义格式 (需要 strptime 格式字符串)
    Custom,
}

/// 单个时间步的风场数据
#[derive(Debug, Clone)]
struct WindRecord {
    time_seconds: f64,
    u: f64,
    v: f64,
    x: Option<f64>,
    y: Option<f64>,
}

/// 文本格式风场读取器
/// 
/// 支持格式示例：
/// ```text
/// # time(s) u10(m/s) v10(m/s)
/// 0.0    5.0   3.0
/// 3600.0 6.0   2.0
/// 7200.0 4.0   4.0
/// ```
pub struct TextWindReader {
    records: Vec<WindRecord>,
    n_cells: usize,
    start_time: DateTime<Utc>,
    scale_factor: f64,
    /// 是否为空间变化风场
    is_spatial: bool,
}

impl TextWindReader {
    /// 从文件路径打开
    pub fn open(path: &Path, n_cells: usize, start_time: DateTime<Utc>) -> MhResult<Self> {
        Self::open_with_config(path, n_cells, start_time, TextWindConfig::default())
    }

    /// 使用自定义配置打开
    pub fn open_with_config(
        path: &Path,
        n_cells: usize,
        start_time: DateTime<Utc>,
        config: TextWindConfig,
    ) -> MhResult<Self> {
        let file = File::open(path).map_err(|e| MhError::Io(e.to_string()))?;
        let reader = BufReader::new(file);
        
        let mut records = Vec::new();
        let is_spatial = config.x_column.is_some() && config.y_column.is_some();
        
        for (line_no, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(|e| MhError::Io(e.to_string()))?;
            let line = line.trim();
            
            // 跳过头部和注释行
            if line_no < config.skip_lines || line.is_empty() 
               || line.starts_with(config.comment_prefix) {
                continue;
            }
            
            // 解析列
            let columns: Vec<&str> = if config.delimiter == ' ' {
                line.split_whitespace().collect()
            } else {
                line.split(config.delimiter).map(|s| s.trim()).collect()
            };
            
            // 解析时间
            let time_str = columns.get(config.time_column)
                .ok_or_else(|| MhError::Parse(format!("Missing time column at line {}", line_no + 1)))?;
            let time_seconds = Self::parse_time(time_str, &config.time_format, start_time)?;
            
            // 解析U, V
            let u: f64 = columns.get(config.u_column)
                .ok_or_else(|| MhError::Parse(format!("Missing U column at line {}", line_no + 1)))?
                .parse()
                .map_err(|_| MhError::Parse(format!("Invalid U value at line {}", line_no + 1)))?;
            
            let v: f64 = columns.get(config.v_column)
                .ok_or_else(|| MhError::Parse(format!("Missing V column at line {}", line_no + 1)))?
                .parse()
                .map_err(|_| MhError::Parse(format!("Invalid V value at line {}", line_no + 1)))?;
            
            // 解析可选的空间坐标
            let x = if let Some(col) = config.x_column {
                Some(columns.get(col)
                    .ok_or_else(|| MhError::Parse(format!("Missing X column at line {}", line_no + 1)))?
                    .parse()
                    .map_err(|_| MhError::Parse(format!("Invalid X value at line {}", line_no + 1)))?)
            } else { None };
            
            let y = if let Some(col) = config.y_column {
                Some(columns.get(col)
                    .ok_or_else(|| MhError::Parse(format!("Missing Y column at line {}", line_no + 1)))?
                    .parse()
                    .map_err(|_| MhError::Parse(format!("Invalid Y value at line {}", line_no + 1)))?)
            } else { None };
            
            records.push(WindRecord {
                time_seconds,
                u: u * config.scale_factor,
                v: v * config.scale_factor,
                x, y,
            });
        }
        
        if records.is_empty() {
            return Err(MhError::Parse("No valid wind records found in file".into()));
        }
        
        // 按时间排序
        records.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());
        
        Ok(Self {
            records,
            n_cells,
            start_time,
            scale_factor: config.scale_factor,
            is_spatial,
        })
    }

    fn parse_time(s: &str, format: &TimeFormat, start_time: DateTime<Utc>) -> MhResult<f64> {
        match format {
            TimeFormat::Seconds => {
                s.parse::<f64>()
                    .map_err(|_| MhError::Parse(format!("Invalid time value: {}", s)))
            }
            TimeFormat::Hours => {
                s.parse::<f64>()
                    .map(|h| h * 3600.0)
                    .map_err(|_| MhError::Parse(format!("Invalid time value: {}", s)))
            }
            TimeFormat::Iso8601 => {
                DateTime::parse_from_rfc3339(s)
                    .map(|dt| (dt.with_timezone(&Utc) - start_time).num_seconds() as f64)
                    .map_err(|_| MhError::Parse(format!("Invalid ISO8601 time: {}", s)))
            }
            TimeFormat::Custom => {
                // 尝试常见格式
                let formats = [
                    "%Y-%m-%d %H:%M:%S",
                    "%Y/%m/%d %H:%M:%S",
                    "%d-%m-%Y %H:%M:%S",
                ];
                for fmt in formats {
                    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, fmt) {
                        let dt = Utc.from_utc_datetime(&ndt);
                        return Ok((dt - start_time).num_seconds() as f64);
                    }
                }
                Err(MhError::Parse(format!("Unable to parse time: {}", s)))
            }
        }
    }

    /// 时间插值获取风速
    fn interpolate_at_time(&self, time_seconds: f64) -> (f64, f64) {
        if self.records.is_empty() {
            return (0.0, 0.0);
        }
        
        // 查找时间区间
        if time_seconds <= self.records[0].time_seconds {
            return (self.records[0].u, self.records[0].v);
        }
        
        let last = self.records.len() - 1;
        if time_seconds >= self.records[last].time_seconds {
            return (self.records[last].u, self.records[last].v);
        }
        
        // 二分查找
        let idx = self.records.partition_point(|r| r.time_seconds < time_seconds);
        if idx == 0 {
            return (self.records[0].u, self.records[0].v);
        }
        
        let r0 = &self.records[idx - 1];
        let r1 = &self.records[idx];
        
        let dt = r1.time_seconds - r0.time_seconds;
        if dt.abs() < 1e-10 {
            return (r0.u, r0.v);
        }
        
        let alpha = (time_seconds - r0.time_seconds) / dt;
        let u = r0.u * (1.0 - alpha) + r1.u * alpha;
        let v = r0.v * (1.0 - alpha) + r1.v * alpha;
        
        (u, v)
    }
}

impl WindProvider for TextWindReader {
    fn get_wind_at(&self, time: DateTime<Utc>, u: &mut [f64], v: &mut [f64]) -> MhResult<()> {
        let time_seconds = (time - self.start_time).num_seconds() as f64;
        let (wind_u, wind_v) = self.interpolate_at_time(time_seconds);
        
        // 均匀风场填充所有单元
        let n = self.n_cells.min(u.len()).min(v.len());
        u[..n].fill(wind_u);
        v[..n].fill(wind_v);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_interpolation() {
        let records = vec![
            WindRecord { time_seconds: 0.0, u: 5.0, v: 3.0, x: None, y: None },
            WindRecord { time_seconds: 3600.0, u: 10.0, v: 6.0, x: None, y: None },
        ];
        
        let reader = TextWindReader {
            records,
            n_cells: 100,
            start_time: Utc::now(),
            scale_factor: 1.0,
            is_spatial: false,
        };
        
        let (u, v) = reader.interpolate_at_time(1800.0);
        assert!((u - 7.5).abs() < 1e-10);
        assert!((v - 4.5).abs() < 1e-10);
    }
}
