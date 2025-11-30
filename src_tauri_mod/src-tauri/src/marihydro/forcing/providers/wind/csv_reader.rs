// src-tauri/src/marihydro/forcing/providers/wind/csv_reader.rs
//! CSV 格式风场读取器
//! 支持标准 CSV 格式，带可选表头

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// CSV 风场配置
#[derive(Debug, Clone)]
pub struct CsvWindConfig {
    /// 时间列名或索引
    pub time_column: ColumnRef,
    /// U分量列名或索引
    pub u_column: ColumnRef,
    /// V分量列名或索引
    pub v_column: ColumnRef,
    /// 时间格式
    pub time_format: CsvTimeFormat,
    /// 是否有表头
    pub has_header: bool,
    /// 分隔符
    pub delimiter: u8,
    /// 缩放因子
    pub scale_factor: f64,
}

impl Default for CsvWindConfig {
    fn default() -> Self {
        Self {
            time_column: ColumnRef::Name("datetime".into()),
            u_column: ColumnRef::Name("u10".into()),
            v_column: ColumnRef::Name("v10".into()),
            time_format: CsvTimeFormat::Iso8601,
            has_header: true,
            delimiter: b',',
            scale_factor: 1.0,
        }
    }
}

/// 列引用方式
#[derive(Debug, Clone)]
pub enum ColumnRef {
    /// 按列名引用
    Name(String),
    /// 按索引引用（从0开始）
    Index(usize),
}

/// CSV 时间格式
#[derive(Debug, Clone)]
pub enum CsvTimeFormat {
    /// 秒数（相对起始时间）
    Seconds,
    /// 小时数
    Hours,
    /// ISO 8601 格式
    Iso8601,
    /// 自定义 strftime 格式
    Custom(String),
}

/// 风场数据记录
#[derive(Debug, Clone)]
struct WindRecord {
    time_seconds: f64,
    u: f64,
    v: f64,
}

/// CSV 格式风场读取器
///
/// 支持格式示例：
/// ```csv
/// datetime,u10,v10
/// 2024-01-01T00:00:00,5.0,3.0
/// 2024-01-01T01:00:00,6.0,2.0
/// ```
pub struct CsvWindReader {
    records: Vec<WindRecord>,
    n_cells: usize,
    start_time: DateTime<Utc>,
}

impl CsvWindReader {
    /// 从文件路径打开（使用默认配置）
    pub fn open(path: &Path, n_cells: usize, start_time: DateTime<Utc>) -> MhResult<Self> {
        Self::open_with_config(path, n_cells, start_time, CsvWindConfig::default())
    }

    /// 使用自定义配置打开
    pub fn open_with_config(
        path: &Path,
        n_cells: usize,
        start_time: DateTime<Utc>,
        config: CsvWindConfig,
    ) -> MhResult<Self> {
        let file = File::open(path).map_err(|e| MhError::Io(e.to_string()))?;
        let reader = BufReader::new(file);

        let mut csv_reader = csv::ReaderBuilder::new()
            .has_headers(config.has_header)
            .delimiter(config.delimiter)
            .flexible(true)
            .from_reader(reader);

        // 获取列索引
        let (time_idx, u_idx, v_idx) = if config.has_header {
            let headers = csv_reader
                .headers()
                .map_err(|e| MhError::Parse(format!("Failed to read CSV headers: {}", e)))?;
            
            let time_idx = Self::resolve_column(&config.time_column, headers)?;
            let u_idx = Self::resolve_column(&config.u_column, headers)?;
            let v_idx = Self::resolve_column(&config.v_column, headers)?;
            (time_idx, u_idx, v_idx)
        } else {
            match (&config.time_column, &config.u_column, &config.v_column) {
                (ColumnRef::Index(t), ColumnRef::Index(u), ColumnRef::Index(v)) => (*t, *u, *v),
                _ => return Err(MhError::Config("Column names require header row".into())),
            }
        };

        let mut records = Vec::new();

        for (row_idx, result) in csv_reader.records().enumerate() {
            let record = result.map_err(|e| {
                MhError::Parse(format!("CSV parse error at row {}: {}", row_idx + 1, e))
            })?;

            // 解析时间
            let time_str = record.get(time_idx).ok_or_else(|| {
                MhError::Parse(format!("Missing time column at row {}", row_idx + 1))
            })?;
            let time_seconds =
                Self::parse_time(time_str, &config.time_format, start_time)?;

            // 解析 U, V
            let u: f64 = record
                .get(u_idx)
                .ok_or_else(|| MhError::Parse(format!("Missing U column at row {}", row_idx + 1)))?
                .parse()
                .map_err(|_| MhError::Parse(format!("Invalid U value at row {}", row_idx + 1)))?;

            let v: f64 = record
                .get(v_idx)
                .ok_or_else(|| MhError::Parse(format!("Missing V column at row {}", row_idx + 1)))?
                .parse()
                .map_err(|_| MhError::Parse(format!("Invalid V value at row {}", row_idx + 1)))?;

            records.push(WindRecord {
                time_seconds,
                u: u * config.scale_factor,
                v: v * config.scale_factor,
            });
        }

        if records.is_empty() {
            return Err(MhError::Parse("No valid wind records found in CSV".into()));
        }

        // 按时间排序
        records.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());

        Ok(Self {
            records,
            n_cells,
            start_time,
        })
    }

    fn resolve_column(col_ref: &ColumnRef, headers: &csv::StringRecord) -> MhResult<usize> {
        match col_ref {
            ColumnRef::Index(idx) => Ok(*idx),
            ColumnRef::Name(name) => headers
                .iter()
                .position(|h| h.eq_ignore_ascii_case(name))
                .ok_or_else(|| MhError::Config(format!("Column '{}' not found in CSV", name))),
        }
    }

    fn parse_time(
        s: &str,
        format: &CsvTimeFormat,
        start_time: DateTime<Utc>,
    ) -> MhResult<f64> {
        match format {
            CsvTimeFormat::Seconds => s
                .parse::<f64>()
                .map_err(|_| MhError::Parse(format!("Invalid time value: {}", s))),
            CsvTimeFormat::Hours => s
                .parse::<f64>()
                .map(|h| h * 3600.0)
                .map_err(|_| MhError::Parse(format!("Invalid time value: {}", s))),
            CsvTimeFormat::Iso8601 => {
                // 尝试多种 ISO 8601 变体
                if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
                    return Ok((dt.with_timezone(&Utc) - start_time).num_seconds() as f64);
                }
                // 尝试不带时区
                if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
                    let dt = Utc.from_utc_datetime(&ndt);
                    return Ok((dt - start_time).num_seconds() as f64);
                }
                // 尝试带空格分隔
                if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
                    let dt = Utc.from_utc_datetime(&ndt);
                    return Ok((dt - start_time).num_seconds() as f64);
                }
                Err(MhError::Parse(format!("Invalid ISO8601 time: {}", s)))
            }
            CsvTimeFormat::Custom(fmt) => {
                NaiveDateTime::parse_from_str(s, fmt)
                    .map(|ndt| {
                        let dt = Utc.from_utc_datetime(&ndt);
                        (dt - start_time).num_seconds() as f64
                    })
                    .map_err(|_| MhError::Parse(format!("Time '{}' doesn't match format '{}'", s, fmt)))
            }
        }
    }

    /// 时间插值获取风速
    fn interpolate_at_time(&self, time_seconds: f64) -> (f64, f64) {
        if self.records.is_empty() {
            return (0.0, 0.0);
        }

        // 边界检查
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

impl WindProvider for CsvWindReader {
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

/// 构建器模式配置
impl CsvWindConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn time_column_name(mut self, name: &str) -> Self {
        self.time_column = ColumnRef::Name(name.into());
        self
    }

    pub fn time_column_index(mut self, idx: usize) -> Self {
        self.time_column = ColumnRef::Index(idx);
        self
    }

    pub fn u_column_name(mut self, name: &str) -> Self {
        self.u_column = ColumnRef::Name(name.into());
        self
    }

    pub fn v_column_name(mut self, name: &str) -> Self {
        self.v_column = ColumnRef::Name(name.into());
        self
    }

    pub fn with_delimiter(mut self, delim: char) -> Self {
        self.delimiter = delim as u8;
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale_factor = scale;
        self
    }

    pub fn with_time_format(mut self, format: CsvTimeFormat) -> Self {
        self.time_format = format;
        self
    }

    pub fn no_header(mut self) -> Self {
        self.has_header = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation() {
        let records = vec![
            WindRecord { time_seconds: 0.0, u: 5.0, v: 3.0 },
            WindRecord { time_seconds: 3600.0, u: 10.0, v: 6.0 },
        ];

        let reader = CsvWindReader {
            records,
            n_cells: 100,
            start_time: Utc::now(),
        };

        let (u, v) = reader.interpolate_at_time(1800.0);
        assert!((u - 7.5).abs() < 1e-10);
        assert!((v - 4.5).abs() < 1e-10);
    }
}
