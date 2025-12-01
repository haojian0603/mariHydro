// src-tauri/src/marihydro/forcing/providers/wind/excel_reader.rs
//! Excel 格式风场读取器
//! 支持 .xlsx 和 .xls 格式

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::forcing::manager::WindProvider;
use chrono::{DateTime, NaiveDateTime, TimeZone, Utc};
use std::path::Path;

/// Excel 风场配置
#[derive(Debug, Clone)]
pub struct ExcelWindConfig {
    /// 工作表名称（None表示使用第一个）
    pub sheet_name: Option<String>,
    /// 时间列（从0开始）
    pub time_column: usize,
    /// U分量列
    pub u_column: usize,
    /// V分量列
    pub v_column: usize,
    /// 起始行（跳过表头）
    pub start_row: usize,
    /// 时间格式
    pub time_format: ExcelTimeFormat,
    /// 缩放因子
    pub scale_factor: f64,
}

impl Default for ExcelWindConfig {
    fn default() -> Self {
        Self {
            sheet_name: None,
            time_column: 0,
            u_column: 1,
            v_column: 2,
            start_row: 1, // 跳过表头
            time_format: ExcelTimeFormat::ExcelSerial,
            scale_factor: 1.0,
        }
    }
}

/// Excel 时间格式
#[derive(Debug, Clone)]
pub enum ExcelTimeFormat {
    /// Excel 序列号（自1900年1月1日的天数）
    ExcelSerial,
    /// 秒数（相对起始时间）
    Seconds,
    /// 小时数
    Hours,
    /// 文本格式（ISO 8601）
    TextIso8601,
}

/// 风场数据记录
#[derive(Debug, Clone)]
struct WindRecord {
    time_seconds: f64,
    u: f64,
    v: f64,
}

/// Excel 格式风场读取器
///
/// 支持 .xlsx (Office Open XML) 和 .xls (BIFF) 格式
/// 使用 calamine 库进行读取
pub struct ExcelWindReader {
    records: Vec<WindRecord>,
    n_cells: usize,
    start_time: DateTime<Utc>,
}

impl ExcelWindReader {
    /// 从文件路径打开（使用默认配置）
    pub fn open(path: &Path, n_cells: usize, start_time: DateTime<Utc>) -> MhResult<Self> {
        Self::open_with_config(path, n_cells, start_time, ExcelWindConfig::default())
    }

    /// 使用自定义配置打开
    pub fn open_with_config(
        path: &Path,
        n_cells: usize,
        start_time: DateTime<Utc>,
        config: ExcelWindConfig,
    ) -> MhResult<Self> {
        use calamine::{open_workbook_auto, Data, Reader};

        let mut workbook = open_workbook_auto(path)
            .map_err(|e| MhError::io(format!("Failed to open Excel file: {}", e)))?;

        // 获取工作表
        let sheet_name = if let Some(name) = &config.sheet_name {
            name.clone()
        } else {
            workbook
                .sheet_names()
                .first()
                .cloned()
                .ok_or_else(|| MhError::parse_simple("Excel file has no sheets"))?
        };

        let range = workbook
            .worksheet_range(&sheet_name)
            .map_err(|e| MhError::parse_simple(format!("Failed to read sheet '{}': {}", sheet_name, e)))?;;

        let mut records = Vec::new();

        for (row_idx, row) in range.rows().enumerate() {
            // 跳过表头行
            if row_idx < config.start_row {
                continue;
            }

            // 获取单元格值
            let time_cell = row.get(config.time_column);
            let u_cell = row.get(config.u_column);
            let v_cell = row.get(config.v_column);

            // 解析时间
            let time_seconds = match time_cell {
                Some(Data::Float(f)) => {
                    Self::parse_time_value(*f, &config.time_format, start_time)?
                }
                Some(Data::Int(i)) => {
                    Self::parse_time_value(*i as f64, &config.time_format, start_time)?
                }
                Some(Data::String(s)) => Self::parse_time_string(s, start_time)?,
                Some(Data::DateTime(dt)) => {
                    // calamine DateTime: days since 1899-12-30
                    Self::excel_datetime_to_seconds(dt.as_f64(), start_time)?
                }
                Some(Data::DateTimeIso(s)) => Self::parse_time_string(s, start_time)?,
                Some(Data::Empty) | None => continue, // 跳过空行
                _ => {
                    return Err(MhError::parse_simple(format!(
                        "Unsupported time format at row {}",
                        row_idx + 1
                    )))
                }
            };

            // 解析 U 值
            let u = Self::parse_numeric_cell(u_cell, row_idx, "U")?;

            // 解析 V 值
            let v = Self::parse_numeric_cell(v_cell, row_idx, "V")?;

            records.push(WindRecord {
                time_seconds,
                u: u * config.scale_factor,
                v: v * config.scale_factor,
            });
        }

        if records.is_empty() {
            return Err(MhError::parse_simple("No valid wind records found in Excel"));
        }

        // 按时间排序
        records.sort_by(|a, b| a.time_seconds.partial_cmp(&b.time_seconds).unwrap());

        Ok(Self {
            records,
            n_cells,
            start_time,
        })
    }

    fn parse_numeric_cell(
        cell: Option<&calamine::Data>,
        row_idx: usize,
        col_name: &str,
    ) -> MhResult<f64> {
        use calamine::Data;
        match cell {
            Some(Data::Float(f)) => Ok(*f),
            Some(Data::Int(i)) => Ok(*i as f64),
            Some(Data::String(s)) => s
                .parse::<f64>()
                .map_err(|_| MhError::parse_simple(format!("Invalid {} value at row {}", col_name, row_idx + 1))),
            Some(Data::Empty) | None => Ok(0.0),
            _ => Err(MhError::parse_simple(format!(
                "Unsupported {} format at row {}",
                col_name,
                row_idx + 1
            ))),
        }
    }

    fn parse_time_value(
        value: f64,
        format: &ExcelTimeFormat,
        start_time: DateTime<Utc>,
    ) -> MhResult<f64> {
        match format {
            ExcelTimeFormat::ExcelSerial => {
                // Excel 序列号：自1899-12-30的天数
                Self::excel_datetime_to_seconds(value, start_time)
            }
            ExcelTimeFormat::Seconds => Ok(value),
            ExcelTimeFormat::Hours => Ok(value * 3600.0),
            ExcelTimeFormat::TextIso8601 => Err(MhError::parse_simple(
                "Expected text format but got numeric",
            )),
        }
    }

    fn excel_datetime_to_seconds(excel_date: f64, start_time: DateTime<Utc>) -> MhResult<f64> {
        // Excel 日期：自1899-12-30的天数（Excel 错误地认为1900是闰年）
        // 1899-12-30 00:00:00 UTC 的 Unix 时间戳
        const EXCEL_EPOCH_OFFSET: f64 = -2209161600.0;
        const SECONDS_PER_DAY: f64 = 86400.0;

        // 转换为 Unix 时间戳
        let unix_timestamp = EXCEL_EPOCH_OFFSET + excel_date * SECONDS_PER_DAY;
        
        // 计算相对于 start_time 的秒数
        let start_unix = start_time.timestamp() as f64;
        Ok(unix_timestamp - start_unix)
    }

    fn parse_time_string(s: &str, start_time: DateTime<Utc>) -> MhResult<f64> {
        // 尝试多种格式
        let formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%Y-%m-%d",
        ];

        for fmt in formats {
            if let Ok(ndt) = NaiveDateTime::parse_from_str(s.trim(), fmt) {
                let dt = Utc.from_utc_datetime(&ndt);
                return Ok((dt - start_time).num_seconds() as f64);
            }
        }

        // 尝试 RFC3339
        if let Ok(dt) = DateTime::parse_from_rfc3339(s.trim()) {
            return Ok((dt.with_timezone(&Utc) - start_time).num_seconds() as f64);
        }

        Err(MhError::parse_simple(format!("Unable to parse time: {}", s)))
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

impl WindProvider for ExcelWindReader {
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
impl ExcelWindConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sheet(mut self, name: &str) -> Self {
        self.sheet_name = Some(name.to_string());
        self
    }

    pub fn with_columns(mut self, time: usize, u: usize, v: usize) -> Self {
        self.time_column = time;
        self.u_column = u;
        self.v_column = v;
        self
    }

    pub fn with_start_row(mut self, row: usize) -> Self {
        self.start_row = row;
        self
    }

    pub fn with_time_format(mut self, format: ExcelTimeFormat) -> Self {
        self.time_format = format;
        self
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale_factor = scale;
        self
    }
}
