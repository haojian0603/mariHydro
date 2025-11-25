// src-tauri/src/marihydro/io/drivers/nc_adapter/time.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use chrono::{DateTime, Duration, NaiveDateTime, Utc};

/// 解析 NetCDF 的时间单位字符串
/// 返回基准时间 (Base Time) 和 单位乘数 (Multiplier to Seconds)
/// 例如: "hours since 1900-01-01" -> (1900-01-01T00:00:00Z, 3600.0)
pub fn parse_cf_time_units(units_str: &str) -> MhResult<(DateTime<Utc>, f64)> {
    let lower = units_str.to_lowercase();
    let parts: Vec<&str> = lower.split(" since ").collect();
    if parts.len() != 2 {
        return Err(MhError::DataLoad {
            file: "NetCDF".into(),
            message: format!("无法解析时间单位字符串: {}", units_str),
        });
    }

    // 1. 确定时间步长单位
    let multiplier = if parts[0].contains("second") {
        1.0
    } else if parts[0].contains("minute") {
        60.0
    } else if parts[0].contains("hour") {
        3600.0
    } else if parts[0].contains("day") {
        86400.0
    } else {
        return Err(MhError::DataLoad {
            file: "NetCDF".into(),
            message: format!("未知的时间步长单位: {}", parts[0]),
        });
    };

    // 2. 解析基准日期
    // CF 约定允许非常复杂的日期格式，这里处理最常见的 ISO 格式
    // 简单清理一下字符串，去掉可能的时区后缀 (暂时强制视为 UTC)
    let date_str = parts[1].trim();

    // 尝试多种格式解析
    let naive_dt = NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d %H:%M:%S.%f"))
        .or_else(|_| NaiveDateTime::parse_from_str(date_str, "%Y-%m-%d")) // 仅日期
        .map_err(|e| MhError::DataLoad {
            file: "NetCDF".into(),
            message: format!("基准时间解析失败 '{}': {}", date_str, e),
        })?;

    let base_utc = DateTime::<Utc>::from_utc(naive_dt, Utc);

    Ok((base_utc, multiplier))
}

/// 计算特定时间值对应的真实 UTC 时间
pub fn calculate_utc_time(value: f64, base: DateTime<Utc>, multiplier: f64) -> DateTime<Utc> {
    let seconds = (value * multiplier) as i64;
    base + Duration::seconds(seconds)
}
