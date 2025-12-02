// src-tauri/src/marihydro/io/drivers/netcdf/time.rs
use chrono::{DateTime, Utc, NaiveDateTime, TimeZone};

pub fn parse_cf_time_units(units: &str) -> Option<(String, DateTime<Utc>)> {
    let parts: Vec<&str> = units.splitn(2, " since ").collect();
    if parts.len() != 2 { return None; }
    let unit = parts[0].to_lowercase();
    let base_str = parts[1].trim();
    let base = NaiveDateTime::parse_from_str(base_str, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| NaiveDateTime::parse_from_str(base_str, "%Y-%m-%dT%H:%M:%S"))
        .or_else(|_| NaiveDateTime::parse_from_str(&format!("{} 00:00:00", base_str), "%Y-%m-%d %H:%M:%S"))
        .ok()?;
    Some((unit, Utc.from_utc_datetime(&base)))
}

pub fn calculate_utc_time(value: f64, unit: &str, base: DateTime<Utc>) -> DateTime<Utc> {
    let secs = match unit {
        "seconds" | "second" | "s" => value,
        "minutes" | "minute" | "min" => value * 60.0,
        "hours" | "hour" | "h" => value * 3600.0,
        "days" | "day" | "d" => value * 86400.0,
        _ => value,
    };
    base + chrono::Duration::milliseconds((secs * 1000.0) as i64)
}
