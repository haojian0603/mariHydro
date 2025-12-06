// crates/mh_io/src/drivers/netcdf/time.rs

//! CF 时间格式解析
//!
//! 解析 NetCDF 文件中的 CF 约定时间格式。
//!
//! # CF 时间约定
//!
//! CF (Climate and Forecast) 约定使用 "units since reference_time" 格式：
//! - `seconds since 1970-01-01 00:00:00`
//! - `hours since 2020-01-01`
//! - `days since 1900-01-01 00:00:00`
//!
//! # 支持的日历
//!
//! - Standard/Gregorian: 标准格里高利历
//! - NoLeap/365_day: 无闰年（每年365天）
//! - AllLeap/366_day: 全闰年（每年366天）
//! - 360_day: 360天历（每月30天）
//! - Julian: 儒略历
//! - Proleptic Gregorian: 预期格里高利历
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use mh_io::drivers::netcdf::time::{CfTimeUnits, CfCalendar};
//!
//! let units = CfTimeUnits::parse("hours since 2020-01-01 00:00:00").unwrap();
//! let datetime = units.to_datetime(24.0);  // 1天后
//! ```

use std::fmt;

// ============================================================
// 错误类型
// ============================================================

/// CF 时间解析错误
#[derive(Debug, Clone)]
pub enum CfTimeError {
    /// 无效的单位字符串
    InvalidUnits(String),
    /// 无效的日期
    InvalidDate(String),
    /// 无效的日历类型
    InvalidCalendar(String),
    /// 数值溢出
    Overflow(String),
}

impl fmt::Display for CfTimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CfTimeError::InvalidUnits(msg) => write!(f, "无效的时间单位: {}", msg),
            CfTimeError::InvalidDate(msg) => write!(f, "无效的日期: {}", msg),
            CfTimeError::InvalidCalendar(msg) => write!(f, "无效的日历类型: {}", msg),
            CfTimeError::Overflow(msg) => write!(f, "数值溢出: {}", msg),
        }
    }
}

impl std::error::Error for CfTimeError {}

/// CF 时间解析结果
pub type CfTimeResult<T> = Result<T, CfTimeError>;

// ============================================================
// 日历类型
// ============================================================

/// CF 日历类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CfCalendar {
    /// 标准格里高利历
    #[default]
    Standard,
    /// 无闰年（每年365天）
    NoLeap,
    /// 全闰年（每年366天）
    AllLeap,
    /// 360 天历（每月30天）
    Day360,
    /// 儒略历
    Julian,
    /// 预期格里高利历
    Proleptic,
}

impl CfCalendar {
    /// 从字符串解析日历类型
    pub fn from_str(s: &str) -> CfTimeResult<Self> {
        match s.to_lowercase().as_str() {
            "standard" | "gregorian" => Ok(Self::Standard),
            "noleap" | "365_day" | "no_leap" => Ok(Self::NoLeap),
            "all_leap" | "366_day" | "allleap" => Ok(Self::AllLeap),
            "360_day" => Ok(Self::Day360),
            "julian" => Ok(Self::Julian),
            "proleptic_gregorian" | "proleptic" => Ok(Self::Proleptic),
            _ => Err(CfTimeError::InvalidCalendar(s.to_string())),
        }
    }

    /// 判断某年是否为闰年
    pub fn is_leap_year(&self, year: i32) -> bool {
        match self {
            Self::NoLeap => false,
            Self::AllLeap => true,
            Self::Day360 => false, // 360天历不考虑闰年
            Self::Julian => year % 4 == 0,
            Self::Standard | Self::Proleptic => {
                (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
            }
        }
    }

    /// 获取某年某月的天数
    pub fn days_in_month(&self, year: i32, month: u32) -> u32 {
        if *self == Self::Day360 {
            return 30;
        }

        match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if self.is_leap_year(year) {
                    29
                } else {
                    28
                }
            }
            _ => 30, // 默认
        }
    }

    /// 获取某年的天数
    pub fn days_in_year(&self, year: i32) -> u32 {
        match self {
            Self::NoLeap => 365,
            Self::AllLeap => 366,
            Self::Day360 => 360,
            _ => {
                if self.is_leap_year(year) {
                    366
                } else {
                    365
                }
            }
        }
    }
}

impl fmt::Display for CfCalendar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard => write!(f, "standard"),
            Self::NoLeap => write!(f, "noleap"),
            Self::AllLeap => write!(f, "all_leap"),
            Self::Day360 => write!(f, "360_day"),
            Self::Julian => write!(f, "julian"),
            Self::Proleptic => write!(f, "proleptic_gregorian"),
        }
    }
}

// ============================================================
// 时间单位
// ============================================================

/// 时间单位类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    /// 秒
    Seconds,
    /// 分钟
    Minutes,
    /// 小时
    Hours,
    /// 天
    Days,
}

impl TimeUnit {
    /// 从字符串解析
    pub fn from_str(s: &str) -> CfTimeResult<Self> {
        match s.to_lowercase().as_str() {
            "second" | "seconds" | "s" => Ok(Self::Seconds),
            "minute" | "minutes" | "min" => Ok(Self::Minutes),
            "hour" | "hours" | "h" | "hr" => Ok(Self::Hours),
            "day" | "days" | "d" => Ok(Self::Days),
            _ => Err(CfTimeError::InvalidUnits(format!("未知时间单位: {}", s))),
        }
    }

    /// 转换为秒
    pub fn to_seconds(&self, value: f64) -> f64 {
        match self {
            Self::Seconds => value,
            Self::Minutes => value * 60.0,
            Self::Hours => value * 3600.0,
            Self::Days => value * 86400.0,
        }
    }

    /// 从秒转换
    pub fn from_seconds(&self, seconds: f64) -> f64 {
        match self {
            Self::Seconds => seconds,
            Self::Minutes => seconds / 60.0,
            Self::Hours => seconds / 3600.0,
            Self::Days => seconds / 86400.0,
        }
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seconds => write!(f, "seconds"),
            Self::Minutes => write!(f, "minutes"),
            Self::Hours => write!(f, "hours"),
            Self::Days => write!(f, "days"),
        }
    }
}

// ============================================================
// 日期时间
// ============================================================

/// 简单日期时间结构
///
/// 不依赖外部库，支持各种日历
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DateTime {
    /// 年
    pub year: i32,
    /// 月 (1-12)
    pub month: u32,
    /// 日 (1-31)
    pub day: u32,
    /// 时 (0-23)
    pub hour: u32,
    /// 分 (0-59)
    pub minute: u32,
    /// 秒 (0.0-60.0，支持闰秒）
    pub second: f64,
}

impl DateTime {
    /// 创建新的日期时间
    pub fn new(year: i32, month: u32, day: u32, hour: u32, minute: u32, second: f64) -> Self {
        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }
    }

    /// Unix 纪元 (1970-01-01 00:00:00)
    pub fn unix_epoch() -> Self {
        Self::new(1970, 1, 1, 0, 0, 0.0)
    }

    /// 从 ISO 8601 字符串解析
    pub fn parse(s: &str) -> CfTimeResult<Self> {
        // 支持格式: YYYY-MM-DD HH:MM:SS 或 YYYY-MM-DDTHH:MM:SS
        let s = s.replace('T', " ");
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.is_empty() {
            return Err(CfTimeError::InvalidDate("空日期字符串".into()));
        }

        // 解析日期部分
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        if date_parts.len() != 3 {
            return Err(CfTimeError::InvalidDate(format!(
                "无效的日期格式: {}",
                parts[0]
            )));
        }

        let year: i32 = date_parts[0]
            .parse()
            .map_err(|_| CfTimeError::InvalidDate(format!("无效的年份: {}", date_parts[0])))?;
        let month: u32 = date_parts[1]
            .parse()
            .map_err(|_| CfTimeError::InvalidDate(format!("无效的月份: {}", date_parts[1])))?;
        let day: u32 = date_parts[2]
            .parse()
            .map_err(|_| CfTimeError::InvalidDate(format!("无效的日期: {}", date_parts[2])))?;

        // 解析时间部分（可选）
        let (hour, minute, second) = if parts.len() > 1 {
            let time_str = parts[1].trim_end_matches('Z'); // 移除可能的 Z 后缀
            let time_parts: Vec<&str> = time_str.split(':').collect();
            let h: u32 = time_parts
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let m: u32 = time_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
            let s: f64 = time_parts
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0);
            (h, m, s)
        } else {
            (0, 0, 0.0)
        };

        Ok(Self::new(year, month, day, hour, minute, second))
    }

    /// 格式化为字符串
    pub fn format(&self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:06.3}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

impl fmt::Display for DateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.format())
    }
}

// ============================================================
// CF 时间单位
// ============================================================

/// CF 时间单位
///
/// 表示 "units since reference_time" 格式
#[derive(Debug, Clone)]
pub struct CfTimeUnits {
    /// 时间单位类型
    pub unit: TimeUnit,
    /// 参考时间
    pub reference_time: DateTime,
    /// 日历类型
    pub calendar: CfCalendar,
}

impl CfTimeUnits {
    /// 从 units 属性解析
    ///
    /// # 参数
    ///
    /// - `units_str`: 时间单位字符串，如 "seconds since 1970-01-01 00:00:00"
    ///
    /// # 示例
    ///
    /// ```rust,ignore
    /// let units = CfTimeUnits::parse("hours since 2020-01-01 00:00:00")?;
    /// ```
    pub fn parse(units_str: &str) -> CfTimeResult<Self> {
        // 查找 "since" 关键字
        let lower = units_str.to_lowercase();
        let since_pos = lower.find(" since ");
        if since_pos.is_none() {
            return Err(CfTimeError::InvalidUnits(format!(
                "缺少 'since' 关键字: {}",
                units_str
            )));
        }
        let since_pos = since_pos.unwrap();

        // 解析单位
        let unit_str = units_str[..since_pos].trim();
        let unit = TimeUnit::from_str(unit_str)?;

        // 解析参考时间
        let ref_time_str = units_str[since_pos + 7..].trim(); // 跳过 " since "
        let reference_time = DateTime::parse(ref_time_str)?;

        Ok(Self {
            unit,
            reference_time,
            calendar: CfCalendar::Standard,
        })
    }

    /// 同时解析单位和日历
    pub fn parse_with_calendar(
        units_str: &str,
        calendar_str: Option<&str>,
    ) -> CfTimeResult<Self> {
        let mut result = Self::parse(units_str)?;
        if let Some(cal) = calendar_str {
            result.calendar = CfCalendar::from_str(cal)?;
        }
        Ok(result)
    }

    /// 设置日历类型
    pub fn with_calendar(mut self, calendar: CfCalendar) -> Self {
        self.calendar = calendar;
        self
    }

    /// 将 CF 时间值转换为 DateTime
    pub fn to_datetime(&self, value: f64) -> DateTime {
        let total_seconds = self.unit.to_seconds(value);
        self.add_seconds_to_datetime(&self.reference_time, total_seconds)
    }

    /// 将 DateTime 转换为 CF 时间值
    pub fn from_datetime(&self, dt: &DateTime) -> f64 {
        let seconds = self.seconds_between(&self.reference_time, dt);
        self.unit.from_seconds(seconds)
    }

    /// 批量转换时间值
    pub fn to_datetimes(&self, values: &[f64]) -> Vec<DateTime> {
        values.iter().map(|&v| self.to_datetime(v)).collect()
    }

    /// 批量转换为 CF 时间值
    pub fn from_datetimes(&self, datetimes: &[DateTime]) -> Vec<f64> {
        datetimes.iter().map(|dt| self.from_datetime(dt)).collect()
    }

    /// 计算两个日期时间之间的秒数
    fn seconds_between(&self, from: &DateTime, to: &DateTime) -> f64 {
        // 转换为儒略日再相减
        let jd_from = self.datetime_to_julian_day(from);
        let jd_to = self.datetime_to_julian_day(to);
        (jd_to - jd_from) * 86400.0
    }

    /// 在日期时间上增加秒数
    fn add_seconds_to_datetime(&self, base: &DateTime, seconds: f64) -> DateTime {
        let jd = self.datetime_to_julian_day(base);
        let new_jd = jd + seconds / 86400.0;
        self.julian_day_to_datetime(new_jd)
    }

    /// 日期时间转换为儒略日
    fn datetime_to_julian_day(&self, dt: &DateTime) -> f64 {
        let (y, m, d) = (dt.year, dt.month as i32, dt.day as i32);

        // 使用改进的儒略日算法
        let a = (14 - m) / 12;
        let y_adj = y + 4800 - a;
        let m_adj = m + 12 * a - 3;

        let jdn = match self.calendar {
            CfCalendar::Julian => {
                d + (153 * m_adj + 2) / 5 + 365 * y_adj + y_adj / 4 - 32083
            }
            CfCalendar::Day360 => {
                // 360天历
                y * 360 + (m - 1) * 30 + d
            }
            CfCalendar::NoLeap => {
                // 无闰年
                let days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
                y * 365 + days_before_month[(m - 1) as usize] + d
            }
            CfCalendar::AllLeap => {
                // 全闰年
                let days_before_month = [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
                y * 366 + days_before_month[(m - 1) as usize] + d
            }
            _ => {
                // Standard/Proleptic Gregorian
                d + (153 * m_adj + 2) / 5 + 365 * y_adj + y_adj / 4 - y_adj / 100 + y_adj / 400
                    - 32045
            }
        };

        // 添加时间部分
        let time_fraction =
            (dt.hour as f64 + dt.minute as f64 / 60.0 + dt.second / 3600.0) / 24.0;
        jdn as f64 + time_fraction - 0.5
    }

    /// 儒略日转换为日期时间
    fn julian_day_to_datetime(&self, jd: f64) -> DateTime {
        let jd_int = (jd + 0.5).floor() as i64;
        let time_fraction = jd + 0.5 - jd_int as f64;

        let (year, month, day) = match self.calendar {
            CfCalendar::Day360 => {
                let total_days = jd_int as i32;
                let y = total_days / 360;
                let remaining = total_days % 360;
                let m = remaining / 30 + 1;
                let d = remaining % 30 + 1;
                (y, m as u32, d as u32)
            }
            CfCalendar::NoLeap => {
                let days_before_month = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365];
                let total_days = jd_int as i32;
                let mut y = total_days / 365;
                let mut doy = total_days % 365; // 1..=365 ideally
                if doy == 0 {
                    doy = 365;
                    y -= 1;
                }
                // find month where cumulative days >= doy
                let m = (1..=12).find(|&i| days_before_month[i] >= doy).unwrap_or(12);
                let d = doy - days_before_month[m - 1];
                (y, m as u32, d as u32)
            }
            CfCalendar::AllLeap => {
                let days_before_month =
                    [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366];
                let total_days = jd_int as i32;
                let y = total_days / 366;
                let doy = total_days % 366;
                let m = (1..=12).find(|&i| days_before_month[i] > doy).unwrap_or(12);
                let d = doy - days_before_month[m - 1] + 1;
                (y, m as u32, d as u32)
            }
            _ => {
                // Gregorian/Julian 算法
                let a = jd_int + 32044;
                let b = (4 * a + 3) / 146097;
                let c = a - (146097 * b) / 4;
                let d_val = (4 * c + 3) / 1461;
                let e = c - (1461 * d_val) / 4;
                let m = (5 * e + 2) / 153;
                let day = e - (153 * m + 2) / 5 + 1;
                let month = m + 3 - 12 * (m / 10);
                let year = 100 * b + d_val - 4800 + m / 10;
                (year as i32, month as u32, day as u32)
            }
        };

        // 时间部分
        let total_hours = time_fraction * 24.0;
        let hour = total_hours.floor() as u32;
        let remaining_minutes = (total_hours - hour as f64) * 60.0;
        let minute = remaining_minutes.floor() as u32;
        let second = (remaining_minutes - minute as f64) * 60.0;

        DateTime::new(year, month, day, hour, minute, second)
    }
}

impl fmt::Display for CfTimeUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} since {}", self.unit, self.reference_time)
    }
}

// ============================================================
// 辅助函数
// ============================================================

/// 解析日历属性
pub fn parse_calendar(calendar_str: &str) -> CfTimeResult<CfCalendar> {
    CfCalendar::from_str(calendar_str)
}

/// 解析日历属性（容错版本，失败时返回 Standard）
pub fn parse_calendar_or_default(calendar_str: &str) -> CfCalendar {
    CfCalendar::from_str(calendar_str).unwrap_or(CfCalendar::Standard)
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cf_time_units() {
        let units = CfTimeUnits::parse("hours since 2020-01-01 00:00:00").unwrap();
        assert_eq!(units.unit, TimeUnit::Hours);
        assert_eq!(units.reference_time.year, 2020);
        assert_eq!(units.reference_time.month, 1);
        assert_eq!(units.reference_time.day, 1);
    }

    #[test]
    fn test_parse_seconds_since() {
        let units = CfTimeUnits::parse("seconds since 1970-01-01").unwrap();
        assert_eq!(units.unit, TimeUnit::Seconds);
        assert_eq!(units.reference_time.year, 1970);
    }

    #[test]
    fn test_parse_days_since() {
        let units = CfTimeUnits::parse("days since 1900-01-01 00:00:00").unwrap();
        assert_eq!(units.unit, TimeUnit::Days);
        assert_eq!(units.reference_time.year, 1900);
    }

    #[test]
    fn test_time_conversion() {
        let units = CfTimeUnits::parse("hours since 2020-01-01 00:00:00").unwrap();

        let dt = units.to_datetime(24.0); // 1天后
        assert_eq!(dt.day, 2);
        assert_eq!(dt.month, 1);
        assert_eq!(dt.year, 2020);

        // 反向转换
        let value = units.from_datetime(&dt);
        assert!((value - 24.0).abs() < 0.001);
    }

    #[test]
    fn test_days_conversion() {
        let units = CfTimeUnits::parse("days since 2020-01-01").unwrap();

        let dt = units.to_datetime(1.0);
        assert_eq!(dt.day, 2);
        assert_eq!(dt.month, 1);
    }

    #[test]
    fn test_calendar_types() {
        assert_eq!(
            CfCalendar::from_str("standard").unwrap(),
            CfCalendar::Standard
        );
        assert_eq!(CfCalendar::from_str("noleap").unwrap(), CfCalendar::NoLeap);
        assert_eq!(
            CfCalendar::from_str("365_day").unwrap(),
            CfCalendar::NoLeap
        );
        assert_eq!(
            CfCalendar::from_str("360_day").unwrap(),
            CfCalendar::Day360
        );
    }

    #[test]
    fn test_leap_year() {
        let std = CfCalendar::Standard;
        assert!(std.is_leap_year(2000));
        assert!(!std.is_leap_year(1900));
        assert!(std.is_leap_year(2004));
        assert!(!std.is_leap_year(2001));

        let noleap = CfCalendar::NoLeap;
        assert!(!noleap.is_leap_year(2000));

        let allleap = CfCalendar::AllLeap;
        assert!(allleap.is_leap_year(2001));
    }

    #[test]
    fn test_days_in_month() {
        let std = CfCalendar::Standard;
        assert_eq!(std.days_in_month(2020, 2), 29); // 闰年
        assert_eq!(std.days_in_month(2021, 2), 28); // 非闰年
        assert_eq!(std.days_in_month(2020, 1), 31);
        assert_eq!(std.days_in_month(2020, 4), 30);

        let day360 = CfCalendar::Day360;
        assert_eq!(day360.days_in_month(2020, 1), 30);
        assert_eq!(day360.days_in_month(2020, 2), 30);
    }

    #[test]
    fn test_datetime_parse() {
        let dt = DateTime::parse("2020-06-15 12:30:45").unwrap();
        assert_eq!(dt.year, 2020);
        assert_eq!(dt.month, 6);
        assert_eq!(dt.day, 15);
        assert_eq!(dt.hour, 12);
        assert_eq!(dt.minute, 30);
        assert!((dt.second - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_datetime_parse_iso() {
        let dt = DateTime::parse("2020-06-15T12:30:45Z").unwrap();
        assert_eq!(dt.year, 2020);
        assert_eq!(dt.month, 6);
        assert_eq!(dt.day, 15);
    }

    #[test]
    fn test_batch_conversion() {
        let units = CfTimeUnits::parse("hours since 2020-01-01 00:00:00").unwrap();
        let values = vec![0.0, 24.0, 48.0, 72.0];

        let datetimes = units.to_datetimes(&values);
        assert_eq!(datetimes.len(), 4);
        assert_eq!(datetimes[0].day, 1);
        assert_eq!(datetimes[1].day, 2);
        assert_eq!(datetimes[2].day, 3);
        assert_eq!(datetimes[3].day, 4);
    }

    #[test]
    fn test_noleap_calendar() {
        let units = CfTimeUnits::parse("days since 2020-02-28")
            .unwrap()
            .with_calendar(CfCalendar::NoLeap);

        let dt = units.to_datetime(1.0);
        // NoLeap 日历中，2月只有28天，所以2月28日+1天应该是3月1日
        assert_eq!(dt.month, 3);
        assert_eq!(dt.day, 1);
    }

    #[test]
    fn test_time_unit_display() {
        assert_eq!(format!("{}", TimeUnit::Seconds), "seconds");
        assert_eq!(format!("{}", TimeUnit::Hours), "hours");
        assert_eq!(format!("{}", TimeUnit::Days), "days");
    }

    #[test]
    fn test_cf_time_units_display() {
        let units = CfTimeUnits::parse("hours since 2020-01-01 00:00:00").unwrap();
        let s = format!("{}", units);
        assert!(s.contains("hours"));
        assert!(s.contains("since"));
        assert!(s.contains("2020"));
    }

    #[test]
    fn test_invalid_units() {
        assert!(CfTimeUnits::parse("invalid format").is_err());
        assert!(CfTimeUnits::parse("hours after 2020-01-01").is_err());
    }
}
