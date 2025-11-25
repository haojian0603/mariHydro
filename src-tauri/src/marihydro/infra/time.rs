use crate::marihydro::infra::error::{MhError, MhResult};
use chrono::{DateTime, Duration, FixedOffset, Local, TimeZone, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

/// 用户配置的时区策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimezoneConfig {
    Utc,           // 强制 UTC
    Local,         // 跟随运行机器的系统时间
    Fixed(i32),    // 固定偏移 (小时)，如 +8
    Named(String), // IANA 名称，如 "Asia/Shanghai"
}

/// 运行时时间管理器
#[derive(Debug, Clone)]
pub struct TimeManager {
    // 内部始终使用 UTC 存储基准时间，保证计算绝对准确
    base_utc: DateTime<Utc>,
    // 经过的物理秒数
    elapsed_seconds: f64,
    // 缓存的时区转换器 (用于显示和 IO)
    display_offset: Option<FixedOffset>,
    display_tz: Option<Tz>,
    // 标记使用哪种显示策略
    mode: TimezoneConfig,
}

impl TimeManager {
    /// 初始化时间管理器
    pub fn new(start_iso: &str, tz_config: TimezoneConfig) -> MhResult<Self> {
        // 1. 解析起始时间 (统一转为 UTC 存储)
        let base_utc = DateTime::parse_from_rfc3339(start_iso)
            .map_err(|e| MhError::Config(format!("时间格式错误 (需 ISO8601): {}", e)))?
            .with_timezone(&Utc);

        // 2. 预处理时区 (Fail fast: 如果时区名错了，启动时就报错)
        let (offset, tz_obj) = match &tz_config {
            TimezoneConfig::Fixed(h) => {
                let secs = h * 3600;
                if secs.abs() > 86400 {
                    return Err(MhError::Timezone(format!("偏移量 {} 小时超出范围", h)));
                }
                (Some(FixedOffset::east_opt(secs).unwrap()), None)
            }
            TimezoneConfig::Named(name) => {
                let tz: Tz = name
                    .parse()
                    .map_err(|_| MhError::Timezone(format!("未知时区名: {}", name)))?;
                (None, Some(tz))
            }
            _ => (None, None), // Local 和 UTC 动态处理
        };

        Ok(Self {
            base_utc,
            elapsed_seconds: 0.0,
            display_offset: offset,
            display_tz: tz_obj,
            mode: tz_config,
        })
    }

    /// 推进时间
    pub fn advance(&mut self, dt: f64) {
        self.elapsed_seconds += dt;
    }

    /// 获取已运行的物理秒数（公开访问器）
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_seconds
    }

    /// 获取当前仿真时刻的 ISO8601 字符串 (带正确时区)
    /// 用于 UI 显示和日志
    pub fn current_display_str(&self) -> String {
        let current_utc = self.current_utc();

        match self.mode {
            TimezoneConfig::Utc => current_utc.to_rfc3339(),
            TimezoneConfig::Local => DateTime::<Local>::from(current_utc).to_rfc3339(),
            TimezoneConfig::Fixed(_) => {
                if let Some(off) = self.display_offset {
                    current_utc.with_timezone(&off).to_rfc3339()
                } else {
                    current_utc.to_rfc3339() // Fallback
                }
            }
            TimezoneConfig::Named(_) => {
                if let Some(tz) = self.display_tz {
                    current_utc.with_timezone(&tz).to_rfc3339()
                } else {
                    current_utc.to_rfc3339()
                }
            }
        }
    }

    /// 获取内部 UTC 时间 (用于物理计算/数据对齐)
    pub fn current_utc(&self) -> DateTime<Utc> {
        let millis = (self.elapsed_seconds * 1000.0) as i64;
        self.base_utc + Duration::milliseconds(millis)
    }

    /// 获取显示时区配置（用于调试和配置验证）
    pub fn timezone_config(&self) -> &TimezoneConfig {
        &self.mode
    }
}
