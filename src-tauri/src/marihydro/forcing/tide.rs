use crate::marihydro::domain::boundary::BoundaryForcing;
use crate::marihydro::infra::error::MhResult;
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

/// 潮位提供者接口
pub trait TideProvider: Send + Sync {
    fn get_forcing(&self, time: DateTime<Utc>) -> MhResult<BoundaryForcing>;
}

/// [实现 1] 静态水位
pub struct ConstantTide {
    pub level: f64,
}

impl TideProvider for ConstantTide {
    fn get_forcing(&self, _time: DateTime<Utc>) -> MhResult<BoundaryForcing> {
        Ok(BoundaryForcing {
            left_level: Some(self.level),
            right_level: Some(self.level),
            top_level: Some(self.level),
            bottom_level: Some(self.level),
        })
    }
}

/// [实现 2] 简谐潮 (模拟单一分潮，如 M2)
/// H(t) = Mean + Amp * cos(omega * t - phase)
pub struct HarmonicTide {
    pub mean_level: f64,
    pub amplitude: f64,
    pub period_hours: f64,
    pub phase_deg: f64,
    pub start_time: DateTime<Utc>,
}

impl TideProvider for HarmonicTide {
    fn get_forcing(&self, time: DateTime<Utc>) -> MhResult<BoundaryForcing> {
        let elapsed_hours = (time - self.start_time).num_seconds() as f64 / 3600.0;

        let omega = if self.period_hours.abs() > 1e-6 {
            2.0 * PI / self.period_hours
        } else {
            0.0
        };

        let phase_rad = self.phase_deg.to_radians();

        let h = self.mean_level + self.amplitude * (omega * elapsed_hours - phase_rad).cos();

        // 默认仅应用于左边界 (West) 作为测试，实际应用需根据 Manifest 配置分发
        Ok(BoundaryForcing {
            left_level: Some(h),
            right_level: Some(self.mean_level),
            top_level: None,
            bottom_level: None,
        })
    }
}
