// src-tauri/src/marihydro/core/traits/forcing.rs

//! 强迫场提供者接口
//!
//! 定义外部驱动数据（风场、潮汐、河流等）的统一抽象。

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::CellIndex;
use glam::DVec2;

/// 强迫场数据
#[derive(Debug, Clone, Copy, Default)]
pub struct ForcingData {
    /// 时间戳 [s]
    pub time: f64,
    /// 数据是否有效
    pub valid: bool,
}

// ============================================================
// 风场提供者
// ============================================================

/// 风场数据（单点）
#[derive(Debug, Clone, Copy, Default)]
pub struct WindData {
    /// 10m高度风速 u 分量 [m/s]
    pub u10: f64,
    /// 10m高度风速 v 分量 [m/s]
    pub v10: f64,
    /// 气压 [Pa]（可选）
    pub pressure: Option<f64>,
}

impl WindData {
    pub const ZERO: Self = Self {
        u10: 0.0,
        v10: 0.0,
        pressure: None,
    };

    /// 风速大小
    pub fn speed(&self) -> f64 {
        (self.u10 * self.u10 + self.v10 * self.v10).sqrt()
    }

    /// 风向（气象惯例：风来自的方向，北为0度，顺时针）
    pub fn direction_from(&self) -> f64 {
        let dir_to = self.v10.atan2(self.u10);
        let dir_from = dir_to + std::f64::consts::PI;
        // 转换为气象惯例 (0 = 北)
        (90.0_f64.to_radians() - dir_from)
            .to_degrees()
            .rem_euclid(360.0)
    }

    /// 风速向量
    pub fn velocity(&self) -> DVec2 {
        DVec2::new(self.u10, self.v10)
    }
}

/// 风场提供者接口
///
/// # 实现要求
///
/// 1. 支持时空插值
/// 2. 处理数据边界（时间范围外）
/// 3. 可选地提供气压场
pub trait WindProvider: Send + Sync {
    /// 提供者名称
    fn name(&self) -> &str;

    /// 更新到指定时间
    fn update_time(&mut self, time: f64) -> MhResult<()>;

    /// 获取单点风场数据
    fn get_wind(&self, x: f64, y: f64) -> WindData;

    /// 批量获取风场数据（填充到数组）
    fn get_wind_field(
        &self,
        centroids: &[DVec2],
        u10: &mut [f64],
        v10: &mut [f64],
    ) -> MhResult<()> {
        if centroids.len() != u10.len() || centroids.len() != v10.len() {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "wind field arrays",
                centroids.len(),
                u10.len(),
            ));
        }

        for (i, centroid) in centroids.iter().enumerate() {
            let wind = self.get_wind(centroid.x, centroid.y);
            u10[i] = wind.u10;
            v10[i] = wind.v10;
        }

        Ok(())
    }

    /// 获取气压场（如果可用）
    fn get_pressure_field(&self, _centroids: &[DVec2], _pressure: &mut [f64]) -> MhResult<bool> {
        Ok(false) // 默认不支持
    }

    /// 数据时间范围
    fn time_range(&self) -> (f64, f64);

    /// 当前时间是否在范围内
    fn is_time_valid(&self, time: f64) -> bool {
        let (start, end) = self.time_range();
        time >= start && time <= end
    }
}

// ============================================================
// 潮汐边界提供者
// ============================================================

/// 潮汐数据（单边界）
#[derive(Debug, Clone, Copy, Default)]
pub struct TideData {
    /// 水位 [m]
    pub elevation: f64,
    /// 法向流速 [m/s]（正向为入流）
    pub normal_velocity: Option<f64>,
}

impl TideData {
    pub const ZERO: Self = Self {
        elevation: 0.0,
        normal_velocity: None,
    };
}

/// 潮汐边界提供者接口
pub trait TideProvider: Send + Sync {
    /// 提供者名称
    fn name(&self) -> &str;

    /// 更新到指定时间
    fn update_time(&mut self, time: f64) -> MhResult<()>;

    /// 获取边界潮位
    fn get_tide(&self, boundary_name: &str) -> Option<TideData>;

    /// 获取所有边界名称
    fn boundary_names(&self) -> &[String];

    /// 数据时间范围
    fn time_range(&self) -> (f64, f64);
}

// ============================================================
// 河流入流提供者
// ============================================================

/// 河流入流数据
#[derive(Debug, Clone, Copy, Default)]
pub struct RiverData {
    /// 流量 [m³/s]
    pub discharge: f64,
    /// 温度 [°C]（可选）
    pub temperature: Option<f64>,
    /// 盐度 [PSU]（可选）
    pub salinity: Option<f64>,
}

impl RiverData {
    pub const ZERO: Self = Self {
        discharge: 0.0,
        temperature: None,
        salinity: None,
    };
}

/// 河流入流提供者接口
pub trait RiverProvider: Send + Sync {
    /// 提供者名称
    fn name(&self) -> &str;

    /// 更新到指定时间
    fn update_time(&mut self, time: f64) -> MhResult<()>;

    /// 获取河流数据
    fn get_river(&self, river_name: &str) -> Option<RiverData>;

    /// 获取所有河流名称
    fn river_names(&self) -> &[String];

    /// 数据时间范围
    fn time_range(&self) -> (f64, f64);
}

// ============================================================
// 常量强迫场（用于测试或简单场景）
// ============================================================

/// 常量风场
pub struct ConstantWindProvider {
    wind: WindData,
}

impl ConstantWindProvider {
    pub fn new(u10: f64, v10: f64) -> Self {
        Self {
            wind: WindData {
                u10,
                v10,
                pressure: None,
            },
        }
    }

    pub fn with_pressure(mut self, pressure: f64) -> Self {
        self.wind.pressure = Some(pressure);
        self
    }
}

impl WindProvider for ConstantWindProvider {
    fn name(&self) -> &str {
        "ConstantWind"
    }

    fn update_time(&mut self, _time: f64) -> MhResult<()> {
        Ok(())
    }

    fn get_wind(&self, _x: f64, _y: f64) -> WindData {
        self.wind
    }

    fn time_range(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

/// 无风场
pub struct NoWindProvider;

impl WindProvider for NoWindProvider {
    fn name(&self) -> &str {
        "NoWind"
    }

    fn update_time(&mut self, _time: f64) -> MhResult<()> {
        Ok(())
    }

    fn get_wind(&self, _x: f64, _y: f64) -> WindData {
        WindData::ZERO
    }

    fn time_range(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_data() {
        let wind = WindData {
            u10: 3.0,
            v10: 4.0,
            pressure: None,
        };
        assert!((wind.speed() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_constant_wind_provider() {
        let provider = ConstantWindProvider::new(10.0, 0.0);
        let wind = provider.get_wind(0.0, 0.0);
        assert!((wind.u10 - 10.0).abs() < 1e-10);
        assert!((wind.v10 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wind_direction() {
        // 东风（从东向西吹）: u < 0
        let wind = WindData {
            u10: -5.0,
            v10: 0.0,
            pressure: None,
        };
        let dir = wind.direction_from();
        assert!((dir - 90.0).abs() < 1.0); // 东 = 90度

        // 北风（从北向南吹）: v < 0
        let wind = WindData {
            u10: 0.0,
            v10: -5.0,
            pressure: None,
        };
        let dir = wind.direction_from();
        assert!((dir - 0.0).abs() < 1.0 || (dir - 360.0).abs() < 1.0); // 北 = 0度
    }
}
