// crates/mh_physics/src/forcing/wind.rs

//! 风场数据提供者
//!
//! 提供时变风场数据，可用于驱动风应力计算。
//!
//! # 支持的数据源
//!
//! - 恒定风场
//! - 时间序列风场（线性插值）
//! - 空间变化风场
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::forcing::WindProvider;
//!
//! // 创建恒定风场
//! let wind = WindProvider::constant(10.0, 225.0); // 10 m/s, 西南风
//!
//! // 获取某时刻的风速
//! let (u, v) = wind.get_wind_at(3600.0); // t = 1 hour
//! ```

use glam::DVec2;

/// 风场数据提供者
#[derive(Debug, Clone)]
pub struct WindProvider {
    /// 数据类型
    data: WindData,
    /// 最后更新时间
    last_update: f64,
    /// 缓存的风速
    cached_wind: DVec2,
}

/// 风场数据类型
#[derive(Debug, Clone)]
pub enum WindData {
    /// 恒定风场
    Constant { u: f64, v: f64 },
    /// 时间序列（时间、U 分量、V 分量）
    TimeSeries {
        times: Vec<f64>,
        u_values: Vec<f64>,
        v_values: Vec<f64>,
    },
    /// 周期性风场（日变化）
    Periodic {
        /// 平均风速
        mean_speed: f64,
        /// 振幅
        amplitude: f64,
        /// 主风向 [弧度]
        direction: f64,
        /// 周期 [s]
        period: f64,
        /// 相位 [弧度]
        phase: f64,
    },
}

impl WindProvider {
    /// 创建恒定风场
    ///
    /// # 参数
    /// - `speed`: 风速 [m/s]
    /// - `direction_deg`: 风向 [度]，0=北，90=东，180=南，270=西
    pub fn constant(speed: f64, direction_deg: f64) -> Self {
        let dir_rad = direction_deg.to_radians();
        // 风向定义：风从哪里来，所以速度方向相反
        let u = -speed * dir_rad.sin();
        let v = -speed * dir_rad.cos();

        Self {
            data: WindData::Constant { u, v },
            last_update: 0.0,
            cached_wind: DVec2::new(u, v),
        }
    }

    /// 创建静风
    pub fn calm() -> Self {
        Self::constant(0.0, 0.0)
    }

    /// 创建时间序列风场
    ///
    /// # 参数
    /// - `times`: 时间点 [s]
    /// - `u_values`: U 分量 [m/s]
    /// - `v_values`: V 分量 [m/s]
    pub fn time_series(times: Vec<f64>, u_values: Vec<f64>, v_values: Vec<f64>) -> Self {
        let initial_u = u_values.first().copied().unwrap_or(0.0);
        let initial_v = v_values.first().copied().unwrap_or(0.0);

        Self {
            data: WindData::TimeSeries { times, u_values, v_values },
            last_update: 0.0,
            cached_wind: DVec2::new(initial_u, initial_v),
        }
    }

    /// 创建周期性风场（如海陆风）
    ///
    /// # 参数
    /// - `mean_speed`: 平均风速 [m/s]
    /// - `amplitude`: 振幅 [m/s]
    /// - `direction_deg`: 主风向 [度]
    /// - `period_hours`: 周期 [小时]
    pub fn periodic(mean_speed: f64, amplitude: f64, direction_deg: f64, period_hours: f64) -> Self {
        Self {
            data: WindData::Periodic {
                mean_speed,
                amplitude,
                direction: direction_deg.to_radians(),
                period: period_hours * 3600.0,
                phase: 0.0,
            },
            last_update: 0.0,
            cached_wind: DVec2::new(mean_speed, 0.0),
        }
    }

    /// 创建日变化海陆风
    pub fn sea_land_breeze(mean_speed: f64, amplitude: f64) -> Self {
        Self::periodic(mean_speed, amplitude, 0.0, 24.0)
    }

    /// 获取指定时刻的风速
    ///
    /// 返回 (u, v) 分量 [m/s]
    pub fn get_wind_at(&self, time: f64) -> (f64, f64) {
        match &self.data {
            WindData::Constant { u, v } => (*u, *v),

            WindData::TimeSeries { times, u_values, v_values } => {
                if times.is_empty() {
                    return (0.0, 0.0);
                }

                // 查找时间区间
                if time <= times[0] {
                    return (u_values[0], v_values[0]);
                }
                if time >= *times.last().unwrap() {
                    return (*u_values.last().unwrap(), *v_values.last().unwrap());
                }

                // 线性插值
                for i in 0..times.len() - 1 {
                    if time >= times[i] && time < times[i + 1] {
                        let t = (time - times[i]) / (times[i + 1] - times[i]);
                        let u = u_values[i] + t * (u_values[i + 1] - u_values[i]);
                        let v = v_values[i] + t * (v_values[i + 1] - v_values[i]);
                        return (u, v);
                    }
                }

                (0.0, 0.0)
            }

            WindData::Periodic { mean_speed, amplitude, direction, period, phase } => {
                let omega = 2.0 * std::f64::consts::PI / period;
                let speed = mean_speed + amplitude * (omega * time + phase).sin();

                let u = -speed * direction.sin();
                let v = -speed * direction.cos();
                (u, v)
            }
        }
    }

    /// 获取指定时刻的风速向量
    pub fn get_wind_vector(&self, time: f64) -> DVec2 {
        let (u, v) = self.get_wind_at(time);
        DVec2::new(u, v)
    }

    /// 获取指定时刻的风速标量
    pub fn get_wind_speed(&self, time: f64) -> f64 {
        let (u, v) = self.get_wind_at(time);
        (u * u + v * v).sqrt()
    }

    /// 获取指定时刻的风向（风从哪里来）[度]
    pub fn get_wind_direction(&self, time: f64) -> f64 {
        let (u, v) = self.get_wind_at(time);
        let dir = (-u).atan2(-v).to_degrees();
        if dir < 0.0 { dir + 360.0 } else { dir }
    }

    /// 更新缓存
    pub fn update(&mut self, time: f64) {
        if (time - self.last_update).abs() > 1e-10 {
            self.cached_wind = self.get_wind_vector(time);
            self.last_update = time;
        }
    }

    /// 获取缓存的风速
    pub fn cached(&self) -> DVec2 {
        self.cached_wind
    }
}

/// 空间变化的风场提供者
#[derive(Debug, Clone)]
pub struct SpatialWindProvider {
    /// 基础风场
    pub base: WindProvider,
    /// 空间修正因子（每个单元）
    pub spatial_factor: Vec<f64>,
    /// 遮蔽因子（每个单元，0=完全遮蔽，1=无遮蔽）
    pub shelter_factor: Vec<f64>,
}

impl SpatialWindProvider {
    /// 创建新的空间变化风场
    pub fn new(base: WindProvider, n_cells: usize) -> Self {
        Self {
            base,
            spatial_factor: vec![1.0; n_cells],
            shelter_factor: vec![1.0; n_cells],
        }
    }

    /// 设置遮蔽因子
    pub fn set_shelter(&mut self, cell: usize, factor: f64) {
        if cell < self.shelter_factor.len() {
            self.shelter_factor[cell] = factor.max(0.0).min(1.0);
        }
    }

    /// 批量设置遮蔽因子
    pub fn set_shelter_zone(&mut self, cells: &[usize], factor: f64) {
        let f = factor.max(0.0).min(1.0);
        for &cell in cells {
            if cell < self.shelter_factor.len() {
                self.shelter_factor[cell] = f;
            }
        }
    }

    /// 获取单元风速
    pub fn get_cell_wind(&self, cell: usize, time: f64) -> (f64, f64) {
        let (u, v) = self.base.get_wind_at(time);

        let spatial = self.spatial_factor.get(cell).copied().unwrap_or(1.0);
        let shelter = self.shelter_factor.get(cell).copied().unwrap_or(1.0);
        let factor = spatial * shelter;

        (u * factor, v * factor)
    }

    /// 更新所有单元的风场
    pub fn update_all(&mut self, time: f64, wind_u: &mut [f64], wind_v: &mut [f64]) {
        self.base.update(time);
        let base_wind = self.base.cached();

        let n = self.shelter_factor.len().min(wind_u.len()).min(wind_v.len());
        for i in 0..n {
            let spatial = self.spatial_factor[i];
            let shelter = self.shelter_factor[i];
            let factor = spatial * shelter;

            wind_u[i] = base_wind.x * factor;
            wind_v[i] = base_wind.y * factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_wind_north() {
        let wind = WindProvider::constant(10.0, 0.0); // 北风
        let (u, v) = wind.get_wind_at(0.0);

        // 北风：向南吹，u=0, v=-10
        assert!(u.abs() < 1e-10);
        assert!((v - (-10.0)).abs() < 1e-10);
    }

    #[test]
    fn test_constant_wind_east() {
        let wind = WindProvider::constant(10.0, 90.0); // 东风
        let (u, v) = wind.get_wind_at(0.0);

        // 东风：向西吹，u=-10, v=0
        assert!((u - (-10.0)).abs() < 1e-10);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_calm_wind() {
        let wind = WindProvider::calm();
        let (u, v) = wind.get_wind_at(0.0);
        assert!(u.abs() < 1e-10);
        assert!(v.abs() < 1e-10);
    }

    #[test]
    fn test_time_series_interpolation() {
        let wind = WindProvider::time_series(
            vec![0.0, 100.0, 200.0],
            vec![0.0, 10.0, 5.0],
            vec![0.0, 0.0, 5.0],
        );

        // t=0
        let (u, v) = wind.get_wind_at(0.0);
        assert!(u.abs() < 1e-10);
        assert!(v.abs() < 1e-10);

        // t=50 (中点)
        let (u, _) = wind.get_wind_at(50.0);
        assert!((u - 5.0).abs() < 1e-10);

        // t=100
        let (u, _) = wind.get_wind_at(100.0);
        assert!((u - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_time_series_boundary() {
        let wind = WindProvider::time_series(
            vec![100.0, 200.0],
            vec![5.0, 10.0],
            vec![0.0, 0.0],
        );

        // 时间范围外使用边界值
        let (u, _) = wind.get_wind_at(0.0);
        assert!((u - 5.0).abs() < 1e-10);

        let (u, _) = wind.get_wind_at(300.0);
        assert!((u - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_wind_speed() {
        let wind = WindProvider::constant(10.0, 45.0);
        let speed = wind.get_wind_speed(0.0);
        assert!((speed - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_wind_direction() {
        let wind = WindProvider::constant(10.0, 180.0); // 南风
        let dir = wind.get_wind_direction(0.0);
        assert!((dir - 180.0).abs() < 1.0);
    }

    #[test]
    fn test_spatial_wind_shelter() {
        let base = WindProvider::constant(10.0, 0.0);
        let mut spatial = SpatialWindProvider::new(base, 10);

        spatial.set_shelter(0, 0.5); // 50% 遮蔽

        let (_, v) = spatial.get_cell_wind(0, 0.0);
        assert!((v - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_spatial_wind_update() {
        let base = WindProvider::constant(10.0, 0.0);
        let mut spatial = SpatialWindProvider::new(base, 10);

        spatial.set_shelter(0, 0.5);
        spatial.set_shelter(1, 0.0); // 完全遮蔽

        let mut wind_u = vec![0.0; 10];
        let mut wind_v = vec![0.0; 10];

        spatial.update_all(0.0, &mut wind_u, &mut wind_v);

        assert!((wind_v[0] - (-5.0)).abs() < 1e-10);
        assert!((wind_v[1]).abs() < 1e-10); // 完全遮蔽
        assert!((wind_v[2] - (-10.0)).abs() < 1e-10); // 无遮蔽
    }
}
