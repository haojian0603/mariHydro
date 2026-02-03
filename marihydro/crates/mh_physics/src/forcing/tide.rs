// crates/mh_physics/src/forcing/tide.rs

//! 潮汐数据提供者
//!
//! 提供潮汐边界条件数据，包括：
//! - 简谐分潮合成
//! - 时间序列插值
//! - 潮汐预报
//!
//! # 简谐分潮
//!
//! 潮位使用分潮叠加计算：
//! ```text
//! η(t) = η₀ + Σ Hᵢ cos(ωᵢt - gᵢ)
//! ```
//!
//! 其中：
//! - η₀ 是平均潮位
//! - Hᵢ 是分潮振幅
//! - ωᵢ 是分潮角频率
//! - gᵢ 是分潮迟角

use std::f64::consts::PI;

/// 主要分潮常数
pub mod tidal_constituents {
    use std::f64::consts::PI;
    
    /// M2 半日潮（主太阴半日潮）
    pub const M2_PERIOD: f64 = 12.4206012;  // 小时
    pub const M2_SPEED: f64 = 28.9841042;    // 度/小时

    /// S2 半日潮（主太阳半日潮）
    pub const S2_PERIOD: f64 = 12.0;
    pub const S2_SPEED: f64 = 30.0;

    /// N2 半日潮（较大月椭圆潮）
    pub const N2_PERIOD: f64 = 12.6583482;
    pub const N2_SPEED: f64 = 28.4397295;

    /// K1 日潮（主日月合成日潮）
    pub const K1_PERIOD: f64 = 23.9344694;
    pub const K1_SPEED: f64 = 15.0410686;

    /// O1 日潮（主太阴日潮）
    pub const O1_PERIOD: f64 = 25.8193417;
    pub const O1_SPEED: f64 = 13.9430356;

    /// P1 日潮（主太阳日潮）
    pub const P1_PERIOD: f64 = 24.0658896;
    pub const P1_SPEED: f64 = 14.9589314;

    /// 角速度转换：度/小时 -> 弧度/秒
    pub fn speed_to_omega(speed_deg_per_hour: f64) -> f64 {
        speed_deg_per_hour * PI / 180.0 / 3600.0
    }
}

/// 单个分潮
#[derive(Debug, Clone, Copy)]
pub struct TidalConstituent {
    /// 分潮名称
    pub name: &'static str,
    /// 振幅 [m]
    pub amplitude: f64,
    /// 角速度 [rad/s]
    pub omega: f64,
    /// 迟角 [rad]
    pub phase: f64,
}

impl TidalConstituent {
    /// 创建新分潮
    pub fn new(name: &'static str, amplitude: f64, period_hours: f64, phase_deg: f64) -> Self {
        Self {
            name,
            amplitude,
            omega: 2.0 * PI / (period_hours * 3600.0),
            phase: phase_deg.to_radians(),
        }
    }

    /// 使用角速度创建分潮
    pub fn with_speed(name: &'static str, amplitude: f64, speed_deg_per_hour: f64, phase_deg: f64) -> Self {
        Self {
            name,
            amplitude,
            omega: tidal_constituents::speed_to_omega(speed_deg_per_hour),
            phase: phase_deg.to_radians(),
        }
    }

    /// 创建 M2 分潮
    pub fn m2(amplitude: f64, phase_deg: f64) -> Self {
        Self::with_speed("M2", amplitude, tidal_constituents::M2_SPEED, phase_deg)
    }

    /// 创建 S2 分潮
    pub fn s2(amplitude: f64, phase_deg: f64) -> Self {
        Self::with_speed("S2", amplitude, tidal_constituents::S2_SPEED, phase_deg)
    }

    /// 创建 K1 分潮
    pub fn k1(amplitude: f64, phase_deg: f64) -> Self {
        Self::with_speed("K1", amplitude, tidal_constituents::K1_SPEED, phase_deg)
    }

    /// 创建 O1 分潮
    pub fn o1(amplitude: f64, phase_deg: f64) -> Self {
        Self::with_speed("O1", amplitude, tidal_constituents::O1_SPEED, phase_deg)
    }

    /// 计算分潮贡献
    #[inline]
    pub fn compute(&self, time: f64) -> f64 {
        self.amplitude * (self.omega * time - self.phase).cos()
    }
}

/// 潮汐数据提供者
#[derive(Debug, Clone)]
pub struct TideProvider {
    /// 数据类型
    data: TideData,
    /// 平均潮位 [m]
    mean_level: f64,
    /// 最后更新时间
    last_update: f64,
    /// 缓存的潮位
    cached_level: f64,
}

/// 潮汐数据类型
#[derive(Debug, Clone)]
pub enum TideData {
    /// 恒定水位
    Constant(f64),
    /// 简谐分潮
    Harmonic(Vec<TidalConstituent>),
    /// 时间序列
    TimeSeries {
        times: Vec<f64>,
        levels: Vec<f64>,
    },
    /// 简单正弦潮
    SimpleSine {
        amplitude: f64,
        period: f64,
        phase: f64,
    },
}

impl TideProvider {
    fn normalize_timeseries(times: Vec<f64>, levels: Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        let n = times.len().min(levels.len());
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut tuples: Vec<(f64, f64)> = (0..n)
            .filter_map(|i| {
                let t = times[i];
                let v = levels[i];
                if t.is_finite() && v.is_finite() {
                    Some((t, v))
                } else {
                    None
                }
            })
            .collect();
        tuples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut out_t = Vec::with_capacity(tuples.len());
        let mut out_v = Vec::with_capacity(tuples.len());
        let mut last_t = None;
        for (t, v) in tuples {
            if last_t.map_or(true, |lt| t > lt) {
                out_t.push(t);
                out_v.push(v);
                last_t = Some(t);
            }
        }

        (out_t, out_v)
    }

    /// 创建恒定水位
    pub fn constant(level: f64) -> Self {
        Self {
            data: TideData::Constant(level),
            mean_level: level,
            last_update: 0.0,
            cached_level: level,
        }
    }

    /// 创建简谐分潮组合
    pub fn harmonic(mean_level: f64, constituents: Vec<TidalConstituent>) -> Self {
        Self {
            data: TideData::Harmonic(constituents),
            mean_level,
            last_update: 0.0,
            cached_level: mean_level,
        }
    }

    /// 创建简单正弦潮（单分潮）
    pub fn simple_sine(mean_level: f64, amplitude: f64, period_hours: f64, phase_deg: f64) -> Self {
        Self {
            data: TideData::SimpleSine {
                amplitude,
                period: period_hours * 3600.0,
                phase: phase_deg.to_radians(),
            },
            mean_level,
            last_update: 0.0,
            cached_level: mean_level,
        }
    }

    /// 创建典型半日潮（M2 + S2）
    pub fn semidiurnal(mean_level: f64, m2_amp: f64, m2_phase: f64, s2_amp: f64, s2_phase: f64) -> Self {
        Self::harmonic(mean_level, vec![
            TidalConstituent::m2(m2_amp, m2_phase),
            TidalConstituent::s2(s2_amp, s2_phase),
        ])
    }

    /// 创建典型混合潮（M2 + S2 + K1 + O1）
    pub fn mixed(
        mean_level: f64,
        m2_amp: f64, m2_phase: f64,
        s2_amp: f64, s2_phase: f64,
        k1_amp: f64, k1_phase: f64,
        o1_amp: f64, o1_phase: f64,
    ) -> Self {
        Self::harmonic(mean_level, vec![
            TidalConstituent::m2(m2_amp, m2_phase),
            TidalConstituent::s2(s2_amp, s2_phase),
            TidalConstituent::k1(k1_amp, k1_phase),
            TidalConstituent::o1(o1_amp, o1_phase),
        ])
    }

    /// 创建时间序列
    pub fn time_series(times: Vec<f64>, levels: Vec<f64>) -> Self {
        let (times, levels) = Self::normalize_timeseries(times, levels);
        let mean = if levels.is_empty() {
            0.0
        } else {
            levels.iter().sum::<f64>() / levels.len() as f64
        };
        let initial = levels.first().copied().unwrap_or(mean);

        Self {
            data: TideData::TimeSeries { times, levels },
            mean_level: mean,
            last_update: 0.0,
            cached_level: initial,
        }
    }

    /// 获取指定时刻的潮位
    pub fn get_level_at(&self, time: f64) -> f64 {
        match &self.data {
            TideData::Constant(level) => *level,

            TideData::Harmonic(constituents) => {
                let mut level = self.mean_level;
                for c in constituents {
                    level += c.compute(time);
                }
                level
            }

            TideData::TimeSeries { times, levels } => {
                if times.is_empty() {
                    return self.mean_level;
                }

                if time <= times[0] {
                    return levels[0];
                }
                if time >= *times.last().unwrap() {
                    return *levels.last().unwrap();
                }

                // 线性插值
                for i in 0..times.len() - 1 {
                    if time >= times[i] && time < times[i + 1] {
                        let t = (time - times[i]) / (times[i + 1] - times[i]);
                        return levels[i] + t * (levels[i + 1] - levels[i]);
                    }
                }

                self.mean_level
            }

            TideData::SimpleSine { amplitude, period, phase } => {
                let omega = 2.0 * PI / period;
                self.mean_level + amplitude * (omega * time - phase).cos()
            }
        }
    }

    /// 获取潮位变化率 [m/s]
    pub fn get_rate_at(&self, time: f64) -> f64 {
        match &self.data {
            TideData::Constant(_) => 0.0,

            TideData::Harmonic(constituents) => {
                let mut rate = 0.0;
                for c in constituents {
                    // d/dt [H cos(ωt - g)] = -H ω sin(ωt - g)
                    rate += -c.amplitude * c.omega * (c.omega * time - c.phase).sin();
                }
                rate
            }

            TideData::TimeSeries { times, levels } => {
                let n = times.len().min(levels.len());
                if n < 2 {
                    return 0.0;
                }

                // 简单差分
                for i in 0..n - 1 {
                    if time >= times[i] && time < times[i + 1] {
                        let dt = times[i + 1] - times[i];
                        if dt.abs() < 1e-12 {
                            return 0.0;
                        }
                        return (levels[i + 1] - levels[i]) / dt;
                    }
                }

                0.0
            }

            TideData::SimpleSine { amplitude, period, phase } => {
                let omega = 2.0 * PI / period;
                -amplitude * omega * (omega * time - phase).sin()
            }
        }
    }

    /// 获取平均潮位
    pub fn mean_level(&self) -> f64 {
        self.mean_level
    }

    /// 更新缓存
    pub fn update(&mut self, time: f64) {
        if (time - self.last_update).abs() > 1e-10 {
            self.cached_level = self.get_level_at(time);
            self.last_update = time;
        }
    }

    /// 获取缓存的潮位
    pub fn cached(&self) -> f64 {
        self.cached_level
    }

    /// 计算潮差
    pub fn tidal_range(&self) -> f64 {
        match &self.data {
            TideData::Constant(_) => 0.0,
            TideData::Harmonic(constituents) => {
                // 粗略估计：所有振幅之和的两倍
                2.0 * constituents.iter().map(|c| c.amplitude).sum::<f64>()
            }
            TideData::TimeSeries { levels, .. } => {
                if levels.is_empty() { return 0.0; }
                let max = levels.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min = levels.iter().cloned().fold(f64::INFINITY, f64::min);
                max - min
            }
            TideData::SimpleSine { amplitude, .. } => 2.0 * amplitude,
        }
    }
}

/// 多点潮汐边界
#[derive(Debug, Clone)]
pub struct TideBoundary {
    /// 边界单元
    pub cells: Vec<usize>,
    /// 潮汐提供者
    pub provider: TideProvider,
    /// 空间插值权重（可选）
    pub weights: Option<Vec<f64>>,
}

// ============================================================
// 天文潮（转接到完整 astronomical_tide 实现）
// ============================================================

pub type TidePrediction = super::astronomical_tide::TidePrediction;
pub type AstronomicalTideEngine = super::astronomical_tide::AstronomicalTideEngine;

impl TideBoundary {
    /// 创建均匀潮汐边界
    pub fn uniform(cells: Vec<usize>, provider: TideProvider) -> Self {
        Self {
            cells,
            provider,
            weights: None,
        }
    }

    /// 创建带权重的潮汐边界
    pub fn with_weights(cells: Vec<usize>, provider: TideProvider, weights: Vec<f64>) -> Self {
        Self {
            cells,
            provider,
            weights: Some(weights),
        }
    }

    /// 获取指定单元的潮位
    pub fn get_cell_level(&self, cell_index: usize, time: f64) -> f64 {
        let base_level = self.provider.get_level_at(time);

        match &self.weights {
            Some(w) if cell_index < w.len() => base_level * w[cell_index],
            _ => base_level,
        }
    }

    /// 更新所有单元水位
    pub fn update_levels(&mut self, time: f64, water_levels: &mut [f64]) {
        // 直接计算当前潮位（不依赖缓存）
        let base_level = self.provider.get_level_at(time);

        for (i, &cell) in self.cells.iter().enumerate() {
            if cell < water_levels.len() {
                let level = match &self.weights {
                    Some(w) if i < w.len() => base_level * w[i],
                    _ => base_level,
                };
                water_levels[cell] = level;
            }
        }
    }
}

// ============================================================================
// 潮汐联合边界（水位 + 流速）
// ============================================================================

/// 潮汐联合边界条件
///
/// 同时提供水位和流速边界，支持相位差管理与 Flather 修正。
#[derive(Debug, Clone)]
pub struct TidalJointBoundary {
    /// 水位提供者
    level_provider: TideProvider,
    /// 流速模型
    velocity_model: TidalVelocityCalculator,
    /// 边界法向方向 [rad]
    normal_direction: f64,
    /// Flather 系数 (sqrt(h/g))
    flather_coeff: Option<f64>,
}

impl TidalJointBoundary {
    /// 创建联合边界
    pub fn new(level_provider: TideProvider, velocity_model: TidalVelocityCalculator) -> Self {
        Self {
            level_provider,
            velocity_model,
            normal_direction: 0.0,
            flather_coeff: None,
        }
    }

    /// 设置边界法向方向
    pub fn with_normal_direction(mut self, direction_deg: f64) -> Self {
        self.normal_direction = direction_deg.to_radians();
        self
    }

    /// 启用 Flather 边界条件
    pub fn with_flather(mut self, mean_depth: f64) -> Self {
        let g = 9.81;
        self.flather_coeff = Some((mean_depth / g).sqrt());
        self
    }

    /// 获取水位边界值
    pub fn get_level(&self, time: f64) -> f64 {
        self.level_provider.get_level_at(time)
    }

    /// 获取流速边界值（返回 u, v 分量）
    pub fn get_velocity(&self, time: f64) -> (f64, f64) {
        let _ = self.level_provider.get_level_at(time);
        self.velocity_model.compute(time)
    }

    /// 获取法向流速
    pub fn get_normal_velocity(&self, time: f64) -> f64 {
        let (u, v) = self.get_velocity(time);
        u * self.normal_direction.cos() + v * self.normal_direction.sin()
    }

    /// Flather 边界条件修正（水位）
    pub fn flather_correction(
        &self,
        time: f64,
        _interior_level: f64,
        interior_velocity: f64,
    ) -> Option<f64> {
        let coeff = self.flather_coeff?;
        let tidal_level = self.get_level(time);
        let tidal_velocity = self.get_normal_velocity(time);

        let correction = coeff * (interior_velocity - tidal_velocity);
        Some(tidal_level - correction)
    }
}

/// 分潮流速计算器
#[derive(Debug, Clone)]
pub struct TidalVelocityCalculator {
    constituents: Vec<TidalVelocityConstituent>,
}

/// 流速分潮
#[derive(Debug, Clone, Copy)]
pub struct TidalVelocityConstituent {
    pub name: &'static str,
    pub u_amplitude: f64,
    pub v_amplitude: f64,
    pub omega: f64,
    pub u_phase: f64,
    pub v_phase: f64,
}

impl TidalVelocityCalculator {
    pub fn new() -> Self {
        Self { constituents: Vec::new() }
    }

    pub fn add_constituent(&mut self, constituent: TidalVelocityConstituent) {
        self.constituents.push(constituent);
    }

    /// 从水位分潮估算流速分潮（长波近似）
    pub fn from_level_constituents(
        level_constituents: &[TidalConstituent],
        mean_depth: f64,
        direction: f64,
    ) -> Self {
        let g = 9.81;
        let factor = (g / mean_depth).sqrt();
        let dir = direction.to_radians();

        let constituents = level_constituents
            .iter()
            .map(|lc| {
                let amp = lc.amplitude * factor;
                let u_amp = amp * dir.cos();
                let v_amp = amp * dir.sin();
                TidalVelocityConstituent {
                    name: lc.name,
                    u_amplitude: u_amp,
                    v_amplitude: v_amp,
                    omega: lc.omega,
                    u_phase: lc.phase,
                    v_phase: lc.phase,
                }
            })
            .collect();

        Self { constituents }
    }

    /// 计算指定时刻的流速
    pub fn compute(&self, time: f64) -> (f64, f64) {
        let mut u = 0.0;
        let mut v = 0.0;

        for c in &self.constituents {
            u += c.u_amplitude * (c.omega * time - c.u_phase).cos();
            v += c.v_amplitude * (c.omega * time - c.v_phase).cos();
        }

        (u, v)
    }
}

impl Default for TidalVelocityCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_tide() {
        let tide = TideProvider::constant(1.5);
        assert!((tide.get_level_at(0.0) - 1.5).abs() < 1e-10);
        assert!((tide.get_level_at(1000.0) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_simple_sine() {
        let tide = TideProvider::simple_sine(0.0, 1.0, 12.0, 0.0);

        // t=0: cos(0) = 1
        assert!((tide.get_level_at(0.0) - 1.0).abs() < 1e-10);

        // t=6h: cos(π) = -1
        assert!((tide.get_level_at(6.0 * 3600.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_m2_constituent() {
        let m2 = TidalConstituent::m2(1.0, 0.0);
        assert_eq!(m2.name, "M2");
        assert!((m2.amplitude - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_tide() {
        let tide = TideProvider::harmonic(0.0, vec![
            TidalConstituent::m2(1.0, 0.0),
            TidalConstituent::s2(0.5, 0.0),
        ]);

        // t=0: 两个分潮都是最大值
        let level = tide.get_level_at(0.0);
        assert!((level - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_time_series() {
        let tide = TideProvider::time_series(
            vec![0.0, 3600.0, 7200.0],
            vec![0.0, 1.0, 0.5],
        );

        assert!((tide.get_level_at(0.0) - 0.0).abs() < 1e-10);
        assert!((tide.get_level_at(1800.0) - 0.5).abs() < 1e-10); // 插值
        assert!((tide.get_level_at(3600.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_tidal_range() {
        let tide = TideProvider::simple_sine(0.0, 2.0, 12.0, 0.0);
        assert!((tide.tidal_range() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_tide_rate() {
        let tide = TideProvider::constant(1.0);
        assert!((tide.get_rate_at(0.0)).abs() < 1e-10);

        let tide_sine = TideProvider::simple_sine(0.0, 1.0, 12.0, 0.0);
        // t=0: rate = 0 (at maximum)
        assert!(tide_sine.get_rate_at(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_tide_boundary() {
        let provider = TideProvider::simple_sine(1.0, 0.5, 12.0, 0.0);
        let mut boundary = TideBoundary::uniform(vec![0, 1, 2], provider);

        let mut levels = vec![0.0; 10];
        boundary.update_levels(0.0, &mut levels);

        assert!((levels[0] - 1.5).abs() < 1e-6);
        assert!((levels[1] - 1.5).abs() < 1e-6);
        assert!((levels[2] - 1.5).abs() < 1e-6);
        assert!((levels[3]).abs() < 1e-10); // 非边界单元
    }

    #[test]
    fn test_semidiurnal_tide() {
        let tide = TideProvider::semidiurnal(0.0, 1.0, 0.0, 0.3, 30.0);

        match &tide.data {
            TideData::Harmonic(c) => {
                assert_eq!(c.len(), 2);
            }
            _ => panic!("Expected Harmonic"),
        }
    }
}
