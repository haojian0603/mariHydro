// crates/mh_physics/src/forcing/spatial.rs

//! 空间时序数据集
//!
//! 提供多站点空间插值的时间序列数据，支持：
//! - IDW（反距离加权）空间插值
//! - 多站点数据管理
//! - 与时间序列结合的时空插值
//!
//! # 使用示例
//!
//! ```ignore
//! use glam::DVec2;
//! use mh_physics::forcing::spatial::SpatialTimeSeries;
//! use mh_physics::forcing::timeseries::TimeSeries;
//!
//! // 创建多个气象站点的降雨数据
//! let station1 = (
//!     DVec2::new(0.0, 0.0),
//!     TimeSeries::new(vec![0.0, 1.0, 2.0], vec![0.0, 10.0, 5.0]),
//! );
//! let station2 = (
//!     DVec2::new(100.0, 0.0),
//!     TimeSeries::new(vec![0.0, 1.0, 2.0], vec![0.0, 20.0, 10.0]),
//! );
//!
//! let spatial = SpatialTimeSeries::new(vec![station1, station2]);
//!
//! // 获取某位置和时间的插值
//! let value = spatial.get_value_at(DVec2::new(50.0, 0.0), 1.0);
//! // value ≈ 15.0 (两站点的平均)
//! ```

use glam::DVec2;
use serde::{Deserialize, Serialize};

use super::timeseries::TimeSeries;

/// 可序列化的站点数据
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StationData {
    /// 站点位置 (x, y)
    x: f64,
    y: f64,
    /// 时间序列数据
    series: TimeSeries,
}

impl StationData {
    fn position(&self) -> DVec2 {
        DVec2::new(self.x, self.y)
    }

    fn from_dvec2(pos: DVec2, series: TimeSeries) -> Self {
        Self {
            x: pos.x,
            y: pos.y,
            series,
        }
    }
}

/// 空间时序数据集（支持多站点IDW插值）
///
/// 管理多个空间站点的时间序列数据，提供任意位置和时间的插值功能。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialTimeSeries {
    /// 站点数据
    stations: Vec<StationData>,
    /// IDW 权重指数（默认 2.0）
    power: f64,
    /// 最小距离阈值（避免除以零）
    min_distance: f64,
}

impl SpatialTimeSeries {
    /// 从站点列表创建空间时序数据集
    ///
    /// # 参数
    ///
    /// - `stations`: 站点列表，每个站点包含 (位置, 时间序列)
    ///
    /// # Panics
    ///
    /// 如果站点列表为空
    pub fn new(stations: Vec<(DVec2, TimeSeries)>) -> Self {
        assert!(
            !stations.is_empty(),
            "SpatialTimeSeries requires at least one station"
        );
        let stations = stations
            .into_iter()
            .map(|(pos, series)| StationData::from_dvec2(pos, series))
            .collect();
        Self {
            stations,
            power: 2.0,
            min_distance: 1e-6,
        }
    }

    /// 设置 IDW 权重指数
    ///
    /// 较高的指数会使距离近的站点权重更大。
    /// 典型值：1.0-3.0，默认 2.0
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power.max(0.5);
        self
    }

    /// 设置最小距离阈值
    pub fn with_min_distance(mut self, min_dist: f64) -> Self {
        self.min_distance = min_dist.max(1e-10);
        self
    }

    /// 获取站点数量
    pub fn n_stations(&self) -> usize {
        self.stations.len()
    }

    /// 获取站点位置列表
    pub fn station_positions(&self) -> Vec<DVec2> {
        self.stations.iter().map(|s| s.position()).collect()
    }

    /// 获取指定站点的时间序列引用
    pub fn station_series(&self, idx: usize) -> Option<&TimeSeries> {
        self.stations.get(idx).map(|s| &s.series)
    }

    /// 获取所有站点在给定时间的值
    pub fn all_values_at_time(&self, time: f64) -> Vec<f64> {
        self.stations
            .iter()
            .map(|s| s.series.get_value(time))
            .collect()
    }

    /// IDW 插值获取指定位置和时间的值
    ///
    /// # 参数
    ///
    /// - `pos`: 空间位置
    /// - `time`: 查询时间 [s]
    ///
    /// # 返回
    ///
    /// 空间和时间插值后的值
    pub fn get_value_at(&self, pos: DVec2, time: f64) -> f64 {
        // 只有一个站点时直接返回
        if self.stations.len() == 1 {
            return self.stations[0].series.get_value(time);
        }

        let mut sum_weight = 0.0;
        let mut sum_weighted_value = 0.0;

        for station in &self.stations {
            let loc = station.position();
            let dist = pos.distance(loc);

            // 距离极小时直接返回该站点值
            if dist < self.min_distance {
                return station.series.get_value(time);
            }

            let weight = 1.0 / dist.powf(self.power);
            sum_weight += weight;
            sum_weighted_value += weight * station.series.get_value(time);
        }

        if sum_weight < 1e-14 {
            // 所有站点距离都很远，返回简单平均
            let total: f64 = self.stations.iter().map(|s| s.series.get_value(time)).sum();
            total / self.stations.len() as f64
        } else {
            sum_weighted_value / sum_weight
        }
    }

    /// 高精度 IDW 插值（使用 Kahan 求和）
    ///
    /// 当站点数量很多或权重差异很大时使用此方法。
    pub fn get_value_at_precise(&self, pos: DVec2, time: f64) -> f64 {
        // 只有一个站点时直接返回
        if self.stations.len() == 1 {
            return self.stations[0].series.get_value(time);
        }

        // Kahan 求和状态
        let mut sum_weight = 0.0;
        let mut comp_weight = 0.0;
        let mut sum_weighted = 0.0;
        let mut comp_weighted = 0.0;

        for station in &self.stations {
            let loc = station.position();
            let dist = pos.distance(loc);

            // 距离极小时直接返回该站点值
            if dist < self.min_distance {
                return station.series.get_value(time);
            }

            let weight = 1.0 / dist.powf(self.power);
            let value = station.series.get_value(time);

            // Kahan 本: 权重求和
            let y_w = weight - comp_weight;
            let t_w = sum_weight + y_w;
            comp_weight = (t_w - sum_weight) - y_w;
            sum_weight = t_w;

            // Kahan 加: 加权值求和
            let wv = weight * value;
            let y_wv = wv - comp_weighted;
            let t_wv = sum_weighted + y_wv;
            comp_weighted = (t_wv - sum_weighted) - y_wv;
            sum_weighted = t_wv;
        }

        if sum_weight < 1e-14 {
            // 回退到简单平均
            let mut sum = 0.0;
            let mut comp = 0.0;
            for station in &self.stations {
                let v = station.series.get_value(time);
                let y = v - comp;
                let t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
            sum / self.stations.len() as f64
        } else {
            sum_weighted / sum_weight
        }
    }

    /// 批量计算多个位置的 IDW 插值值（并行）
    pub fn get_values_at_parallel(&self, positions: &[DVec2], time: f64) -> Vec<f64> {
        use rayon::prelude::*;

        positions
            .par_iter()
            .map(|&pos| self.get_value_at(pos, time))
            .collect()
    }

    /// 获取最近站点的值（无插值）
    pub fn get_nearest_value(&self, pos: DVec2, time: f64) -> f64 {
        let mut min_dist = f64::MAX;
        let mut nearest_idx = 0;

        for (i, station) in self.stations.iter().enumerate() {
            let dist = pos.distance(station.position());
            if dist < min_dist {
                min_dist = dist;
                nearest_idx = i;
            }
        }

        self.stations[nearest_idx].series.get_value(time)
    }

    /// 计算指定位置的 IDW 权重
    pub fn compute_weights(&self, pos: DVec2) -> Vec<f64> {
        let mut weights = Vec::with_capacity(self.stations.len());
        let mut sum = 0.0;

        for station in &self.stations {
            let dist = pos.distance(station.position()).max(self.min_distance);
            let w = 1.0 / dist.powf(self.power);
            weights.push(w);
            sum += w;
        }

        // 归一化
        if sum > 1e-14 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// 添加站点
    pub fn add_station(&mut self, pos: DVec2, series: TimeSeries) {
        self.stations.push(StationData::from_dvec2(pos, series));
    }

    /// 获取数据范围
    pub fn spatial_bounds(&self) -> (DVec2, DVec2) {
        let mut min = DVec2::splat(f64::MAX);
        let mut max = DVec2::splat(f64::MIN);

        for station in &self.stations {
            let pos = station.position();
            min = min.min(pos);
            max = max.max(pos);
        }

        (min, max)
    }

    /// 获取时间范围
    pub fn time_range(&self) -> (f64, f64) {
        let mut t_min = f64::MAX;
        let mut t_max = f64::MIN;

        for station in &self.stations {
            let (t0, t1) = station.series.time_range();
            t_min = t_min.min(t0);
            t_max = t_max.max(t1);
        }

        (t_min, t_max)
    }
}

/// 空间向量时序数据集
///
/// 用于风场等向量数据的空间插值。
#[derive(Debug, Clone)]
pub struct SpatialVectorTimeSeries {
    /// X 分量空间时序
    x_spatial: SpatialTimeSeries,
    /// Y 分量空间时序
    y_spatial: SpatialTimeSeries,
}

impl SpatialVectorTimeSeries {
    /// 从分量创建
    pub fn new(x_spatial: SpatialTimeSeries, y_spatial: SpatialTimeSeries) -> Self {
        assert_eq!(
            x_spatial.n_stations(),
            y_spatial.n_stations(),
            "X and Y spatial series must have same number of stations"
        );
        Self { x_spatial, y_spatial }
    }

    /// 从站点数据创建
    pub fn from_stations(
        positions: Vec<DVec2>,
        x_series: Vec<TimeSeries>,
        y_series: Vec<TimeSeries>,
    ) -> Self {
        assert_eq!(positions.len(), x_series.len());
        assert_eq!(positions.len(), y_series.len());

        let x_stations: Vec<_> = positions
            .iter()
            .zip(x_series)
            .map(|(&pos, series)| (pos, series))
            .collect();

        let y_stations: Vec<_> = positions
            .into_iter()
            .zip(y_series)
            .collect();

        Self {
            x_spatial: SpatialTimeSeries::new(x_stations),
            y_spatial: SpatialTimeSeries::new(y_stations),
        }
    }

    /// 获取指定位置和时间的向量值
    pub fn get_value_at(&self, pos: DVec2, time: f64) -> (f64, f64) {
        (
            self.x_spatial.get_value_at(pos, time),
            self.y_spatial.get_value_at(pos, time),
        )
    }

    /// 获取指定位置和时间的模长
    pub fn get_magnitude_at(&self, pos: DVec2, time: f64) -> f64 {
        let (x, y) = self.get_value_at(pos, time);
        (x * x + y * y).sqrt()
    }

    /// 获取指定位置和时间的方向（弧度）
    pub fn get_direction_at(&self, pos: DVec2, time: f64) -> f64 {
        let (x, y) = self.get_value_at(pos, time);
        y.atan2(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_station(x: f64, y: f64, values: Vec<f64>) -> (DVec2, TimeSeries) {
        let times: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();
        (DVec2::new(x, y), TimeSeries::new(times, values))
    }

    #[test]
    fn test_single_station() {
        let station = make_station(0.0, 0.0, vec![0.0, 1.0, 2.0]);
        let spatial = SpatialTimeSeries::new(vec![station]);

        let value = spatial.get_value_at(DVec2::new(100.0, 100.0), 1.0);
        assert!((value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_stations_equal_distance() {
        let station1 = make_station(0.0, 0.0, vec![0.0, 10.0, 0.0]);
        let station2 = make_station(100.0, 0.0, vec![0.0, 20.0, 0.0]);

        let spatial = SpatialTimeSeries::new(vec![station1, station2]);

        // 中点应该是两站点的平均
        let value = spatial.get_value_at(DVec2::new(50.0, 0.0), 1.0);
        assert!((value - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_idw_weights() {
        let station1 = make_station(0.0, 0.0, vec![0.0]);
        let station2 = make_station(100.0, 0.0, vec![0.0]);

        let spatial = SpatialTimeSeries::new(vec![station1, station2]);

        // 中点权重应该相等
        let weights = spatial.compute_weights(DVec2::new(50.0, 0.0));
        assert!((weights[0] - weights[1]).abs() < 1e-10);

        // 靠近站点1的权重应该更大
        let weights = spatial.compute_weights(DVec2::new(25.0, 0.0));
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_very_close_to_station() {
        let station1 = make_station(0.0, 0.0, vec![0.0, 100.0, 0.0]);
        let station2 = make_station(100.0, 0.0, vec![0.0, 0.0, 0.0]);

        let spatial = SpatialTimeSeries::new(vec![station1, station2]);

        // 非常接近站点1时应该返回站点1的值
        let value = spatial.get_value_at(DVec2::new(1e-8, 0.0), 1.0);
        assert!((value - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_spatial_vector() {
        let x_stations = vec![
            make_station(0.0, 0.0, vec![1.0]),
            make_station(100.0, 0.0, vec![-1.0]),
        ];
        let y_stations = vec![
            make_station(0.0, 0.0, vec![0.0]),
            make_station(100.0, 0.0, vec![0.0]),
        ];

        let x_spatial = SpatialTimeSeries::new(x_stations);
        let y_spatial = SpatialTimeSeries::new(y_stations);
        let vector = SpatialVectorTimeSeries::new(x_spatial, y_spatial);

        let (x, y) = vector.get_value_at(DVec2::new(50.0, 0.0), 0.0);
        assert!((x - 0.0).abs() < 1e-10);  // 中点 x 分量应该是 0
        assert!((y - 0.0).abs() < 1e-10);
    }
}
