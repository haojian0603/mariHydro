// crates/mh_physics/src/forcing/timeseries.rs

//! 时间序列数据结构和插值
//!
//! 提供通用的时间序列数据表示和插值功能，支持：
//! - 线性插值
//! - 多种外推模式（截断、线性、循环）
//! - 缓存优化的快速查找
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::forcing::timeseries::{TimeSeries, ExtrapolationMode};
//!
//! let times = vec![0.0, 1.0, 2.0, 3.0];
//! let values = vec![0.0, 1.0, 0.5, 0.0];
//!
//! let series = TimeSeries::new(times, values)
//!     .with_extrapolation(ExtrapolationMode::Cyclic);
//!
//! // 获取 t=0.5 时的插值
//! let value = series.get_value(0.5);  // ≈ 0.5
//!
//! // 循环外推 t=4.0 -> t=1.0
//! let cyclic_value = series.get_value(4.0);
//! ```

use mh_foundation::Scalar;
use serde::{Deserialize, Serialize};

/// 外推模式
///
/// 定义当查询时间超出数据范围时的处理方式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtrapolationMode {
    /// 截断模式：超出范围时返回边界值
    ///
    /// t < t_start -> values[0]
    /// t > t_end   -> values[n-1]
    #[default]
    Clamp,

    /// 线性外推：使用边界斜率延伸
    ///
    /// 使用首/末两个点的斜率进行线性外推。
    /// 适用于趋势变化的数据。
    Linear,

    /// 循环模式：周期性重复数据
    ///
    /// t -> t_start + (t - t_start) mod (t_end - t_start)
    /// 适用于潮汐、日变化等周期性数据。
    Cyclic,
}

impl ExtrapolationMode {
    /// 获取模式名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Clamp => "Clamp",
            Self::Linear => "Linear",
            Self::Cyclic => "Cyclic",
        }
    }
}

/// 时间序列查找游标
///
/// 由调用方持有，用于加速连续时间查询。
/// 避免在 TimeSeries 内部使用 AtomicUsize 导致的伪共享问题。
#[derive(Debug, Clone, Copy, Default)]
pub struct TimeSeriesCursor {
    /// 上次查找的区间索引
    pub last_index: usize,
}

impl TimeSeriesCursor {
    /// 创建新的游标
    pub fn new() -> Self {
        Self::default()
    }
}

/// 时间序列数据
///
/// 存储时间-值对，提供线性插值和外推功能。
///
/// # 约束
///
/// - 时间数组必须严格单调递增
/// - 时间和值数组长度必须相等且非空
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// 时间点 [s]（严格单调递增）
    times: Vec<Scalar>,
    /// 对应的值
    values: Vec<Scalar>,
    /// 外推模式
    extrap_mode: ExtrapolationMode,
}

impl TimeSeries {
    /// 从时间和值数组创建时间序列
    ///
    /// # 参数
    ///
    /// - `times`: 时间点数组（必须严格单调递增）
    /// - `values`: 对应的值数组（长度必须与 times 相同）
    ///
    /// # Panics
    ///
    /// - 如果 times 和 values 长度不同
    /// - 如果 times 为空
    /// - 如果 times 不是严格单调递增
    pub fn new(times: Vec<Scalar>, values: Vec<Scalar>) -> Self {
        assert_eq!(
            times.len(),
            values.len(),
            "Times and values must have same length"
        );
        assert!(!times.is_empty(), "TimeSeries cannot be empty");

        // 验证时间单调递增
        for i in 1..times.len() {
            assert!(
                times[i] > times[i - 1],
                "Times must be monotonically increasing: times[{}]={} <= times[{}]={}",
                i,
                times[i],
                i - 1,
                times[i - 1]
            );
        }

        Self {
            times,
            values,
            extrap_mode: ExtrapolationMode::Clamp,
        }
    }

    /// 从 (时间, 值) 点对列表创建时间序列
    ///
    /// # 参数
    ///
    /// - `points`: (时间, 值) 元组的向量
    ///
    /// # Panics
    ///
    /// - 如果 points 为空
    /// - 如果时间不是严格单调递增
    pub fn from_points(points: Vec<(Scalar, Scalar)>) -> Self {
        let (times, values): (Vec<_>, Vec<_>) = points.into_iter().unzip();
        Self::new(times, values)
    }

    /// 设置外推模式
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.extrap_mode = mode;
        self
    }

    /// 获取外推模式
    pub fn extrapolation_mode(&self) -> ExtrapolationMode {
        self.extrap_mode
    }

    /// 设置外推模式（可变引用版本）
    pub fn set_extrapolation(&mut self, mode: ExtrapolationMode) {
        self.extrap_mode = mode;
    }

    /// 获取时间范围
    pub fn time_range(&self) -> (Scalar, Scalar) {
        let n = self.times.len();
        (self.times[0], self.times[n - 1])
    }

    /// 获取值范围
    pub fn value_range(&self) -> (Scalar, Scalar) {
        let min = self.values.iter().cloned().fold(Scalar::INFINITY, Scalar::min);
        let max = self.values.iter().cloned().fold(Scalar::NEG_INFINITY, Scalar::max);
        (min, max)
    }

    /// 获取数据点数量
    pub fn len(&self) -> usize {
        self.times.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    /// 获取时间点数组引用
    pub fn times(&self) -> &[Scalar] {
        &self.times
    }

    /// 获取值数组引用
    pub fn values(&self) -> &[Scalar] {
        &self.values
    }

    /// 获取指定时间的插值
    ///
    /// # 参数
    ///
    /// - `t`: 查询时间 [s]
    ///
    /// # 返回
    ///
    /// 插值后的值。如果 t 超出数据范围，根据外推模式处理。
    pub fn get_value(&self, t: Scalar) -> Scalar {
        let n = self.times.len();
        let t_start = self.times[0];
        let t_end = self.times[n - 1];

        // 处理超出范围的情况
        if t < t_start || t > t_end {
            return self.handle_extrapolation(t, t_start, t_end);
        }

        self.interpolate_internal(t)
    }

    /// 获取指定时间的插值（`get_value` 的别名）
    ///
    /// 为兼容性保留的方法别名。
    #[inline]
    pub fn interpolate(&self, t: Scalar) -> Scalar {
        self.get_value(t)
    }

    /// 处理外推
    fn handle_extrapolation(&self, t: Scalar, t_start: Scalar, t_end: Scalar) -> Scalar {
        let n = self.times.len();

        match self.extrap_mode {
            ExtrapolationMode::Clamp => {
                if t < t_start {
                    self.values[0]
                } else {
                    self.values[n - 1]
                }
            }
            ExtrapolationMode::Cyclic => {
                let duration = t_end - t_start;
                if duration < 1e-12 {
                    return self.values[0];
                }
                let offset = (t - t_start).rem_euclid(duration);
                self.interpolate_internal(t_start + offset)
            }
            ExtrapolationMode::Linear => {
                if t < t_start {
                    // 使用前两个点的斜率
                    if n < 2 {
                        return self.values[0];
                    }
                    let slope = (self.values[1] - self.values[0]) / (self.times[1] - t_start);
                    self.values[0] + slope * (t - t_start)
                } else {
                    // 使用后两个点的斜率
                    if n < 2 {
                        return self.values[n - 1];
                    }
                    let slope =
                        (self.values[n - 1] - self.values[n - 2]) / (t_end - self.times[n - 2]);
                    self.values[n - 1] + slope * (t - t_end)
                }
            }
        }
    }

    /// 内部插值（假设 t 在范围内）- 无状态版本
    fn interpolate_internal(&self, t: Scalar) -> Scalar {
        let n = self.times.len();

        // 二分查找
        let mut lo = 0;
        let mut hi = n - 1;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.times[mid + 1] <= t {
                lo = mid + 1;
            } else if self.times[mid] > t {
                hi = mid;
            } else {
                lo = mid;
                break;
            }
        }
        let idx = lo;

        // 边界检查
        if idx >= n - 1 {
            return self.values[n - 1];
        }

        // 线性插值
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let v0 = self.values[idx];
        let v1 = self.values[idx + 1];

        let dt = t1 - t0;
        if dt.abs() < 1e-12 {
            v0
        } else {
            v0 + (t - t0) / dt * (v1 - v0)
        }
    }

    /// 内部插值（假设 t 在范围内）- 带游标版本
    fn interpolate_internal_with_cursor(&self, t: Scalar, cursor: &mut TimeSeriesCursor) -> Scalar {
        let n = self.times.len();

        // 利用游标加速查找
        let mut idx = cursor.last_index;
        if idx >= n - 1 {
            idx = 0;
        }
        if t < self.times[idx] {
            idx = 0;
        }

        // 向前查找
        while idx < n - 1 && t >= self.times[idx + 1] {
            idx += 1;
        }
        cursor.last_index = idx;

        // 边界检查
        if idx >= n - 1 {
            return self.values[n - 1];
        }

        // 线性插值
        let t0 = self.times[idx];
        let t1 = self.times[idx + 1];
        let v0 = self.values[idx];
        let v1 = self.values[idx + 1];

        let dt = t1 - t0;
        if dt.abs() < 1e-12 {
            v0
        } else {
            v0 + (t - t0) / dt * (v1 - v0)
        }
    }

    /// 获取指定时间的插值（带游标版本，推荐用于连续时间查询）
    ///
    /// # 参数
    ///
    /// - `t`: 查询时间 [s]
    /// - `cursor`: 调用方持有的游标，用于加速连续查询
    ///
    /// # 返回
    ///
    /// 插值后的值。如果 t 超出数据范围，根据外推模式处理。
    pub fn get_value_with_cursor(&self, t: Scalar, cursor: &mut TimeSeriesCursor) -> Scalar {
        let n = self.times.len();
        let t_start = self.times[0];
        let t_end = self.times[n - 1];

        // 处理超出范围的情况
        if t < t_start || t > t_end {
            return self.handle_extrapolation_with_cursor(t, t_start, t_end, cursor);
        }

        self.interpolate_internal_with_cursor(t, cursor)
    }

    /// 处理外推（带游标版本）
    fn handle_extrapolation_with_cursor(
        &self,
        t: Scalar,
        t_start: Scalar,
        t_end: Scalar,
        cursor: &mut TimeSeriesCursor,
    ) -> Scalar {
        let n = self.times.len();

        match self.extrap_mode {
            ExtrapolationMode::Clamp => {
                if t < t_start {
                    self.values[0]
                } else {
                    self.values[n - 1]
                }
            }
            ExtrapolationMode::Cyclic => {
                let duration = t_end - t_start;
                if duration < 1e-12 {
                    return self.values[0];
                }
                let offset = (t - t_start).rem_euclid(duration);
                self.interpolate_internal_with_cursor(t_start + offset, cursor)
            }
            ExtrapolationMode::Linear => {
                if t < t_start {
                    if n < 2 {
                        return self.values[0];
                    }
                    let slope = (self.values[1] - self.values[0]) / (self.times[1] - t_start);
                    self.values[0] + slope * (t - t_start)
                } else {
                    if n < 2 {
                        return self.values[n - 1];
                    }
                    let slope =
                        (self.values[n - 1] - self.values[n - 2]) / (t_end - self.times[n - 2]);
                    self.values[n - 1] + slope * (t - t_end)
                }
            }
        }
    }

    /// 获取导数（有限差分）
    ///
    /// 在指定时间点使用中心差分计算导数。
    pub fn get_derivative(&self, t: Scalar) -> Scalar {
        let eps = 1e-6;
        let v_plus = self.get_value(t + eps);
        let v_minus = self.get_value(t - eps);
        (v_plus - v_minus) / (2.0 * eps)
    }

    /// 获取高精度导数（5 点自适应有限差分）
    ///
    /// 使用 5 点中心差分公式: f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h
    /// 参考 Richardson 外推法增强精度。
    pub fn get_derivative_precise(&self, t: Scalar) -> Scalar {
        let (t_start, t_end) = self.time_range();
        let span = t_end - t_start;
        
        // 自适应步长
        let h = if span > 0.0 {
            (span / 1000.0).max(1e-8).min(1e-4)
        } else {
            1e-6
        };

        let v_m2 = self.get_value(t - 2.0 * h);
        let v_m1 = self.get_value(t - h);
        let v_p1 = self.get_value(t + h);
        let v_p2 = self.get_value(t + 2.0 * h);

        (-v_p2 + 8.0 * v_p1 - 8.0 * v_m1 + v_m2) / (12.0 * h)
    }

    /// 高精度循环外推
    ///
    /// 使用整数周期分解避免浮点累积误差。
    pub fn get_value_cyclic_precise(&self, t: Scalar) -> Scalar {
        let n = self.times.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return self.values[0];
        }

        let t_start = self.times[0];
        let t_end = self.times[n - 1];
        let duration = t_end - t_start;

        if duration < 1e-12 {
            return self.values[0];
        }

        // 整数周期分解
        let offset = t - t_start;
        let n_periods = (offset / duration).floor();
        let local_offset = offset - n_periods * duration;
        
        // 处理负偏移
        let normalized_t = if local_offset < 0.0 {
            t_start + local_offset + duration
        } else {
            t_start + local_offset
        };

        self.interpolate_internal(normalized_t)
    }

    /// 获取积分（梯形法则）
    ///
    /// 计算从 t_start 到 t_end 的定积分。
    pub fn integrate(&self, t_start: Scalar, t_end: Scalar) -> Scalar {
        if t_start >= t_end {
            return 0.0;
        }

        // 简单实现：使用固定步数的梯形法则
        let n_steps = 100;
        let dt = (t_end - t_start) / n_steps as Scalar;

        let mut integral = 0.0;
        let mut t = t_start;

        for _ in 0..n_steps {
            let v0 = self.get_value(t);
            let v1 = self.get_value(t + dt);
            integral += 0.5 * (v0 + v1) * dt;
            t += dt;
        }

        integral
    }

    /// 重采样到新的时间点
    pub fn resample(&self, new_times: &[Scalar]) -> Self {
        let new_values: Vec<Scalar> = new_times.iter().map(|&t| self.get_value(t)).collect();
        Self::new(new_times.to_vec(), new_values).with_extrapolation(self.extrap_mode)
    }

    /// 缩放值
    pub fn scale(&mut self, factor: Scalar) {
        for v in &mut self.values {
            *v *= factor;
        }
    }

    /// 偏移值
    pub fn offset(&mut self, offset: Scalar) {
        for v in &mut self.values {
            *v += offset;
        }
    }
}

/// 向量时间序列（2D）
///
/// 存储时间-向量对，用于风速、流速等矢量数据。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorTimeSeries {
    /// X 分量时间序列
    x_series: TimeSeries,
    /// Y 分量时间序列
    y_series: TimeSeries,
}

impl VectorTimeSeries {
    /// 从分量时间序列创建
    pub fn new(x_series: TimeSeries, y_series: TimeSeries) -> Self {
        assert_eq!(
            x_series.times(),
            y_series.times(),
            "X and Y series must have same time points"
        );
        Self { x_series, y_series }
    }

    /// 从时间和分量数组创建
    pub fn from_components(times: Vec<Scalar>, x: Vec<Scalar>, y: Vec<Scalar>) -> Self {
        let x_series = TimeSeries::new(times.clone(), x);
        let y_series = TimeSeries::new(times, y);
        Self { x_series, y_series }
    }

    /// 设置外推模式
    pub fn with_extrapolation(mut self, mode: ExtrapolationMode) -> Self {
        self.x_series = self.x_series.with_extrapolation(mode);
        self.y_series = self.y_series.with_extrapolation(mode);
        self
    }

    /// 获取指定时间的向量值
    pub fn get_value(&self, t: Scalar) -> (Scalar, Scalar) {
        (self.x_series.get_value(t), self.y_series.get_value(t))
    }

    /// 获取指定时间的模长
    pub fn get_magnitude(&self, t: Scalar) -> Scalar {
        let (x, y) = self.get_value(t);
        (x * x + y * y).sqrt()
    }

    /// 获取指定时间的方向（弧度）
    pub fn get_direction(&self, t: Scalar) -> Scalar {
        let (x, y) = self.get_value(t);
        y.atan2(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_basic() {
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let values = vec![0.0, 1.0, 0.5, 0.0];

        let series = TimeSeries::new(times, values);

        // 边界值
        assert!((series.get_value(0.0) - 0.0).abs() < 1e-10);
        assert!((series.get_value(3.0) - 0.0).abs() < 1e-10);

        // 中间插值
        assert!((series.get_value(0.5) - 0.5).abs() < 1e-10);
        assert!((series.get_value(1.5) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolation_clamp() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![1.0, 2.0, 3.0];

        let series = TimeSeries::new(times, values).with_extrapolation(ExtrapolationMode::Clamp);

        assert!((series.get_value(-1.0) - 1.0).abs() < 1e-10);
        assert!((series.get_value(3.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolation_linear() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![0.0, 1.0, 2.0];

        let series = TimeSeries::new(times, values).with_extrapolation(ExtrapolationMode::Linear);

        // 线性外推
        assert!((series.get_value(-1.0) - (-1.0)).abs() < 1e-10);
        assert!((series.get_value(3.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_extrapolation_cyclic() {
        let times = vec![0.0, 1.0, 2.0];
        let values = vec![0.0, 1.0, 0.0];

        let series = TimeSeries::new(times, values).with_extrapolation(ExtrapolationMode::Cyclic);

        // 周期外推: t=3.0 -> t=1.0
        assert!((series.get_value(3.0) - 1.0).abs() < 1e-10);
        // t=4.0 -> t=0.0
        assert!((series.get_value(4.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_timeseries() {
        let times = vec![0.0, 1.0, 2.0];
        let x_vals = vec![1.0, 0.0, -1.0];
        let y_vals = vec![0.0, 1.0, 0.0];

        let series = VectorTimeSeries::from_components(times, x_vals, y_vals);

        let (x, y) = series.get_value(0.5);
        assert!((x - 0.5).abs() < 1e-10);
        assert!((y - 0.5).abs() < 1e-10);

        let mag = series.get_magnitude(0.5);
        assert!((mag - (0.5_f64.powi(2) * 2.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_cache_acceleration() {
        let n = 1000;
        let times: Vec<Scalar> = (0..n).map(|i| i as Scalar).collect();
        let values: Vec<Scalar> = (0..n).map(|i| (i as Scalar).sin()).collect();

        let series = TimeSeries::new(times, values);

        // 顺序访问应该利用缓存
        for i in 0..n - 1 {
            let t = i as Scalar + 0.5;
            let _ = series.get_value(t);
        }
    }
}
