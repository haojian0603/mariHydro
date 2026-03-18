// marihydro\crates\mh_terrain\src/interpolation/temporal.rs

//! 时间插值器
//!
//! 用于在不同时刻之间插值物理量。
//!
//! # 示例
//!
//! ```
//! use mh_terrain::interpolation::temporal::{TemporalInterpolator, TemporalMethod};
//!
//! let interp = TemporalInterpolator::linear();
//!
//! // 在 t=0.5 时在 t=0.0 (v=10.0) 和 t=1.0 (v=20.0) 之间插值
//! let result = interp.interpolate_scalar(0.5, 0.0, 1.0, 10.0, 20.0);
//! assert!((result - 15.0).abs() < 1e-10);
//! ```

use mh_foundation::error::{MhError, MhResult};

/// 时间插值方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TemporalMethod {
    /// 最近时刻
    Nearest,
    /// 线性插值
    #[default]
    Linear,
    /// 保持前值（阶跃函数）
    Previous,
    /// 保持后值
    Next,
}

impl TemporalMethod {
    /// 获取方法名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Linear => "linear",
            Self::Previous => "previous",
            Self::Next => "next",
        }
    }
}

/// 时间插值器
#[derive(Debug, Clone)]
pub struct TemporalInterpolator {
    method: TemporalMethod,
}

impl Default for TemporalInterpolator {
    fn default() -> Self {
        Self::linear()
    }
}

impl TemporalInterpolator {
    /// 创建插值器
    pub fn new(method: TemporalMethod) -> Self {
        Self { method }
    }

    /// 线性插值器
    pub fn linear() -> Self {
        Self::new(TemporalMethod::Linear)
    }

    /// 最近邻插值器
    pub fn nearest() -> Self {
        Self::new(TemporalMethod::Nearest)
    }

    /// 前值插值器
    pub fn previous() -> Self {
        Self::new(TemporalMethod::Previous)
    }

    /// 后值插值器
    pub fn next() -> Self {
        Self::new(TemporalMethod::Next)
    }

    /// 获取插值方法
    #[inline]
    pub fn method(&self) -> TemporalMethod {
        self.method
    }

    /// 标量插值
    ///
    /// 在时刻 t 处，根据 t0 和 t1 时刻的值 v0 和 v1 进行插值。
    ///
    /// # 参数
    /// - `t`: 目标时刻
    /// - `t0`: 前一时刻
    /// - `t1`: 后一时刻
    /// - `v0`: t0 时刻的值
    /// - `v1`: t1 时刻的值
    #[inline]
    pub fn interpolate_scalar(&self, t: f64, t0: f64, t1: f64, v0: f64, v1: f64) -> f64 {
        match self.method {
            TemporalMethod::Nearest => {
                if (t - t0).abs() <= (t - t1).abs() {
                    v0
                } else {
                    v1
                }
            }
            TemporalMethod::Linear => {
                let dt = t1 - t0;
                if dt.abs() < 1e-14 {
                    v0
                } else {
                    let alpha = (t - t0) / dt;
                    v0 + alpha * (v1 - v0)
                }
            }
            TemporalMethod::Previous => v0,
            TemporalMethod::Next => v1,
        }
    }

    /// 批量标量插值
    ///
    /// 对整个数组进行时间插值。
    pub fn interpolate_batch(
        &self,
        t: f64,
        t0: f64,
        t1: f64,
        v0: &[f64],
        v1: &[f64],
        output: &mut [f64],
    ) -> MhResult<()> {
        if v0.len() != v1.len() || v0.len() != output.len() {
            return Err(MhError::size_mismatch(
                "temporal interpolation",
                v0.len(),
                output.len(),
            ));
        }

        match self.method {
            TemporalMethod::Nearest => {
                let use_v0 = (t - t0).abs() <= (t - t1).abs();
                let src = if use_v0 { v0 } else { v1 };
                output.copy_from_slice(src);
            }
            TemporalMethod::Linear => {
                let dt = t1 - t0;
                let alpha = if dt.abs() < 1e-14 { 0.0 } else { (t - t0) / dt };

                for i in 0..output.len() {
                    output[i] = v0[i] + alpha * (v1[i] - v0[i]);
                }
            }
            TemporalMethod::Previous => {
                output.copy_from_slice(v0);
            }
            TemporalMethod::Next => {
                output.copy_from_slice(v1);
            }
        }

        Ok(())
    }

    /// 批量原地插值
    ///
    /// 将结果直接写入 output 数组。
    pub fn interpolate_batch_inplace(
        &self,
        t: f64,
        t0: f64,
        t1: f64,
        v0: &[f64],
        v1: &[f64],
    ) -> Vec<f64> {
        let mut output = vec![0.0; v0.len().min(v1.len())];
        let _ = self.interpolate_batch(t, t0, t1, v0, v1, &mut output);
        output
    }
}

/// 时间帧：带时间戳的数据
#[derive(Debug, Clone)]
pub struct TimeFrame<T> {
    /// 时间戳
    pub time: f64,
    /// 数据
    pub data: T,
}

impl<T> TimeFrame<T> {
    /// 创建时间帧
    pub fn new(time: f64, data: T) -> Self {
        Self { time, data }
    }

    /// 转换数据类型
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> TimeFrame<U> {
        TimeFrame {
            time: self.time,
            data: f(self.data),
        }
    }
}

/// 双缓冲时间帧
///
/// 用于流式读取时间序列数据，保持前后两个时刻的数据用于插值。
pub struct DoubleBufferTimeFrame {
    frame0: Option<TimeFrame<Vec<f64>>>,
    frame1: Option<TimeFrame<Vec<f64>>>,
    interpolator: TemporalInterpolator,
}

impl DoubleBufferTimeFrame {
    /// 创建双缓冲
    pub fn new(method: TemporalMethod) -> Self {
        Self {
            frame0: None,
            frame1: None,
            interpolator: TemporalInterpolator::new(method),
        }
    }

    /// 使用线性插值创建
    pub fn linear() -> Self {
        Self::new(TemporalMethod::Linear)
    }

    /// 推入新帧
    ///
    /// 将 frame1 移动到 frame0，新帧成为 frame1。
    pub fn push(&mut self, frame: TimeFrame<Vec<f64>>) {
        self.frame0 = self.frame1.take();
        self.frame1 = Some(frame);
    }

    /// 是否可以进行插值
    pub fn can_interpolate(&self) -> bool {
        self.frame0.is_some() && self.frame1.is_some()
    }

    /// 获取有效时间范围
    pub fn time_range(&self) -> Option<(f64, f64)> {
        match (&self.frame0, &self.frame1) {
            (Some(f0), Some(f1)) => Some((f0.time, f1.time)),
            _ => None,
        }
    }

    /// 在给定时刻插值
    pub fn interpolate_at(&self, t: f64) -> Option<Vec<f64>> {
        match (&self.frame0, &self.frame1) {
            (Some(f0), Some(f1)) => {
                Some(self.interpolator.interpolate_batch_inplace(
                    t,
                    f0.time,
                    f1.time,
                    &f0.data,
                    &f1.data,
                ))
            }
            _ => None,
        }
    }

    /// 在给定时刻插值到输出数组
    pub fn interpolate_at_into(&self, t: f64, output: &mut [f64]) -> MhResult<()> {
        match (&self.frame0, &self.frame1) {
            (Some(f0), Some(f1)) => {
                self.interpolator.interpolate_batch(t, f0.time, f1.time, &f0.data, &f1.data, output)
            }
            _ => Err(MhError::invalid_input("No frames available for interpolation")),
        }
    }

    /// 清空缓冲
    pub fn clear(&mut self) {
        self.frame0 = None;
        self.frame1 = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let interp = TemporalInterpolator::linear();

        // 中点
        assert!((interp.interpolate_scalar(0.5, 0.0, 1.0, 10.0, 20.0) - 15.0).abs() < 1e-10);

        // 起点
        assert!((interp.interpolate_scalar(0.0, 0.0, 1.0, 10.0, 20.0) - 10.0).abs() < 1e-10);

        // 终点
        assert!((interp.interpolate_scalar(1.0, 0.0, 1.0, 10.0, 20.0) - 20.0).abs() < 1e-10);

        // 外推
        assert!((interp.interpolate_scalar(1.5, 0.0, 1.0, 10.0, 20.0) - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_interpolation() {
        let interp = TemporalInterpolator::nearest();

        // 靠近起点
        assert!((interp.interpolate_scalar(0.3, 0.0, 1.0, 10.0, 20.0) - 10.0).abs() < 1e-10);

        // 靠近终点
        assert!((interp.interpolate_scalar(0.7, 0.0, 1.0, 10.0, 20.0) - 20.0).abs() < 1e-10);

        // 中点（取前值）
        assert!((interp.interpolate_scalar(0.5, 0.0, 1.0, 10.0, 20.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_interpolation() {
        let interp = TemporalInterpolator::linear();
        let v0 = vec![10.0, 20.0, 30.0];
        let v1 = vec![20.0, 40.0, 60.0];
        let mut output = vec![0.0; 3];

        interp.interpolate_batch(0.5, 0.0, 1.0, &v0, &v1, &mut output).unwrap();

        assert!((output[0] - 15.0).abs() < 1e-10);
        assert!((output[1] - 30.0).abs() < 1e-10);
        assert!((output[2] - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_buffer() {
        let mut buffer = DoubleBufferTimeFrame::linear();

        assert!(!buffer.can_interpolate());

        buffer.push(TimeFrame::new(0.0, vec![10.0, 20.0]));
        assert!(!buffer.can_interpolate());

        buffer.push(TimeFrame::new(1.0, vec![20.0, 40.0]));
        assert!(buffer.can_interpolate());

        let result = buffer.interpolate_at(0.5).unwrap();
        assert!((result[0] - 15.0).abs() < 1e-10);
        assert!((result[1] - 30.0).abs() < 1e-10);
    }
}
