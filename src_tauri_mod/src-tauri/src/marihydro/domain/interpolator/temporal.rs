// src-tauri/src/marihydro/domain/interpolator/temporal.rs

//! 时间插值器

use crate::marihydro::core::error::{MhError, MhResult};

/// 时间插值方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TemporalMethod {
    /// 最近时刻
    Nearest,
    /// 线性插值
    #[default]
    Linear,
    /// 保持前值
    Previous,
    /// 保持后值
    Next,
}

/// 时间插值器
pub struct TemporalInterpolator {
    method: TemporalMethod,
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

    /// 标量插值
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
                if (t1 - t0).abs() < 1e-14 {
                    v0
                } else {
                    let alpha = (t - t0) / (t1 - t0);
                    v0 + alpha * (v1 - v0)
                }
            }
            TemporalMethod::Previous => v0,
            TemporalMethod::Next => v1,
        }
    }

    /// 批量标量插值
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
                let alpha = if (t1 - t0).abs() < 1e-14 {
                    0.0
                } else {
                    (t - t0) / (t1 - t0)
                };

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

    /// 获取插值方法
    pub fn method(&self) -> TemporalMethod {
        self.method
    }
}

impl Default for TemporalInterpolator {
    fn default() -> Self {
        Self::linear()
    }
}

/// 时间序列数据帧
#[derive(Debug, Clone)]
pub struct TimeFrame<T> {
    pub time: f64,
    pub data: T,
}

impl<T> TimeFrame<T> {
    pub fn new(time: f64, data: T) -> Self {
        Self { time, data }
    }
}

/// 双缓冲时间帧（用于流式读取）
pub struct DoubleBufferTimeFrame {
    frame0: Option<TimeFrame<Vec<f64>>>,
    frame1: Option<TimeFrame<Vec<f64>>>,
    interpolator: TemporalInterpolator,
}

impl DoubleBufferTimeFrame {
    pub fn new(method: TemporalMethod) -> Self {
        Self {
            frame0: None,
            frame1: None,
            interpolator: TemporalInterpolator::new(method),
        }
    }

    /// 是否需要更新
    pub fn needs_update(&self, t: f64) -> bool {
        match (&self.frame0, &self.frame1) {
            (Some(f0), Some(f1)) => t < f0.time || t > f1.time,
            _ => true,
        }
    }

    /// 推入新帧
    pub fn push_frame(&mut self, frame: TimeFrame<Vec<f64>>) {
        self.frame0 = self.frame1.take();
        self.frame1 = Some(frame);
    }

    /// 插值到指定时间
    pub fn interpolate_to(&self, t: f64, output: &mut [f64]) -> MhResult<()> {
        match (&self.frame0, &self.frame1) {
            (Some(f0), Some(f1)) => self
                .interpolator
                .interpolate_batch(t, f0.time, f1.time, &f0.data, &f1.data, output),
            (None, Some(f1)) => {
                if output.len() != f1.data.len() {
                    return Err(MhError::size_mismatch(
                        "output",
                        f1.data.len(),
                        output.len(),
                    ));
                }
                output.copy_from_slice(&f1.data);
                Ok(())
            }
            (Some(f0), None) => {
                if output.len() != f0.data.len() {
                    return Err(MhError::size_mismatch(
                        "output",
                        f0.data.len(),
                        output.len(),
                    ));
                }
                output.copy_from_slice(&f0.data);
                Ok(())
            }
            (None, None) => Err(MhError::DataLoad {
                file: "temporal".into(),
                message: "无可用时间帧".into(),
            }),
        }
    }

    /// 当前时间范围
    pub fn time_range(&self) -> Option<(f64, f64)> {
        match (&self.frame0, &self.frame1) {
            (Some(f0), Some(f1)) => Some((f0.time, f1.time)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_interpolation() {
        let interp = TemporalInterpolator::linear();

        let result = interp.interpolate_scalar(0.5, 0.0, 1.0, 0.0, 10.0);
        assert!((result - 5.0).abs() < 1e-10);

        let result = interp.interpolate_scalar(0.25, 0.0, 1.0, 0.0, 10.0);
        assert!((result - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_interpolation() {
        let interp = TemporalInterpolator::nearest();

        let result = interp.interpolate_scalar(0.4, 0.0, 1.0, 0.0, 10.0);
        assert!((result - 0.0).abs() < 1e-10);

        let result = interp.interpolate_scalar(0.6, 0.0, 1.0, 0.0, 10.0);
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_batch_interpolation() {
        let interp = TemporalInterpolator::linear();

        let v0 = vec![0.0, 10.0, 20.0];
        let v1 = vec![10.0, 20.0, 30.0];
        let mut output = vec![0.0; 3];

        interp
            .interpolate_batch(0.5, 0.0, 1.0, &v0, &v1, &mut output)
            .unwrap();

        assert!((output[0] - 5.0).abs() < 1e-10);
        assert!((output[1] - 15.0).abs() < 1e-10);
        assert!((output[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_double_buffer() {
        let mut buffer = DoubleBufferTimeFrame::new(TemporalMethod::Linear);

        assert!(buffer.needs_update(0.5));

        buffer.push_frame(TimeFrame::new(0.0, vec![0.0, 0.0]));
        buffer.push_frame(TimeFrame::new(1.0, vec![10.0, 20.0]));

        assert!(!buffer.needs_update(0.5));

        let mut output = vec![0.0; 2];
        buffer.interpolate_to(0.5, &mut output).unwrap();

        assert!((output[0] - 5.0).abs() < 1e-10);
        assert!((output[1] - 10.0).abs() < 1e-10);
    }
}
