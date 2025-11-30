// src-tauri/src/marihydro/core/traits/interpolator.rs

//! 插值器接口
//!
//! 定义空间和时间插值的统一抽象。

use crate::marihydro::core::error::MhResult;
use glam::DVec2;

/// 空间插值器接口
pub trait SpatialInterpolator: Send + Sync {
    /// 插值器名称
    fn name(&self) -> &str;

    /// 从规则网格插值到点
    fn interpolate_from_grid(
        &self,
        grid_data: &[f64],
        grid_origin: DVec2,
        grid_spacing: DVec2,
        grid_dims: (usize, usize),
        target_points: &[DVec2],
        output: &mut [f64],
    ) -> MhResult<()>;

    /// 从非结构点插值到点（IDW等）
    fn interpolate_from_points(
        &self,
        source_points: &[DVec2],
        source_values: &[f64],
        target_points: &[DVec2],
        output: &mut [f64],
    ) -> MhResult<()>;
}

/// 时间插值方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TemporalInterpolationMethod {
    /// 最近邻（阶跃）
    Nearest,
    /// 线性插值
    #[default]
    Linear,
    /// 三次样条（需要更多时间点）
    CubicSpline,
}

/// 时间插值器接口
pub trait TemporalInterpolator: Send + Sync {
    /// 插值方法
    fn method(&self) -> TemporalInterpolationMethod;

    /// 线性时间插值
    fn interpolate_linear(t: f64, t0: f64, t1: f64, v0: f64, v1: f64) -> f64 {
        if (t1 - t0).abs() < 1e-14 {
            return v0;
        }
        let alpha = (t - t0) / (t1 - t0);
        v0 + alpha * (v1 - v0)
    }

    /// 批量线性时间插值
    fn interpolate_linear_batch(
        t: f64,
        t0: f64,
        t1: f64,
        v0: &[f64],
        v1: &[f64],
        output: &mut [f64],
    ) -> MhResult<()> {
        if v0.len() != v1.len() || v0.len() != output.len() {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "temporal interpolation arrays",
                v0.len(),
                output.len(),
            ));
        }

        let alpha = if (t1 - t0).abs() < 1e-14 {
            0.0
        } else {
            (t - t0) / (t1 - t0)
        };

        for i in 0..output.len() {
            output[i] = v0[i] + alpha * (v1[i] - v0[i]);
        }

        Ok(())
    }
}

/// 双线性网格插值
pub struct BilinearInterpolator;

impl SpatialInterpolator for BilinearInterpolator {
    fn name(&self) -> &str {
        "Bilinear"
    }

    fn interpolate_from_grid(
        &self,
        grid_data: &[f64],
        grid_origin: DVec2,
        grid_spacing: DVec2,
        grid_dims: (usize, usize),
        target_points: &[DVec2],
        output: &mut [f64],
    ) -> MhResult<()> {
        let (nx, ny) = grid_dims;

        if grid_data.len() != nx * ny {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "grid_data",
                nx * ny,
                grid_data.len(),
            ));
        }

        if target_points.len() != output.len() {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "target_points/output",
                target_points.len(),
                output.len(),
            ));
        }

        for (i, point) in target_points.iter().enumerate() {
            // 计算网格坐标
            let gx = (point.x - grid_origin.x) / grid_spacing.x;
            let gy = (point.y - grid_origin.y) / grid_spacing.y;

            // 边界检查
            if gx < 0.0 || gy < 0.0 || gx >= (nx - 1) as f64 || gy >= (ny - 1) as f64 {
                output[i] = f64::NAN; // 超出范围
                continue;
            }

            let ix = gx.floor() as usize;
            let iy = gy.floor() as usize;
            let fx = gx - ix as f64;
            let fy = gy - iy as f64;

            // 四个角点值
            let v00 = grid_data[iy * nx + ix];
            let v10 = grid_data[iy * nx + ix + 1];
            let v01 = grid_data[(iy + 1) * nx + ix];
            let v11 = grid_data[(iy + 1) * nx + ix + 1];

            // 双线性插值
            let v0 = v00 * (1.0 - fx) + v10 * fx;
            let v1 = v01 * (1.0 - fx) + v11 * fx;
            output[i] = v0 * (1.0 - fy) + v1 * fy;
        }

        Ok(())
    }

    fn interpolate_from_points(
        &self,
        source_points: &[DVec2],
        source_values: &[f64],
        target_points: &[DVec2],
        output: &mut [f64],
    ) -> MhResult<()> {
        // 点到点插值使用IDW
        if source_points.len() != source_values.len() {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "source arrays",
                source_points.len(),
                source_values.len(),
            ));
        }

        if target_points.len() != output.len() {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "target arrays",
                target_points.len(),
                output.len(),
            ));
        }

        const POWER: f64 = 2.0;
        const EPSILON: f64 = 1e-10;

        for (i, target) in target_points.iter().enumerate() {
            let mut weight_sum = 0.0;
            let mut value_sum = 0.0;

            for (j, source) in source_points.iter().enumerate() {
                let dist = (*target - *source).length();

                if dist < EPSILON {
                    // 精确命中
                    output[i] = source_values[j];
                    weight_sum = 1.0; // 标记已处理
                    break;
                }

                let weight = 1.0 / dist.powf(POWER);
                weight_sum += weight;
                value_sum += weight * source_values[j];
            }

            if weight_sum > EPSILON {
                output[i] = value_sum / weight_sum;
            } else {
                output[i] = f64::NAN;
            }
        }

        Ok(())
    }
}

/// 默认时间插值器
pub struct LinearTemporalInterpolator;

impl TemporalInterpolator for LinearTemporalInterpolator {
    fn method(&self) -> TemporalInterpolationMethod {
        TemporalInterpolationMethod::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_interpolation() {
        let interpolator = BilinearInterpolator;

        // 2x2 网格
        let grid_data = vec![0.0, 1.0, 2.0, 3.0];
        let grid_origin = DVec2::new(0.0, 0.0);
        let grid_spacing = DVec2::new(1.0, 1.0);
        let grid_dims = (2, 2);

        let target_points = vec![DVec2::new(0.5, 0.5)];
        let mut output = vec![0.0];

        interpolator
            .interpolate_from_grid(
                &grid_data,
                grid_origin,
                grid_spacing,
                grid_dims,
                &target_points,
                &mut output,
            )
            .unwrap();

        // 中心点应该是四个角的平均值
        assert!((output[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_linear_interpolation() {
        let result = LinearTemporalInterpolator::interpolate_linear(0.5, 0.0, 1.0, 0.0, 10.0);
        assert!((result - 5.0).abs() < 1e-10);
    }
}
