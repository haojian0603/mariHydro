// src-tauri/src/marihydro/domain/interpolator/spatial.rs

//! 空间插值器

use glam::DVec2;
use rayon::prelude::*;

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::GeoTransform;

/// 插值方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMethod {
    /// 最近邻
    NearestNeighbor,
    /// 双线性
    #[default]
    Bilinear,
    /// 双三次
    Bicubic,
}

impl InterpolationMethod {
    /// 并行阈值
    pub fn parallel_threshold(&self) -> usize {
        match self {
            Self::NearestNeighbor => 10000,
            Self::Bilinear => 2000,
            Self::Bicubic => 500,
        }
    }

    /// 预期权重数
    pub fn expected_weights(&self) -> usize {
        match self {
            Self::NearestNeighbor => 1,
            Self::Bilinear => 4,
            Self::Bicubic => 16,
        }
    }
}

/// 无数据处理策略
#[derive(Debug, Clone, Copy, Default)]
pub enum NoDataStrategy {
    /// 使用回退值
    UseFallback(f64),
    /// 保持原值
    KeepOriginal,
    /// 设为 NaN
    #[default]
    SetNaN,
}

impl NoDataStrategy {
    pub fn for_wind() -> Self {
        Self::UseFallback(0.0)
    }

    pub fn for_pressure() -> Self {
        Self::UseFallback(101325.0)
    }

    pub fn for_temperature() -> Self {
        Self::UseFallback(15.0)
    }
}

/// 插值权重
#[derive(Debug, Clone, Copy)]
struct Weight {
    src_idx: u32,
    val: f32,
}

/// 空间插值器配置
#[derive(Debug, Clone)]
pub struct InterpolatorConfig {
    pub method: InterpolationMethod,
    pub parallel_enabled: bool,
    pub parallel_threshold: Option<usize>,
}

impl Default for InterpolatorConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Bilinear,
            parallel_enabled: true,
            parallel_threshold: None,
        }
    }
}

/// 空间插值器
pub struct SpatialInterpolator {
    flat_weights: Vec<Weight>,
    offsets: Vec<usize>,
    rotation_angles: Vec<f32>,
    src_dims: (usize, usize),
    config: InterpolatorConfig,
}

impl SpatialInterpolator {
    /// 从目标点和源栅格创建插值器
    pub fn new(
        target_points: &[DVec2],
        source_transform: &GeoTransform,
        source_dims: (usize, usize),
        config: InterpolatorConfig,
    ) -> MhResult<Self> {
        let (src_w, src_h) = source_dims;

        let threshold = config
            .parallel_threshold
            .unwrap_or_else(|| config.method.parallel_threshold());

        let should_parallel = config.parallel_enabled && target_points.len() >= threshold;

        let results: Vec<Vec<Weight>> = if should_parallel {
            target_points
                .par_iter()
                .map(|point| {
                    Self::compute_point_weights(
                        point,
                        source_transform,
                        src_w,
                        src_h,
                        config.method,
                    )
                })
                .collect()
        } else {
            target_points
                .iter()
                .map(|point| {
                    Self::compute_point_weights(
                        point,
                        source_transform,
                        src_w,
                        src_h,
                        config.method,
                    )
                })
                .collect()
        };

        // 展平权重
        let expected_total = results.len() * config.method.expected_weights();
        let mut flat_weights = Vec::with_capacity(expected_total);
        let mut offsets = Vec::with_capacity(results.len() + 1);

        offsets.push(0);
        for weights in results {
            flat_weights.extend_from_slice(&weights);
            offsets.push(flat_weights.len());
        }

        flat_weights.shrink_to_fit();

        Ok(Self {
            flat_weights,
            offsets,
            rotation_angles: vec![0.0; target_points.len()],
            src_dims: source_dims,
            config,
        })
    }

    /// 设置旋转角度（用于矢量场）
    pub fn set_rotation_angles(&mut self, angles: Vec<f32>) {
        debug_assert_eq!(angles.len(), self.offsets.len() - 1);
        self.rotation_angles = angles;
    }

    /// 计算单点权重
    fn compute_point_weights(
        point: &DVec2,
        transform: &GeoTransform,
        src_w: usize,
        src_h: usize,
        method: InterpolationMethod,
    ) -> Vec<Weight> {
        // 地理坐标转像素坐标
        let (u, v) = match transform.geo_to_pixel(point.x, point.y) {
            Some((u, v)) => (u, v),
            None => return vec![],
        };

        match method {
            InterpolationMethod::NearestNeighbor => Self::nearest_weights(u, v, src_w, src_h),
            InterpolationMethod::Bilinear => Self::bilinear_weights(u, v, src_w, src_h),
            InterpolationMethod::Bicubic => Self::bicubic_weights(u, v, src_w, src_h),
        }
    }

    /// 最近邻权重
    fn nearest_weights(u: f64, v: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let col = u.round() as isize;
        let row = v.round() as isize;

        if col < 0 || row < 0 || col as usize >= src_w || row as usize >= src_h {
            return vec![];
        }

        vec![Weight {
            src_idx: (row as usize * src_w + col as usize) as u32,
            val: 1.0,
        }]
    }

    /// 双线性权重
    fn bilinear_weights(u: f64, v: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let u0 = u.floor();
        let v0 = v.floor();

        let col_i = u0 as isize;
        let row_i = v0 as isize;

        if col_i < 0 || row_i < 0 || (col_i as usize) >= src_w - 1 || (row_i as usize) >= src_h - 1
        {
            return vec![];
        }

        let u_ratio = (u - u0) as f32;
        let v_ratio = (v - v0) as f32;
        let u_inv = 1.0 - u_ratio;
        let v_inv = 1.0 - v_ratio;

        let idx_base = (row_i as usize) * src_w + (col_i as usize);

        vec![
            Weight {
                src_idx: idx_base as u32,
                val: u_inv * v_inv,
            },
            Weight {
                src_idx: (idx_base + 1) as u32,
                val: u_ratio * v_inv,
            },
            Weight {
                src_idx: (idx_base + src_w) as u32,
                val: u_inv * v_ratio,
            },
            Weight {
                src_idx: (idx_base + src_w + 1) as u32,
                val: u_ratio * v_ratio,
            },
        ]
    }

    /// 双三次权重
    fn bicubic_weights(u: f64, v: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let u0 = u.floor();
        let v0 = v.floor();

        let col_i = u0 as isize;
        let row_i = v0 as isize;

        // 边界检查
        if col_i < 1 || row_i < 1 || (col_i as usize) >= src_w - 2 || (row_i as usize) >= src_h - 2
        {
            // 回退到双线性
            return Self::bilinear_weights(u, v, src_w, src_h);
        }

        let fx = (u - u0) as f32;
        let fy = (v - v0) as f32;

        // 三次核函数
        let cubic = |t: f32| -> [f32; 4] {
            let t2 = t * t;
            let t3 = t2 * t;
            [
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2,
            ]
        };

        let wx = cubic(fx);
        let wy = cubic(fy);

        let mut weights = Vec::with_capacity(16);
        let base_row = (row_i - 1) as usize;
        let base_col = (col_i - 1) as usize;

        for dy in 0..4 {
            for dx in 0..4 {
                let idx = (base_row + dy) * src_w + (base_col + dx);
                let val = wx[dx] * wy[dy];

                if val.abs() > 1e-6 {
                    weights.push(Weight {
                        src_idx: idx as u32,
                        val,
                    });
                }
            }
        }

        weights
    }

    /// 插值标量场
    pub fn interpolate(
        &self,
        source_data: &[f64],
        target: &mut [f64],
        strategy: NoDataStrategy,
    ) -> MhResult<()> {
        let (sw, sh) = self.src_dims;

        if source_data.len() != sw * sh {
            return Err(MhError::size_mismatch(
                "source_data",
                sw * sh,
                source_data.len(),
            ));
        }

        let n_points = self.offsets.len() - 1;
        if target.len() != n_points {
            return Err(MhError::size_mismatch("target", n_points, target.len()));
        }

        let threshold = self
            .config
            .parallel_threshold
            .unwrap_or_else(|| self.config.method.parallel_threshold());

        if self.config.parallel_enabled && target.len() >= threshold {
            self.interpolate_parallel(source_data, target, strategy);
        } else {
            self.interpolate_serial(source_data, target, strategy);
        }

        Ok(())
    }

    fn interpolate_serial(&self, src: &[f64], target: &mut [f64], strategy: NoDataStrategy) {
        for k in 0..target.len() {
            let weights = &self.flat_weights[self.offsets[k]..self.offsets[k + 1]];

            match self.compute_value(weights, src) {
                Some(val) => target[k] = val,
                None => match strategy {
                    NoDataStrategy::UseFallback(v) => target[k] = v,
                    NoDataStrategy::KeepOriginal => {}
                    NoDataStrategy::SetNaN => target[k] = f64::NAN,
                },
            }
        }
    }

    fn interpolate_parallel(&self, src: &[f64], target: &mut [f64], strategy: NoDataStrategy) {
        let results: Vec<Option<f64>> = (0..target.len())
            .into_par_iter()
            .map(|k| {
                let weights = &self.flat_weights[self.offsets[k]..self.offsets[k + 1]];
                self.compute_value(weights, src)
            })
            .collect();

        for (k, result) in results.into_iter().enumerate() {
            match result {
                Some(val) => target[k] = val,
                None => match strategy {
                    NoDataStrategy::UseFallback(v) => target[k] = v,
                    NoDataStrategy::KeepOriginal => {}
                    NoDataStrategy::SetNaN => target[k] = f64::NAN,
                },
            }
        }
    }

    #[inline]
    fn compute_value(&self, weights: &[Weight], src: &[f64]) -> Option<f64> {
        if weights.is_empty() {
            return None;
        }

        let mut sum = 0.0;
        let mut w_total = 0.0;

        for w in weights {
            if let Some(&val) = src.get(w.src_idx as usize) {
                if val.is_finite() {
                    sum += val * w.val as f64;
                    w_total += w.val as f64;
                }
            }
        }

        (w_total > 1e-6).then(|| sum / w_total)
    }

    /// 插值矢量场（带旋转修正）
    pub fn interpolate_vector(
        &self,
        u_source: &[f64],
        v_source: &[f64],
        u_target: &mut [f64],
        v_target: &mut [f64],
        strategy: NoDataStrategy,
    ) -> MhResult<()> {
        self.interpolate(u_source, u_target, strategy)?;
        self.interpolate(v_source, v_target, strategy)?;

        // 应用旋转修正
        for k in 0..u_target.len() {
            if k >= self.rotation_angles.len() {
                break;
            }

            let angle = self.rotation_angles[k] as f64;
            if angle.abs() > 1e-6 {
                let u = u_target[k];
                let v = v_target[k];

                if u.is_finite() && v.is_finite() {
                    let (sin_a, cos_a) = angle.sin_cos();
                    u_target[k] = u * cos_a - v * sin_a;
                    v_target[k] = u * sin_a + v * cos_a;
                }
            }
        }

        Ok(())
    }

    /// 获取旋转角度
    pub fn rotation_angles(&self) -> &[f32] {
        &self.rotation_angles
    }

    /// 获取源数据尺寸
    pub fn source_dims(&self) -> (usize, usize) {
        self.src_dims
    }

    /// 目标点数量
    pub fn n_targets(&self) -> usize {
        self.offsets.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_weights() {
        let weights = SpatialInterpolator::bilinear_weights(1.5, 1.5, 10, 10);
        assert_eq!(weights.len(), 4);

        let sum: f32 = weights.iter().map(|w| w.val).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_nearest_weights() {
        let weights = SpatialInterpolator::nearest_weights(1.5, 1.5, 10, 10);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].val, 1.0);
        // round(1.5) = 2, index = 2*10+2 = 22
        assert_eq!(weights[0].src_idx, 22);
    }

    #[test]
    fn test_out_of_bounds() {
        let weights = SpatialInterpolator::bilinear_weights(-1.0, 0.0, 10, 10);
        assert!(weights.is_empty());

        let weights = SpatialInterpolator::bilinear_weights(10.0, 0.0, 10, 10);
        assert!(weights.is_empty());
    }
}
