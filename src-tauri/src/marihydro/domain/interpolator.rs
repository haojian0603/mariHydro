// src-tauri/src/marihydro/domain/interpolator.rs

use crate::marihydro::geo::crs::Crs;
use crate::marihydro::geo::transform::GeoTransformer;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::types::RasterMetadata;
use glam::DVec2;
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct Weight {
    src_idx: u32,
    val: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    Bilinear,
    NearestNeighbor,
    Bicubic,
}

impl InterpolationMethod {
    pub fn parallel_threshold(&self) -> usize {
        match self {
            Self::NearestNeighbor => 10000,
            Self::Bilinear => 2000,
            Self::Bicubic => 500,
        }
    }

    pub fn expected_weights(&self) -> usize {
        match self {
            Self::NearestNeighbor => 1,
            Self::Bilinear => 4,
            Self::Bicubic => 16,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NoDataStrategy {
    UseFallback(f64),
    KeepOriginal,
    SetNaN,
}

impl Default for NoDataStrategy {
    fn default() -> Self {
        Self::SetNaN
    }
}

impl NoDataStrategy {
    pub fn for_wind() -> Self {
        Self::UseFallback(0.0)
    }
    pub fn for_pressure() -> Self {
        Self::UseFallback(1013.25)
    }
    pub fn for_temperature_celsius() -> Self {
        Self::UseFallback(15.0)
    }
}

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

pub struct SpatialInterpolator {
    flat_weights: Vec<Weight>,
    offsets: Vec<usize>,
    rotation_angles: Vec<f32>,
    src_dims: (usize, usize),
    config: InterpolatorConfig,
}

impl SpatialInterpolator {
    pub fn new_from_points(
        target_points: &[DVec2],
        target_crs_def: &str,
        source_meta: &RasterMetadata,
    ) -> MhResult<Self> {
        Self::new_from_points_with_config(
            target_points,
            target_crs_def,
            source_meta,
            InterpolatorConfig::default(),
        )
    }

    pub fn new_from_points_with_config(
        target_points: &[DVec2],
        target_crs_def: &str,
        source_meta: &RasterMetadata,
        config: InterpolatorConfig,
    ) -> MhResult<Self> {
        let (src_w, src_h) = (source_meta.width, source_meta.height);
        let src_gt = &source_meta.transform;

        let target_crs = Crs::from_string(target_crs_def);
        let src_crs = if source_meta.crs_wkt.is_empty() {
            Crs::wgs84()
        } else {
            Crs::from_string(&source_meta.crs_wkt)
        };

        let transformer = GeoTransformer::new(&target_crs, &src_crs)?;
        let angle_transformer = GeoTransformer::new(&src_crs, &target_crs)?;

        let (sx0, sy0, sdx, sdy) = (src_gt.0[0], src_gt.0[3], src_gt.0[1], src_gt.0[5]);

        let threshold = config
            .parallel_threshold
            .unwrap_or_else(|| config.method.parallel_threshold());
        let should_parallel = config.parallel_enabled && target_points.len() >= threshold;

        let results: Vec<(Vec<Weight>, f32)> = if should_parallel {
            target_points
                .par_iter()
                .map(|point| {
                    Self::compute_point_weights(
                        point,
                        &transformer,
                        &angle_transformer,
                        sx0,
                        sy0,
                        sdx,
                        sdy,
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
                        &transformer,
                        &angle_transformer,
                        sx0,
                        sy0,
                        sdx,
                        sdy,
                        src_w,
                        src_h,
                        config.method,
                    )
                })
                .collect()
        };

        let expected_total = results.len() * config.method.expected_weights();
        let mut flat_weights = Vec::with_capacity(expected_total);
        let mut offsets = Vec::with_capacity(results.len() + 1);
        let mut rotation_angles = Vec::with_capacity(results.len());

        offsets.push(0);
        for (weights, angle) in results {
            flat_weights.extend_from_slice(&weights);
            offsets.push(flat_weights.len());
            rotation_angles.push(angle);
        }

        flat_weights.shrink_to_fit();

        Ok(Self {
            flat_weights,
            offsets,
            rotation_angles,
            src_dims: (src_w, src_h),
            config,
        })
    }

    #[inline]
    fn compute_point_weights(
        point: &DVec2,
        transformer: &GeoTransformer,
        angle_transformer: &GeoTransformer,
        sx0: f64,
        sy0: f64,
        sdx: f64,
        sdy: f64,
        src_w: usize,
        src_h: usize,
        method: InterpolationMethod,
    ) -> (Vec<Weight>, f32) {
        let angle = angle_transformer
            .calculate_convergence_angle(point.x, point.y)
            .unwrap_or(0.0) as f32;

        let (sx, sy) = match transformer.transform_point(point.x, point.y) {
            Ok(p) => p,
            Err(_) => return (vec![], angle),
        };

        let u = (sx - sx0) / sdx;
        let v = (sy - sy0) / sdy;

        let weights = match method {
            InterpolationMethod::Bilinear => Self::bilinear_weights(u, v, src_w, src_h),
            InterpolationMethod::NearestNeighbor => Self::nearest_weights(u, v, src_w, src_h),
            InterpolationMethod::Bicubic => Self::bicubic_weights(u, v, src_w, src_h),
        };

        (weights, angle)
    }

    #[inline]
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

    #[inline]
    fn nearest_weights(u: f64, v: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let col = u.round() as usize;
        let row = v.round() as usize;

        if col >= src_w || row >= src_h {
            return vec![];
        }

        vec![Weight {
            src_idx: (row * src_w + col) as u32,
            val: 1.0,
        }]
    }

    #[inline]
    fn bicubic_weights(u: f64, v: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let u0 = u.floor();
        let v0 = v.floor();

        let col_i = u0 as isize;
        let row_i = v0 as isize;

        if col_i < 1 || row_i < 1 || (col_i as usize) >= src_w - 2 || (row_i as usize) >= src_h - 2
        {
            return Self::bilinear_weights(u, v, src_w, src_h);
        }

        let fx = (u - u0) as f32;
        let fy = (v - v0) as f32;

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

    pub fn interpolate_to_vec(
        &self,
        source_data: &[f64],
        target: &mut [f64],
        strategy: NoDataStrategy,
    ) -> MhResult<()> {
        let (sh, sw) = self.src_dims;
        if source_data.len() != sw * sh {
            return Err(MhError::DataLoad {
                file: "Interpolation Source".into(),
                message: format!(
                    "源数据尺寸不匹配: 期望 {}, 实际 {}",
                    sw * sh,
                    source_data.len()
                ),
            });
        }

        let expected_points = self.offsets.len() - 1;
        if target.len() != expected_points {
            return Err(MhError::InvalidMesh {
                message: format!(
                    "目标数组长度不匹配: 期望 {}, 实际 {}",
                    expected_points,
                    target.len()
                ),
            });
        }

        let threshold = self
            .config
            .parallel_threshold
            .unwrap_or_else(|| self.config.method.parallel_threshold());

        if self.config.parallel_enabled && target.len() >= threshold {
            self.interpolate_parallel(source_data, target, strategy)
        } else {
            self.interpolate_serial(source_data, target, strategy)
        }

        Ok(())
    }

    #[inline]
    fn interpolate_serial(&self, src_slice: &[f64], target: &mut [f64], strategy: NoDataStrategy) {
        for k in 0..target.len() {
            let weights = &self.flat_weights[self.offsets[k]..self.offsets[k + 1]];

            match self.compute_value(weights, src_slice) {
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
    fn interpolate_parallel(
        &self,
        src_slice: &[f64],
        target: &mut [f64],
        strategy: NoDataStrategy,
    ) {
        let results: Vec<Option<f64>> = (0..target.len())
            .into_par_iter()
            .map(|k| {
                let weights = &self.flat_weights[self.offsets[k]..self.offsets[k + 1]];
                self.compute_value(weights, src_slice)
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

    #[inline(always)]
    fn compute_value(&self, weights: &[Weight], src: &[f64]) -> Option<f64> {
        if weights.is_empty() {
            return None;
        }

        let mut sum = 0.0;
        let mut w_total = 0.0;

        for w in weights {
            if let Some(&val) = src.get(w.src_idx as usize) {
                if !val.is_nan() {
                    sum += val * w.val as f64;
                    w_total += w.val as f64;
                }
            }
        }

        (w_total > 1e-6).then(|| sum / w_total)
    }

    pub fn interpolate_vector_field(
        &self,
        u_source: &[f64],
        v_source: &[f64],
        u_target: &mut [f64],
        v_target: &mut [f64],
        strategy: NoDataStrategy,
    ) -> MhResult<()> {
        self.interpolate_to_vec(u_source, u_target, strategy)?;
        self.interpolate_to_vec(v_source, v_target, strategy)?;

        for k in 0..u_target.len() {
            if k >= self.rotation_angles.len() {
                break;
            }

            let angle = self.rotation_angles[k] as f64;
            if angle.abs() > 1e-6 {
                let u = u_target[k];
                let v = v_target[k];

                if !u.is_nan() && !v.is_nan() {
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    u_target[k] = u * cos_a - v * sin_a;
                    v_target[k] = u * sin_a + v * cos_a;
                }
            }
        }

        Ok(())
    }

    pub fn rotation_angles(&self) -> &[f32] {
        &self.rotation_angles
    }

    pub fn source_dimensions(&self) -> (usize, usize) {
        self.src_dims
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
        assert_eq!(weights[0].src_idx, 22);
    }
}
