// src-tauri/src/marihydro/domain/interpolator.rs

use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::geo::crs::Crs;
use crate::marihydro::geo::transform::GeoTransformer;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::types::RasterMetadata;
use ndarray::Array2;
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
    pub fn for_temperature_kelvin() -> Self {
        Self::UseFallback(288.15)
    }
    pub fn for_precipitation() -> Self {
        Self::UseFallback(0.0)
    }
    pub fn for_water_depth() -> Self {
        Self::KeepOriginal
    }
    pub fn for_salinity() -> Self {
        Self::UseFallback(35.0)
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

/// 空间插值器
pub struct SpatialInterpolator {
    flat_weights: Vec<Weight>,
    offsets: Vec<usize>,
    rotation_angles: Vec<f32>,
    src_dims: (usize, usize),
    config: InterpolatorConfig,
}

impl SpatialInterpolator {
    pub fn new(mesh: &Mesh, target_crs_def: &str, source_meta: &RasterMetadata) -> MhResult<Self> {
        Self::with_config(
            mesh,
            target_crs_def,
            source_meta,
            InterpolatorConfig::default(),
        )
    }

    pub fn with_config(
        mesh: &Mesh,
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
        let should_parallel = config.parallel_enabled && mesh.active_indices.len() >= threshold;

        let results: Vec<(Vec<Weight>, f32)> = if should_parallel {
            mesh.active_indices
                .par_iter()
                .map(|(j, i)| {
                    Self::compute_point_weights(
                        mesh,
                        j,
                        i,
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
            mesh.active_indices
                .iter()
                .map(|(j, i)| {
                    Self::compute_point_weights(
                        mesh,
                        j,
                        i,
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
        mesh: &Mesh,
        j: usize,
        i: usize,
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
        let (mx, my) = mesh.transform.pixel_to_world(i as f64, j as f64);

        let angle = angle_transformer
            .calculate_convergence_angle(mx, my)
            .unwrap_or(0.0) as f32;

        let (sx, sy) = match transformer.transform_point(mx, my) {
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

        // Bilinear weights: W = (1-dx)(1-dy), dx(1-dy), (1-dx)dy, dx*dy
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

        // Catmull-Rom spline kernel
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

    pub fn interpolate(
        &self,
        source_data: &Array2<f64>,
        target: &mut Array2<f64>,
        active_indices: &[(usize, usize)],
        strategy: NoDataStrategy,
    ) -> MhResult<()> {
        let (sh, sw) = source_data.dim();
        if (sw, sh) != self.src_dims {
            return Err(MhError::DataLoad {
                file: "Interpolation Source".into(),
                message: format!(
                    "源数据尺寸不匹配: 期望 {:?}, 实际 ({}, {})",
                    self.src_dims, sw, sh
                ),
            });
        }

        let expected_points = self.offsets.len() - 1;
        if active_indices.len() != expected_points {
            return Err(MhError::InvalidMesh {
                message: format!(
                    "索引数量不匹配: 期望 {}, 实际 {}",
                    expected_points,
                    active_indices.len()
                ),
            });
        }

        let src_slice =
            source_data
                .as_slice_memory_order()
                .ok_or_else(|| MhError::InvalidMesh {
                    message: "源数据内存不连续".into(),
                })?;

        let threshold = self
            .config
            .parallel_threshold
            .unwrap_or_else(|| self.config.method.parallel_threshold());

        if self.config.parallel_enabled && active_indices.len() >= threshold {
            self.interpolate_parallel(src_slice, target, active_indices, strategy)
        } else {
            self.interpolate_serial(src_slice, target, active_indices, strategy)
        }

        Ok(())
    }

    #[inline]
    fn interpolate_serial(
        &self,
        src_slice: &[f64],
        target: &mut Array2<f64>,
        active_indices: &[(usize, usize)],
        strategy: NoDataStrategy,
    ) {
        for (k, &(j, i)) in active_indices.iter().enumerate() {
            let weights = &self.flat_weights[self.offsets[k]..self.offsets[k + 1]];

            match self.compute_value(weights, src_slice) {
                Some(val) => target[[j, i]] = val,
                None => match strategy {
                    NoDataStrategy::UseFallback(v) => target[[j, i]] = v,
                    NoDataStrategy::KeepOriginal => {}
                    NoDataStrategy::SetNaN => target[[j, i]] = f64::NAN,
                },
            }
        }
    }

    #[inline]
    fn interpolate_parallel(
        &self,
        src_slice: &[f64],
        target: &mut Array2<f64>,
        active_indices: &[(usize, usize)],
        strategy: NoDataStrategy,
    ) {
        let results: Vec<Option<f64>> = (0..active_indices.len())
            .into_par_iter()
            .map(|k| {
                let weights = &self.flat_weights[self.offsets[k]..self.offsets[k + 1]];
                self.compute_value(weights, src_slice)
            })
            .collect();

        for (k, &(j, i)) in active_indices.iter().enumerate() {
            match results[k] {
                Some(val) => target[[j, i]] = val,
                None => match strategy {
                    NoDataStrategy::UseFallback(v) => target[[j, i]] = v,
                    NoDataStrategy::KeepOriginal => {}
                    NoDataStrategy::SetNaN => target[[j, i]] = f64::NAN,
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
        u_source: &Array2<f64>,
        v_source: &Array2<f64>,
        u_target: &mut Array2<f64>,
        v_target: &mut Array2<f64>,
        active_indices: &[(usize, usize)],
        strategy: NoDataStrategy,
    ) -> MhResult<()> {
        self.interpolate(u_source, u_target, active_indices, strategy)?;
        self.interpolate(v_source, v_target, active_indices, strategy)?;

        for (k, &(j, i)) in active_indices.iter().enumerate() {
            if k >= self.rotation_angles.len() {
                break;
            }

            let angle = self.rotation_angles[k] as f64;
            if angle.abs() > 1e-6 {
                let u = u_target[[j, i]];
                let v = v_target[[j, i]];

                if !u.is_nan() && !v.is_nan() {
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();

                    // Rotation formula: u' = u*cos(θ) - v*sin(θ), v' = u*sin(θ) + v*cos(θ)
                    u_target[[j, i]] = u * cos_a - v * sin_a;
                    v_target[[j, i]] = u * sin_a + v * cos_a;
                }
            }
        }

        Ok(())
    }

    pub fn interpolate_batch(
        &self,
        sources: &[Array2<f64>],
        targets: &mut [Array2<f64>],
        active_indices: &[(usize, usize)],
        strategies: &[NoDataStrategy],
    ) -> MhResult<()> {
        if sources.len() != targets.len() || sources.len() != strategies.len() {
            return Err(MhError::InvalidMesh {
                message: "源数据、目标数据和策略数量不匹配".into(),
            });
        }

        for ((source, target), &strategy) in sources
            .iter()
            .zip(targets.iter_mut())
            .zip(strategies.iter())
        {
            self.interpolate(source, target, active_indices, strategy)?;
        }

        Ok(())
    }

    pub fn rotation_angles(&self) -> &[f32] {
        &self.rotation_angles
    }

    pub fn source_dimensions(&self) -> (usize, usize) {
        self.src_dims
    }

    pub fn validate_source(&self, source: &Array2<f64>) -> bool {
        source.dim() == (self.src_dims.1, self.src_dims.0)
    }

    pub fn config(&self) -> &InterpolatorConfig {
        &self.config
    }

    pub fn memory_usage(&self) -> usize {
        self.flat_weights.len() * std::mem::size_of::<Weight>()
            + self.offsets.len() * std::mem::size_of::<usize>()
            + self.rotation_angles.len() * std::mem::size_of::<f32>()
            + std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_bilinear_weights() {
        let weights = SpatialInterpolator::bilinear_weights(1.5, 1.5, 10, 10);
        assert_eq!(weights.len(), 4);

        let sum: f32 = weights.iter().map(|w| w.val).sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // 中心点权重应该相等
        for w in &weights {
            assert!((w.val - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nearest_weights() {
        let weights = SpatialInterpolator::nearest_weights(1.5, 1.5, 10, 10);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0].val, 1.0);
        assert_eq!(weights[0].src_idx, 22); // row=2, col=2, idx=2*10+2
    }

    #[test]
    fn test_bicubic_boundary() {
        // 测试边界处退化为双线性
        let weights = SpatialInterpolator::bicubic_weights(0.5, 0.5, 10, 10);
        assert_eq!(weights.len(), 4); // 应该退化为双线性
    }

    #[test]
    fn test_compute_value_with_nan() {
        let source = arr2(&[[1.0, 2.0], [f64::NAN, 4.0]]);
        let weights = vec![
            Weight {
                src_idx: 0,
                val: 0.25,
            },
            Weight {
                src_idx: 1,
                val: 0.25,
            },
            Weight {
                src_idx: 2,
                val: 0.25,
            },
            Weight {
                src_idx: 3,
                val: 0.25,
            },
        ];

        let interpolator = SpatialInterpolator {
            flat_weights: weights,
            offsets: vec![0, 4],
            rotation_angles: vec![0.0],
            src_dims: (2, 2),
            config: InterpolatorConfig::default(),
        };

        let result = interpolator.compute_value(
            &interpolator.flat_weights[0..4],
            source.as_slice_memory_order().unwrap(),
        );

        assert!(result.is_some());
        // 只有3个有效值参与计算: (1 + 2 + 4) / 3 = 7/3
        assert!((result.unwrap() - 7.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_data_strategies() {
        assert_eq!(
            match NoDataStrategy::for_wind() {
                NoDataStrategy::UseFallback(v) => v,
                _ => -1.0,
            },
            0.0
        );

        assert_eq!(
            match NoDataStrategy::for_pressure() {
                NoDataStrategy::UseFallback(v) => v,
                _ => -1.0,
            },
            1013.25
        );

        assert!(matches!(NoDataStrategy::default(), NoDataStrategy::SetNaN));
    }

    #[test]
    fn test_interpolation_method_thresholds() {
        assert!(
            InterpolationMethod::Bicubic.parallel_threshold()
                < InterpolationMethod::Bilinear.parallel_threshold()
        );
        assert!(
            InterpolationMethod::Bilinear.parallel_threshold()
                < InterpolationMethod::NearestNeighbor.parallel_threshold()
        );
    }

    #[test]
    fn test_memory_usage() {
        let interpolator = SpatialInterpolator {
            flat_weights: vec![
                Weight {
                    src_idx: 0,
                    val: 1.0
                };
                100
            ],
            offsets: vec![0, 25, 50, 75, 100],
            rotation_angles: vec![0.0; 4],
            src_dims: (10, 10),
            config: InterpolatorConfig::default(),
        };

        let usage = interpolator.memory_usage();
        assert!(usage > 0);

        // 验证计算正确性
        let expected = 100 * std::mem::size_of::<Weight>()
            + 5 * std::mem::size_of::<usize>()
            + 4 * std::mem::size_of::<f32>()
            + std::mem::size_of::<SpatialInterpolator>();
        assert_eq!(usage, expected);
    }
}
