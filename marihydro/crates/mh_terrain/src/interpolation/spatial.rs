// marihydro\crates\mh_terrain\src/interpolation/spatial.rs

//! 空间插值器
//!
//! 用于将栅格数据插值到网格单元中心。
//! 预计算插值权重以加速批量插值。
//!
//! # 示例
//!
//! ```ignore
//! use mh_terrain::interpolation::spatial::{SpatialInterpolator, InterpolationMethod};
//!
//! // 创建插值器（预计算权重）
//! let interp = SpatialInterpolator::new(
//!     &target_points,
//!     &source_transform,
//!     (width, height),
//!     InterpolationMethod::Bilinear,
//! )?;
//!
//! // 批量插值
//! let mut output = vec![0.0; target_points.len()];
//! interp.interpolate(&source_data, &mut output)?;
//! ```

use mh_foundation::error::{MhError, MhResult};
use mh_geo::Point2D;

/// 插值方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpolationMethod {
    /// 最近邻
    NearestNeighbor,
    /// 双线性
    #[default]
    Bilinear,
    /// 双三次（16点）
    Bicubic,
}

impl InterpolationMethod {
    /// 获取方法名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::NearestNeighbor => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
        }
    }

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
    /// 风速无数据策略
    pub fn for_wind() -> Self {
        Self::UseFallback(0.0)
    }

    /// 气压无数据策略
    pub fn for_pressure() -> Self {
        Self::UseFallback(101325.0)
    }

    /// 温度无数据策略
    pub fn for_temperature() -> Self {
        Self::UseFallback(15.0)
    }
}

/// 仿射变换（栅格坐标 ↔ 地理坐标）
#[derive(Debug, Clone, Copy)]
pub struct GeoTransform {
    /// 左上角 x 坐标
    pub origin_x: f64,
    /// 左上角 y 坐标
    pub origin_y: f64,
    /// 像素宽度（x 方向分辨率）
    pub pixel_width: f64,
    /// 像素高度（y 方向分辨率，通常为负）
    pub pixel_height: f64,
    /// x 方向旋转（通常为 0）
    pub rotation_x: f64,
    /// y 方向旋转（通常为 0）
    pub rotation_y: f64,
}

impl GeoTransform {
    /// 创建简单的仿射变换（无旋转）
    pub fn new(origin_x: f64, origin_y: f64, pixel_width: f64, pixel_height: f64) -> Self {
        Self {
            origin_x,
            origin_y,
            pixel_width,
            pixel_height,
            rotation_x: 0.0,
            rotation_y: 0.0,
        }
    }

    /// 地理坐标转栅格坐标
    #[inline]
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        // 简化版（无旋转）
        let px = (x - self.origin_x) / self.pixel_width;
        let py = (y - self.origin_y) / self.pixel_height;
        (px, py)
    }

    /// 栅格坐标转地理坐标
    #[inline]
    pub fn pixel_to_geo(&self, px: f64, py: f64) -> (f64, f64) {
        let x = self.origin_x + px * self.pixel_width;
        let y = self.origin_y + py * self.pixel_height;
        (x, y)
    }
}

/// 插值权重
#[derive(Debug, Clone, Copy)]
struct Weight {
    /// 源像素索引
    src_idx: u32,
    /// 权重值
    val: f32,
}

/// 空间插值器配置
#[derive(Debug, Clone)]
pub struct InterpolatorConfig {
    /// 插值方法
    pub method: InterpolationMethod,
    /// 无数据策略
    pub nodata_strategy: NoDataStrategy,
    /// 无数据值
    pub nodata_value: f64,
    /// 是否启用并行
    pub parallel_enabled: bool,
    /// 并行阈值
    pub parallel_threshold: Option<usize>,
}

impl Default for InterpolatorConfig {
    fn default() -> Self {
        Self {
            method: InterpolationMethod::Bilinear,
            nodata_strategy: NoDataStrategy::SetNaN,
            nodata_value: f64::NAN,
            parallel_enabled: true,
            parallel_threshold: None,
        }
    }
}

/// 空间插值器
///
/// 预计算目标点到源栅格的插值权重。
pub struct SpatialInterpolator {
    /// 展平的权重数组
    flat_weights: Vec<Weight>,
    /// 每个目标点的权重偏移
    offsets: Vec<usize>,
    /// 源栅格尺寸
    src_dims: (usize, usize),
    /// 配置
    config: InterpolatorConfig,
}

impl SpatialInterpolator {
    /// 从目标点和源栅格创建插值器
    pub fn new(
        target_points: &[Point2D],
        source_transform: &GeoTransform,
        source_dims: (usize, usize),
        config: InterpolatorConfig,
    ) -> MhResult<Self> {
        let (src_w, src_h) = source_dims;
        let n_targets = target_points.len();

        // 预分配权重数组
        let expected_total = n_targets * config.method.expected_weights();
        let mut flat_weights = Vec::with_capacity(expected_total);
        let mut offsets = Vec::with_capacity(n_targets + 1);
        offsets.push(0);

        // 计算每个目标点的权重
        for point in target_points {
            let weights = Self::compute_point_weights(
                point,
                source_transform,
                src_w,
                src_h,
                config.method,
            );

            flat_weights.extend(weights);
            offsets.push(flat_weights.len());
        }

        Ok(Self {
            flat_weights,
            offsets,
            src_dims: source_dims,
            config,
        })
    }

    /// 计算单点的权重
    fn compute_point_weights(
        point: &Point2D,
        transform: &GeoTransform,
        src_w: usize,
        src_h: usize,
        method: InterpolationMethod,
    ) -> Vec<Weight> {
        let (px, py) = transform.geo_to_pixel(point.x, point.y);

        match method {
            InterpolationMethod::NearestNeighbor => {
                Self::nearest_weights(px, py, src_w, src_h)
            }
            InterpolationMethod::Bilinear => {
                Self::bilinear_weights(px, py, src_w, src_h)
            }
            InterpolationMethod::Bicubic => {
                Self::bicubic_weights(px, py, src_w, src_h)
            }
        }
    }

    /// 最近邻权重
    fn nearest_weights(px: f64, py: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let ix = px.round() as isize;
        let iy = py.round() as isize;

        if ix < 0 || iy < 0 || ix >= src_w as isize || iy >= src_h as isize {
            return Vec::new();
        }

        let idx = (iy as usize) * src_w + (ix as usize);
        vec![Weight { src_idx: idx as u32, val: 1.0 }]
    }

    /// 双线性权重
    fn bilinear_weights(px: f64, py: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let x0 = px.floor() as isize;
        let y0 = py.floor() as isize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        // 检查边界
        if x0 < 0 || y0 < 0 || x1 >= src_w as isize || y1 >= src_h as isize {
            // 回退到最近邻
            return Self::nearest_weights(px, py, src_w, src_h);
        }

        let dx = (px - x0 as f64) as f32;
        let dy = (py - y0 as f64) as f32;

        let x0 = x0 as usize;
        let y0 = y0 as usize;
        let x1 = x1 as usize;
        let y1 = y1 as usize;

        vec![
            Weight { src_idx: (y0 * src_w + x0) as u32, val: (1.0 - dx) * (1.0 - dy) },
            Weight { src_idx: (y0 * src_w + x1) as u32, val: dx * (1.0 - dy) },
            Weight { src_idx: (y1 * src_w + x0) as u32, val: (1.0 - dx) * dy },
            Weight { src_idx: (y1 * src_w + x1) as u32, val: dx * dy },
        ]
    }

    /// 双三次权重
    fn bicubic_weights(px: f64, py: f64, src_w: usize, src_h: usize) -> Vec<Weight> {
        let x0 = px.floor() as isize;
        let y0 = py.floor() as isize;

        // 检查边界（需要 4x4 窗口）
        if x0 < 1 || y0 < 1 || x0 + 2 >= src_w as isize || y0 + 2 >= src_h as isize {
            // 回退到双线性
            return Self::bilinear_weights(px, py, src_w, src_h);
        }

        let dx = (px - x0 as f64) as f32;
        let dy = (py - y0 as f64) as f32;

        let mut weights = Vec::with_capacity(16);

        // Catmull-Rom 权重
        let wx = [
            cubic_weight(dx + 1.0),
            cubic_weight(dx),
            cubic_weight(1.0 - dx),
            cubic_weight(2.0 - dx),
        ];

        let wy = [
            cubic_weight(dy + 1.0),
            cubic_weight(dy),
            cubic_weight(1.0 - dy),
            cubic_weight(2.0 - dy),
        ];

        for j in 0..4 {
            for i in 0..4 {
                let x = (x0 - 1 + i as isize) as usize;
                let y = (y0 - 1 + j as isize) as usize;
                let w = wx[i] * wy[j];
                if w.abs() > 1e-10 {
                    weights.push(Weight {
                        src_idx: (y * src_w + x) as u32,
                        val: w,
                    });
                }
            }
        }

        weights
    }

    /// 使用预计算的权重进行插值
    pub fn interpolate(&self, source: &[f64], output: &mut [f64]) -> MhResult<()> {
        let (src_w, src_h) = self.src_dims;
        let expected_size = src_w * src_h;

        if source.len() != expected_size {
            return Err(MhError::size_mismatch(
                "source raster",
                expected_size,
                source.len(),
            ));
        }

        if output.len() + 1 != self.offsets.len() {
            return Err(MhError::size_mismatch(
                "output",
                self.offsets.len() - 1,
                output.len(),
            ));
        }

        let nodata = self.config.nodata_value;

        for i in 0..output.len() {
            let start = self.offsets[i];
            let end = self.offsets[i + 1];

            if start == end {
                // 没有权重
                output[i] = match self.config.nodata_strategy {
                    NoDataStrategy::UseFallback(v) => v,
                    NoDataStrategy::KeepOriginal => output[i],
                    NoDataStrategy::SetNaN => f64::NAN,
                };
                continue;
            }

            let mut sum = 0.0;
            let mut weight_sum = 0.0f32;
            let mut has_nodata = false;

            for w in &self.flat_weights[start..end] {
                let val = source[w.src_idx as usize];
                if val.is_nan() || (nodata.is_finite() && (val - nodata).abs() < 1e-10) {
                    has_nodata = true;
                    continue;
                }
                sum += val * (w.val as f64);
                weight_sum += w.val;
            }

            if weight_sum.abs() < 1e-10 || has_nodata && weight_sum < 0.5 {
                output[i] = match self.config.nodata_strategy {
                    NoDataStrategy::UseFallback(v) => v,
                    NoDataStrategy::KeepOriginal => output[i],
                    NoDataStrategy::SetNaN => f64::NAN,
                };
            } else {
                output[i] = sum / (weight_sum as f64);
            }
        }

        Ok(())
    }

    /// 获取目标点数量
    pub fn n_targets(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// 获取总权重数量
    pub fn n_weights(&self) -> usize {
        self.flat_weights.len()
    }

    /// 获取配置
    pub fn config(&self) -> &InterpolatorConfig {
        &self.config
    }
}

/// Catmull-Rom 三次权重函数
#[inline]
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        (1.5 * t - 2.5) * t * t + 1.0
    } else if t <= 2.0 {
        ((-0.5 * t + 2.5) * t - 4.0) * t + 2.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_raster() -> (Vec<f64>, GeoTransform, (usize, usize)) {
        // 4x4 栅格
        // 值 = x + y * 10
        let mut data = Vec::with_capacity(16);
        for y in 0..4 {
            for x in 0..4 {
                data.push((x + y * 10) as f64);
            }
        }

        let transform = GeoTransform::new(0.0, 40.0, 10.0, -10.0);
        (data, transform, (4, 4))
    }

    #[test]
    fn test_nearest_neighbor() {
        let (data, transform, dims) = create_test_raster();

        // 测试点在 (14, 26) - 转换为像素 (1.4, 1.4)，最近邻到 (1, 1) = 11
        let points = vec![Point2D::new(14.0, 26.0)];

        let config = InterpolatorConfig {
            method: InterpolationMethod::NearestNeighbor,
            ..Default::default()
        };

        let interp = SpatialInterpolator::new(&points, &transform, dims, config).unwrap();
        let mut output = vec![0.0];
        interp.interpolate(&data, &mut output).unwrap();

        assert!((output[0] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear() {
        let (data, transform, dims) = create_test_raster();

        // 测试点在 (5, 35) - 在 (0,0), (1,0), (0,1), (1,1) 中间
        // 值分别是 0, 1, 10, 11
        // 中心应该是 (0 + 1 + 10 + 11) / 4 = 5.5
        let points = vec![Point2D::new(5.0, 35.0)];

        let config = InterpolatorConfig {
            method: InterpolationMethod::Bilinear,
            ..Default::default()
        };

        let interp = SpatialInterpolator::new(&points, &transform, dims, config).unwrap();
        let mut output = vec![0.0];
        interp.interpolate(&data, &mut output).unwrap();

        assert!((output[0] - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_geo_transform() {
        let transform = GeoTransform::new(0.0, 100.0, 10.0, -10.0);

        // (0, 100) -> (0, 0)
        let (px, py) = transform.geo_to_pixel(0.0, 100.0);
        assert!((px - 0.0).abs() < 1e-10);
        assert!((py - 0.0).abs() < 1e-10);

        // (10, 90) -> (1, 1)
        let (px, py) = transform.geo_to_pixel(10.0, 90.0);
        assert!((px - 1.0).abs() < 1e-10);
        assert!((py - 1.0).abs() < 1e-10);
    }
}
