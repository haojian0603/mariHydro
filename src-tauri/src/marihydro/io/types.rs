// src-tauri/src/marihydro/io/types.rs

use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// 标准仿射变换六参数 (GDAL Convention)
/// 用于描述栅格像素坐标 (Pixel/Line) 与 地理空间坐标 (X/Y) 的关系
/// X_geo = gt[0] + pixel * gt[1] + line * gt[2]
/// Y_geo = gt[3] + pixel * gt[4] + line * gt[5]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct GeoTransform(pub [f64; 6]);

impl GeoTransform {
    /// 创建一个标准的正北朝向变换
    /// origin_x, origin_y: 左上角坐标
    /// pixel_width: 像素宽度 (正数)
    /// pixel_height: 像素高度 (通常为负数，表示Y轴向下)
    pub fn new(origin_x: f64, origin_y: f64, pixel_width: f64, pixel_height: f64) -> Self {
        Self([
            origin_x,     // 0: top left x
            pixel_width,  // 1: w-e pixel resolution
            0.0,          // 2: rotation, 0 if image is "north up"
            origin_y,     // 3: top left y
            0.0,          // 4: rotation, 0 if image is "north up"
            pixel_height, // 5: n-s pixel resolution (negative)
        ])
    }

    /// 像素坐标 -> 地理坐标 (中心点)
    pub fn pixel_to_world(&self, px: f64, py: f64) -> (f64, f64) {
        let gt = self.0;
        let x = gt[0] + px * gt[1] + py * gt[2];
        let y = gt[3] + px * gt[4] + py * gt[5];
        (x, y)
    }

    /// 获取分辨率 (dx, dy)
    pub fn resolution(&self) -> (f64, f64) {
        (self.0[1].abs(), self.0[5].abs())
    }
}

impl Default for GeoTransform {
    fn default() -> Self {
        // 默认为原点 0,0，分辨率 1.0
        Self([0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }
}

/// 轻量级元数据 (不包含大数组)
/// 用于快速扫描文件信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterMetadata {
    pub width: usize,
    pub height: usize,
    pub transform: GeoTransform,
    pub crs_wkt: String, // 坐标系 WKT 字符串
    pub no_data_value: Option<f64>,
    pub driver_name: String, // e.g., "GTiff", "NetCDF"
}

/// 完整的地理栅格数据对象 (DTO)
/// 所有的 IO Driver 最终都要产出这个结构
#[derive(Debug, Clone)]
pub struct GeoGridData {
    /// 元数据信息
    pub meta: RasterMetadata,

    /// 实际的二维数据数组
    /// 约定：Row-Major (行优先), [y, x]
    pub data: Array2<f64>,
}

impl GeoGridData {
    /// 检查坐标系是否有效
    pub fn has_valid_crs(&self) -> bool {
        !self.meta.crs_wkt.is_empty()
    }

    /// 获取数据统计信息 (Min, Max, Mean) - 用于前端可视化拉伸
    pub fn compute_stats(&self) -> (f64, f64, f64) {
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        let mut sum = 0.0;
        let mut count = 0;

        for &v in self.data.iter() {
            // 跳过 NoData 和 NaN
            if let Some(nd) = self.meta.no_data_value {
                if (v - nd).abs() < 1e-6 {
                    continue;
                }
            }
            if v.is_nan() {
                continue;
            }

            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            sum += v;
            count += 1;
        }

        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
        (min, max, mean)
    }
}
