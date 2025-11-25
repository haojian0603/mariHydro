// src-tauri/src/marihydro/io/loaders/mod.rs

pub mod raster;

use crate::marihydro::infra::error::MhResult;
use ndarray::Array2;

/// 栅格数据加载器接口
/// 用于读取 地形(DEM)、曼宁系数(Roughness)、初始场等
pub trait RasterLoader {
    /// 读取并返回与目标网格匹配的二维数组
    /// target_shape: (nx, ny) - 加载器负责重采样(Resample)到这个尺寸
    /// target_bounds: (xmin, ymin, xmax, ymax) - 加载器负责裁剪(Clip)到这个范围
    fn load_array(
        &self,
        file_path: &str,
        target_shape: (usize, usize),
        target_bounds: Option<(f64, f64, f64, f64)>,
    ) -> MhResult<Array2<f64>>;
}
