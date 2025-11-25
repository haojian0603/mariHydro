// src-tauri/src/marihydro/io/loaders/raster.rs

use super::RasterLoader;
use crate::marihydro::infra::error::{MhError, MhResult};
use ndarray::Array2;
use std::path::Path;

/// 标准栅格加载器
pub struct StandardRasterLoader;

impl RasterLoader for StandardRasterLoader {
    fn load_array(
        &self,
        file_path: &str,
        target_shape: (usize, usize),
        _target_bounds: Option<(f64, f64, f64, f64)>,
    ) -> MhResult<Array2<f64>> {
        let path = Path::new(file_path);

        if !path.exists() {
            return Err(MhError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("文件不存在: {}", file_path),
            )));
        }

        // TODO: [工程化接入点] 集成 gdal crate
        // 目前阶段为了不破坏编译环境 (GDAL 需要系统库支持)，
        // 我们暂时返回一个 Mock 的数据，或者简单的 ASC 读取器。

        // 真正的逻辑应该是：
        // 1. let dataset = gdal::Dataset::open(path)?;
        // 2. dataset.rasterband(1)?.read_as_array(...)
        // 3. 执行重采样算法 (Bilinear/Cubic) 匹配 target_shape

        println!("(WARN) 使用模拟加载器读取: {}", file_path);

        // 模拟返回一个平坦地形 (用于测试)
        let (nx, ny) = target_shape;
        let array = Array2::from_elem((nx, ny), -10.0); // 水深 10m

        Ok(array)
    }
}
