// src-tauri/src/marihydro/io/drivers/gdal_adapter/converter.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::traits::{CoordinateTransform, GeoTransform}; // ✅ 使用 trait

pub struct GdalCoordinateTransform {
    transform: GeoTransform,
}

impl GdalCoordinateTransform {
    pub fn from_dataset(dataset: &gdal::Dataset) -> MhResult<Self> {
        let gt = dataset.geo_transform().map_err(|e| MhError::DataLoad {
            file: "dataset".into(),
            message: format!("无法获取仿射变换: {}", e),
        })?;

        Ok(Self {
            transform: GeoTransform::from_gdal(&gt),
        })
    }
}

impl CoordinateTransform for GdalCoordinateTransform {
    fn get_transform(&self) -> MhResult<GeoTransform> {
        Ok(self.transform)
    }

    fn geo_to_pixel(&self, x: f64, y: f64) -> MhResult<(usize, usize)> {
        // 简化的逆变换（假设无旋转）
        let col = ((x - self.transform.origin_x) / self.transform.pixel_width).round() as usize;
        let row = ((y - self.transform.origin_y) / self.transform.pixel_height).round() as usize;
        Ok((col, row))
    }
}
