// src-tauri/src/marihydro/io/drivers/gdal_adapter/converter.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::types::GeoTransform;

/// 将 GDAL 的 [f64; 6] 转换为我们的 GeoTransform
pub fn from_gdal_transform(gt: [f64; 6]) -> GeoTransform {
    GeoTransform(gt)
}

/// 将我们的 GeoTransform 转换为 GDAL 需要的 [f64; 6]
pub fn to_gdal_transform(gt: &GeoTransform) -> [f64; 6] {
    gt.0
}

/// 安全地获取 GDAL 的 WKT 投影字符串
pub fn get_wkt(dataset: &gdal::Dataset) -> MhResult<String> {
    dataset
        .projection()
        .map_err(|e| MhError::DatasetLoad(format!("无法获取投影信息: {}", e)))
}

/// 获取数据集的 GeoTransform
pub fn get_transform(dataset: &gdal::Dataset) -> MhResult<GeoTransform> {
    let gt = dataset
        .geo_transform()
        .map_err(|e| MhError::DatasetLoad(format!("无法获取仿射变换参数: {}", e)))?;
    Ok(from_gdal_transform(gt))
}
