// src-tauri/src/marihydro/io/drivers/gdal_loader.rs

use super::gdal_adapter::{converter, core::GdalCore};
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::traits::{RasterDriver, RasterLoader, RasterRequest};
use crate::marihydro::io::types::{GeoGridData, RasterMetadata};
use ndarray::Array2;

pub struct GdalLoader;

impl RasterDriver for GdalLoader {
    /// 仅读取元数据 (快速)
    fn read_metadata(&self, path: &str) -> MhResult<RasterMetadata> {
        let ds = GdalCore::open(path)?;

        let (w, h) = ds.raster_size();
        let transform = converter::get_transform(&ds)?;
        let wkt = converter::get_wkt(&ds)?;

        // 读取第一个波段的 NoData 值
        let band = ds
            .rasterband(1)
            .map_err(|e| MhError::DatasetLoad(e.to_string()))?;
        let no_data = band.no_data_value();

        Ok(RasterMetadata {
            width: w,
            height: h,
            transform,
            crs_wkt: wkt,
            no_data_value: no_data,
            driver_name: ds.driver().short_name(),
        })
    }

    /// 读取完整数据 (可能包含重采样)
    fn read_data(&self, path: &str, request: Option<RasterRequest>) -> MhResult<GeoGridData> {
        // 1. 调用 Adapter 获取数据集 (可能是原始的，也可能是 Warp 过的)
        let ds = GdalCore::open_and_process(path, request.as_ref())?;

        // 2. 提取元数据
        let (w, h) = ds.raster_size();
        let transform = converter::get_transform(&ds)?;
        let wkt = converter::get_wkt(&ds)?;

        // 3. 读取栅格数据到内存 (Band 1)
        let band = ds
            .rasterband(1)
            .map_err(|e| MhError::DatasetLoad(e.to_string()))?;
        let no_data = band.no_data_value();

        // 读取为 Vec<f64>
        let buffer = band
            .read_as::<f64>((0, 0), (w, h), (w, h), None)
            .map_err(|e| MhError::DatasetLoad(format!("读取栅格数据失败: {}", e)))?;

        // 4. 转换为 ndarray::Array2 (Row Major)
        // buffer.data 是平铺的，Array2::from_shape_vec 需要 (rows, cols)
        let array = Array2::from_shape_vec((h, w), buffer.data)
            .map_err(|e| MhError::InvalidMesh(format!("数组形状不匹配: {}", e)))?;

        // 5. 组装 DTO
        Ok(GeoGridData {
            meta: RasterMetadata {
                width: w,
                height: h,
                transform,
                crs_wkt: wkt,
                no_data_value: no_data,
                driver_name: "GDAL".to_string(),
            },
            data: array,
        })
    }

    fn supports_extension(&self, ext: &str) -> bool {
        matches!(ext.to_lowercase().as_str(), "tif" | "tiff" | "asc" | "img")
    }
}
