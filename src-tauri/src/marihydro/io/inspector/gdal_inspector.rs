use super::dto::{DatasetMetadata, VariableInfo};
use super::FileInspector;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::infra::manifest::DataFormat;
use crate::marihydro::io::drivers::gdal_adapter::converter; // 复用之前的 Converter
use gdal::Dataset;

pub struct GdalInspector;

impl FileInspector for GdalInspector {
    fn supports(&self, ext: &str) -> bool {
        matches!(ext.to_lowercase().as_str(), "tif" | "tiff" | "asc" | "img")
    }

    fn inspect(&self, path: &str) -> MhResult<DatasetMetadata> {
        let ds = Dataset::open(path).map_err(|e| MhError::DatasetLoad(e.to_string()))?;

        // 1. 获取投影和坐标变换
        let wkt = converter::get_wkt(&ds).ok(); // 允许为空
        let gt = converter::get_transform(&ds)?;

        // 2. 计算 Bounding Box
        let (w, h) = ds.raster_size();
        let (min_x, max_y) = gt.pixel_to_world(0.0, 0.0);
        let (max_x, min_y) = gt.pixel_to_world(w as f64, h as f64);
        // 注意：GDAL y 轴分辨率通常为负，所以 pixel(0,0) 是 TopLeft (MaxY)

        // 3. 扫描波段 (Bands) 作为变量
        let mut variables = Vec::new();
        let count = ds.raster_count();
        for i in 1..=count {
            let band = ds
                .rasterband(i)
                .map_err(|e| MhError::DatasetLoad(e.to_string()))?;

            // 尝试获取波段描述，如果没有则命名为 Band1, Band2...
            let desc = band.description().unwrap_or_else(|_| String::new());
            let name = if desc.is_empty() {
                format!("Band{}", i)
            } else {
                desc
            };

            variables.push(VariableInfo {
                name,
                dimensions: vec![format!("y:{}", h), format!("x:{}", w)],
                dtype: "float".to_string(), // GDAL 内部一般处理为 float
                standard_name: None,
                units: None, // GDAL 较少包含单位元数据
            });
        }

        Ok(DatasetMetadata {
            format: DataFormat::GeoTIFF,
            variables,
            crs_wkt: wkt,
            time_coverage: None, // GeoTIFF 通常是静态快照
            geo_bounds: Some([
                min_x.min(max_x),
                min_y.min(max_y),
                min_x.max(max_x),
                min_y.max(max_y),
            ]),
        })
    }
}
