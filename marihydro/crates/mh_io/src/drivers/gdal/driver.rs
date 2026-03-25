// crates/mh_io/src/drivers/gdal/driver.rs

//! GDAL 栅格驱动实现
//!
//! IO_SOURCE: GDAL 官方文档中的 `gdalinfo -json` 元数据输出约定，以及 `gdal_translate -of AAIGrid`
//! IO_SCOPE: 当前 CLI 回退路径只接受数值可解析的 AAIGrid 头和栅格载荷；NoData、尺寸和栅格值任一字段解析失败时直接报错，不静默丢弃也不合成默认值。

use super::error::GdalError;
use serde_json;
use std::path::Path;

/// 栅格元数据
#[derive(Debug, Clone)]
pub struct RasterMetadata {
    /// 宽度 (像素)
    pub width: usize,
    /// 高度 (像素)
    pub height: usize,
    /// 波段数
    pub band_count: usize,
    /// 地理变换参数 [x_origin, x_res, x_rot, y_origin, y_rot, y_res]
    pub geo_transform: [f64; 6],
    /// 投影 WKT
    pub projection: Option<String>,
    /// NoData 值
    pub nodata: Option<f64>,
}

impl RasterMetadata {
    /// 获取像素分辨率
    pub fn resolution(&self) -> (f64, f64) {
        (self.geo_transform[1].abs(), self.geo_transform[5].abs())
    }

    /// 获取地理范围 (min_x, min_y, max_x, max_y)
    pub fn extent(&self) -> (f64, f64, f64, f64) {
        let x0 = self.geo_transform[0];
        let y0 = self.geo_transform[3];
        let x1 = x0 + self.width as f64 * self.geo_transform[1];
        let y1 = y0 + self.height as f64 * self.geo_transform[5];

        (x0.min(x1), y0.min(y1), x0.max(x1), y0.max(y1))
    }

    /// 地理坐标转像素坐标
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> (f64, f64) {
        let px = (x - self.geo_transform[0]) / self.geo_transform[1];
        let py = (y - self.geo_transform[3]) / self.geo_transform[5];
        (px, py)
    }

    /// 像素坐标转地理坐标
    pub fn pixel_to_geo(&self, px: f64, py: f64) -> (f64, f64) {
        let x = self.geo_transform[0] + px * self.geo_transform[1];
        let y = self.geo_transform[3] + py * self.geo_transform[5];
        (x, y)
    }
}

/// 栅格波段数据
#[derive(Debug, Clone)]
pub struct RasterBand {
    /// 数据
    pub data: Vec<f64>,
    /// 宽度
    pub width: usize,
    /// 高度
    pub height: usize,
    /// NoData 值
    pub nodata: Option<f64>,
}

impl RasterBand {
    /// 获取指定位置的值
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let val = self.data[y * self.width + x];
        if let Some(nd) = self.nodata {
            if (val - nd).abs() < 1e-10 {
                return None;
            }
        }
        Some(val)
    }

    /// 双线性插值
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        if x < 0.0 || y < 0.0 {
            return None;
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = x0 + 1;
        let y1 = y0 + 1;

        if x1 >= self.width || y1 >= self.height {
            return None;
        }

        let v00 = self.get(x0, y0)?;
        let v10 = self.get(x1, y0)?;
        let v01 = self.get(x0, y1)?;
        let v11 = self.get(x1, y1)?;

        let fx = x - x0 as f64;
        let fy = y - y0 as f64;

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        Some(v0 * (1.0 - fy) + v1 * fy)
    }
}

/// GDAL 栅格驱动
#[cfg(feature = "gdal")]
pub struct GdalDriver {
    dataset: gdal::Dataset,
    metadata: RasterMetadata,
}

#[cfg(feature = "gdal")]
impl GdalDriver {
    /// 打开栅格文件
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GdalError> {
        use gdal::Dataset;

        let path = path.as_ref();
        if !path.exists() {
            return Err(GdalError::FileNotFound(path.display().to_string()));
        }

        let dataset = Dataset::open(path)?;
        let (width, height) = dataset.raster_size();
        let band_count = dataset.raster_count();
        let geo_transform = dataset.geo_transform()?;
        let projection = dataset.projection().ok();

        let nodata = if band_count > 0 {
            dataset.rasterband(1).ok().and_then(|b| b.no_data_value())
        } else {
            None
        };

        let metadata = RasterMetadata {
            width,
            height,
            band_count,
            geo_transform,
            projection,
            nodata,
        };

        Ok(Self { dataset, metadata })
    }

    /// 获取元数据
    pub fn metadata(&self) -> &RasterMetadata {
        &self.metadata
    }

    /// 读取波段数据
    pub fn read_band(&self, band_idx: usize) -> Result<RasterBand, GdalError> {
        use gdal::raster::GdalType;

        if band_idx == 0 || band_idx > self.metadata.band_count {
            return Err(GdalError::BandNotFound(band_idx));
        }

        let band = self.dataset.rasterband(band_idx)?;
        let nodata = band.no_data_value();

        let (width, height) = (self.metadata.width, self.metadata.height);
        let data: Vec<f64> = band.read_as::<f64>((0, 0), (width, height), (width, height), None)?;

        Ok(RasterBand {
            data,
            width,
            height,
            nodata,
        })
    }

    /// 读取所有波段
    pub fn read_all_bands(&self) -> Result<Vec<RasterBand>, GdalError> {
        let mut bands = Vec::new();
        for i in 1..=self.metadata.band_count {
            bands.push(self.read_band(i)?);
        }
        Ok(bands)
    }
}

/// 无 GDAL 支持时的 CLI 驱动实现
#[cfg(not(feature = "gdal"))]
pub struct GdalDriver {
    path: std::path::PathBuf,
    metadata: RasterMetadata,
}

#[cfg(not(feature = "gdal"))]
impl GdalDriver {
    /// 打开栅格文件 (CLI 驱动)
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GdalError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(GdalError::FileNotFound(path.display().to_string()));
        }

        let metadata = cli_read_metadata(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            metadata,
        })
    }

    /// 获取元数据
    pub fn metadata(&self) -> &RasterMetadata {
        &self.metadata
    }

    /// 读取波段数据
    pub fn read_band(&self, band_idx: usize) -> Result<RasterBand, GdalError> {
        if band_idx == 0 || band_idx > self.metadata.band_count {
            return Err(GdalError::BandNotFound(band_idx));
        }
        cli_read_band(&self.path, band_idx, &self.metadata)
    }

    /// 读取所有波段
    pub fn read_all_bands(&self) -> Result<Vec<RasterBand>, GdalError> {
        let mut bands = Vec::new();
        for i in 1..=self.metadata.band_count {
            bands.push(self.read_band(i)?);
        }
        Ok(bands)
    }
}

#[cfg(not(feature = "gdal"))]
fn parse_json_usize(value: &serde_json::Value, field: &str) -> Result<usize, GdalError> {
    value
        .as_u64()
        .map(|raw| raw as usize)
        .ok_or_else(|| GdalError::ReadFailed(format!("gdalinfo 输出中的 {field} 不是无符号整数")))
}

#[cfg(not(feature = "gdal"))]
fn parse_json_f64(value: &serde_json::Value, field: &str) -> Result<f64, GdalError> {
    value
        .as_f64()
        .ok_or_else(|| GdalError::ReadFailed(format!("gdalinfo 输出中的 {field} 不是数值")))
}

#[cfg(not(feature = "gdal"))]
fn parse_gdalinfo_metadata(value: &serde_json::Value) -> Result<RasterMetadata, GdalError> {
    let size = value
        .get("size")
        .and_then(|v| v.as_array())
        .ok_or_else(|| GdalError::ReadFailed("gdalinfo 输出缺少 size".to_string()))?;
    if size.len() != 2 {
        return Err(GdalError::ReadFailed(
            "gdalinfo 输出中的 size 长度不正确".to_string(),
        ));
    }

    let width = parse_json_usize(&size[0], "size[0]")?;
    let height = parse_json_usize(&size[1], "size[1]")?;

    let geo_transform = value
        .get("geoTransform")
        .and_then(|v| v.as_array())
        .ok_or_else(|| GdalError::ReadFailed("gdalinfo 输出缺少 geoTransform".to_string()))?;
    if geo_transform.len() != 6 {
        return Err(GdalError::ReadFailed("geoTransform 长度不正确".to_string()));
    }

    let geo_transform = [
        parse_json_f64(&geo_transform[0], "geoTransform[0]")?,
        parse_json_f64(&geo_transform[1], "geoTransform[1]")?,
        parse_json_f64(&geo_transform[2], "geoTransform[2]")?,
        parse_json_f64(&geo_transform[3], "geoTransform[3]")?,
        parse_json_f64(&geo_transform[4], "geoTransform[4]")?,
        parse_json_f64(&geo_transform[5], "geoTransform[5]")?,
    ];

    let projection = value
        .get("coordinateSystem")
        .and_then(|v| v.get("wkt"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let empty_bands: Vec<serde_json::Value> = Vec::new();
    let bands = value
        .get("bands")
        .and_then(|v| v.as_array())
        .unwrap_or(&empty_bands);
    let band_count = bands.len().max(1);
    let nodata = bands
        .first()
        .and_then(|b| b.get("noDataValue"))
        .and_then(|v| v.as_f64());

    Ok(RasterMetadata {
        width,
        height,
        band_count,
        geo_transform,
        projection,
        nodata,
    })
}

#[cfg(not(feature = "gdal"))]
fn cli_read_metadata(path: &Path) -> Result<RasterMetadata, GdalError> {
    let output = std::process::Command::new(gdalinfo_bin())
        .arg("-json")
        .arg(path)
        .output()
        .map_err(|_| GdalError::NotAvailable)?;

    if !output.status.success() {
        return Err(GdalError::OpenFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| GdalError::Other(e.to_string()))?;
    parse_gdalinfo_metadata(&value)
}

#[cfg(not(feature = "gdal"))]
fn cli_read_band(
    path: &Path,
    band_idx: usize,
    meta: &RasterMetadata,
) -> Result<RasterBand, GdalError> {
    let output = std::process::Command::new(gdal_translate_bin())
        .arg("-of")
        .arg("AAIGrid")
        .arg("-b")
        .arg(band_idx.to_string())
        .arg(path)
        .arg("/vsistdout/")
        .output()
        .map_err(|_| GdalError::NotAvailable)?;

    if !output.status.success() {
        return Err(GdalError::ReadFailed(
            String::from_utf8_lossy(&output.stderr).to_string(),
        ));
    }

    let text = String::from_utf8_lossy(&output.stdout);
    parse_ascii_grid_band(&text, meta)
}

#[cfg(not(feature = "gdal"))]
fn parse_ascii_grid_band(text: &str, meta: &RasterMetadata) -> Result<RasterBand, GdalError> {
    let mut ncols = 0usize;
    let mut nrows = 0usize;
    let mut nodata = None;
    let mut data_start = 0usize;
    for (idx, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.to_lowercase().starts_with("ncols") {
            let raw = line
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| GdalError::ReadFailed("AAIGrid 头缺少 ncols 值".to_string()))?;
            ncols = raw
                .parse()
                .map_err(|_| GdalError::ReadFailed(format!("AAIGrid ncols 无法解析: {raw}")))?;
        } else if line.to_lowercase().starts_with("nrows") {
            let raw = line
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| GdalError::ReadFailed("AAIGrid 头缺少 nrows 值".to_string()))?;
            nrows = raw
                .parse()
                .map_err(|_| GdalError::ReadFailed(format!("AAIGrid nrows 无法解析: {raw}")))?;
        } else if line.to_lowercase().starts_with("nodata_value") {
            let raw = line.split_whitespace().nth(1).ok_or_else(|| {
                GdalError::ReadFailed("AAIGrid 头缺少 nodata_value 值".to_string())
            })?;
            nodata = Some(raw.parse().map_err(|_| {
                GdalError::ReadFailed(format!("AAIGrid nodata_value 无法解析: {raw}"))
            })?);
        } else if !line.is_empty() {
            data_start = idx;
            break;
        }
    }

    if ncols == 0 || nrows == 0 {
        ncols = meta.width;
        nrows = meta.height;
    }

    let mut data = Vec::with_capacity(ncols * nrows);
    for line in text.lines().skip(data_start) {
        for token in line.split_whitespace() {
            let value = token.parse::<f64>().map_err(|_| {
                GdalError::ReadFailed(format!("AAIGrid 栅格值无法解析: {token}"))
            })?;
            data.push(value);
        }
    }

    if data.len() != ncols * nrows {
        return Err(GdalError::ReadFailed(
            "栅格数据长度与维度不匹配".to_string(),
        ));
    }

    Ok(RasterBand {
        data,
        width: ncols,
        height: nrows,
        nodata: nodata.or(meta.nodata),
    })
}

#[cfg(not(feature = "gdal"))]
fn gdalinfo_bin() -> String {
    std::env::var("GDALINFO_BIN").unwrap_or_else(|_| "gdalinfo".to_string())
}

#[cfg(not(feature = "gdal"))]
fn gdal_translate_bin() -> String {
    std::env::var("GDAL_TRANSLATE_BIN").unwrap_or_else(|_| "gdal_translate".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raster_metadata_extent() {
        let meta = RasterMetadata {
            width: 100,
            height: 100,
            band_count: 1,
            geo_transform: [0.0, 1.0, 0.0, 100.0, 0.0, -1.0],
            projection: None,
            nodata: None,
        };

        let (min_x, min_y, max_x, max_y) = meta.extent();
        assert!((min_x - 0.0).abs() < 1e-10);
        assert!((max_x - 100.0).abs() < 1e-10);
        assert!((min_y - 0.0).abs() < 1e-10);
        assert!((max_y - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_raster_band_interpolate() {
        let band = RasterBand {
            data: vec![0.0, 1.0, 2.0, 3.0],
            width: 2,
            height: 2,
            nodata: None,
        };

        // 中心点应该是 4 个角的平均值
        let val = band.interpolate(0.5, 0.5).unwrap();
        assert!((val - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_gdalinfo_metadata_rejects_non_numeric_geotransform() {
        let value = serde_json::json!({
            "size": [100, 50],
            "geoTransform": [0.0, 1.0, "bad", 100.0, 0.0, -1.0]
        });
        assert!(parse_gdalinfo_metadata(&value).is_err());
    }

    #[cfg(not(feature = "gdal"))]
    #[test]
    fn test_parse_ascii_grid_rejects_invalid_nodata_value() {
        let meta = RasterMetadata {
            width: 2,
            height: 2,
            band_count: 1,
            geo_transform: [0.0, 1.0, 0.0, 2.0, 0.0, -1.0],
            projection: None,
            nodata: None,
        };
        let text = "ncols 2\nnrows 2\nNODATA_value bad\n1 2\n3 4\n";
        assert!(parse_ascii_grid_band(text, &meta).is_err());
    }

    #[cfg(not(feature = "gdal"))]
    #[test]
    fn test_parse_ascii_grid_rejects_invalid_data_token() {
        let meta = RasterMetadata {
            width: 2,
            height: 2,
            band_count: 1,
            geo_transform: [0.0, 1.0, 0.0, 2.0, 0.0, -1.0],
            projection: None,
            nodata: None,
        };
        let text = "ncols 2\nnrows 2\n1 2\n3 bad\n";
        assert!(parse_ascii_grid_band(text, &meta).is_err());
    }
}
