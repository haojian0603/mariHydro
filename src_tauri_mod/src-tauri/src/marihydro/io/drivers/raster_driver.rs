// src-tauri/src/marihydro/io/drivers/raster_driver.rs
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::io::traits::{GeoTransform, RasterDriver, RasterMetadata};
use std::path::Path;

pub struct GenericRasterDriver;

impl RasterDriver for GenericRasterDriver {
    fn read_metadata(&self, path: &Path) -> MhResult<RasterMetadata> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "tif" | "tiff" => Err(MhError::io("GDAL support not compiled")),
            "nc" | "nc4" => Err(MhError::io("NetCDF support not compiled")),
            _ => Err(MhError::io(format!("Unsupported format: {}", ext))),
        }
    }
    fn read_band(&self, path: &Path, _band: usize) -> MhResult<Vec<f64>> {
        Err(MhError::io(format!("Cannot read {:?}", path)))
    }
}
