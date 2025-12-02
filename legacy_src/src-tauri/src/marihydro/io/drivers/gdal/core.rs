// src-tauri/src/marihydro/io/drivers/gdal/core.rs
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::io::traits::{RasterDriver, RasterMetadata};
use std::path::Path;

pub struct GdalDriver;

impl RasterDriver for GdalDriver {
    fn read_metadata(&self, _path: &Path) -> MhResult<RasterMetadata> {
        Err(MhError::io("GDAL not available"))
    }
    fn read_band(&self, _path: &Path, _band: usize) -> MhResult<Vec<f64>> {
        Err(MhError::io("GDAL not available"))
    }
}
