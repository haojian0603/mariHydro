// src-tauri/src/marihydro/io/drivers/netcdf/core.rs
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::io::traits::{RasterDriver, RasterMetadata};
use std::path::Path;

pub struct NetCdfDriver;

impl RasterDriver for NetCdfDriver {
    fn read_metadata(&self, _path: &Path) -> MhResult<RasterMetadata> {
        Err(MhError::Io("NetCDF not available".into()))
    }
    fn read_band(&self, _path: &Path, _band: usize) -> MhResult<Vec<f64>> {
        Err(MhError::Io("NetCDF not available".into()))
    }
}
