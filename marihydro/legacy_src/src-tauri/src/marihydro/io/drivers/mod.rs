// src-tauri/src/marihydro/io/drivers/mod.rs
pub mod gdal;
pub mod netcdf;
pub mod raster_driver;

pub use raster_driver::GenericRasterDriver;
