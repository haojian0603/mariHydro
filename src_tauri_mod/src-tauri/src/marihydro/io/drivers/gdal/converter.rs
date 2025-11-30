// src-tauri/src/marihydro/io/drivers/gdal/converter.rs
use crate::marihydro::io::traits::GeoTransform;

pub fn gdal_to_geo_transform(gt: &[f64; 6]) -> GeoTransform { GeoTransform::from_gdal(gt) }
pub fn geo_transform_to_gdal(gt: &GeoTransform) -> [f64; 6] { gt.to_gdal() }
