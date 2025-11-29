// src-tauri/src/marihydro/io/traits.rs

use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::infra::error::MhResult;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct GeoTransform {
    pub origin_x: f64,
    pub origin_y: f64,
    pub pixel_width: f64,
    pub pixel_height: f64,
    pub rotation_x: f64,
    pub rotation_y: f64,
}

impl GeoTransform {
    pub fn from_gdal(gt: &[f64; 6]) -> Self {
        Self {
            origin_x: gt[0],
            pixel_width: gt[1],
            rotation_x: gt[2],
            origin_y: gt[3],
            rotation_y: gt[4],
            pixel_height: gt[5],
        }
    }

    pub fn to_gdal(&self) -> [f64; 6] {
        [
            self.origin_x,
            self.pixel_width,
            self.rotation_x,
            self.origin_y,
            self.rotation_y,
            self.pixel_height,
        ]
    }

    #[inline]
    pub fn apply(&self, col: usize, row: usize) -> (f64, f64) {
        let x = self.origin_x + col as f64 * self.pixel_width + row as f64 * self.rotation_x;
        let y = self.origin_y + col as f64 * self.rotation_y + row as f64 * self.pixel_height;
        (x, y)
    }

    #[inline]
    pub fn inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let det = self.pixel_width * self.pixel_height - self.rotation_x * self.rotation_y;
        if det.abs() < 1e-12 {
            return None;
        }
        let dx = x - self.origin_x;
        let dy = y - self.origin_y;
        let col = (self.pixel_height * dx - self.rotation_x * dy) / det;
        let row = (-self.rotation_y * dx + self.pixel_width * dy) / det;
        Some((col, row))
    }

    #[inline]
    pub fn resolution(&self) -> (f64, f64) {
        (self.pixel_width.abs(), self.pixel_height.abs())
    }
}

impl Default for GeoTransform {
    fn default() -> Self {
        Self {
            origin_x: 0.0,
            origin_y: 0.0,
            pixel_width: 1.0,
            pixel_height: -1.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RasterMetadata {
    pub width: usize,
    pub height: usize,
    pub n_bands: usize,
    pub crs_wkt: Option<String>,
    pub geo_transform: Option<GeoTransform>,
    pub no_data_value: Option<f64>,
}

impl RasterMetadata {
    pub fn total_pixels(&self) -> usize {
        self.width * self.height
    }
}

pub trait RasterDriver: Send + Sync {
    fn read_metadata(&self, path: &Path) -> MhResult<RasterMetadata>;
    fn read_band(&self, path: &Path, band: usize) -> MhResult<Vec<f64>>;
    fn read_band_to_slice(&self, path: &Path, band: usize, output: &mut [f64]) -> MhResult<()>;
}

pub trait MeshLoader: Send + Sync {
    fn load(&self, path: &Path) -> MhResult<UnstructuredMesh>;
    fn supports_extension(&self, ext: &str) -> bool;
}

pub trait TimeSeriesFieldProvider: Send + Sync {
    fn get_field_at(&self, time: f64, output: &mut [f64]) -> MhResult<()>;
    fn get_time_range(&self) -> (f64, f64);
    fn n_cells(&self) -> usize;
    fn preload(&mut self, t_start: f64, t_end: f64) -> MhResult<()> {
        let _ = (t_start, t_end);
        Ok(())
    }
}

pub trait VectorFieldProvider: Send + Sync {
    fn get_vector_at(&self, time: f64, out_u: &mut [f64], out_v: &mut [f64]) -> MhResult<()>;
    fn get_time_range(&self) -> (f64, f64);
    fn n_cells(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_transform_roundtrip() {
        let gt = GeoTransform {
            origin_x: 100.0,
            origin_y: 200.0,
            pixel_width: 10.0,
            pixel_height: -10.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
        };
        let (x, y) = gt.apply(5, 3);
        let (col, row) = gt.inverse(x, y).unwrap();
        assert!((col - 5.0).abs() < 1e-10);
        assert!((row - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_gdal_conversion() {
        let gdal_gt = [100.0, 10.0, 0.0, 200.0, 0.0, -10.0];
        let gt = GeoTransform::from_gdal(&gdal_gt);
        let back = gt.to_gdal();
        assert_eq!(gdal_gt, back);
    }
}
