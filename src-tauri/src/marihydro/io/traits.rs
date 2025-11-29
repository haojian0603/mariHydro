use crate::marihydro::domain::mesh::unstructured::UnstructuredMesh;
use crate::marihydro::infra::error::MhResult;
use ndarray::Array2;
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

    pub fn apply(&self, col: usize, row: usize) -> (f64, f64) {
        let x = self.origin_x + col as f64 * self.pixel_width + row as f64 * self.rotation_x;
        let y = self.origin_y + col as f64 * self.rotation_y + row as f64 * self.pixel_height;
        (x, y)
    }
}

pub trait CoordinateTransform {
    fn get_transform(&self) -> MhResult<GeoTransform>;

    fn pixel_to_geo(&self, col: usize, row: usize) -> MhResult<(f64, f64)> {
        let transform = self.get_transform()?;
        Ok(transform.apply(col, row))
    }

    fn geo_to_pixel(&self, x: f64, y: f64) -> MhResult<(usize, usize)>;
}

#[derive(Debug, Clone)]
pub struct RasterMetadata {
    pub width: usize,
    pub height: usize,
    pub n_bands: usize,
    pub crs_wkt: Option<String>,
    pub geo_transform: Option<GeoTransform>,
}

pub trait RasterLoader {
    fn read_metadata(&self, path: &Path) -> MhResult<RasterMetadata>;

    fn load_array(
        &self,
        path: &Path,
        target_shape: (usize, usize),
        band: Option<usize>,
    ) -> MhResult<Array2<f64>>;
}

pub trait MeshLoader {
    fn load(&self, path: &Path) -> MhResult<UnstructuredMesh>;

    fn supports_extension(&self, ext: &str) -> bool;
}

pub trait TimeSeriesFieldProvider: Send + Sync {
    fn get_field_at(&self, time: f64, output: &mut Array2<f64>) -> MhResult<()>;

    fn get_time_range(&self) -> (f64, f64);

    fn preload(&mut self, t_start: f64, t_end: f64) -> MhResult<()>;
}
