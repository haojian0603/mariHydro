// src-tauri/src/marihydro/io/drivers/nc_adapter/core.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::traits::{CoordinateTransform, GeoTransform}; // ✅ 使用 trait
use ndarray::{Array2, ArrayD};
use netcdf::{File, Variable};

pub struct NcCore {
    file: File,
    path_str: String,
}

impl NcCore {
    pub fn open(path: &str) -> MhResult<Self> {
        let file = File::open(path).map_err(MhError::NetCdf)?;
        Ok(Self {
            file,
            path_str: path.to_string(),
        })
    }

    pub fn list_variables(&self) -> Vec<String> {
        self.file
            .variables()
            .map(|v| v.name().to_string())
            .collect()
    }

    pub fn read_2d_slice(
        &self,
        var_name: &str,
        time_index: Option<usize>,
    ) -> MhResult<Array2<f64>> {
        let var = self
            .file
            .variable(var_name)
            .ok_or_else(|| MhError::DataLoad {
                file: self.path_str.clone(),
                message: format!("变量 '{}' 不存在", var_name),
            })?;

        let dims = var.dimensions();

        let data: ArrayD<f64> = if dims.len() == 2 {
            var.values::<f64>(None, None).map_err(MhError::NetCdf)?
        } else if dims.len() == 3 {
            let t = time_index.unwrap_or(0);
            let start = [t, 0, 0];
            let count = [1, dims[1].len(), dims[2].len()];
            var.values::<f64>(Some(&start), Some(&count))
                .map_err(MhError::NetCdf)?
        } else {
            return Err(MhError::DataLoad {
                file: self.path_str.clone(),
                message: format!("不支持的维度数量: {}", dims.len()),
            });
        };

        data.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| MhError::DataLoad {
                file: self.path_str.clone(),
                message: format!("无法转换为2D数组: {}", e),
            })
    }
}

/// ✅ NetCDF 坐标转换（假设 CF-Conventions）
pub struct NcCoordinateTransform {
    lon_min: f64,
    lat_min: f64,
    d_lon: f64,
    d_lat: f64,
}

impl NcCoordinateTransform {
    pub fn from_nc_file(nc: &NcCore, lon_var: &str, lat_var: &str) -> MhResult<Self> {
        // TODO: 从 NC 文件读取经纬度变量
        // 这里简化实现
        Ok(Self {
            lon_min: 0.0,
            lat_min: 0.0,
            d_lon: 0.01,
            d_lat: 0.01,
        })
    }
}

impl CoordinateTransform for NcCoordinateTransform {
    fn get_transform(&self) -> MhResult<GeoTransform> {
        Ok(GeoTransform {
            origin_x: self.lon_min,
            origin_y: self.lat_min,
            pixel_width: self.d_lon,
            pixel_height: self.d_lat,
            rotation_x: 0.0,
            rotation_y: 0.0,
        })
    }

    fn geo_to_pixel(&self, x: f64, y: f64) -> MhResult<(usize, usize)> {
        let col = ((x - self.lon_min) / self.d_lon).round() as usize;
        let row = ((y - self.lat_min) / self.d_lat).round() as usize;
        Ok((col, row))
    }
}
