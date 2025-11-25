// src-tauri/src/marihydro/io/drivers/nc_adapter/core.rs

use crate::marihydro::infra::error::{MhError, MhResult};
use ndarray::{Array2, ArrayD};
use netcdf::{File, Variable};

pub struct NcCore<'a> {
    file: File,
    path_str: String,
    // 保持文件句柄存活的生命周期
    _marker: std::marker::PhantomData<&'a ()>,
}

impl<'a> NcCore<'a> {
    pub fn open(path: &str) -> MhResult<Self> {
        let file = File::open(path).map_err(|e| MhError::NetCdf(e))?;
        Ok(Self {
            file,
            path_str: path.to_string(),
            _marker: std::marker::PhantomData,
        })
    }

    /// 获取所有变量名
    pub fn list_variables(&self) -> Vec<String> {
        self.file
            .variables()
            .map(|v| v.name().to_string())
            .collect()
    }

    /// 获取特定变量 (如 "u10")
    pub fn get_variable(&self, name: &str) -> MhResult<Variable> {
        self.file.variable(name).ok_or_else(|| MhError::DataLoad {
            file: self.path_str.clone(),
            message: format!("变量 '{}' 不存在", name),
        })
    }

    /// 读取 2D 切片
    /// 如果变量是 3D (Time, Lat, Lon)，需要指定 time_index
    /// 如果变量是 2D (Lat, Lon)，忽略 index
    pub fn read_2d_slice(
        &self,
        var_name: &str,
        time_index: Option<usize>,
    ) -> MhResult<Array2<f64>> {
        let var = self.get_variable(var_name)?;
        let dims = var.dimensions();

        // 简单的维度推断策略
        let data: ArrayD<f64> = if dims.len() == 2 {
            // 2D 直接读取
            var.values::<f64>(None, None).map_err(MhError::NetCdf)?
        } else if dims.len() == 3 {
            // 3D 读取切片: [time_index, :, :]
            let t = time_index.unwrap_or(0); // 默认取第0帧
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

        // 将 ArrayD 转换为 Array2
        // 假设最后两个维度是 (Y, X)
        // 注意：NetCDF 通常是 (Lat, Lon)，即 (Y, X)
        let shape = data.shape();
        let (ny, nx) = (shape[shape.len() - 2], shape[shape.len() - 1]);

        data.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| MhError::DataLoad {
                file: self.path_str.clone(),
                message: format!("无法转换为2D数组: {}", e),
            })
    }
}
