// src-tauri/src/marihydro/io/drivers/nc_loader.rs

use super::nc_adapter::core::NcCore;
use crate::marihydro::infra::error::{MhError, MhResult};
use crate::marihydro::io::traits::{RasterDriver, RasterRequest};
use crate::marihydro::io::types::{GeoGridData, GeoTransform, RasterMetadata};

pub struct NetCdfLoader;

impl RasterDriver for NetCdfLoader {
    fn read_metadata(&self, path: &str) -> MhResult<RasterMetadata> {
        let nc = NcCore::open(path)?;

        // 尝试自动探测坐标变量 (lon/lat 或 x/y)
        // 这里简化处理：假设存在 'lon' 和 'lat' 变量用于计算分辨率
        // 在真实的工程代码中，这里需要复杂的 CF-Convention 解析逻辑
        let lon_var = nc
            .get_variable("longitude")
            .or_else(|_| nc.get_variable("lon"))
            .or_else(|_| nc.get_variable("x"))?;

        let lat_var = nc
            .get_variable("latitude")
            .or_else(|_| nc.get_variable("lat"))
            .or_else(|_| nc.get_variable("y"))?;

        let w = lon_var.len();
        let h = lat_var.len();

        // 简单的 GeoTransform 估算 (假设等间距)
        let lons = lon_var.values::<f64>(None, None).map_err(MhError::NetCdf)?;
        let lats = lat_var.values::<f64>(None, None).map_err(MhError::NetCdf)?;

        let x0 = lons[[0]];
        let y0 = lats[[0]];
        let dx = (lons[[w - 1]] - lons[[0]]) / (w as f64 - 1.0);
        let dy = (lats[[h - 1]] - lats[[0]]) / (h as f64 - 1.0);

        let transform = GeoTransform::new(x0, y0, dx, dy);

        Ok(RasterMetadata {
            width: w,
            height: h,
            transform,
            crs_wkt: "EPSG:4326".into(), // NC 文件通常是 LatLon，除非有 Grid Mapping
            no_data_value: None,         // 需读取 _FillValue 属性
            driver_name: "NetCDF".into(),
        })
    }

    fn read_data(&self, path: &str, _request: Option<RasterRequest>) -> MhResult<GeoGridData> {
        let nc = NcCore::open(path)?;

        // 1. 获取元数据 (调用上面的逻辑)
        let meta = self.read_metadata(path)?;

        // 2. 猜测要读取的数据变量
        // 这里是一个启发式搜索列表
        let candidates = ["elevation", "z", "h", "u10", "v10", "Band1"];
        let var_name = candidates
            .iter()
            .find(|&&name| nc.get_variable(name).is_ok())
            .ok_or_else(|| MhError::DataLoad {
                file: path.into(),
                message: "无法自动找到主数据变量(z, elevation, etc.)，请手动指定".into(),
            })?;

        // 3. 读取数据 (默认取第0帧)
        let array = nc.read_2d_slice(var_name, Some(0))?;

        // TODO: 如果 request 中包含重采样请求，这里需要类似 GDAL 那样处理
        // 但 netcdf crate 不支持 warp，所以这里如果需要重采样，通常建议使用 gdal_loader 读取 netcdf
        // 本 loader主要用于读取原始数据矩阵

        Ok(GeoGridData { meta, data: array })
    }

    fn supports_extension(&self, ext: &str) -> bool {
        matches!(ext.to_lowercase().as_str(), "nc" | "cdf" | "netcdf")
    }
}
