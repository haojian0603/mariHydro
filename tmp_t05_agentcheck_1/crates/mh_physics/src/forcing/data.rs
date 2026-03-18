// crates/mh_physics/src/forcing/data.rs
//! 强迫数据格式抽象层
//!
//! 提供强迫数据的通用读取接口与元信息描述。

#[cfg(feature = "netcdf")]
use std::path::Path;
#[cfg(feature = "netcdf")]
use mh_io::{NetCdfDriver, NetCdfError, Variable};
use mh_foundation::MhError;

/// 强迫数据读取错误
#[derive(Debug, thiserror::Error)]
pub enum ForcingDataError {
    #[error("文件未找到: {0}")]
    FileNotFound(String),
    #[error("不支持的数据格式: {0}")]
    UnsupportedFormat(String),
    #[error("变量未找到: {0}")]
    VariableNotFound(String),
    #[error("时间维度无效")]
    InvalidTimeDimension,
    #[error("空间维度不匹配")]
    SpatialMismatch,
    #[error("数据读取失败: {0}")]
    ReadError(String),
    #[error("插值失败: {0}")]
    InterpolationError(String),
}

impl From<ForcingDataError> for MhError {
    fn from(err: ForcingDataError) -> Self {
        match err {
            ForcingDataError::FileNotFound(path) => MhError::file_not_found(path),
            ForcingDataError::UnsupportedFormat(format) => {
                MhError::invalid_input(format!("强迫数据格式不支持: {format}"))
            }
            ForcingDataError::VariableNotFound(variable) => {
                MhError::invalid_input(format!("强迫变量不存在: {variable}"))
            }
            ForcingDataError::InvalidTimeDimension => {
                MhError::invalid_input("强迫数据时间维度无效".to_string())
            }
            ForcingDataError::SpatialMismatch => {
                MhError::invalid_input("强迫数据空间维度不匹配".to_string())
            }
            ForcingDataError::ReadError(message) => {
                MhError::io(format!("强迫数据读取失败: {message}"))
            }
            ForcingDataError::InterpolationError(message) => {
                MhError::invalid_input(format!("强迫数据插值失败: {message}"))
            }
        }
    }
}

/// 强迫数据元信息
#[derive(Debug, Clone)]
pub struct ForcingMetadata {
    /// 变量名
    pub variable_name: String,
    /// 物理单位
    pub units: String,
    /// 时间范围 [s]
    pub time_range: (f64, f64),
    /// 时间步
    pub time_step: f64,
    /// 空间范围 (lon_min, lon_max, lat_min, lat_max)
    pub spatial_extent: Option<(f64, f64, f64, f64)>,
    /// 空间分辨率
    pub spatial_resolution: Option<(f64, f64)>,
    /// 填充值
    pub fill_value: f64,
    /// 缩放和偏移
    pub scale_offset: (f64, f64),
}

impl Default for ForcingMetadata {
    fn default() -> Self {
        Self {
            variable_name: String::new(),
            units: String::new(),
            time_range: (0.0, 0.0),
            time_step: 3600.0,
            spatial_extent: None,
            spatial_resolution: None,
            fill_value: -9999.0,
            scale_offset: (1.0, 0.0),
        }
    }
}

/// 强迫数据格式 trait
pub trait ForcingDataReader: Send + Sync {
    /// 获取元信息
    fn metadata(&self) -> &ForcingMetadata;

    /// 读取单个时刻的场数据
    fn read_field_at_time(&self, time: f64) -> Result<ForcingField, ForcingDataError>;

    /// 读取时间序列（单点）
    fn read_timeseries_at_point(
        &self,
        lon: f64,
        lat: f64,
        start_time: f64,
        end_time: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), ForcingDataError>;

    /// 获取可用时间点
    fn available_times(&self) -> &[f64];
}

/// 强迫场数据
#[derive(Debug, Clone)]
pub struct ForcingField {
    /// 数据网格
    pub values: Vec<Vec<f64>>,
    /// 经度坐标
    pub lons: Vec<f64>,
    /// 纬度坐标
    pub lats: Vec<f64>,
    /// 时间戳
    pub time: f64,
    /// 填充值
    pub fill_value: f64,
}

impl ForcingField {
    /// 双线性插值
    pub fn interpolate_bilinear(&self, lon: f64, lat: f64) -> Option<f64> {
        if !lon.is_finite() || !lat.is_finite() {
            return None;
        }
        if self.lons.is_empty() || self.lats.is_empty() || self.values.is_empty() {
            return None;
        }
        let rows = self.values.len();
        let cols = self.values[0].len();
        if cols == 0 {
            return None;
        }
        if self.lats.len() != rows || self.lons.len() != cols {
            return None;
        }
        if self.values.iter().any(|row| row.len() != cols) {
            return None;
        }
        let (i0, i1, fx) = self.find_index_and_frac(&self.lons, lon)?;
        let (j0, j1, fy) = self.find_index_and_frac(&self.lats, lat)?;

        let v00 = self.values.get(j0)?.get(i0).copied()?;
        let v01 = self.values.get(j0)?.get(i1).copied()?;
        let v10 = self.values.get(j1)?.get(i0).copied()?;
        let v11 = self.values.get(j1)?.get(i1).copied()?;

        if !Self::value_is_valid(v00, self.fill_value)
            || !Self::value_is_valid(v01, self.fill_value)
            || !Self::value_is_valid(v10, self.fill_value)
            || !Self::value_is_valid(v11, self.fill_value)
        {
            return None;
        }

        let fx = fx.clamp(0.0, 1.0);
        let fy = fy.clamp(0.0, 1.0);
        let v0 = v00 * (1.0 - fx) + v01 * fx;
        let v1 = v10 * (1.0 - fx) + v11 * fx;

        Some(v0 * (1.0 - fy) + v1 * fy)
    }

    #[inline]
    fn value_is_valid(value: f64, fill_value: f64) -> bool {
        if !value.is_finite() {
            return false;
        }
        if fill_value.is_nan() {
            return !value.is_nan();
        }
        value != fill_value
    }

    fn find_index_and_frac(&self, coords: &[f64], val: f64) -> Option<(usize, usize, f64)> {
        if coords.len() < 2 {
            return None;
        }

        if !val.is_finite() {
            return None;
        }

        let first = *coords.first()?;
        let last = *coords.last()?;
        let ascending = last >= first;

        if ascending {
            if val < first || val > last {
                return None;
            }
        } else if val > first || val < last {
            return None;
        }

        for i in 0..coords.len() - 1 {
            let a = coords[i];
            let b = coords[i + 1];
            if (ascending && a <= val && val <= b) || (!ascending && a >= val && val >= b) {
                let denom = b - a;
                if denom.abs() <= 1e-14 {
                    return None;
                }
                let frac = (val - a) / denom;
                return Some((i, i + 1, frac));
            }
        }

        None
    }
}

/// NetCDF 数据读取器（需启用 netcdf feature）
#[cfg(feature = "netcdf")]
pub struct NetCdfReader {
    path: std::path::PathBuf,
    variable: String,
    metadata: ForcingMetadata,
    times: Vec<f64>,
    lons: Vec<f64>,
    lats: Vec<f64>,
    variable_dims: Vec<usize>,
    has_time_dim: bool,
    driver: NetCdfDriver,
}

#[cfg(feature = "netcdf")]
impl NetCdfReader {
    pub fn open(path: impl AsRef<Path>, variable: &str) -> Result<Self, ForcingDataError> {
        let path = path.as_ref();
        let driver = NetCdfDriver::open(path).map_err(map_netcdf_error)?;

        let vars = driver.variables().map_err(map_netcdf_error)?;
        let var_info = vars
            .iter()
            .find(|v| v.name == variable || v.standard_name.as_deref() == Some(variable))
            .or_else(|| vars.iter().find(|v| v.long_name.as_deref() == Some(variable)))
            .ok_or_else(|| ForcingDataError::VariableNotFound(variable.to_string()))?;

        let variable_name = var_info.name.clone();
        let variable_dims = var_info
            .dimensions
            .iter()
            .map(|d| driver.dimension(d).map_err(map_netcdf_error).map(|dim| dim.len))
            .collect::<Result<Vec<_>, _>>()?;

        let has_time_dim = var_info
            .dimensions
            .first()
            .map(|name| name.to_lowercase().contains("time"))
            .unwrap_or(false);

        let lons = read_coord_variable_lon(&driver, &["lon", "longitude", "x", "XLONG", "LON"])?;
        let lats = read_coord_variable_lat(&driver, &["lat", "latitude", "y", "XLAT", "LAT"])?;

        let times = read_time_variable(&driver, &["time", "Times", "t"]).unwrap_or_else(|_| vec![0.0]);

        let mut metadata = ForcingMetadata::default();
        metadata.variable_name = variable_name.clone();
        metadata.units = var_info.units.clone().unwrap_or_default();
        if !times.is_empty() {
            metadata.time_range = (*times.first().unwrap(), *times.last().unwrap());
            metadata.time_step = if times.len() > 1 { times[1] - times[0] } else { metadata.time_step };
        }
        if !lons.is_empty() && !lats.is_empty() {
            let lon_min = lons.iter().cloned().fold(f64::INFINITY, f64::min);
            let lon_max = lons.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lat_min = lats.iter().cloned().fold(f64::INFINITY, f64::min);
            let lat_max = lats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            metadata.spatial_extent = Some((lon_min, lon_max, lat_min, lat_max));
            if lons.len() > 1 && lats.len() > 1 {
                metadata.spatial_resolution = Some((
                    (lons[1] - lons[0]).abs(),
                    (lats[1] - lats[0]).abs(),
                ));
            }
        }

        Ok(Self {
            path: path.to_path_buf(),
            variable: variable_name,
            metadata,
            times,
            lons,
            lats,
            variable_dims,
            has_time_dim,
            driver,
        })
    }
}

#[cfg(feature = "netcdf")]
impl ForcingDataReader for NetCdfReader {
    fn metadata(&self) -> &ForcingMetadata {
        &self.metadata
    }

    fn read_field_at_time(&self, time: f64) -> Result<ForcingField, ForcingDataError> {
        if self.variable_dims.len() < 2 {
            return Err(ForcingDataError::SpatialMismatch);
        }

        let (time_idx, next_idx, frac) = select_time_indices(&self.times, time);

        let mut field = read_slice_as_field(self, time_idx)?;
        if self.has_time_dim && next_idx != time_idx {
            let field_next = read_slice_as_field(self, next_idx)?;
            field.values = interpolate_fields(&field.values, &field_next.values, frac)?;
            field.time = time;
        }

        Ok(field)
    }

    fn read_timeseries_at_point(
        &self,
        lon: f64,
        lat: f64,
        start_time: f64,
        end_time: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), ForcingDataError> {
        let mut times = Vec::new();
        let mut values = Vec::new();

        for &t in &self.times {
            if t < start_time || t > end_time {
                continue;
            }
            let field = self.read_field_at_time(t)?;
            if let Some(v) = field.interpolate_bilinear(lon, lat) {
                times.push(t);
                values.push(v);
            }
        }

        if times.is_empty() {
            return Err(ForcingDataError::InterpolationError("无可用时间点".to_string()));
        }

        Ok((times, values))
    }

    fn available_times(&self) -> &[f64] {
        &self.times
    }
}

#[cfg(feature = "netcdf")]
fn read_slice_as_field(reader: &NetCdfReader, time_idx: usize) -> Result<ForcingField, ForcingDataError> {
    let variable = if reader.has_time_dim {
        reader
            .driver
            .read_variable_slice(&reader.variable, time_idx)
            .map_err(map_netcdf_error)?
    } else {
        reader
            .driver
            .read_variable(&reader.variable)
            .map_err(map_netcdf_error)?
    };

    let (values, dims) = (variable.data, variable.dims);
    if dims.len() < 2 {
        return Err(ForcingDataError::SpatialMismatch);
    }

    let (rows, cols) = (dims[0], dims[1]);
    if rows * cols != values.len() {
        return Err(ForcingDataError::SpatialMismatch);
    }

    let mut grid = vec![vec![0.0; cols]; rows];
    for j in 0..rows {
        for i in 0..cols {
            grid[j][i] = values[j * cols + i];
        }
    }

    Ok(ForcingField {
        values: grid,
        lons: reader.lons.clone(),
        lats: reader.lats.clone(),
        time: reader.times.get(time_idx).copied().unwrap_or(0.0),
        fill_value: reader.metadata.fill_value,
    })
}

#[cfg(feature = "netcdf")]
fn interpolate_fields(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    frac: f64,
) -> Result<Vec<Vec<f64>>, ForcingDataError> {
    if a.len() != b.len() {
        return Err(ForcingDataError::SpatialMismatch);
    }
    let mut out = Vec::with_capacity(a.len());
    for (row_a, row_b) in a.iter().zip(b.iter()) {
        if row_a.len() != row_b.len() {
            return Err(ForcingDataError::SpatialMismatch);
        }
        let mut row = Vec::with_capacity(row_a.len());
        for (&va, &vb) in row_a.iter().zip(row_b.iter()) {
            row.push(va * (1.0 - frac) + vb * frac);
        }
        out.push(row);
    }
    Ok(out)
}

#[cfg(feature = "netcdf")]
fn select_time_indices(times: &[f64], time: f64) -> (usize, usize, f64) {
    if times.is_empty() {
        return (0, 0, 0.0);
    }
    if times.len() == 1 {
        return (0, 0, 0.0);
    }

    for i in 0..times.len() - 1 {
        let t0 = times[i];
        let t1 = times[i + 1];
        if (t0 <= time && time <= t1) || (t1 <= time && time <= t0) {
            let denom = t1 - t0;
            let frac = if denom.abs() < 1e-12 { 0.0 } else { (time - t0) / denom };
            return (i, i + 1, frac.clamp(0.0, 1.0));
        }
    }
    let last = times.len() - 1;
    (last, last, 0.0)
}

#[cfg(feature = "netcdf")]
fn read_coord_variable_lon(driver: &NetCdfDriver, candidates: &[&str]) -> Result<Vec<f64>, ForcingDataError> {
    for &name in candidates {
        if let Ok(var) = driver.read_variable(name) {
            return Ok(flatten_coord_lon(var));
        }
    }
    Err(ForcingDataError::SpatialMismatch)
}

#[cfg(feature = "netcdf")]
fn read_time_variable(driver: &NetCdfDriver, candidates: &[&str]) -> Result<Vec<f64>, ForcingDataError> {
    for &name in candidates {
        if let Ok(var) = driver.read_variable(name) {
            if !var.data.is_empty() {
                return Ok(var.data);
            }
        }
    }
    Err(ForcingDataError::InvalidTimeDimension)
}

#[cfg(feature = "netcdf")]
fn read_coord_variable_lat(driver: &NetCdfDriver, candidates: &[&str]) -> Result<Vec<f64>, ForcingDataError> {
    for &name in candidates {
        if let Ok(var) = driver.read_variable(name) {
            return Ok(flatten_coord_lat(var));
        }
    }
    Err(ForcingDataError::SpatialMismatch)
}

#[cfg(feature = "netcdf")]
fn flatten_coord_lon(var: Variable) -> Vec<f64> {
    if var.dims.len() == 1 {
        return var.data;
    }
    if var.dims.len() == 2 {
        let (rows, cols) = (var.dims[0], var.dims[1]);
        if var.data.len() == rows * cols {
            return var.data[..cols].to_vec();
        }
    }
    var.data
}

#[cfg(feature = "netcdf")]
fn flatten_coord_lat(var: Variable) -> Vec<f64> {
    if var.dims.len() == 1 {
        return var.data;
    }
    if var.dims.len() == 2 {
        let (rows, cols) = (var.dims[0], var.dims[1]);
        if var.data.len() == rows * cols {
            let mut lat = Vec::with_capacity(rows);
            for r in 0..rows {
                lat.push(var.data[r * cols]);
            }
            return lat;
        }
    }
    var.data
}

#[cfg(feature = "netcdf")]
fn map_netcdf_error(err: NetCdfError) -> ForcingDataError {
    match err {
        NetCdfError::FileNotFound(path) => ForcingDataError::FileNotFound(path),
        NetCdfError::OpenFailed(msg) => ForcingDataError::ReadError(msg),
        NetCdfError::VariableNotFound(name) => ForcingDataError::VariableNotFound(name),
        NetCdfError::DimensionNotFound(name) => ForcingDataError::ReadError(name),
        NetCdfError::ReadFailed(msg) => ForcingDataError::ReadError(msg),
        NetCdfError::AttributeNotFound(msg) => ForcingDataError::ReadError(msg),
        NetCdfError::TimeParseError(msg) => ForcingDataError::ReadError(msg),
        NetCdfError::NotAvailable => ForcingDataError::UnsupportedFormat("netcdf not available".to_string()),
        NetCdfError::Other(msg) => ForcingDataError::ReadError(msg),
    }
}

/// 工业标准变量名映射（CF Conventions）
pub struct CFConventions;

impl CFConventions {
    pub const WIND_U: &'static str = "eastward_wind";
    pub const WIND_V: &'static str = "northward_wind";
    pub const AIR_PRESSURE: &'static str = "air_pressure_at_sea_level";
    pub const PRECIPITATION: &'static str = "precipitation_flux";
    pub const SEA_SURFACE_HEIGHT: &'static str = "sea_surface_height_above_geoid";

    pub fn standard_name(common_name: &str) -> Option<&'static str> {
        match common_name.to_lowercase().as_str() {
            "u10" | "wind_u" => Some(Self::WIND_U),
            "v10" | "wind_v" => Some(Self::WIND_V),
            "msl" | "slp" | "pressure" => Some(Self::AIR_PRESSURE),
            "precip" | "rain" => Some(Self::PRECIPITATION),
            "ssh" | "eta" | "water_level" => Some(Self::SEA_SURFACE_HEIGHT),
            _ => None,
        }
    }
}
