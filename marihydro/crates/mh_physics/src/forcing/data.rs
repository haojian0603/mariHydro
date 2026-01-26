// crates/mh_physics/src/forcing/data.rs
//! 强迫数据格式抽象层
//!
//! 提供强迫数据的通用读取接口与元信息描述。

#[cfg(feature = "netcdf")]
use std::path::Path;

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
        let (i0, i1, fx) = self.find_index_and_frac(&self.lons, lon)?;
        let (j0, j1, fy) = self.find_index_and_frac(&self.lats, lat)?;

        let v00 = self.values.get(j0)?.get(i0).copied()?;
        let v01 = self.values.get(j0)?.get(i1).copied()?;
        let v10 = self.values.get(j1)?.get(i0).copied()?;
        let v11 = self.values.get(j1)?.get(i1).copied()?;

        if v00 == self.fill_value || v01 == self.fill_value || v10 == self.fill_value || v11 == self.fill_value {
            return None;
        }

        let fx = fx.clamp(0.0, 1.0);
        let fy = fy.clamp(0.0, 1.0);
        let v0 = v00 * (1.0 - fx) + v01 * fx;
        let v1 = v10 * (1.0 - fx) + v11 * fx;

        Some(v0 * (1.0 - fy) + v1 * fy)
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

/// NetCDF 数据读取器（占位，需启用 netcdf feature）
#[cfg(feature = "netcdf")]
pub struct NetCdfReader {
    path: std::path::PathBuf,
    variable: String,
    metadata: ForcingMetadata,
    times: Vec<f64>,
}

#[cfg(feature = "netcdf")]
impl NetCdfReader {
    pub fn open(path: impl AsRef<Path>, variable: &str) -> Result<Self, ForcingDataError> {
        let _ = path.as_ref();
        let _ = variable;
        todo!("需要 netcdf feature")
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
