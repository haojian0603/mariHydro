// crates/mh_io/src/netcdf_tide.rs

//! NetCDF 潮汐数据 I/O
//!
//! 当前主链只接入两类真实布局：
//! - 单个 TPXO/ATLAS 规则网格水位分潮文件（`hRe`/`hIm`）
//! - FES2014 按分潮拆分的规则网格文件目录（`amplitude`/`phase` 或 `Ha`/`Hg`）
//!
//! IO_SOURCE: TPXO/ATLAS 单分潮 NetCDF 布局约定（`hRe`/`hIm` 复数分量），以及 FES2014 分潮拆分文件约定（`amplitude`/`phase`、`Ha`/`Hg`、`Ua`/`Ug`、`Va`/`Vg`）。
//! IO_SCOPE: 仅支持规则经纬网格的单分潮 TPXO/ATLAS 文件与按分潮拆分的 FES2014 目录；布局、变量或坐标不符时显式报错，不回退为默认网格、模拟潮汐常数或零值分潮。
//!
//! # 支持的格式
//!
//! - TPXO9/ATLAS 单分潮 NetCDF 文件
//! - FES2014 按分潮拆分的 NetCDF 目录
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_io::netcdf_tide::{TidalDataReader, TpxoReader};
//!
//! let reader = TpxoReader::open("path/to/tpxo9.nc")?;
//! let constituents = reader.available_constituents();
//! let (amp, phase) = reader.read_constituent("M2", lon, lat)?;
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::drivers::{NetCdfDriver, NetCdfError, Variable};

// ============================================================================
// 潮汐数据元数据
// ============================================================================

/// 潮汐分潮常数
#[derive(Debug, Clone)]
pub struct TidalConstituent {
    /// 分潮名称（如 M2, S2, K1）
    pub name: String,
    /// 达尔文符号编号
    pub darwin_number: Option<usize>,
    /// 角频率 (rad/s)
    pub frequency: f64,
    /// Doodson 数
    pub doodson: [i8; 6],
}

/// 全球潮汐模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TidalModel {
    /// TPXO9 模型
    Tpxo9,
    /// TPXO 地区模型
    TpxoLocal,
    /// FES2014 模型
    Fes2014,
    /// 未知模型
    Unknown,
}

impl TidalModel {
    /// 从路径提示推断模型类型
    ///
    /// 该逻辑只用于路径不存在时的内部错误分流，以及 TPXO 文件的提示性元数据。
    /// 真实打开路径时，主入口必须优先依据实际文件类型、目录结构和布局校验结果分发读取器。
    fn infer_from_path_hint(path: &Path) -> Self {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();

        if name.contains("tpxo9") {
            Self::Tpxo9
        } else if name.contains("tpxo") {
            Self::TpxoLocal
        } else if name.contains("fes2014") || name.contains("fes") {
            Self::Fes2014
        } else {
            Self::Unknown
        }
    }
}

/// 潮汐数据网格信息
#[derive(Debug, Clone)]
pub struct TidalGrid {
    /// 经度范围 (min, max)
    pub lon_range: (f64, f64),
    /// 纬度范围 (min, max)
    pub lat_range: (f64, f64),
    /// 经度分辨率 (度)
    pub lon_resolution: f64,
    /// 纬度分辨率 (度)
    pub lat_resolution: f64,
    /// 经度点数
    pub n_lon: usize,
    /// 纬度点数
    pub n_lat: usize,
}

impl TidalGrid {
    /// 获取插值索引和权重
    pub fn interpolation_indices(&self, lon: f64, lat: f64) -> Option<InterpolationIndices> {
        if !lon.is_finite() || !lat.is_finite() {
            return None;
        }
        if !self.lon_range.0.is_finite()
            || !self.lon_range.1.is_finite()
            || !self.lat_range.0.is_finite()
            || !self.lat_range.1.is_finite()
        {
            return None;
        }
        if self.lon_range.1 <= self.lon_range.0 {
            return None;
        }
        if !self.lon_resolution.is_finite() || !self.lat_resolution.is_finite() {
            return None;
        }
        if self.lon_resolution <= 0.0 || self.lat_resolution <= 0.0 {
            return None;
        }

        let lon_normalized = wrap_longitude_to_range(lon, self.lon_range)?;

        // 检查范围
        if self.lat_range.0 > self.lat_range.1 {
            return None;
        }
        if lat < self.lat_range.0 || lat > self.lat_range.1 {
            return None;
        }

        // 计算索引
        let i_lon = ((lon_normalized - self.lon_range.0) / self.lon_resolution).floor() as isize;
        let j_lat = ((lat - self.lat_range.0) / self.lat_resolution).floor() as isize;
        if i_lon < 0 || j_lat < 0 {
            return None;
        }
        let i_lon = i_lon as usize;
        let j_lat = j_lat as usize;

        if i_lon >= self.n_lon - 1 || j_lat >= self.n_lat - 1 {
            return None;
        }

        // 计算权重（双线性插值）
        let x = (lon_normalized - self.lon_range.0 - (i_lon as f64) * self.lon_resolution)
            / self.lon_resolution;
        let y =
            (lat - self.lat_range.0 - (j_lat as f64) * self.lat_resolution) / self.lat_resolution;

        let x = x.clamp(0.0, 1.0);
        let y = y.clamp(0.0, 1.0);

        Some(InterpolationIndices {
            i: [i_lon, i_lon + 1, i_lon, i_lon + 1],
            j: [j_lat, j_lat, j_lat + 1, j_lat + 1],
            weights: [(1.0 - x) * (1.0 - y), x * (1.0 - y), (1.0 - x) * y, x * y],
        })
    }
}

/// 双线性插值索引和权重
#[derive(Debug, Clone)]
pub struct InterpolationIndices {
    /// 经度索引（4个角点）
    pub i: [usize; 4],
    /// 纬度索引（4个角点）
    pub j: [usize; 4],
    /// 权重（4个角点）
    pub weights: [f64; 4],
}

#[derive(Debug, Clone)]
struct GridLayout {
    lon_name: String,
    lat_name: String,
}

const TPXO_COORD_CANDIDATES: &[(&str, &str)] = &[
    ("lon_z", "lat_z"),
    ("lon", "lat"),
    ("longitude", "latitude"),
    ("x", "y"),
];

const FES_COORD_CANDIDATES: &[(&str, &str)] =
    &[("lon", "lat"), ("longitude", "latitude"), ("x", "y")];

const TPXO_HEIGHT_COMPONENTS: (&str, &str) = ("hRe", "hIm");
const FES_HEIGHT_COMPONENTS: &[(&str, &str)] = &[("amplitude", "phase"), ("Ha", "Hg")];
const FES_U_COMPONENTS: &[(&str, &str)] = &[("Ua", "Ug"), ("u_amplitude", "u_phase")];
const FES_V_COMPONENTS: &[(&str, &str)] = &[("Va", "Vg"), ("v_amplitude", "v_phase")];

const TPXO_CONSTITUENTS: &[&str] = &[
    "m2", "s2", "n2", "k2", "k1", "o1", "p1", "q1", "mf", "mm", "m4", "ms4", "mn4", "2n2", "s1",
];

const FES_CONSTITUENTS: &[&str] = &[
    "2n2", "eps2", "j1", "k1", "k2", "l2", "la2", "m2", "m3", "m4", "m6", "m8", "mf", "mks2", "mm",
    "mn4", "ms4", "msf", "msqm", "mtm", "mu2", "n2", "n4", "nu2", "o1", "p1", "q1", "r2", "s1",
    "s2", "s4", "sa", "ssa", "t2",
];

fn wrap_longitude_to_range(lon: f64, range: (f64, f64)) -> Option<f64> {
    if !lon.is_finite() || !range.0.is_finite() || !range.1.is_finite() || range.1 <= range.0 {
        return None;
    }

    [lon, lon + 360.0, lon - 360.0]
        .into_iter()
        .find(|candidate| *candidate >= range.0 && *candidate <= range.1)
}

fn infer_regular_spacing(values: &[f64], axis_name: &str) -> Result<f64, TidalIoError> {
    if values.len() < 2 {
        return Err(TidalIoError::FormatError(format!(
            "坐标轴 {axis_name} 至少需要两个点"
        )));
    }

    let first = values[0];
    let second = values[1];
    if !first.is_finite() || !second.is_finite() || second <= first {
        return Err(TidalIoError::FormatError(format!(
            "坐标轴 {axis_name} 必须是严格递增的规则网格"
        )));
    }

    let step = second - first;
    let tolerance = step.abs().max(1.0) * 1.0e-6;
    for window in values.windows(2) {
        let current = window[1] - window[0];
        if !current.is_finite() || current <= 0.0 || (current - step).abs() > tolerance {
            return Err(TidalIoError::FormatError(format!(
                "坐标轴 {axis_name} 不是规则网格，当前主链不支持该布局"
            )));
        }
    }

    Ok(step)
}

fn detect_regular_grid(
    driver: &NetCdfDriver,
    candidates: &[(&str, &str)],
) -> Result<(GridLayout, TidalGrid), TidalIoError> {
    for (lon_name, lat_name) in candidates {
        let lon = match driver.read_variable(lon_name) {
            Ok(variable) if variable.dims.len() == 1 => variable.data,
            _ => continue,
        };
        let lat = match driver.read_variable(lat_name) {
            Ok(variable) if variable.dims.len() == 1 => variable.data,
            _ => continue,
        };

        let lon_resolution = infer_regular_spacing(&lon, lon_name)?;
        let lat_resolution = infer_regular_spacing(&lat, lat_name)?;
        let lon_range = axis_bounds(&lon, lon_name)?;
        let lat_range = axis_bounds(&lat, lat_name)?;

        return Ok((
            GridLayout {
                lon_name: (*lon_name).to_string(),
                lat_name: (*lat_name).to_string(),
            },
            TidalGrid {
                lon_range,
                lat_range,
                lon_resolution,
                lat_resolution,
                n_lon: lon.len(),
                n_lat: lat.len(),
            },
        ));
    }

    Err(TidalIoError::FormatError(
        "未找到受支持的规则经纬度坐标变量".to_string(),
    ))
}

fn axis_bounds(values: &[f64], axis_name: &str) -> Result<(f64, f64), TidalIoError> {
    let first = values.first().copied().ok_or_else(|| {
        TidalIoError::FormatError(format!(
            "坐标轴 {axis_name} 为空，无法构建规则网格范围"
        ))
    })?;
    let last = values.last().copied().ok_or_else(|| {
        TidalIoError::FormatError(format!(
            "坐标轴 {axis_name} 缺少末端值，无法构建规则网格范围"
        ))
    })?;

    Ok((first, last))
}

fn infer_constituent_from_path(path: &Path, candidates: &[&str]) -> Option<String> {
    let stem = path.file_stem()?.to_string_lossy().to_lowercase();
    let padded = format!("_{stem}_");
    candidates
        .iter()
        .find(|candidate| padded.contains(&format!("_{}_", candidate)))
        .map(|candidate| (*candidate).to_string())
}

fn complex_components_to_amplitude_phase(real: f64, imag: f64) -> (f64, f64) {
    let amplitude = real.hypot(imag);
    let phase = (-imag.atan2(real).to_degrees()).rem_euclid(360.0);
    (amplitude, phase)
}

fn sample_variable(
    variable: &Variable,
    variable_name: &str,
    dimensions: &[String],
    layout: &GridLayout,
    indices: &InterpolationIndices,
) -> Result<f64, TidalIoError> {
    let sample_at = |i: usize, j: usize| -> Result<f64, TidalIoError> {
        let value = match dimensions {
            dims if dims.len() == 2 && dims[0] == layout.lat_name && dims[1] == layout.lon_name => {
                variable.get(&[j, i])
            }
            dims if dims.len() == 2 && dims[0] == layout.lon_name && dims[1] == layout.lat_name => {
                variable.get(&[i, j])
            }
            dims if dims.len() == 3
                && matches!(variable.dims.first(), Some(&1))
                && dims[1] == layout.lat_name
                && dims[2] == layout.lon_name =>
            {
                variable.get(&[0, j, i])
            }
            dims if dims.len() == 3
                && matches!(variable.dims.first(), Some(&1))
                && dims[1] == layout.lon_name
                && dims[2] == layout.lat_name =>
            {
                variable.get(&[0, i, j])
            }
            _ => {
                return Err(TidalIoError::FormatError(format!(
                    "变量 {variable_name} 的维度 {:?} 与规则经纬度网格不匹配",
                    dimensions
                )));
            }
        };

        value.ok_or_else(|| {
            TidalIoError::FormatError(format!("变量 {variable_name} 的插值索引超出数据范围"))
        })
    };

    let mut result = 0.0;
    for k in 0..4 {
        result += indices.weights[k] * sample_at(indices.i[k], indices.j[k])?;
    }
    Ok(result)
}

fn sample_named_variable(
    driver: &NetCdfDriver,
    layout: &GridLayout,
    indices: &InterpolationIndices,
    variable_name: &str,
) -> Result<f64, TidalIoError> {
    let variable = driver.read_variable(variable_name)?;
    let info = driver.variable_info(variable_name)?;
    sample_variable(&variable, variable_name, &info.dimensions, layout, indices)
}

fn sample_complex_components(
    driver: &NetCdfDriver,
    layout: &GridLayout,
    indices: &InterpolationIndices,
    real_name: &str,
    imag_name: &str,
) -> Result<(f64, f64), TidalIoError> {
    let real = sample_named_variable(driver, layout, indices, real_name)?;
    let imag = sample_named_variable(driver, layout, indices, imag_name)?;
    Ok(complex_components_to_amplitude_phase(real, imag))
}

fn sample_component_pair(
    driver: &NetCdfDriver,
    layout: &GridLayout,
    indices: &InterpolationIndices,
    pairs: &[(&str, &str)],
) -> Result<(f64, f64), TidalIoError> {
    for (first_name, second_name) in pairs {
        if driver.has_variable(first_name) && driver.has_variable(second_name) {
            let first = sample_named_variable(driver, layout, indices, first_name)?;
            let second = sample_named_variable(driver, layout, indices, second_name)?;
            return Ok((first, second));
        }
    }

    Err(TidalIoError::Unsupported(format!(
        "当前 NetCDF 文件不包含受支持的变量对 {:?}",
        pairs
    )))
}

// ============================================================================
// 潮汐数据读取器 trait
// ============================================================================

/// 潮汐数据读取器接口
pub trait TidalDataReader {
    /// 获取模型类型
    fn model_type(&self) -> TidalModel;

    /// 获取可用分潮列表
    fn available_constituents(&self) -> &[String];

    /// 获取网格信息
    fn grid(&self) -> &TidalGrid;

    /// 读取单个分潮的振幅和相位
    ///
    /// # 参数
    ///
    /// * `name` - 分潮名称
    /// * `lon` - 经度 (度)
    /// * `lat` - 纬度 (度)
    ///
    /// # 返回
    ///
    /// (振幅 m, 相位 度) 或错误
    fn read_constituent(&self, name: &str, lon: f64, lat: f64) -> Result<(f64, f64), TidalIoError>;

    /// 批量读取多个点的分潮数据
    fn read_constituent_batch(
        &self,
        name: &str,
        lons: &[f64],
        lats: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), TidalIoError>;

    /// 读取速度分潮（如 u, v 分量）
    fn read_velocity_constituent(
        &self,
        name: &str,
        lon: f64,
        lat: f64,
    ) -> Result<((f64, f64), (f64, f64)), TidalIoError>;
}

// ============================================================================
// TPXO 读取器实现
// ============================================================================

/// TPXO 模型读取器
///
/// 当前只接入单个 TPXO/ATLAS 规则网格水位分潮文件。
pub struct TpxoReader {
    /// 文件路径
    path: PathBuf,
    /// 模型类型
    model_type: TidalModel,
    /// 网格信息
    grid: TidalGrid,
    /// 可用分潮
    constituents: Vec<String>,
}

impl TpxoReader {
    /// 打开 TPXO 文件
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TidalIoError> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(TidalIoError::FileNotFound(path.to_path_buf()));
        }

        // 检测模型类型
        let model_type = TidalModel::infer_from_path_hint(path);
        let driver = NetCdfDriver::open(path)?;
        if !(driver.has_variable(TPXO_HEIGHT_COMPONENTS.0)
            && driver.has_variable(TPXO_HEIGHT_COMPONENTS.1))
        {
            return Err(TidalIoError::FormatError(
                "TPXO 入口当前只支持包含 hRe/hIm 的规则网格水位分潮文件".to_string(),
            ));
        }

        let (_, grid) = detect_regular_grid(&driver, TPXO_COORD_CANDIDATES)?;
        let constituent =
            infer_constituent_from_path(path, TPXO_CONSTITUENTS).ok_or_else(|| {
                TidalIoError::FormatError(
                    "无法从 TPXO 文件名识别单个分潮，请使用带分潮名的 TPXO/ATLAS NetCDF 文件"
                        .to_string(),
                )
            })?;

        Ok(Self {
            path: path.to_path_buf(),
            model_type,
            grid,
            constituents: vec![constituent],
        })
    }

    fn read_constituent_with_driver(
        &self,
        driver: &NetCdfDriver,
        name: &str,
        lon: f64,
        lat: f64,
    ) -> Result<(f64, f64), TidalIoError> {
        let name_lower = name.to_lowercase();
        if !self.constituents.iter().any(|c| c == &name_lower) {
            return Err(TidalIoError::ConstituentNotFound(name.to_string()));
        }

        let (layout, _) = detect_regular_grid(driver, TPXO_COORD_CANDIDATES)?;
        let indices = self
            .grid
            .interpolation_indices(lon, lat)
            .ok_or(TidalIoError::OutOfBounds { lon, lat })?;

        sample_complex_components(
            driver,
            &layout,
            &indices,
            TPXO_HEIGHT_COMPONENTS.0,
            TPXO_HEIGHT_COMPONENTS.1,
        )
    }
}

impl TidalDataReader for TpxoReader {
    fn model_type(&self) -> TidalModel {
        self.model_type
    }

    fn available_constituents(&self) -> &[String] {
        &self.constituents
    }

    fn grid(&self) -> &TidalGrid {
        &self.grid
    }

    fn read_constituent(&self, name: &str, lon: f64, lat: f64) -> Result<(f64, f64), TidalIoError> {
        let driver = NetCdfDriver::open(&self.path)?;
        self.read_constituent_with_driver(&driver, name, lon, lat)
    }

    fn read_constituent_batch(
        &self,
        name: &str,
        lons: &[f64],
        lats: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), TidalIoError> {
        let n = lons.len().min(lats.len());
        let mut amplitudes = Vec::with_capacity(n);
        let mut phases = Vec::with_capacity(n);
        let driver = NetCdfDriver::open(&self.path)?;

        for i in 0..n {
            let (amp, phase) =
                self.read_constituent_with_driver(&driver, name, lons[i], lats[i])?;
            amplitudes.push(amp);
            phases.push(phase);
        }

        Ok((amplitudes, phases))
    }

    fn read_velocity_constituent(
        &self,
        name: &str,
        lon: f64,
        lat: f64,
    ) -> Result<((f64, f64), (f64, f64)), TidalIoError> {
        let _ = (name, lon, lat);
        Err(TidalIoError::Unsupported(
            "当前 TPXO 入口只接入水位分潮文件；速度分量需要独立已接线数据源".to_string(),
        ))
    }
}

// ============================================================================
// FES2014 读取器
// ============================================================================

/// FES2014 模型读取器
pub struct Fes2014Reader {
    /// 网格信息
    grid: TidalGrid,
    /// 可用分潮
    constituents: Vec<String>,
    /// 分潮文件映射
    files: HashMap<String, PathBuf>,
}

impl Fes2014Reader {
    /// 打开 FES2014 数据目录
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, TidalIoError> {
        let dir = dir.as_ref();

        if !dir.exists() || !dir.is_dir() {
            return Err(TidalIoError::FileNotFound(dir.to_path_buf()));
        }

        let mut files = HashMap::new();
        let mut grid = None;

        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let is_netcdf = path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("nc"))
                .unwrap_or(false);
            if !is_netcdf {
                continue;
            }

            let Some(constituent) = infer_constituent_from_path(&path, FES_CONSTITUENTS) else {
                continue;
            };

            let driver = match NetCdfDriver::open(&path) {
                Ok(driver) => driver,
                Err(_) => continue,
            };
            if !FES_HEIGHT_COMPONENTS.iter().any(|(amp_name, phase_name)| {
                driver.has_variable(amp_name) && driver.has_variable(phase_name)
            }) {
                continue;
            }

            if grid.is_none() {
                grid = Some(detect_regular_grid(&driver, FES_COORD_CANDIDATES)?.1);
            }
            files.entry(constituent).or_insert(path);
        }

        let mut constituents = files.keys().cloned().collect::<Vec<_>>();
        constituents.sort();
        let grid = grid.ok_or_else(|| {
            TidalIoError::FormatError("目录中未找到受支持的 FES2014 分潮高度文件".to_string())
        })?;

        Ok(Self {
            grid,
            constituents,
            files,
        })
    }

    fn constituent_file(&self, name: &str) -> Result<&PathBuf, TidalIoError> {
        self.files
            .get(&name.to_lowercase())
            .ok_or_else(|| TidalIoError::ConstituentNotFound(name.to_string()))
    }

    fn read_constituent_with_driver(
        &self,
        driver: &NetCdfDriver,
        lon: f64,
        lat: f64,
    ) -> Result<(f64, f64), TidalIoError> {
        let (layout, _) = detect_regular_grid(driver, FES_COORD_CANDIDATES)?;
        let indices = self
            .grid
            .interpolation_indices(lon, lat)
            .ok_or(TidalIoError::OutOfBounds { lon, lat })?;
        sample_component_pair(driver, &layout, &indices, FES_HEIGHT_COMPONENTS)
    }
}

impl TidalDataReader for Fes2014Reader {
    fn model_type(&self) -> TidalModel {
        TidalModel::Fes2014
    }

    fn available_constituents(&self) -> &[String] {
        &self.constituents
    }

    fn grid(&self) -> &TidalGrid {
        &self.grid
    }

    fn read_constituent(&self, name: &str, lon: f64, lat: f64) -> Result<(f64, f64), TidalIoError> {
        let file_path = self.constituent_file(name)?;
        let driver = NetCdfDriver::open(file_path)?;
        self.read_constituent_with_driver(&driver, lon, lat)
    }

    fn read_constituent_batch(
        &self,
        name: &str,
        lons: &[f64],
        lats: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), TidalIoError> {
        let n = lons.len().min(lats.len());
        let mut amps = Vec::with_capacity(n);
        let mut phases = Vec::with_capacity(n);
        let file_path = self.constituent_file(name)?;
        let driver = NetCdfDriver::open(file_path)?;

        for i in 0..n {
            let (a, p) = self.read_constituent_with_driver(&driver, lons[i], lats[i])?;
            amps.push(a);
            phases.push(p);
        }

        Ok((amps, phases))
    }

    fn read_velocity_constituent(
        &self,
        name: &str,
        lon: f64,
        lat: f64,
    ) -> Result<((f64, f64), (f64, f64)), TidalIoError> {
        let file_path = self.constituent_file(name)?;
        let driver = NetCdfDriver::open(file_path)?;
        let (layout, _) = detect_regular_grid(&driver, FES_COORD_CANDIDATES)?;
        let indices = self
            .grid
            .interpolation_indices(lon, lat)
            .ok_or(TidalIoError::OutOfBounds { lon, lat })?;
        let u = sample_component_pair(&driver, &layout, &indices, FES_U_COMPONENTS)?;
        let v = sample_component_pair(&driver, &layout, &indices, FES_V_COMPONENTS)?;
        Ok((u, v))
    }
}

// ============================================================================
// 潮汐数据工厂
// ============================================================================

/// 自动检测并打开潮汐数据
pub fn open_tidal_data(path: impl AsRef<Path>) -> Result<Box<dyn TidalDataReader>, TidalIoError> {
    let path = path.as_ref();

    if path.exists() {
        if path.is_dir() {
            return Ok(Box::new(Fes2014Reader::open(path)?));
        }
        if path.is_file() {
            return Ok(Box::new(TpxoReader::open(path)?));
        }
        return Err(TidalIoError::Unsupported(format!(
            "不支持的潮汐数据路径类型: {}",
            path.display()
        )));
    }

    match TidalModel::infer_from_path_hint(path) {
        TidalModel::Fes2014 => Ok(Box::new(Fes2014Reader::open(path)?)),
        TidalModel::Tpxo9 | TidalModel::TpxoLocal | TidalModel::Unknown => {
            Ok(Box::new(TpxoReader::open(path)?))
        }
    }
}

// ============================================================================
// 边界提取
// ============================================================================

/// 沿边界提取潮汐常数
pub struct TidalBoundaryExtractor<'a> {
    reader: &'a dyn TidalDataReader,
}

impl<'a> TidalBoundaryExtractor<'a> {
    /// 创建边界提取器
    pub fn new(reader: &'a dyn TidalDataReader) -> Self {
        Self { reader }
    }

    /// 沿边界点提取所有分潮常数
    pub fn extract_boundary(
        &self,
        lons: &[f64],
        lats: &[f64],
        constituents: &[&str],
    ) -> Result<BoundaryTidalConstants, TidalIoError> {
        let n_points = lons.len().min(lats.len());
        let mut constants = HashMap::new();

        for &name in constituents {
            let (amps, phases) = self.reader.read_constituent_batch(name, lons, lats)?;
            constants.insert(name.to_string(), amps.into_iter().zip(phases).collect());
        }

        Ok(BoundaryTidalConstants {
            n_points,
            constituents: constituents.iter().map(|s| s.to_string()).collect(),
            constants,
        })
    }
}

/// 边界潮汐常数
#[derive(Debug, Clone)]
pub struct BoundaryTidalConstants {
    /// 边界点数
    pub n_points: usize,
    /// 分潮列表
    pub constituents: Vec<String>,
    /// 常数数据：分潮名 -> [(振幅, 相位), ...]
    pub constants: HashMap<String, Vec<(f64, f64)>>,
}

impl BoundaryTidalConstants {
    /// 预测特定时刻的边界水位
    pub fn predict(&self, time_hours: f64, frequencies: &HashMap<String, f64>) -> Vec<f64> {
        let mut levels = vec![0.0; self.n_points];

        for (name, data) in &self.constants {
            if let Some(&freq) = frequencies.get(name) {
                if !freq.is_finite() || !time_hours.is_finite() {
                    continue;
                }
                for (i, &(amp, phase)) in data.iter().enumerate() {
                    if i >= self.n_points {
                        break;
                    }
                    let phase_rad = phase.to_radians();
                    levels[i] += amp * (freq * time_hours - phase_rad).cos();
                }
            }
        }

        levels
    }
}

// ============================================================================
// 错误类型
// ============================================================================

/// 潮汐数据 IO 错误
#[derive(Debug)]
pub enum TidalIoError {
    /// 文件未找到
    FileNotFound(PathBuf),
    /// NetCDF 错误
    NetcdfError(String),
    /// 分潮未找到
    ConstituentNotFound(String),
    /// 坐标超出范围
    OutOfBounds { lon: f64, lat: f64 },
    /// 数据格式错误
    FormatError(String),
    /// 当前入口明确不支持
    Unsupported(String),
    /// IO 错误
    IoError(std::io::Error),
}

impl std::fmt::Display for TidalIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "文件未找到: {}", p.display()),
            Self::NetcdfError(e) => write!(f, "NetCDF 错误: {}", e),
            Self::ConstituentNotFound(c) => write!(f, "分潮未找到: {}", c),
            Self::OutOfBounds { lon, lat } => write!(f, "坐标超出范围: ({}, {})", lon, lat),
            Self::FormatError(e) => write!(f, "数据格式错误: {}", e),
            Self::Unsupported(e) => write!(f, "当前入口不支持: {}", e),
            Self::IoError(e) => write!(f, "IO 错误: {}", e),
        }
    }
}

impl std::error::Error for TidalIoError {}

impl From<std::io::Error> for TidalIoError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError(err)
    }
}

impl From<NetCdfError> for TidalIoError {
    fn from(err: NetCdfError) -> Self {
        match err {
            NetCdfError::FileNotFound(path) => Self::FileNotFound(PathBuf::from(path)),
            NetCdfError::UnsupportedLayout(message) => Self::FormatError(message),
            other => Self::NetcdfError(other.to_string()),
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_detection() {
        assert_eq!(
            TidalModel::infer_from_path_hint(Path::new("tpxo9.nc")),
            TidalModel::Tpxo9
        );
        assert_eq!(
            TidalModel::infer_from_path_hint(Path::new("fes2014.nc")),
            TidalModel::Fes2014
        );
        assert_eq!(
            TidalModel::infer_from_path_hint(Path::new("unknown.nc")),
            TidalModel::Unknown
        );
    }

    #[test]
    fn test_grid_interpolation() {
        let grid = TidalGrid {
            lon_range: (0.0, 360.0),
            lat_range: (-90.0, 90.0),
            lon_resolution: 1.0,
            lat_resolution: 1.0,
            n_lon: 360,
            n_lat: 180,
        };

        let indices = grid.interpolation_indices(122.5, 30.5);
        assert!(indices.is_some());

        let idx = indices.unwrap();
        assert_eq!(idx.i[0], 122);
        assert_eq!(idx.j[0], 120); // 30.5 + 90 = 120.5, floor = 120
    }

    #[test]
    fn test_grid_interpolation_wraps_negative_longitude() {
        let grid = TidalGrid {
            lon_range: (0.0, 360.0),
            lat_range: (-90.0, 90.0),
            lon_resolution: 1.0,
            lat_resolution: 1.0,
            n_lon: 361,
            n_lat: 181,
        };

        let wrapped = grid.interpolation_indices(-75.25, 10.5).unwrap();
        assert_eq!(wrapped.i[0], 284);
    }

    #[test]
    fn test_infer_constituent_from_filename() {
        let constituent =
            infer_constituent_from_path(Path::new("h_m2_tpxo9_atlas_30_v5.nc"), TPXO_CONSTITUENTS)
                .unwrap();
        assert_eq!(constituent, "m2");
    }

    #[test]
    fn test_complex_components_to_amplitude_phase() {
        let (amp, phase) = complex_components_to_amplitude_phase(0.0, -2.0);
        assert!((amp - 2.0).abs() < 1.0e-12);
        assert!((phase - 90.0).abs() < 1.0e-12);
    }

    #[test]
    fn test_axis_bounds_rejects_empty_axis() {
        let err = axis_bounds(&[], "lon").unwrap_err();
        assert!(matches!(err, TidalIoError::FormatError(_)));
        assert!(err.to_string().contains("坐标轴 lon 为空"));
    }

    #[test]
    fn test_axis_bounds_uses_real_endpoints() {
        let bounds = axis_bounds(&[120.0, 121.5, 123.0], "lon").unwrap();
        assert_eq!(bounds, (120.0, 123.0));
    }

    #[test]
    fn test_open_tidal_data_existing_directory_uses_directory_reader() {
        let base = std::env::temp_dir().join(format!(
            "mh_io_tide_dir_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();

        let err = match open_tidal_data(&base) {
            Ok(_) => panic!("existing directory should dispatch to FES reader and fail explicitly"),
            Err(err) => err,
        };
        assert!(matches!(err, TidalIoError::FormatError(_)));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_open_tidal_data_existing_file_uses_file_reader() {
        let path = std::env::temp_dir().join(format!(
            "mh_io_tide_file_{}.nc",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, b"not a netcdf file").unwrap();

        let err = match open_tidal_data(&path) {
            Ok(_) => panic!("existing file should dispatch to TPXO reader and fail explicitly"),
            Err(err) => err,
        };
        assert!(!matches!(err, TidalIoError::Unsupported(_)));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_open_tidal_data_missing_path_reports_file_not_found() {
        let path = Path::new("unknown_layout.bin");
        let err = match open_tidal_data(path) {
            Ok(_) => panic!("missing path should not open successfully"),
            Err(err) => err,
        };
        assert!(matches!(err, TidalIoError::FileNotFound(_)));
    }

    #[test]
    fn test_tpxo_velocity_reader_fails_explicitly() {
        let reader = TpxoReader {
            path: PathBuf::from("unused.nc"),
            model_type: TidalModel::Tpxo9,
            grid: TidalGrid {
                lon_range: (0.0, 360.0),
                lat_range: (-90.0, 90.0),
                lon_resolution: 1.0,
                lat_resolution: 1.0,
                n_lon: 361,
                n_lat: 181,
            },
            constituents: vec!["m2".to_string()],
        };

        let err = reader
            .read_velocity_constituent("m2", 120.0, 30.0)
            .unwrap_err();
        assert!(matches!(err, TidalIoError::Unsupported(_)));
    }

    #[test]
    fn test_boundary_constants() {
        let constants = BoundaryTidalConstants {
            n_points: 3,
            constituents: vec!["m2".to_string(), "s2".to_string()],
            constants: [
                ("m2".to_string(), vec![(1.0, 0.0), (1.2, 30.0), (0.8, 60.0)]),
                ("s2".to_string(), vec![(0.5, 0.0), (0.6, 45.0), (0.4, 90.0)]),
            ]
            .into_iter()
            .collect(),
        };

        let mut freqs = HashMap::new();
        freqs.insert("m2".to_string(), 0.5058681); // deg/hr
        freqs.insert("s2".to_string(), 0.5000000);

        let levels = constants.predict(12.0, &freqs);
        assert_eq!(levels.len(), 3);
    }

    #[test]
    fn test_sample_variable_rejects_missing_singleton_axis() {
        let variable = Variable {
            data: vec![1.0, 2.0, 3.0, 4.0],
            dims: vec![2, 2],
        };
        let layout = GridLayout {
            lon_name: "lon".to_string(),
            lat_name: "lat".to_string(),
        };
        let indices = InterpolationIndices {
            i: [0, 1, 0, 1],
            j: [0, 0, 1, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
        };

        let err = sample_variable(
            &variable,
            "hRe",
            &["time".to_string(), "lat".to_string(), "lon".to_string()],
            &layout,
            &indices,
        )
        .unwrap_err();

        assert!(matches!(err, TidalIoError::FormatError(_)));
    }
}
