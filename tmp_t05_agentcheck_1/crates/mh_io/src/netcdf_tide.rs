// crates/mh_io/src/netcdf_tide.rs

//! NetCDF 潮汐数据 I/O
//!
//! 支持 TPXO9、FES2014 等全球潮汐模型数据的读取和插值。
//!
//! # 支持的格式
//!
//! - TPXO9: OSU 全球潮汐模型
//! - FES2014: LEGOS/CNES 全球潮汐模型
//! - GOT4.10: GSFC 全球潮汐模型
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
use std::sync::RwLock;

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
    /// GOT4.10 模型
    Got410,
    /// 未知模型
    Unknown,
}

impl TidalModel {
    /// 从文件检测模型类型
    pub fn detect(path: &Path) -> Self {
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_lowercase();
        
        if name.contains("tpxo9") {
            Self::Tpxo9
        } else if name.contains("tpxo") {
            Self::TpxoLocal
        } else if name.contains("fes2014") || name.contains("fes") {
            Self::Fes2014
        } else if name.contains("got") {
            Self::Got410
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

        // 归一化经度到 [0, 360)
        let mut lon_normalized = lon.rem_euclid(360.0);
        if lon_normalized < self.lon_range.0 {
            lon_normalized += 360.0;
        }
        
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
        let y = (lat - self.lat_range.0 - (j_lat as f64) * self.lat_resolution)
            / self.lat_resolution;

        let x = x.clamp(0.0, 1.0);
        let y = y.clamp(0.0, 1.0);
        
        Some(InterpolationIndices {
            i: [i_lon, i_lon + 1, i_lon, i_lon + 1],
            j: [j_lat, j_lat, j_lat + 1, j_lat + 1],
            weights: [
                (1.0 - x) * (1.0 - y),
                x * (1.0 - y),
                (1.0 - x) * y,
                x * y,
            ],
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
/// 支持 TPXO9 及其地区模型
pub struct TpxoReader {
    /// 文件路径
    #[allow(dead_code)]
    path: PathBuf,
    /// 模型类型
    model_type: TidalModel,
    /// 网格信息
    grid: TidalGrid,
    /// 可用分潮
    constituents: Vec<String>,
    /// 缓存的数据（分潮名 -> (振幅, 相位)）
    cache: RwLock<HashMap<String, (Vec<Vec<f64>>, Vec<Vec<f64>>) >>,
}

impl TpxoReader {
    /// 打开 TPXO 文件
    pub fn open(path: impl AsRef<Path>) -> Result<Self, TidalIoError> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(TidalIoError::FileNotFound(path.to_path_buf()));
        }
        
        // 检测模型类型
        let model_type = TidalModel::detect(path);
        
        // 从文件读取元数据（这里是模拟实现）
        // 实际实现需要 netcdf crate
        let grid = Self::read_grid_info(path)?;
        let constituents = Self::read_constituents(path)?;
        
        Ok(Self {
            path: path.to_path_buf(),
            model_type,
            grid,
            constituents,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// 读取网格信息
    fn read_grid_info(_path: &Path) -> Result<TidalGrid, TidalIoError> {
        // 模拟 TPXO9 全球网格（1/30 度分辨率）
        // 实际实现需要从 NetCDF 读取
        Ok(TidalGrid {
            lon_range: (0.0, 360.0),
            lat_range: (-90.0, 90.0),
            lon_resolution: 1.0 / 30.0, // 1/30 度 ≈ 2 arcmin
            lat_resolution: 1.0 / 30.0,
            n_lon: 10800,
            n_lat: 5400,
        })
    }

    /// 读取分潮列表
    fn read_constituents(_path: &Path) -> Result<Vec<String>, TidalIoError> {
        // TPXO9 标准 15 分潮
        Ok(vec![
            "m2", "s2", "n2", "k2",
            "k1", "o1", "p1", "q1",
            "mf", "mm", "m4", "ms4", "mn4",
            "2n2", "s1",
        ].into_iter().map(String::from).collect())
    }

    /// 加载分潮数据到缓存
    #[allow(dead_code)]
    fn load_constituent(&self, name: &str) -> Result<(), TidalIoError> {
        if !self.constituents.iter().any(|c| c == name) {
            return Err(TidalIoError::ConstituentNotFound(name.to_string()));
        }
        if self.cache.read().map(|c| c.contains_key(name)).unwrap_or(false) {
            return Ok(());
        }
        
        // 模拟数据加载
        // 实际实现需要从 NetCDF 读取
        const MAX_CACHE_CELLS: usize = 5_000_000;
        let n_lon = self.grid.n_lon;
        let n_lat = self.grid.n_lat;
        let total = n_lon.saturating_mul(n_lat);

        // 生成模拟数据（测试用），避免超大内存分配
        let (amplitude, phase) = if total == 0 || total > MAX_CACHE_CELLS {
            (Vec::new(), Vec::new())
        } else {
            (vec![vec![0.0; n_lon]; n_lat], vec![vec![0.0; n_lon]; n_lat])
        };
        
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(name.to_lowercase(), (amplitude, phase));
        }
        
        Ok(())
    }

    /// 双线性插值
    fn interpolate(
        &self,
        data: &[Vec<f64>],
        indices: &InterpolationIndices,
    ) -> f64 {
        let mut result = 0.0;
        for k in 0..4 {
            let i = indices.i[k];
            let j = indices.j[k];
            if j < data.len() && i < data[j].len() {
                result += indices.weights[k] * data[j][i];
            }
        }
        result
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
    
    fn read_constituent(&self, name: &str, _lon: f64, _lat: f64) -> Result<(f64, f64), TidalIoError> {
        let name_lower = name.to_lowercase();
        if !self.constituents.iter().any(|c| c == &name_lower) {
            return Err(TidalIoError::ConstituentNotFound(name.to_string()));
        }

        if !_lon.is_finite() || !_lat.is_finite() {
            return Err(TidalIoError::OutOfBounds { lon: _lon, lat: _lat });
        }
        
        let indices = self.grid.interpolation_indices(_lon, _lat)
            .ok_or(TidalIoError::OutOfBounds { lon: _lon, lat: _lat })?;
        
        if self.cache.read().map(|c| !c.contains_key(&name_lower)).unwrap_or(true) {
            self.load_constituent(&name_lower)?;
        }

        // 从缓存读取或返回默认值
        if let Ok(cache) = self.cache.read() {
            if let Some((amp_data, phase_data)) = cache.get(&name_lower) {
                let amp = self.interpolate(amp_data, &indices);
                let phase = self.interpolate(phase_data, &indices);
                return Ok((amp, phase));
            }
        }

        Ok((0.0, 0.0))
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
        
        for i in 0..n {
            let (amp, phase) = self.read_constituent(name, lons[i], lats[i])?;
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
        // U 分量和 V 分量
        // 实际实现需要从 NetCDF 的 u/v 变量读取
        let (amp_u, phase_u) = self.read_constituent(&format!("{}_u", name), lon, lat)?;
        let (amp_v, phase_v) = self.read_constituent(&format!("{}_v", name), lon, lat)?;
        
        Ok(((amp_u, phase_u), (amp_v, phase_v)))
    }
}

// ============================================================================
// FES2014 读取器
// ============================================================================

/// FES2014 模型读取器
pub struct Fes2014Reader {
    /// 高度文件目录
    height_dir: PathBuf,
    /// 网格信息
    grid: TidalGrid,
    /// 可用分潮
    constituents: Vec<String>,
}

impl Fes2014Reader {
    /// 打开 FES2014 数据目录
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, TidalIoError> {
        let dir = dir.as_ref();
        
        if !dir.exists() || !dir.is_dir() {
            return Err(TidalIoError::FileNotFound(dir.to_path_buf()));
        }
        
        // FES2014 使用 1/16 度分辨率
        let grid = TidalGrid {
            lon_range: (0.0, 360.0),
            lat_range: (-90.0, 90.0),
            lon_resolution: 1.0 / 16.0,
            lat_resolution: 1.0 / 16.0,
            n_lon: 5760,
            n_lat: 2880,
        };
        
        // FES2014 34 分潮
        let constituents = vec![
            "2n2", "eps2", "j1", "k1", "k2", "l2", "la2", "m2", "m3", "m4",
            "m6", "m8", "mf", "mks2", "mm", "mn4", "ms4", "msf", "msqm",
            "mtm", "mu2", "n2", "n4", "nu2", "o1", "p1", "q1", "r2", "s1",
            "s2", "s4", "sa", "ssa", "t2",
        ].into_iter().map(String::from).collect();
        
        Ok(Self {
            height_dir: dir.to_path_buf(),
            grid,
            constituents,
        })
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
    
    fn read_constituent(&self, name: &str, _lon: f64, _lat: f64) -> Result<(f64, f64), TidalIoError> {
        let _ = (_lon, _lat);
        // FES2014 每个分潮一个文件
        let file_path = self.height_dir.join(format!("{}.nc", name.to_lowercase()));
        
        if !file_path.exists() {
            return Err(TidalIoError::ConstituentNotFound(name.to_string()));
        }
        
        // 模拟实现
        Ok((0.0, 0.0))
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
        
        for i in 0..n {
            let (a, p) = self.read_constituent(name, lons[i], lats[i])?;
            amps.push(a);
            phases.push(p);
        }
        
        Ok((amps, phases))
    }
    
    fn read_velocity_constituent(
        &self,
        _name: &str,
        _lon: f64,
        _lat: f64,
    ) -> Result<((f64, f64), (f64, f64)), TidalIoError> {
        Ok(((0.0, 0.0), (0.0, 0.0)))
    }
}

// ============================================================================
// 潮汐数据工厂
// ============================================================================

/// 自动检测并打开潮汐数据
pub fn open_tidal_data(path: impl AsRef<Path>) -> Result<Box<dyn TidalDataReader>, TidalIoError> {
    let path = path.as_ref();
    let model = TidalModel::detect(path);
    
    match model {
        TidalModel::Tpxo9 | TidalModel::TpxoLocal => {
            Ok(Box::new(TpxoReader::open(path)?))
        }
        TidalModel::Fes2014 => {
            Ok(Box::new(Fes2014Reader::open(path)?))
        }
        _ => {
            // 尝试作为 TPXO 格式打开
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
            constants.insert(
                name.to_string(),
                amps.into_iter().zip(phases).collect(),
            );
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

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_detection() {
        assert_eq!(TidalModel::detect(Path::new("tpxo9.nc")), TidalModel::Tpxo9);
        assert_eq!(TidalModel::detect(Path::new("fes2014.nc")), TidalModel::Fes2014);
        assert_eq!(TidalModel::detect(Path::new("unknown.nc")), TidalModel::Unknown);
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
    fn test_boundary_constants() {
        let constants = BoundaryTidalConstants {
            n_points: 3,
            constituents: vec!["m2".to_string(), "s2".to_string()],
            constants: [
                ("m2".to_string(), vec![(1.0, 0.0), (1.2, 30.0), (0.8, 60.0)]),
                ("s2".to_string(), vec![(0.5, 0.0), (0.6, 45.0), (0.4, 90.0)]),
            ].into_iter().collect(),
        };

        let mut freqs = HashMap::new();
        freqs.insert("m2".to_string(), 0.5058681); // deg/hr
        freqs.insert("s2".to_string(), 0.5000000);

        let levels = constants.predict(12.0, &freqs);
        assert_eq!(levels.len(), 3);
    }
}
