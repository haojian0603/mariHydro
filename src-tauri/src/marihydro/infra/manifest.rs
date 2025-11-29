// src-tauri/src/marihydro/infra/manifest.rs

use crate::marihydro::domain::feature::GeometricFeature;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use uuid::Uuid;

// ============================================================================
// 数据源配置 (Data Source Configuration)
// ============================================================================

/// 支持的文件格式枚举
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataFormat {
    NetCDF,
    GeoTIFF,
    TPXO,
    CSV,
}

impl DataFormat {
    pub fn from_extension(path: &str) -> Option<Self> {
        let ext = Path::new(path).extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "nc" | "cdf" | "nc4" => Some(Self::NetCDF),
            "tif" | "tiff" | "geotiff" => Some(Self::GeoTIFF),
            "csv" | "txt" => Some(Self::CSV),
            _ => None,
        }
    }

    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Self::NetCDF => &["nc", "cdf", "nc4"],
            Self::GeoTIFF => &["tif", "tiff", "geotiff"],
            Self::TPXO => &["tpxo"],
            Self::CSV => &["csv", "txt"],
        }
    }
}

/// 变量映射规则
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMapping {
    pub target_var: String,
    pub source_var: String,
    #[serde(default = "default_scale_factor")]
    pub scale_factor: f64,
    #[serde(default)]
    pub offset: f64,
    pub fallback_value: Option<f64>,
}

fn default_scale_factor() -> f64 {
    1.0
}

impl Default for VariableMapping {
    fn default() -> Self {
        Self {
            target_var: String::new(),
            source_var: String::new(),
            scale_factor: 1.0,
            offset: 0.0,
            fallback_value: None,
        }
    }
}

impl VariableMapping {
    pub fn new(target: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            target_var: target.into(),
            source_var: source.into(),
            ..Default::default()
        }
    }

    pub fn with_transform(mut self, scale: f64, offset: f64) -> Self {
        self.scale_factor = scale;
        self.offset = offset;
        self
    }

    pub fn with_fallback(mut self, value: f64) -> Self {
        self.fallback_value = Some(value);
        self
    }

    pub fn apply_transform(&self, value: f64) -> f64 {
        value * self.scale_factor + self.offset
    }
}

/// 外部数据源配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    pub file_path: String,
    pub format: DataFormat,
    #[serde(default)]
    pub mappings: Vec<VariableMapping>,
    pub time_dim_name: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl DataSourceConfig {
    pub fn new(name: impl Into<String>, file_path: impl Into<String>) -> Self {
        let path = file_path.into();
        let format = DataFormat::from_extension(&path).unwrap_or(DataFormat::NetCDF);

        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            file_path: path,
            format,
            mappings: Vec::new(),
            time_dim_name: None,
            metadata: HashMap::new(),
        }
    }

    pub fn add_mapping(&mut self, mapping: VariableMapping) -> &mut Self {
        self.mappings.push(mapping);
        self
    }

    pub fn get_mapping(&self, target_var: &str) -> Option<&VariableMapping> {
        self.mappings.iter().find(|m| m.target_var == target_var)
    }
}

// ============================================================================
// 网格格式定义（新增）
// ============================================================================

/// 网格文件格式
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MeshFormat {
    /// Gmsh MSH2 格式（ASCII）
    GmshMSH2,
    /// Gmsh MSH4 格式（ASCII/Binary）
    GmshMSH4,
    /// UGRID 格式（NetCDF）
    UGRID,
}

impl Default for MeshFormat {
    fn default() -> Self {
        Self::GmshMSH4
    }
}

impl MeshFormat {
    pub fn from_extension(path: &str) -> Option<Self> {
        let ext = Path::new(path).extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "msh" => Some(Self::GmshMSH4),
            "nc" => Some(Self::UGRID),
            _ => None,
        }
    }
}

// ============================================================================
// 物理过程配置
// ============================================================================

/// 河流源项定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverSource {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,
    pub location: (f64, f64),
    pub source_id: Option<Uuid>,
    #[serde(default)]
    pub constant_discharge: f64,
}

impl RiverSource {
    pub fn new(name: impl Into<String>, x: f64, y: f64) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            location: (x, y),
            source_id: None,
            constant_discharge: 0.0,
        }
    }

    pub fn with_discharge(mut self, discharge: f64) -> Self {
        self.constant_discharge = discharge;
        self
    }

    pub fn with_source(mut self, source_id: Uuid) -> Self {
        self.source_id = Some(source_id);
        self
    }
}

/// 物理参数集合
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParameters {
    #[serde(default = "default_gravity")]
    pub gravity: f64,
    #[serde(default = "default_water_density")]
    pub water_density: f64,
    #[serde(default = "default_air_density")]
    pub air_density: f64,
    #[serde(default = "default_h_min")]
    pub h_min: f64,
    #[serde(default = "default_bottom_friction")]
    pub bottom_friction_coeff: f64,
    #[serde(default = "default_eddy_viscosity")]
    pub eddy_viscosity: f64,
    #[serde(default)]
    pub use_smagorinsky: bool,
    #[serde(default = "default_enable_coriolis")]
    pub enable_coriolis: bool,
    #[serde(default = "default_latitude_ref")]
    pub latitude_ref: f64,
    #[serde(default)]
    pub enable_sediment: bool,
    #[serde(default = "default_settling_velocity")]
    pub sediment_settling_velocity: f64,
    #[serde(default = "default_critical_shear")]
    pub critical_shear_stress: f64,
}

fn default_gravity() -> f64 {
    9.81
}
fn default_water_density() -> f64 {
    1025.0
}
fn default_air_density() -> f64 {
    1.225
}
fn default_h_min() -> f64 {
    0.05
}
fn default_bottom_friction() -> f64 {
    0.025
}
fn default_eddy_viscosity() -> f64 {
    1.0
}
fn default_enable_coriolis() -> bool {
    true
}
fn default_latitude_ref() -> f64 {
    30.0
}
fn default_settling_velocity() -> f64 {
    0.001
}
fn default_critical_shear() -> f64 {
    0.1
}

impl Default for PhysicsParameters {
    fn default() -> Self {
        Self {
            gravity: default_gravity(),
            water_density: default_water_density(),
            air_density: default_air_density(),
            h_min: default_h_min(),
            bottom_friction_coeff: default_bottom_friction(),
            eddy_viscosity: default_eddy_viscosity(),
            use_smagorinsky: false,
            enable_coriolis: default_enable_coriolis(),
            latitude_ref: default_latitude_ref(),
            enable_sediment: false,
            sediment_settling_velocity: default_settling_velocity(),
            critical_shear_stress: default_critical_shear(),
        }
    }
}

impl PhysicsParameters {
    pub fn river_preset() -> Self {
        Self {
            water_density: 1000.0,
            enable_coriolis: false,
            bottom_friction_coeff: 0.035,
            ..Default::default()
        }
    }

    pub fn ocean_preset() -> Self {
        Self {
            water_density: 1025.0,
            enable_coriolis: true,
            bottom_friction_coeff: 0.015,
            use_smagorinsky: true,
            ..Default::default()
        }
    }
}

// ============================================================================
// 工程总蓝图（修改为非结构化网格）
// ============================================================================

/// 模拟工程总蓝图
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectManifest {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    #[serde(default)]
    pub meta: HashMap<String, String>,

    // --- 时间控制 ---
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    #[serde(default)]
    pub dt: f64,

    // --- 空间定义（改为非结构化网格） ---
    ///  网格文件路径
    pub mesh_file: PathBuf,
    ///  网格格式
    #[serde(default)]
    pub mesh_format: MeshFormat,

    /// 坐标参考系统定义 (WKT / EPSG)
    #[serde(default = "default_crs")]
    pub crs_wkt: String,

    // --- 物理与驱动 ---
    #[serde(default)]
    pub physics: PhysicsParameters,
    #[serde(default)]
    pub sources: Vec<DataSourceConfig>,
    #[serde(default)]
    pub rivers: Vec<RiverSource>,
    #[serde(default)]
    pub features: Vec<GeometricFeature>,
}

fn default_crs() -> String {
    "EPSG:32651".into()
}

impl Default for ProjectManifest {
    fn default() -> Self {
        Self::new("Default Project")
    }
}

impl ProjectManifest {
    pub fn new(name: impl Into<String>) -> Self {
        let mut meta = HashMap::new();
        let name = name.into();
        meta.insert("name".to_string(), name.clone());
        meta.insert("version".to_string(), "2.0.0".to_string());
        meta.insert("created_at".to_string(), Utc::now().to_rfc3339());
        meta.insert(
            "description".to_string(),
            format!("Project {} created by mariHydro Desktop", name),
        );

        Self {
            id: Uuid::new_v4(),
            meta,
            start_time: Utc::now(),
            end_time: Utc::now() + chrono::Duration::days(1),
            dt: 0.0,

            mesh_file: PathBuf::from("mesh/domain.msh"),
            mesh_format: MeshFormat::default(),

            crs_wkt: default_crs(),

            physics: PhysicsParameters::default(),
            sources: Vec::new(),
            rivers: Vec::new(),
            features: Vec::new(),
        }
    }

    pub fn name(&self) -> String {
        self.meta
            .get("name")
            .cloned()
            .unwrap_or_else(|| "Unnamed Project".to_string())
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.meta.insert("name".to_string(), name.into());
    }

    pub fn duration_seconds(&self) -> f64 {
        (self.end_time - self.start_time).num_seconds() as f64
    }

    /// 验证蓝图逻辑完整性
    pub fn validate(&self) -> Result<(), String> {
        self.validate_mesh_params()?;
        self.validate_time_params()?;
        self.validate_physics_params()?;
        self.validate_data_source_mappings()?;
        self.validate_river_sources()?;
        Ok(())
    }

    fn validate_mesh_params(&self) -> Result<(), String> {
        if self.mesh_file.as_os_str().is_empty() {
            return Err("配置错误: 必须指定网格文件路径".into());
        }

        if !self.mesh_file.exists() {
            log::warn!("网格文件不存在: {:?}", self.mesh_file);
        }

        if self.crs_wkt.trim().is_empty() {
            return Err("配置错误: 必须指定坐标参考系统 (CRS)".into());
        }

        Ok(())
    }

    fn validate_time_params(&self) -> Result<(), String> {
        if self.end_time <= self.start_time {
            return Err("配置错误: 模拟结束时间必须晚于开始时间".into());
        }
        if self.dt < 0.0 {
            return Err("配置错误: 时间步长 dt 不能为负数".into());
        }

        let duration_days = self.duration_seconds() / 86400.0;
        if duration_days > 365.0 {
            log::warn!("模拟时长 ({:.1} 天) 超过一年", duration_days);
        }

        Ok(())
    }

    fn validate_physics_params(&self) -> Result<(), String> {
        let p = &self.physics;

        if p.water_density <= 0.0 {
            return Err("物理参数错误: 水体密度必须为正数".into());
        }
        if p.gravity <= 0.0 {
            return Err("物理参数错误: 重力加速度必须为正数".into());
        }
        if p.h_min < 0.0 {
            return Err("物理参数错误: 最小水深阈值不能为负数".into());
        }
        if p.bottom_friction_coeff < 0.0 || p.bottom_friction_coeff > 1.0 {
            return Err("物理参数错误: 曼宁糙率系数范围应在 [0, 1]".into());
        }
        if p.latitude_ref < -90.0 || p.latitude_ref > 90.0 {
            return Err("物理参数错误: 参考纬度必须在 [-90, 90]".into());
        }

        Ok(())
    }

    fn validate_data_source_mappings(&self) -> Result<(), String> {
        let mut target_var_registry: HashMap<String, String> = HashMap::new();

        for source in &self.sources {
            if source.file_path.trim().is_empty() {
                return Err(format!("数据源 '{}' 的文件路径为空", source.name));
            }

            for mapping in &source.mappings {
                if mapping.target_var.trim().is_empty() {
                    continue;
                }

                if let Some(existing_source) = target_var_registry.get(&mapping.target_var) {
                    return Err(format!(
                        "变量映射冲突: 目标变量 '{}' 同时被 '{}' 和 '{}' 定义",
                        mapping.target_var, existing_source, source.name
                    ));
                }

                target_var_registry.insert(mapping.target_var.clone(), source.name.clone());

                if !mapping.scale_factor.is_finite() || mapping.scale_factor == 0.0 {
                    return Err(format!("数据源 '{}' 的缩放因子无效", source.name));
                }
            }
        }

        Ok(())
    }

    fn validate_river_sources(&self) -> Result<(), String> {
        let valid_source_ids: HashSet<Uuid> = self.sources.iter().map(|s| s.id).collect();

        for river in &self.rivers {
            if river.name.trim().is_empty() {
                return Err(format!("河流 ID {} 缺少名称", river.id));
            }

            if let Some(source_id) = river.source_id {
                if !valid_source_ids.contains(&source_id) {
                    return Err(format!(
                        "河流 '{}' 引用了不存在的数据源 {}",
                        river.name, source_id
                    ));
                }
            }

            if river.source_id.is_none() && river.constant_discharge == 0.0 {
                log::warn!("河流 '{}' 既没有数据源也没有固定流量", river.name);
            }
        }

        Ok(())
    }

    pub fn get_required_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for source in &self.sources {
            for mapping in &source.mappings {
                if !mapping.target_var.is_empty() {
                    vars.insert(mapping.target_var.clone());
                }
            }
        }
        vars
    }

    pub fn find_source_for_variable(&self, var_name: &str) -> Option<&DataSourceConfig> {
        self.sources
            .iter()
            .find(|source| source.mappings.iter().any(|m| m.target_var == var_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_creation() {
        let manifest = ProjectManifest::new("Test Project");
        assert_eq!(manifest.name(), "Test Project");
        assert!(manifest.mesh_file.to_str().is_some());
    }

    #[test]
    fn test_data_format_from_extension() {
        assert_eq!(
            DataFormat::from_extension("data.nc"),
            Some(DataFormat::NetCDF)
        );
        assert_eq!(
            DataFormat::from_extension("terrain.tif"),
            Some(DataFormat::GeoTIFF)
        );
    }

    #[test]
    fn test_mesh_format_from_extension() {
        assert_eq!(
            MeshFormat::from_extension("mesh.msh"),
            Some(MeshFormat::GmshMSH4)
        );
        assert_eq!(
            MeshFormat::from_extension("grid.nc"),
            Some(MeshFormat::UGRID)
        );
    }
}
