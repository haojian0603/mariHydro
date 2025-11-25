// src-tauri/src/marihydro/infra/manifest.rs

use crate::marihydro::domain::feature::GeometricFeature;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use uuid::Uuid;

// ============================================================================
// 数据源配置 (Data Source Configuration)
// ============================================================================

/// 支持的文件格式枚举
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataFormat {
    /// NetCDF 气象/海洋通用格式 (.nc, .cdf)
    /// 支持多维数组和时间维度
    NetCDF,

    /// GeoTIFF 栅格数据 (.tif, .tiff)
    /// 通常用于静态地形、糙率场或初始场
    GeoTIFF,

    /// TPXO 潮汐模型专用二进制格式
    /// 用于全球潮汐预报
    TPXO,

    /// 逗号分隔值 (.csv, .txt)
    /// 用于时间序列数据（如河流流量、验潮站水位）
    CSV,
}

impl DataFormat {
    /// 根据文件扩展名推断格式
    pub fn from_extension(path: &str) -> Option<Self> {
        let ext = Path::new(path).extension()?.to_str()?.to_lowercase();

        match ext.as_str() {
            "nc" | "cdf" | "nc4" => Some(Self::NetCDF),
            "tif" | "tiff" | "geotiff" => Some(Self::GeoTIFF),
            "csv" | "txt" => Some(Self::CSV),
            _ => None,
        }
    }

    /// 获取格式的文件扩展名列表
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
/// 用于解决 "文件里的变量名" -> "模型里的物理含义" 的映射问题
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableMapping {
    /// 模型内部目标变量名
    /// 例如: "wind_u", "wind_v", "zb", "roughness", "pressure"
    pub target_var: String,

    /// 文件内部源变量名
    /// 例如: "u10", "Band1", "elevation", "msl"
    pub source_var: String,

    /// 线性变换缩放系数: value = source * scale + offset
    /// 默认为 1.0。用于单位转换 (如 Pa -> hPa, cm -> m)
    #[serde(default = "default_scale_factor")]
    pub scale_factor: f64,

    /// 线性变换偏移量: value = source * scale + offset
    /// 默认为 0.0。
    #[serde(default)]
    pub offset: f64,

    /// 无效值回退 (Fallback Value)
    /// 如果源数据中存在 NoData/NaN，或者插值失败，用此值填充。
    /// 如果为 None，则保持 NaN (Fail-Fast) 或由插值器策略决定。
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
    /// 创建新的变量映射
    pub fn new(target: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            target_var: target.into(),
            source_var: source.into(),
            ..Default::default()
        }
    }

    /// 设置线性变换参数
    pub fn with_transform(mut self, scale: f64, offset: f64) -> Self {
        self.scale_factor = scale;
        self.offset = offset;
        self
    }

    /// 设置回退值
    pub fn with_fallback(mut self, value: f64) -> Self {
        self.fallback_value = Some(value);
        self
    }

    /// 应用变换
    pub fn apply_transform(&self, value: f64) -> f64 {
        value * self.scale_factor + self.offset
    }
}

/// 外部数据源配置
/// 描述一个外部文件及其在模拟中的用途
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// 数据源唯一标识
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,

    /// 显示名称 (e.g., "ERA5 Wind 2024")
    pub name: String,

    /// 文件绝对路径
    pub file_path: String,

    /// 文件格式
    pub format: DataFormat,

    /// 变量映射列表
    /// 一个文件可能提供多个物理变量
    #[serde(default)]
    pub mappings: Vec<VariableMapping>,

    /// 时间维度名称 (仅 NetCDF/HDF5 有效，默认为 "time")
    pub time_dim_name: Option<String>,

    /// 通用元数据/策略配置
    /// 用于存储扩展策略，如:
    /// "nodata_strategy": "nearest" | "zero" | "nan"
    /// "interpolation": "bilinear" | "bicubic"
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl DataSourceConfig {
    /// 创建新的数据源配置
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

    /// 添加变量映射
    pub fn add_mapping(&mut self, mapping: VariableMapping) -> &mut Self {
        self.mappings.push(mapping);
        self
    }

    /// 获取特定目标变量的映射
    pub fn get_mapping(&self, target_var: &str) -> Option<&VariableMapping> {
        self.mappings.iter().find(|m| m.target_var == target_var)
    }
}

// ============================================================================
// 物理过程配置 (Physics Configuration)
// ============================================================================

/// 河流源项定义
/// 描述一个点源流入/流出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiverSource {
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,
    pub name: String,

    /// 河口坐标 (x, y) - 位于模型投影坐标系中
    pub location: (f64, f64),

    /// 关联的数据源 ID (指向 CSV 或 NC 文件中的流量 Q 时间序列)
    /// 如果存在，优先从该数据源读取流量
    pub source_id: Option<Uuid>,

    /// 固定流量 [m^3/s] (当没有外部数据源时使用)
    /// 正值表示流入，负值表示流出(取水)
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

    /// 设置固定流量
    pub fn with_discharge(mut self, discharge: f64) -> Self {
        self.constant_discharge = discharge;
        self
    }

    /// 关联数据源
    pub fn with_source(mut self, source_id: Uuid) -> Self {
        self.source_id = Some(source_id);
        self
    }
}

/// 物理参数集合
/// 包含所有控制方程所需的常数和开关
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParameters {
    // --- 基础参数 ---
    /// 重力加速度 [m/s^2] (默认 9.81)
    #[serde(default = "default_gravity")]
    pub gravity: f64,

    /// 水体密度 [kg/m^3] (默认 1025.0 海水)
    #[serde(default = "default_water_density")]
    pub water_density: f64,

    /// 空气密度 [kg/m^3] (默认 1.225)
    #[serde(default = "default_air_density")]
    pub air_density: f64,

    /// 最小水深阈值 (干湿判别) [m] (默认 0.05)
    #[serde(default = "default_h_min")]
    pub h_min: f64,

    // --- 底部摩擦 ---
    /// 全局曼宁粗糙率系数 (Manning's n) [s/m^(1/3)]
    /// 如果没有提供 roughness 栅格文件，则全场使用此值
    #[serde(default = "default_bottom_friction")]
    pub bottom_friction_coeff: f64,

    // --- 湍流 (Turbulence) ---
    /// 水平涡粘系数 (Horizontal Eddy Viscosity) [m^2/s]
    /// 用于 Smagorinsky 模型或常数模型
    #[serde(default = "default_eddy_viscosity")]
    pub eddy_viscosity: f64,

    /// 是否启用 Smagorinsky 亚网格湍流模型
    /// 如果为 true，eddy_viscosity 将作为 Smagorinsky 常数 Cs 使用
    #[serde(default)]
    pub use_smagorinsky: bool,

    // --- 科氏力 (Coriolis) ---
    /// 是否启用科氏力计算
    #[serde(default = "default_enable_coriolis")]
    pub enable_coriolis: bool,

    /// 参考纬度 (Degrees)
    /// 当网格未提供投影信息或无法反算纬度时，使用此常数计算 f 参数
    #[serde(default = "default_latitude_ref")]
    pub latitude_ref: f64,

    // --- 泥沙 (Sediment) ---
    /// 是否启用泥沙输运计算
    #[serde(default)]
    pub enable_sediment: bool,

    /// 沉降速度 [m/s]
    #[serde(default = "default_settling_velocity")]
    pub sediment_settling_velocity: f64,

    /// 临界切应力 [N/m^2] (用于起悬判断)
    #[serde(default = "default_critical_shear")]
    pub critical_shear_stress: f64,
}

// 默认值函数
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
    /// 预设：河流模拟
    pub fn river_preset() -> Self {
        Self {
            water_density: 1000.0,        // 淡水
            enable_coriolis: false,       // 河流尺度通常忽略科氏力
            bottom_friction_coeff: 0.035, // 天然河道较粗糙
            ..Default::default()
        }
    }

    /// 预设：海洋模拟
    pub fn ocean_preset() -> Self {
        Self {
            water_density: 1025.0, // 海水
            enable_coriolis: true,
            bottom_friction_coeff: 0.015, // 海底相对平滑
            use_smagorinsky: true,
            ..Default::default()
        }
    }
}

// ============================================================================
// 工程总蓝图 (Project Manifest)
// ============================================================================

/// 模拟工程总蓝图
/// 对应磁盘上的 project.json 文件，定义了模拟所需的一切静态信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectManifest {
    /// 项目唯一标识 (UUID)
    #[serde(default = "Uuid::new_v4")]
    pub id: Uuid,

    /// 项目元数据 (名称, 描述, 版本, 作者, 创建时间等)
    #[serde(default)]
    pub meta: HashMap<String, String>,

    // --- 时间控制 ---
    /// 模拟起始时间 (UTC ISO8601)
    pub start_time: DateTime<Utc>,
    /// 模拟结束时间 (UTC ISO8601)
    pub end_time: DateTime<Utc>,
    /// 建议计算时间步长 [s]
    /// 如果设置为 0.0，引擎将基于 CFL 条件自动计算动态步长
    #[serde(default)]
    pub dt: f64,

    // --- 空间定义 ---
    /// 物理网格 X 方向单元数
    pub grid_nx: usize,
    /// 物理网格 Y 方向单元数
    pub grid_ny: usize,
    /// 网格 X 方向分辨率 [m]
    pub grid_dx: f64,
    /// 网格 Y 方向分辨率 [m]
    pub grid_dy: f64,

    /// 网格原点 X (Projected Coordinates)
    pub origin_x: Option<f64>,
    /// 网格原点 Y (Projected Coordinates)
    pub origin_y: Option<f64>,

    /// 坐标参考系统定义 (WKT / EPSG)
    /// 必须是投影坐标系 (单位: 米)，用于物理计算
    #[serde(default = "default_crs")]
    pub crs_wkt: String,

    // --- 物理与驱动 ---
    /// 物理参数配置
    #[serde(default)]
    pub physics: PhysicsParameters,

    /// 驱动数据源列表 (风场、气压场、地形、粗糙率、潮汐)
    #[serde(default)]
    pub sources: Vec<DataSourceConfig>,

    /// 河流源项列表
    #[serde(default)]
    pub rivers: Vec<RiverSource>,

    /// 几何特征列表 (边界定义、结构物、观测点)
    #[serde(default)]
    pub features: Vec<GeometricFeature>,
}

fn default_crs() -> String {
    "EPSG:32651".into() // UTM Zone 51N
}

impl ProjectManifest {
    /// 创建一个新的空白工程模板
    pub fn new(name: impl Into<String>) -> Self {
        let mut meta = HashMap::new();
        let name = name.into();
        meta.insert("name".to_string(), name.clone());
        meta.insert("version".to_string(), "1.2.0".to_string());
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
            dt: 0.0, // Auto CFL

            grid_nx: 200,
            grid_ny: 200,
            grid_dx: 50.0,
            grid_dy: 50.0,
            origin_x: Some(0.0),
            origin_y: Some(0.0),

            crs_wkt: default_crs(),

            physics: PhysicsParameters::default(),
            sources: Vec::new(),
            rivers: Vec::new(),
            features: Vec::new(),
        }
    }

    /// 获取项目名称
    pub fn name(&self) -> String {
        self.meta
            .get("name")
            .cloned()
            .unwrap_or_else(|| "Unnamed Project".to_string())
    }

    /// 设置项目名称
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.meta.insert("name".to_string(), name.into());
    }

    /// 获取模拟时长 (秒)
    pub fn duration_seconds(&self) -> f64 {
        (self.end_time - self.start_time).num_seconds() as f64
    }

    /// 获取网格总单元数
    pub fn total_cells(&self) -> usize {
        self.grid_nx * self.grid_ny
    }

    /// 获取计算域面积 (平方米)
    pub fn domain_area(&self) -> f64 {
        (self.grid_nx as f64 * self.grid_dx) * (self.grid_ny as f64 * self.grid_dy)
    }

    /// 计算 CFL 条件建议的最大时间步长
    pub fn estimate_max_dt(&self, max_depth: f64) -> f64 {
        let c = (self.physics.gravity * max_depth).sqrt(); // 波速
        let dx_min = self.grid_dx.min(self.grid_dy);
        0.5 * dx_min / c // CFL = 0.5 保守估计
    }

    /// 验证蓝图逻辑完整性 (Strict Validation)
    ///
    /// 执行全面的自检，包括：
    /// 1. 空间时间参数合法性
    /// 2. 物理参数范围合理性
    /// 3. 变量映射冲突检测 (确保没有两个文件写入同一个目标变量)
    /// 4. 引用完整性 (河流源项指向的数据源是否存在)
    pub fn validate(&self) -> Result<(), String> {
        // --- 1. 基础参数检查 ---
        self.validate_grid_params()?;
        self.validate_time_params()?;

        // --- 2. 物理参数合理性检查 ---
        self.validate_physics_params()?;

        // --- 3. 数据源变量映射冲突检测 ---
        self.validate_data_source_mappings()?;

        // --- 4. 河流源项引用完整性检查 ---
        self.validate_river_sources()?;

        Ok(())
    }

    /// 验证网格参数
    fn validate_grid_params(&self) -> Result<(), String> {
        if self.grid_nx == 0 || self.grid_ny == 0 {
            return Err("配置错误: 网格尺寸 (nx, ny) 必须大于 0".into());
        }
        if self.grid_dx <= 0.0 || self.grid_dy <= 0.0 {
            return Err("配置错误: 网格间距 (dx, dy) 必须为正数".into());
        }
        if self.crs_wkt.trim().is_empty() {
            return Err("配置错误: 必须指定坐标参考系统 (CRS)".into());
        }

        // 检查网格规模是否过大
        let total_cells = self.total_cells();
        if total_cells > 10_000_000 {
            return Err(format!(
                "配置警告: 网格单元总数 ({}) 超过千万，可能导致内存不足",
                total_cells
            ));
        }

        Ok(())
    }

    /// 验证时间参数
    fn validate_time_params(&self) -> Result<(), String> {
        if self.end_time <= self.start_time {
            return Err("配置错误: 模拟结束时间必须晚于开始时间".into());
        }
        if self.dt < 0.0 {
            return Err("配置错误: 时间步长 dt 不能为负数".into());
        }

        // 检查模拟时长是否合理
        let duration_days = self.duration_seconds() / 86400.0;
        if duration_days > 365.0 {
            return Err(format!(
                "配置警告: 模拟时长 ({:.1} 天) 超过一年，请确认是否正确",
                duration_days
            ));
        }

        Ok(())
    }

    /// 验证物理参数
    fn validate_physics_params(&self) -> Result<(), String> {
        let p = &self.physics;

        // 基础参数
        if p.water_density <= 0.0 {
            return Err("物理参数错误: 水体密度必须为正数".into());
        }
        if p.air_density <= 0.0 {
            return Err("物理参数错误: 空气密度必须为正数".into());
        }
        if p.gravity <= 0.0 {
            return Err("物理参数错误: 重力加速度必须为正数".into());
        }
        if p.h_min < 0.0 {
            return Err("物理参数错误: 最小水深阈值 (h_min) 不能为负数".into());
        }

        // 摩擦参数
        if p.bottom_friction_coeff < 0.0 {
            return Err("物理参数错误: 曼宁糙率系数不能为负数".into());
        }
        if p.bottom_friction_coeff > 1.0 {
            return Err("物理参数错误: 曼宁糙率系数通常小于 1.0".into());
        }

        // 湍流参数
        if p.eddy_viscosity < 0.0 {
            return Err("物理参数错误: 涡粘系数不能为负数".into());
        }

        // 科氏力参数
        if p.latitude_ref < -90.0 || p.latitude_ref > 90.0 {
            return Err("物理参数错误: 参考纬度必须在 [-90, 90] 之间".into());
        }

        // 泥沙参数
        if p.enable_sediment {
            if p.sediment_settling_velocity <= 0.0 {
                return Err("物理参数错误: 泥沙沉降速度必须为正数".into());
            }
            if p.critical_shear_stress < 0.0 {
                return Err("物理参数错误: 临界切应力不能为负数".into());
            }
        }

        Ok(())
    }

    /// 验证数据源映射
    fn validate_data_source_mappings(&self) -> Result<(), String> {
        // 规则：同一个目标变量只能被一个数据源定义
        let mut target_var_registry: HashMap<String, String> = HashMap::new();

        for source in &self.sources {
            // 检查文件路径
            if source.file_path.trim().is_empty() {
                return Err(format!("数据源 '{}' 的文件路径为空", source.name));
            }

            // 检查文件是否存在
            if !Path::new(&source.file_path).exists() {
                return Err(format!(
                    "数据源 '{}' 的文件不存在: {}",
                    source.name, source.file_path
                ));
            }

            for mapping in &source.mappings {
                // 忽略空目标
                if mapping.target_var.trim().is_empty() {
                    continue;
                }

                // 检查冲突
                if let Some(existing_source) = target_var_registry.get(&mapping.target_var) {
                    return Err(format!(
                        "变量映射冲突: 目标变量 '{}' 同时被数据源 '{}' 和 '{}' 定义。\n\
                         请修改配置，确保每个物理变量只由一个数据源驱动。",
                        mapping.target_var, existing_source, source.name
                    ));
                }

                // 注册
                target_var_registry.insert(mapping.target_var.clone(), source.name.clone());

                // 检查变换参数
                if !mapping.scale_factor.is_finite() {
                    return Err(format!(
                        "数据源 '{}' 中变量 '{}' 的缩放因子无效",
                        source.name, mapping.target_var
                    ));
                }
                if !mapping.offset.is_finite() {
                    return Err(format!(
                        "数据源 '{}' 中变量 '{}' 的偏移量无效",
                        source.name, mapping.target_var
                    ));
                }
                if mapping.scale_factor == 0.0 {
                    return Err(format!(
                        "数据源 '{}' 中变量 '{}' 的缩放因子不能为零",
                        source.name, mapping.target_var
                    ));
                }
            }
        }

        Ok(())
    }

    /// 验证河流源项
    fn validate_river_sources(&self) -> Result<(), String> {
        // 收集所有数据源 ID
        let valid_source_ids: HashSet<Uuid> = self.sources.iter().map(|s| s.id).collect();

        for river in &self.rivers {
            // 检查名称
            if river.name.trim().is_empty() {
                return Err(format!("河流 ID {} 缺少名称", river.id));
            }

            // 检查位置是否在计算域内
            let (x, y) = river.location;
            let x_max = self.origin_x.unwrap_or(0.0) + (self.grid_nx as f64 * self.grid_dx);
            let y_max = self.origin_y.unwrap_or(0.0) + (self.grid_ny as f64 * self.grid_dy);
            let x_min = self.origin_x.unwrap_or(0.0);
            let y_min = self.origin_y.unwrap_or(0.0);

            if x < x_min || x > x_max || y < y_min || y > y_max {
                return Err(format!(
                    "河流 '{}' 的位置 ({:.1}, {:.1}) 超出计算域范围",
                    river.name, x, y
                ));
            }

            // 检查关联的数据源是否存在
            if let Some(source_id) = river.source_id {
                if !valid_source_ids.contains(&source_id) {
                    return Err(format!(
                        "配置一致性错误: 河流 '{}' 引用了不存在的数据源 ID ({})。\n\
                         请检查该数据源是否已被删除。",
                        river.name, source_id
                    ));
                }
            }

            // 检查流量合理性
            if river.constant_discharge.abs() > 1_000_000.0 {
                return Err(format!(
                    "河流 '{}' 的流量绝对值 ({:.2}) 异常巨大。\n\
                     请确认单位是否为 m³/s。",
                    river.name, river.constant_discharge
                ));
            }

            // 警告：没有数据源也没有固定流量
            if river.source_id.is_none() && river.constant_discharge == 0.0 {
                return Err(format!(
                    "河流 '{}' 既没有关联数据源，也没有设置固定流量",
                    river.name
                ));
            }
        }

        Ok(())
    }

    /// 获取所有需要的输入变量列表
    pub fn get_required_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();

        // 从数据源映射收集
        for source in &self.sources {
            for mapping in &source.mappings {
                if !mapping.target_var.is_empty() {
                    vars.insert(mapping.target_var.clone());
                }
            }
        }

        // 添加必需的基础变量
        vars.insert("zb".to_string()); // 地形高程

        // 根据物理开关添加可选变量
        if self.physics.enable_coriolis {
            vars.insert("latitude".to_string());
        }
        if self.physics.enable_sediment {
            vars.insert("sediment_conc".to_string());
        }

        vars
    }

    /// 查找提供指定变量的数据源
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
        assert_eq!(manifest.grid_nx, 200);
        assert_eq!(manifest.grid_ny, 200);
    }

    #[test]
    fn test_validation_empty_manifest() {
        let manifest = ProjectManifest::new("Test");
        assert!(manifest.validate().is_ok());
    }

    #[test]
    fn test_validation_invalid_grid() {
        let mut manifest = ProjectManifest::new("Test");
        manifest.grid_nx = 0;
        assert!(manifest.validate().is_err());
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
        assert_eq!(
            DataFormat::from_extension("flow.csv"),
            Some(DataFormat::CSV)
        );
        assert_eq!(DataFormat::from_extension("unknown.xyz"), None);
    }

    #[test]
    fn test_variable_mapping_transform() {
        let mapping = VariableMapping::new("pressure", "msl").with_transform(0.01, -1013.25); // Pa to hPa

        assert_eq!(mapping.apply_transform(101325.0), 0.0);
    }
}
