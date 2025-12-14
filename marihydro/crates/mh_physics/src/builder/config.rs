// crates/mh_physics/src/builder/config.rs

//! 求解器配置（无泛型）
//!
//! App层直接使用的配置类型，完全不包含泛型参数。

use mh_config::Precision;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 黎曼求解器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiemannSolverType {
    /// HLLC求解器（推荐）
    #[default]
    Hllc,
    /// Roe求解器
    Roe,
    /// Rusanov求解器
    Rusanov,
    /// 简单中心格式
    Central,
}

/// 时间积分方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeIntegrationMethod {
    /// 前向欧拉（一阶）
    ForwardEuler,
    /// SSP-RK2（二阶）
    #[default]
    SspRk2,
    /// SSP-RK3（三阶）
    SspRk3,
}

/// 限制器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LimiterType {
    /// 无限制器（一阶精度）
    None,
    /// Minmod限制器
    #[default]
    Minmod,
    /// Van Leer限制器
    VanLeer,
    /// Superbee限制器
    Superbee,
    /// MC限制器
    Mc,
}

/// 求解器配置（完全无泛型）
///
/// 这是App层唯一需要接触的配置类型。所有数值参数使用f64存储，
/// 在构建求解器时会根据选择的精度进行转换。
///
/// # 示例
///
/// ```ignore
/// use mh_physics::builder::{SolverConfig, Precision};
///
/// let config = SolverConfig {
///     precision: Precision::F32,
///     cfl: 0.5,
///     ..Default::default()
/// };
///
/// // 保存到文件
/// config.save("config.yaml")?;
///
/// // 从文件加载
/// let loaded = SolverConfig::load("config.yaml")?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    // ========== 精度选择 ==========
    
    /// 计算精度（运行时决定）
    #[serde(default)]
    pub precision: Precision,

    // ========== 数值参数 ==========
    
    /// CFL数（库朗数）
    #[serde(default = "default_cfl")]
    pub cfl: f64,

    /// 最大速度限制 [m/s]
    #[serde(default = "default_max_velocity")]
    pub max_velocity: f64,

    /// 最小水深 [m]
    #[serde(default = "default_h_min")]
    pub h_min: f64,

    /// 干单元水深阈值 [m]
    #[serde(default = "default_h_dry")]
    pub h_dry: f64,

    /// 重力加速度 [m/s²]
    #[serde(default = "default_gravity")]
    pub gravity: f64,

    // ========== 物理选项 ==========
    
    /// 黎曼求解器类型
    #[serde(default)]
    pub riemann_solver: RiemannSolverType,

    /// 时间积分方法
    #[serde(default)]
    pub time_integration: TimeIntegrationMethod,

    /// 限制器类型
    #[serde(default)]
    pub limiter: LimiterType,

    // ========== 特性开关 ==========
    
    /// 启用干湿边界处理
    #[serde(default = "default_true")]
    pub wetting_drying: bool,

    /// 启用底摩擦
    #[serde(default)]
    pub friction: bool,

    /// 曼宁系数 [s/m^(1/3)]
    #[serde(default = "default_manning")]
    pub manning_coefficient: f64,

    /// 启用科氏力
    #[serde(default)]
    pub coriolis: bool,

    /// 科氏参数 [1/s]
    #[serde(default)]
    pub coriolis_parameter: f64,

    /// 启用风应力
    #[serde(default)]
    pub wind_forcing: bool,

    /// 风阻系数
    #[serde(default = "default_wind_drag")]
    pub wind_drag_coefficient: f64,

    // ========== 输出选项 ==========
    
    /// 启用详细日志
    #[serde(default)]
    pub verbose: bool,

    /// 每N步输出统计
    #[serde(default = "default_stats_interval")]
    pub stats_interval: usize,
}

// ========== 默认值函数 ==========

fn default_cfl() -> f64 { 0.5 }
fn default_max_velocity() -> f64 { 100.0 }
fn default_h_min() -> f64 { 1e-6 }
fn default_h_dry() -> f64 { 1e-4 }
fn default_gravity() -> f64 { 9.81 }
fn default_true() -> bool { true }
fn default_manning() -> f64 { 0.025 }
fn default_wind_drag() -> f64 { 0.0013 }
fn default_stats_interval() -> usize { 100 }

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            precision: Precision::default(),
            cfl: default_cfl(),
            max_velocity: default_max_velocity(),
            h_min: default_h_min(),
            h_dry: default_h_dry(),
            gravity: default_gravity(),
            riemann_solver: RiemannSolverType::default(),
            time_integration: TimeIntegrationMethod::default(),
            limiter: LimiterType::default(),
            wetting_drying: default_true(),
            friction: false,
            manning_coefficient: default_manning(),
            coriolis: false,
            coriolis_parameter: 0.0,
            wind_forcing: false,
            wind_drag_coefficient: default_wind_drag(),
            verbose: false,
            stats_interval: default_stats_interval(),
        }
    }
}

impl SolverConfig {
    /// 创建默认配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建高精度配置
    pub fn high_precision() -> Self {
        Self {
            precision: Precision::F64,
            cfl: 0.4,
            h_min: 1e-9,
            h_dry: 1e-6,
            ..Default::default()
        }
    }

    /// 创建快速计算配置
    pub fn fast() -> Self {
        Self {
            precision: Precision::F32,
            cfl: 0.8,
            h_min: 1e-4,
            h_dry: 1e-3,
            limiter: LimiterType::None,
            time_integration: TimeIntegrationMethod::ForwardEuler,
            ..Default::default()
        }
    }

    /// 从YAML文件加载
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_yaml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// 保存到YAML文件
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| ConfigError::IoError(e.to_string()))
    }

    /// 从JSON字符串解析
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(json)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// 转换为JSON字符串
    pub fn to_json(&self) -> Result<String, ConfigError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(ConfigError::InvalidValue(
                "cfl".to_string(),
                "必须在 (0, 1] 范围内".to_string(),
            ));
        }

        if self.h_min <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "h_min".to_string(),
                "必须大于 0".to_string(),
            ));
        }

        if self.h_dry <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "h_dry".to_string(),
                "必须大于 0".to_string(),
            ));
        }

        if self.h_min > self.h_dry {
            return Err(ConfigError::InvalidValue(
                "h_min".to_string(),
                "必须小于等于 h_dry".to_string(),
            ));
        }

        if self.gravity <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "gravity".to_string(),
                "必须大于 0".to_string(),
            ));
        }

        if self.friction && self.manning_coefficient <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "manning_coefficient".to_string(),
                "启用摩擦时必须大于 0".to_string(),
            ));
        }

        Ok(())
    }

    /// 根据精度调整容差值
    pub fn adjust_for_precision(&mut self) {
        match self.precision {
            Precision::F32 => {
                // F32需要更宽松的容差
                if self.h_min < 1e-4 {
                    self.h_min = 1e-4;
                }
                if self.h_dry < 1e-3 {
                    self.h_dry = 1e-3;
                }
            }
            Precision::F64 => {
                // F64可以使用更严格的容差
            }
        }
    }
}

/// 配置错误
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// IO错误
    IoError(String),
    /// 解析错误
    ParseError(String),
    /// 序列化错误
    SerializeError(String),
    /// 无效值
    InvalidValue(String, String),
    /// 缺少必需字段
    MissingField(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(msg) => write!(f, "IO错误: {}", msg),
            ConfigError::ParseError(msg) => write!(f, "解析错误: {}", msg),
            ConfigError::SerializeError(msg) => write!(f, "序列化错误: {}", msg),
            ConfigError::InvalidValue(field, msg) => write!(f, "无效值 '{}': {}", field, msg),
            ConfigError::MissingField(field) => write!(f, "缺少必需字段: {}", field),
        }
    }
}

impl std::error::Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SolverConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_precision_config() {
        let config = SolverConfig::high_precision();
        assert_eq!(config.precision, Precision::F64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fast_config() {
        let config = SolverConfig::fast();
        assert_eq!(config.precision, Precision::F32);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_invalid_cfl() {
        let config = SolverConfig {
            cfl: 1.5,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_serde_json() {
        let config = SolverConfig::default();
        let json = config.to_json().unwrap();
        let parsed = SolverConfig::from_json(&json).unwrap();
        assert_eq!(config.cfl, parsed.cfl);
    }
}
