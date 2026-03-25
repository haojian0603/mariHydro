// crates/mh_physics/src/builder/config.rs

//! 求解器配置（无泛型入口层）
//!
//! 提供应用层直接使用的配置结构，所有参数使用 f64 存储，在构建时转换到引擎层泛型类型。
//! 这是架构分层的关键边界：Layer 4（配置层）→ Layer 3（引擎层）。


use serde::{Deserialize, Serialize};
use std::path::Path;
pub use mh_config::Precision;

/// 黎曼求解器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RiemannSolverType {
    /// HLLC 求解器（推荐，鲁棒性与精度平衡）
    #[default]
    Hllc,
    /// Roe 求解器（需熵修正）
    Roe,
    /// Rusanov 求解器（最稳定，耗散性较强）
    Rusanov,
    /// 中心差分格式（仅用于测试）
    Central,
}

/// 时间积分方法枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimeIntegrationMethod {
    /// 前向欧拉格式（一阶精度）
    ForwardEuler,
    /// SSP-RK2 格式（二阶强稳定保持）
    #[default]
    SspRk2,
    /// SSP-RK3 格式（三阶强稳定保持）
    SspRk3,
}

/// 限制器类型枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LimiterType {
    /// 无限制器（退化为一阶精度）
    None,
    /// Minmod 限制器（最鲁棒，耗散性最强）
    #[default]
    Minmod,
    /// Van Leer 限制器（二阶精度）
    VanLeer,
    /// Superbee 限制器（最激进，可能产生振荡）
    Superbee,
    /// MC 限制器（单调中心格式）
    Mc,
}

/// 求解器配置（完全无泛型）
///
/// 这是应用层唯一需要接触的配置类型。所有数值参数使用 f64 存储，
/// 在构建求解器时会根据选择的精度转换为 Layer 3 的泛型类型。
/// 使得 CLI/Editor 层完全无泛型语法，提升易用性。
///
/// # 使用示例
///
/// ```ignore
/// use mh_physics::builder::{SolverConfig, Precision};
///
/// // 创建高精度配置
/// let config = SolverConfig::high_precision();
/// config.save("simulation.yaml")?;
///
/// // 从文件加载
/// let loaded = SolverConfig::load("simulation.yaml")?;
/// let params = NumericalParams::<f64>::from_config(&loaded).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    // ========== 精度与性能参数 ==========

    /// 计算精度选择（运行时决定标量类型）
    #[serde(default)]
    pub precision: Precision,

    /// CFL 数（库朗数，控制时间步稳定性）
    #[serde(default = "default_cfl")]
    pub cfl: f64,

    /// 最大速度限制 [m/s]，防止数值爆炸
    #[serde(default = "default_max_velocity")]
    pub max_velocity: f64,

    // ========== 数值阈值参数 ==========

    /// 最小水深 [m]，用于除法保护
    #[serde(default = "default_h_min")]
    pub h_min: f64,

    /// 干单元水深阈值 [m]
    #[serde(default = "default_h_dry")]
    pub h_dry: f64,

    /// 重力加速度 [m/s²]
    #[serde(default = "default_gravity")]
    pub gravity: f64,

    // ========== 物理模块开关 ==========

    /// 黎曼求解器类型选择
    #[serde(default)]
    pub riemann_solver: RiemannSolverType,

    /// 时间积分方法选择
    #[serde(default)]
    pub time_integration: TimeIntegrationMethod,

    /// 梯度限制器类型
    #[serde(default)]
    pub limiter: LimiterType,

    /// 是否启用干湿边界处理
    #[serde(default = "default_true")]
    pub wetting_drying: bool,

    /// 是否启用底摩擦
    #[serde(default)]
    pub friction: bool,

    /// 曼宁粗糙系数 [s/m^(1/3)]
    #[serde(default = "default_manning")]
    pub manning_coefficient: f64,

    /// 是否启用科里奥利力
    #[serde(default)]
    pub coriolis: bool,

    /// 科里奥利参数 [1/s]（通常由纬度计算）
    #[serde(default)]
    pub coriolis_parameter: f64,

    /// 是否启用风应力
    #[serde(default)]
    pub wind_forcing: bool,

    /// 风拖曳系数
    #[serde(default = "default_wind_drag")]
    pub wind_drag_coefficient: f64,

    // ========== 输出与诊断 ==========

    /// 启用详细日志输出
    #[serde(default)]
    pub verbose: bool,

    /// 统计信息输出间隔步数
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

    /// 创建高精度配置（严格阈值）
    pub fn high_precision() -> Self {
        Self {
            precision: Precision::F64,
            cfl: 0.4,
            h_min: 1e-9,
            h_dry: 1e-6,
            ..Default::default()
        }
    }

    /// 创建快速计算配置（宽松阈值，一阶格式）
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

    /// 从 YAML 文件加载配置
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| ConfigError::IoError(e.to_string()))?;
        serde_yaml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// 保存配置到 YAML 文件
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ConfigError> {
        let content = serde_yaml::to_string(self)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))?;
        std::fs::write(path.as_ref(), content)
            .map_err(|e| ConfigError::IoError(e.to_string()))
    }

    /// 从 JSON 字符串解析配置
    pub fn from_json(json: &str) -> Result<Self, ConfigError> {
        serde_json::from_str(json)
            .map_err(|e| ConfigError::ParseError(e.to_string()))
    }

    /// 将配置序列化为 JSON 字符串
    pub fn to_json(&self) -> Result<String, ConfigError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::SerializeError(e.to_string()))
    }

    /// 验证配置参数的有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        // 验证 CFL 数范围
        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(ConfigError::InvalidValue(
                "cfl".to_string(),
                "必须在 (0, 1] 范围内".to_string(),
            ));
        }

        // 验证水深阈值合理性
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

        // 验证物理常数
        if self.gravity <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "gravity".to_string(),
                "必须大于 0".to_string(),
            ));
        }

        // 验证摩擦系数（如果启用摩擦）
        if self.friction && self.manning_coefficient <= 0.0 {
            return Err(ConfigError::InvalidValue(
                "manning_coefficient".to_string(),
                "启用摩擦时必须大于 0".to_string(),
            ));
        }

        if self.friction && (self.manning_coefficient < 0.001 || self.manning_coefficient > 0.5) {
            return Err(ConfigError::InvalidValue(
                "manning_coefficient".to_string(),
                "超出物理范围 [0.001, 0.5]".to_string(),
            ));
        }

        // 科里奥利参数范围
        if self.coriolis && self.coriolis_parameter.abs() > 1.5e-4 {
            return Err(ConfigError::InvalidValue(
                "coriolis_parameter".to_string(),
                "超出范围 [-1.5e-4, 1.5e-4]".to_string(),
            ));
        }

        // 风拖曳系数范围
        if self.wind_forcing && (self.wind_drag_coefficient < 0.0 || self.wind_drag_coefficient > 0.01) {
            return Err(ConfigError::InvalidValue(
                "wind_drag_coefficient".to_string(),
                "超出范围 [0, 0.01]".to_string(),
            ));
        }

        // 最大速度限制
        if self.max_velocity < 1.0 {
            return Err(ConfigError::InvalidValue(
                "max_velocity".to_string(),
                "必须 >= 1.0 m/s".to_string(),
            ));
        }

        // 精度与阈值匹配检查
        if matches!(self.precision, Precision::F32) && self.h_min < 1e-6 {
            return Err(ConfigError::InvalidValue(
                "h_min".to_string(),
                "F32 精度下 h_min 过小，建议 >= 1e-6".to_string(),
            ));
        }

        // 模块依赖检查
        if matches!(self.riemann_solver, RiemannSolverType::Roe)
            && matches!(self.limiter, LimiterType::None)
        {
            return Err(ConfigError::InvalidValue(
                "limiter".to_string(),
                "Roe 求解器建议启用限制器以保证稳定性".to_string(),
            ));
        }

        Ok(())
    }

    /// 根据精度自动调整容差值
    pub fn adjust_for_precision(&mut self) {
        match self.precision {
            Precision::F32 => {
                // F32 需要更宽松的阈值以避免下溢
                if self.h_min < 1e-4 {
                    self.h_min = 1e-4;
                }
                if self.h_dry < 1e-3 {
                    self.h_dry = 1e-3;
                }
                // F32 相对容差也应适当放宽
                if self.cfl > 0.8 {
                    self.cfl = 0.8;
                }
            }
            Precision::F64 => {
                // F64 可以使用更严格的阈值
                // 保持用户设置不变
            }
        }
    }
}


/// 配置错误类型
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// IO 错误（文件读写失败）
    IoError(String),
    /// 解析错误（格式无效）
    ParseError(String),
    /// 序列化错误（结构不匹配）
    SerializeError(String),
    /// 参数值无效（范围或约束违反）
    InvalidValue(String, String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::IoError(msg) => write!(f, "IO错误: {}", msg),
            ConfigError::ParseError(msg) => write!(f, "解析错误: {}", msg),
            ConfigError::SerializeError(msg) => write!(f, "序列化错误: {}", msg),
            ConfigError::InvalidValue(field, msg) => write!(f, "无效值 '{}': {}", field, msg),
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
        assert_eq!(config.precision, Precision::F64);
    }

    #[test]
    fn test_high_precision_config() {
        let config = SolverConfig::high_precision();
        assert_eq!(config.precision, Precision::F64);
        assert_eq!(config.cfl, 0.4);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_fast_config() {
        let config = SolverConfig::fast();
        assert_eq!(config.precision, Precision::F32);
        assert_eq!(config.limiter, LimiterType::None);
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
    fn test_invalid_h_min() {
        let config = SolverConfig {
            h_min: 1e-3,
            h_dry: 1e-6,
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
        assert_eq!(config.h_min, parsed.h_min);
    }
}
