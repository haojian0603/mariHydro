// crates/mh_config/src/solver_config.rs

//! SolverConfig - 求解器配置（全 f64）
//!
//! 定义求解器的所有配置参数，使用纯 f64 类型，
//! 在构建求解器时根据 Precision 转换到相应精度。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::precision::Precision;
use crate::error::ConfigError;

/// 求解器配置（全 f64，Layer 4 唯一真理源）
///
/// 包含所有求解器参数，使用 f64 存储以便 JSON 序列化。
/// 在构建求解器时，根据 `precision` 字段转换到 f32 或 f64。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// 计算精度
    #[serde(default)]
    pub precision: Precision,
    
    /// 物理参数
    #[serde(default)]
    pub physics: PhysicsConfig,
    
    /// 网格配置
    #[serde(default)]
    pub mesh: MeshConfig,
    
    /// 输出配置
    #[serde(default)]
    pub output: OutputConfig,
    
    /// 最大迭代次数
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    
    /// 最大模拟时间 [s]
    #[serde(default = "default_max_time")]
    pub max_time: f64,

    /// 数值格式配置
    #[serde(default)]
    pub numerical: NumericalConfig,

    /// 时间积分配置
    #[serde(default)]
    pub time: TimeConfig,

    /// 并行化配置
    #[serde(default)]
    pub parallel: ParallelConfig,
}

fn default_max_iterations() -> usize { 100000 }
fn default_max_time() -> f64 { 3600.0 }

/// 物理参数配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// 重力加速度 [m/s²]
    #[serde(default = "default_gravity")]
    pub gravity: f64,
    
    /// CFL 数
    #[serde(default = "default_cfl")]
    pub cfl: f64,
    
    /// 干单元水深阈值 [m]
    #[serde(default = "default_h_dry")]
    pub h_dry: f64,
    
    /// 最小水深 [m]
    #[serde(default = "default_h_min")]
    pub h_min: f64,
    
    /// 收敛容差
    #[serde(default = "default_convergence")]
    pub convergence: f64,
    
    /// Manning 糙率系数
    #[serde(default = "default_manning")]
    pub manning_n: f64,
    
    /// 最大速度限制 [m/s]
    #[serde(default = "default_velocity_cap")]
    pub velocity_cap: f64,
    
    /// 最小波速阈值 [m/s]
    #[serde(default = "default_min_wave_speed")]
    pub min_wave_speed: f64,
    
    /// 通量计算零阈值
    #[serde(default = "default_flux_eps")]
    pub flux_eps: f64,
}

fn default_gravity() -> f64 { 9.81 }
fn default_cfl() -> f64 { 0.9 }
fn default_h_dry() -> f64 { 1e-3 }
fn default_h_min() -> f64 { 1e-9 }
fn default_convergence() -> f64 { 1e-8 }
fn default_manning() -> f64 { 0.03 }
fn default_velocity_cap() -> f64 { 100.0 }
fn default_min_wave_speed() -> f64 { 1e-6 }
fn default_flux_eps() -> f64 { 1e-14 }

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            gravity: default_gravity(),
            cfl: default_cfl(),
            h_dry: default_h_dry(),
            h_min: default_h_min(),
            convergence: default_convergence(),
            manning_n: default_manning(),
            velocity_cap: default_velocity_cap(),
            min_wave_speed: default_min_wave_speed(),
            flux_eps: default_flux_eps(),
        }
    }
}

/// 数值格式配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalConfig {
    /// 黎曼求解器类型
    #[serde(default)]
    pub riemann_solver: RiemannSolverType,
    
    /// 时间积分方法
    #[serde(default)]
    pub time_integration: TimeIntegrationMethod,
    
    /// 梯度限制器类型
    #[serde(default)]
    pub limiter: LimiterType,
    
    /// 是否启用二阶精度
    #[serde(default = "default_true")]
    pub second_order: bool,
    
    /// 是否启用静水重构
    #[serde(default = "default_true")]
    pub use_hydrostatic_reconstruction: bool,
    
    /// 是否启用干湿处理
    #[serde(default = "default_true")]
    pub wetting_drying: bool,
    
    /// 是否启用底摩擦
    #[serde(default = "default_false")]
    pub friction: bool,
    
    /// 是否启用科氏力
    #[serde(default = "default_false")]
    pub coriolis: bool,
    
    /// 是否启用风应力
    #[serde(default = "default_false")]
    pub wind_forcing: bool,
    
    /// 最大回退次数
    #[serde(default = "default_max_fallback_attempts")]
    pub max_fallback_attempts: u32,
    
    /// 时间步减小因子（回退时使用）
    #[serde(default = "default_timestep_reduction_factor")]
    pub timestep_reduction_factor: f64,
}

fn default_true() -> bool { true }
fn default_false() -> bool { false }
fn default_max_fallback_attempts() -> u32 { 3 }
fn default_timestep_reduction_factor() -> f64 { 0.5 }

impl Default for NumericalConfig {
    fn default() -> Self {
        Self {
            riemann_solver: RiemannSolverType::default(),
            time_integration: TimeIntegrationMethod::default(),
            limiter: LimiterType::default(),
            second_order: true,
            use_hydrostatic_reconstruction: true,
            wetting_drying: true,
            friction: false,
            coriolis: false,
            wind_forcing: false,
            max_fallback_attempts: default_max_fallback_attempts(),
            timestep_reduction_factor: default_timestep_reduction_factor(),
        }
    }
}

/// 时间积分配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeConfig {
    /// 初始时间步长 [s]
    #[serde(default = "default_initial_dt")]
    pub initial_dt: f64,
    
    /// 最小时间步长 [s]
    #[serde(default = "default_min_dt")]
    pub min_dt: f64,
    
    /// 最大时间步长 [s]
    #[serde(default = "default_max_dt")]
    pub max_dt: f64,
    
    /// 自适应增长因子
    #[serde(default = "default_growth_factor")]
    pub growth_factor: f64,
    
    /// 自适应收缩因子
    #[serde(default = "default_shrink_factor")]
    pub shrink_factor: f64,
    
    /// 稳定增长阈值（步数）
    #[serde(default = "default_stable_threshold")]
    pub stable_threshold: usize,
}

fn default_initial_dt() -> f64 { 0.01 }
fn default_min_dt() -> f64 { 1e-6 }
fn default_max_dt() -> f64 { 1.0 }
fn default_growth_factor() -> f64 { 1.1 }
fn default_shrink_factor() -> f64 { 0.5 }
fn default_stable_threshold() -> usize { 10 }

impl Default for TimeConfig {
    fn default() -> Self {
        Self {
            initial_dt: default_initial_dt(),
            min_dt: default_min_dt(),
            max_dt: default_max_dt(),
            growth_factor: default_growth_factor(),
            shrink_factor: default_shrink_factor(),
            stable_threshold: default_stable_threshold(),
        }
    }
}

/// 并行化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// 并行化阈值（单元数量）
    #[serde(default = "default_parallel_threshold")]
    pub threshold: usize,
    
    /// 是否启用多线程
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_parallel_threshold() -> usize { 1000 }

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            threshold: default_parallel_threshold(),
            enabled: true,
        }
    }
}

/// 黎曼求解器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
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

/// 网格配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    /// 网格文件路径
    #[serde(default)]
    pub file: PathBuf,
    
    /// 最大单元数限制
    #[serde(default)]
    pub max_cells: Option<usize>,
    
    /// 是否使用自适应网格
    #[serde(default)]
    pub adaptive: bool,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            file: PathBuf::from("mesh.msh"),
            max_cells: None,
            adaptive: false,
        }
    }
}

/// 输出配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// 输出目录
    #[serde(default = "default_output_dir")]
    pub directory: PathBuf,
    
    /// 输出间隔 [s]
    #[serde(default = "default_output_interval")]
    pub interval: f64,
    
    /// 输出格式
    #[serde(default)]
    pub format: OutputFormat,
}

fn default_output_dir() -> PathBuf { PathBuf::from("output") }
fn default_output_interval() -> f64 { 1.0 }

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            directory: default_output_dir(),
            interval: default_output_interval(),
            format: OutputFormat::default(),
        }
    }
}

/// 输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    /// VTK 格式
    #[default]
    Vtk,
    /// CSV 格式
    Csv,
    /// 二进制格式
    Binary,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            precision: Precision::default(),
            physics: PhysicsConfig::default(),
            mesh: MeshConfig::default(),
            output: OutputConfig::default(),
            max_iterations: default_max_iterations(),
            max_time: default_max_time(),
            numerical: NumericalConfig::default(),
            time: TimeConfig::default(),
            parallel: ParallelConfig::default(),
        }
    }
}

impl SolverConfig {
    /// 从文件加载配置
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(ConfigError::Io)?;
        
        let config: SolverConfig = serde_json::from_str(&content)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        
        config.validate()?;
        Ok(config)
    }

    /// 验证配置有效性
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_iterations == 0 {
            return Err(ConfigError::InvalidValue {
                key: "max_iterations".into(),
                value: self.max_iterations.to_string(),
                reason: "必须为正数".into(),
            });
        }

        if self.max_time <= 0.0 || !self.max_time.is_finite() {
            return Err(ConfigError::InvalidValue {
                key: "max_time".into(),
                value: self.max_time.to_string(),
                reason: "必须为正的有限值".into(),
            });
        }

        if self.time.min_dt <= 0.0 || !self.time.min_dt.is_finite() {
            return Err(ConfigError::InvalidValue {
                key: "time.min_dt".into(),
                value: self.time.min_dt.to_string(),
                reason: "必须为正".into(),
            });
        }
        if self.time.max_dt < self.time.min_dt {
            return Err(ConfigError::InvalidValue {
                key: "time.max_dt".into(),
                value: self.time.max_dt.to_string(),
                reason: "必须 >= min_dt".into(),
            });
        }
        if !(self.time.initial_dt >= self.time.min_dt && self.time.initial_dt <= self.time.max_dt) {
            return Err(ConfigError::InvalidValue {
                key: "time.initial_dt".into(),
                value: self.time.initial_dt.to_string(),
                reason: "必须在[min_dt, max_dt]".into(),
            });
        }
        if self.time.growth_factor <= 1.0 {
            return Err(ConfigError::InvalidValue {
                key: "time.growth_factor".into(),
                value: self.time.growth_factor.to_string(),
                reason: "必须 > 1".into(),
            });
        }
        if self.time.shrink_factor <= 0.0 || self.time.shrink_factor >= 1.0 {
            return Err(ConfigError::InvalidValue {
                key: "time.shrink_factor".into(),
                value: self.time.shrink_factor.to_string(),
                reason: "必须在(0,1)".into(),
            });
        }

        if self.output.interval <= 0.0 || !self.output.interval.is_finite() {
            return Err(ConfigError::InvalidValue {
                key: "output.interval".into(),
                value: self.output.interval.to_string(),
                reason: "必须为正的有限值".into(),
            });
        }

        if self.mesh.file.as_os_str().is_empty() {
            return Err(ConfigError::Missing("mesh.file".into()));
        }

        // CFL 验证
        if self.physics.cfl <= 0.0 || self.physics.cfl > 2.0 {
            return Err(ConfigError::InvalidValue {
                key: "physics.cfl".to_string(),
                value: self.physics.cfl.to_string(),
                reason: "CFL 必须在 (0, 2] 范围内".to_string(),
            });
        }
        
        // 水深阈值验证
        if self.physics.h_dry < 0.0 {
            return Err(ConfigError::InvalidValue {
                key: "physics.h_dry".to_string(),
                value: self.physics.h_dry.to_string(),
                reason: "h_dry 不能为负".to_string(),
            });
        }
        
        if self.physics.h_min < 0.0 {
            return Err(ConfigError::InvalidValue {
                key: "physics.h_min".to_string(),
                value: self.physics.h_min.to_string(),
                reason: "h_min 不能为负".to_string(),
            });
        }
        
        // 数值参数层级验证
        if self.physics.h_min > self.physics.h_dry {
            return Err(ConfigError::InvalidValue {
                key: "physics.h_min".to_string(),
                value: self.physics.h_min.to_string(),
                reason: "h_min 必须小于 h_dry".to_string(),
            });
        }
        
        // 重力验证
        if self.physics.gravity <= 0.0 {
            return Err(ConfigError::InvalidValue {
                key: "physics.gravity".to_string(),
                value: self.physics.gravity.to_string(),
                reason: "重力必须为正".to_string(),
            });
        }
        
        // 时间步长因子验证
        if self.numerical.timestep_reduction_factor <= 0.0 || 
           self.numerical.timestep_reduction_factor > 1.0 {
            return Err(ConfigError::InvalidValue {
                key: "numerical.timestep_reduction_factor".to_string(),
                value: self.numerical.timestep_reduction_factor.to_string(),
                reason: "时间步减小因子必须在 (0, 1] 范围内".to_string(),
            });
        }
        
        Ok(())
    }

    /// 保存配置到文件
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), ConfigError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| ConfigError::Parse(e.to_string()))?;
        std::fs::write(path, content).map_err(ConfigError::Io)?;
        Ok(())
    }
}

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
    fn test_invalid_cfl() {
        let mut config = SolverConfig::default();
        config.physics.cfl = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_threshold_hierarchy() {
        let mut config = SolverConfig::default();
        config.physics.h_min = 1e-3;  // > h_dry
        config.physics.h_dry = 1e-6;  // < h_min
        assert!(config.validate().is_err());
    }
}
