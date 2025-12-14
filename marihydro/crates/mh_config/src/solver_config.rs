// crates/mh_config/src/solver_config.rs

//! SolverConfig - 求解器配置（全 f64）
//!
//! 定义求解器的所有配置参数，使用纯 f64 类型，
//! 在构建求解器时根据 Precision 转换到相应精度。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::precision::Precision;
use crate::error::ConfigError;

/// 求解器配置（全 f64）
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
}

fn default_gravity() -> f64 { 9.81 }
fn default_cfl() -> f64 { 0.9 }
fn default_h_dry() -> f64 { 1e-3 }
fn default_h_min() -> f64 { 1e-9 }
fn default_convergence() -> f64 { 1e-8 }
fn default_manning() -> f64 { 0.03 }
fn default_velocity_cap() -> f64 { 100.0 }

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
        }
    }
}

/// 网格配置
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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
        
        // 重力验证
        if self.physics.gravity <= 0.0 {
            return Err(ConfigError::InvalidValue {
                key: "physics.gravity".to_string(),
                value: self.physics.gravity.to_string(),
                reason: "重力必须为正".to_string(),
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
    fn test_serialize_deserialize() {
        let config = SolverConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: SolverConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.precision, config.precision);
    }
}
