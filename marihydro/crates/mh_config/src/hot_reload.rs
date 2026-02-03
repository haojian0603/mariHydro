// crates/mh_config/src/hot_reload.rs

//! 配置热更新模块
//!
//! 支持运行时配置更新，无需重启求解器。
//!
//! # 功能
//!
//! 1. **文件监控**：使用 notify 监控配置文件变化
//! 2. **增量应用**：只更新白名单字段
//! 3. **回滚保护**：更新失败自动回滚
//! 4. **防抖动**：10秒间隔防止频繁重载
//!
//! # 白名单字段
//!
//! 以下字段允许热更新：
//! - `cfl_number`: CFL 数
//! - `max_velocity`: 最大速度限制
//! - `output_interval`: 输出间隔
//! - `tide_phase_offset`: 潮汐相位偏移
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_config::hot_reload::{ConfigWatcher, HotReloadConfig};
//!
//! let config = HotReloadConfig::default();
//! let mut watcher = ConfigWatcher::new("config.json", config)?;
//!
//! // 在主循环中检查更新
//! if let Some(updates) = watcher.check_updates()? {
//!     solver.apply_updates(&updates);
//! }
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// 热更新错误
#[derive(Error, Debug)]
pub enum HotReloadError {
    /// 文件监控错误
    #[error("文件监控错误: {0}")]
    WatchError(String),
    
    /// 配置解析错误
    #[error("配置解析错误: {0}")]
    ParseError(String),
    
    /// 验证错误
    #[error("配置验证失败: {field} = {value}, 原因: {reason}")]
    ValidationError {
        /// 字段名称
        field: String,
        /// 字段值
        value: String,
        /// 失败原因
        reason: String,
    },
    
    /// 字段不可热更新
    #[error("字段 '{0}' 不支持热更新")]
    FieldNotHotReloadable(String),
    
    /// IO 错误
    #[error("IO 错误: {0}")]
    IoError(#[from] std::io::Error),
}

/// 热更新结果
pub type HotReloadResult<T> = Result<T, HotReloadError>;

/// 热更新配置
#[derive(Debug, Clone)]
pub struct HotReloadConfig {
    /// 防抖动间隔（秒）
    pub debounce_seconds: f64,
    /// 允许热更新的字段
    pub allowed_fields: HashSet<String>,
    /// 是否启用热更新
    pub enabled: bool,
}

impl Default for HotReloadConfig {
    fn default() -> Self {
        let mut allowed = HashSet::new();
        allowed.insert("physics.cfl".to_string());
        allowed.insert("physics.velocity_cap".to_string());
        allowed.insert("output.interval".to_string());
        allowed.insert("physics.h_min".to_string());
        allowed.insert("numerical.timestep_reduction_factor".to_string());
        
        Self {
            debounce_seconds: 10.0,
            allowed_fields: allowed,
            enabled: true,
        }
    }
}

impl HotReloadConfig {
    /// 创建禁用热更新的配置
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
    
    /// 添加允许的字段
    pub fn allow_field(mut self, field: impl Into<String>) -> Self {
        self.allowed_fields.insert(field.into());
        self
    }
    
    /// 设置防抖动间隔
    pub fn debounce(mut self, seconds: f64) -> Self {
        self.debounce_seconds = seconds;
        self
    }
}

/// 配置更新项
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ConfigValue {
    /// 浮点数值
    Float(f64),
    /// 整数值
    Integer(i64),
    /// 布尔值
    Bool(bool),
    /// 字符串值
    String(String),
}

impl ConfigValue {
    /// 尝试转为 f64
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Integer(v) => Some(*v as f64),
            _ => None,
        }
    }
    
    /// 尝试转为 i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Integer(v) => Some(*v),
            Self::Float(v) => Some(*v as i64),
            _ => None,
        }
    }
    
    /// 尝试转为 bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// 配置更新集
#[derive(Debug, Clone, Default)]
pub struct ConfigUpdates {
    /// 更新的字段
    pub updates: HashMap<String, ConfigValue>,
    /// 更新时间戳
    pub timestamp: Option<Instant>,
}

impl ConfigUpdates {
    /// 创建空更新集
    pub fn new() -> Self {
        Self {
            updates: HashMap::new(),
            timestamp: Some(Instant::now()),
        }
    }
    
    /// 添加更新
    pub fn set(&mut self, key: impl Into<String>, value: ConfigValue) {
        self.updates.insert(key.into(), value);
    }
    
    /// 获取 f64 值
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.updates.get(key).and_then(|v| v.as_f64())
    }
    
    /// 获取 bool 值
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.updates.get(key).and_then(|v| v.as_bool())
    }
    
    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.updates.is_empty()
    }
    
    /// 更新数量
    pub fn len(&self) -> usize {
        self.updates.len()
    }
}

/// 配置验证器
pub trait ConfigValidator: Send + Sync {
    /// 验证单个字段
    fn validate(&self, field: &str, value: &ConfigValue) -> HotReloadResult<()>;
}

/// 默认验证器
#[derive(Debug, Default)]
pub struct DefaultValidator {
    /// 字段范围约束 (min, max)
    ranges: HashMap<String, (f64, f64)>,
}

impl DefaultValidator {
    /// 创建默认验证器
    pub fn new() -> Self {
        let mut ranges = HashMap::new();
        ranges.insert("physics.cfl".to_string(), (0.1, 1.0));
        ranges.insert("physics.velocity_cap".to_string(), (1.0, 1000.0));
        ranges.insert("output.interval".to_string(), (0.1, 86400.0));
        ranges.insert("physics.h_min".to_string(), (1e-8, 1.0));
        ranges.insert("numerical.timestep_reduction_factor".to_string(), (0.0, 1.0));
        
        Self { ranges }
    }
    
    /// 添加范围约束
    pub fn with_range(mut self, field: &str, min: f64, max: f64) -> Self {
        self.ranges.insert(field.to_string(), (min, max));
        self
    }
}

impl ConfigValidator for DefaultValidator {
    fn validate(&self, field: &str, value: &ConfigValue) -> HotReloadResult<()> {
        if let Some(&(min, max)) = self.ranges.get(field) {
            if let Some(v) = value.as_f64() {
                if v < min || v > max {
                    return Err(HotReloadError::ValidationError {
                        field: field.to_string(),
                        value: format!("{}", v),
                        reason: format!("值必须在 [{}, {}] 范围内", min, max),
                    });
                }
            }
        }
        Ok(())
    }
}

/// 配置快照（用于回滚）
#[derive(Debug, Clone)]
pub struct ConfigSnapshot {
    /// 快照数据
    pub data: HashMap<String, ConfigValue>,
    /// 创建时间
    pub created_at: Instant,
}

impl ConfigSnapshot {
    /// 创建新快照
    pub fn new(data: HashMap<String, ConfigValue>) -> Self {
        Self {
            data,
            created_at: Instant::now(),
        }
    }
    
    /// 从更新集创建
    pub fn from_updates(updates: &ConfigUpdates) -> Self {
        Self::new(updates.updates.clone())
    }
}

/// 文件变更事件
#[derive(Debug, Clone)]
pub enum FileEvent {
    /// 文件修改
    Modified(PathBuf),
    /// 文件创建
    Created(PathBuf),
    /// 文件删除
    Removed(PathBuf),
    /// 错误
    Error(String),
}

/// 配置文件监控器
///
/// 使用 polling 方式监控配置文件变化（不依赖 notify crate）
pub struct ConfigWatcher {
    /// 配置文件路径
    path: PathBuf,
    /// 热更新配置
    config: HotReloadConfig,
    /// 验证器
    validator: Box<dyn ConfigValidator>,
    /// 上次检查时间
    last_check: Instant,
    /// 上次修改时间
    last_modified: Option<std::time::SystemTime>,
    /// 当前配置快照
    snapshot: Option<ConfigSnapshot>,
}

impl std::fmt::Debug for ConfigWatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConfigWatcher")
            .field("path", &self.path)
            .field("config", &self.config)
            .field("validator", &"<dyn ConfigValidator>")
            .field("last_check", &self.last_check)
            .field("last_modified", &self.last_modified)
            .field("snapshot", &self.snapshot)
            .finish()
    }
}

impl ConfigWatcher {
    /// 创建配置监控器
    pub fn new(
        path: impl AsRef<Path>,
        config: HotReloadConfig,
    ) -> HotReloadResult<Self> {
        let path = path.as_ref().to_path_buf();
        let last_modified = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok());
        
        Ok(Self {
            path,
            config,
            validator: Box::new(DefaultValidator::new()),
            last_check: Instant::now(),
            last_modified,
            snapshot: None,
        })
    }
    
    /// 设置自定义验证器
    pub fn with_validator(mut self, validator: impl ConfigValidator + 'static) -> Self {
        self.validator = Box::new(validator);
        self
    }
    
    /// 检查更新（非阻塞）
    pub fn check_updates(&mut self) -> HotReloadResult<Option<ConfigUpdates>> {
        if !self.config.enabled {
            return Ok(None);
        }

        if !self.path.exists() {
            return Err(HotReloadError::WatchError("配置文件不存在".into()));
        }
        
        // 防抖动
        let now = Instant::now();
        if now.duration_since(self.last_check).as_secs_f64() < self.config.debounce_seconds {
            return Ok(None);
        }
        self.last_check = now;
        
        // 检查文件修改时间
        let current_modified = std::fs::metadata(&self.path)
            .ok()
            .and_then(|m| m.modified().ok());
        
        if current_modified == self.last_modified {
            return Ok(None);
        }
        self.last_modified = current_modified;
        
        // 读取并解析配置
        let content = std::fs::read_to_string(&self.path)?;
        let updates = self.parse_and_filter(&content)?;
        
        if updates.is_empty() {
            return Ok(None);
        }
        
        // 验证更新
        for (field, value) in &updates.updates {
            self.validator.validate(field, value)?;
        }
        
        // 保存快照
        self.snapshot = Some(ConfigSnapshot::from_updates(&updates));
        
        Ok(Some(updates))
    }
    
    fn flatten_json(
        prefix: &str,
        value: &serde_json::Value,
        out: &mut HashMap<String, serde_json::Value>,
    ) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    let key = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    Self::flatten_json(&key, v, out);
                }
            }
            _ => {
                out.insert(prefix.to_string(), value.clone());
            }
        }
    }

    /// 解析并过滤配置
    fn parse_and_filter(&self, content: &str) -> HotReloadResult<ConfigUpdates> {
        let parsed: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| HotReloadError::ParseError(e.to_string()))?;

        let mut flat = HashMap::new();
        Self::flatten_json("", &parsed, &mut flat);
        let mut updates = ConfigUpdates::new();
        
        for (key, value) in flat {
            // 只处理白名单字段或其子字段
            let allowed = self.config.allowed_fields.iter().any(|allowed_key| {
                key == *allowed_key || key.starts_with(&format!("{allowed_key}."))
            });
            if !allowed {
                continue;
            }
            
            let config_value = match value {
                serde_json::Value::Number(n) => {
                    if let Some(f) = n.as_f64() {
                        ConfigValue::Float(f)
                    } else if let Some(i) = n.as_i64() {
                        ConfigValue::Integer(i)
                    } else {
                        continue;
                    }
                }
                serde_json::Value::Bool(b) => ConfigValue::Bool(b),
                serde_json::Value::String(s) => ConfigValue::String(s),
                _ => continue,
            };
            
            updates.set(key, config_value);
        }
        
        Ok(updates)
    }

    /// 应用更新并在失败时回滚
    pub fn apply_updates_with_rollback<T: HotReloadable>(
        &self,
        target: &mut T,
        updates: &ConfigUpdates,
    ) -> HotReloadResult<()> {
        let snapshot = target.snapshot();
        match target.apply_updates(updates) {
            Ok(()) => Ok(()),
            Err(e) => {
                let _ = target.rollback(&snapshot);
                Err(e)
            }
        }
    }
    
    /// 获取上次快照（用于回滚）
    pub fn last_snapshot(&self) -> Option<&ConfigSnapshot> {
        self.snapshot.as_ref()
    }
    
    /// 获取配置文件路径
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// 可热更新的 trait
pub trait HotReloadable {
    /// 应用配置更新
    fn apply_updates(&mut self, updates: &ConfigUpdates) -> HotReloadResult<()>;
    
    /// 获取当前可热更新的配置快照
    fn snapshot(&self) -> ConfigSnapshot;
    
    /// 从快照回滚
    fn rollback(&mut self, snapshot: &ConfigSnapshot) -> HotReloadResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_value() {
        let v = ConfigValue::Float(3.14);
        assert!((v.as_f64().unwrap() - 3.14).abs() < 1e-10);
        
        let v = ConfigValue::Integer(42);
        assert_eq!(v.as_i64().unwrap(), 42);
        assert!((v.as_f64().unwrap() - 42.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_config_updates() {
        let mut updates = ConfigUpdates::new();
        updates.set("cfl_number", ConfigValue::Float(0.5));
        updates.set("output_interval", ConfigValue::Float(3600.0));
        
        assert_eq!(updates.len(), 2);
        assert!(!updates.is_empty());
        assert!((updates.get_f64("cfl_number").unwrap() - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_default_validator() {
        let validator = DefaultValidator::new();
        
        // 有效值
        assert!(validator.validate("physics.cfl", &ConfigValue::Float(0.5)).is_ok());
        
        // 无效值（超出范围）
        assert!(validator.validate("physics.cfl", &ConfigValue::Float(2.0)).is_err());
    }
    
    #[test]
    fn test_hot_reload_config() {
        let config = HotReloadConfig::default();
        assert!(config.enabled);
        assert!(config.allowed_fields.contains("physics.cfl"));
        assert!(!config.allowed_fields.contains("mesh_file")); // 不可热更新
    }
}
