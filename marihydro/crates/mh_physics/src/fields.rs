// crates/mh_physics/src/fields.rs

//! 动态字段注册系统
//!
//! 提供物理计算字段的元数据管理和注册机制。
//!
//! # 设计原则
//!
//! 1. **Layer 4 职责**：本模块属于应用层配置，保持无泛型设计
//! 2. **运行时注册**：支持插件动态注入新物理量
//! 3. **元数据驱动**：字段携带类型、单位、守恒性等语义信息
//! 4. **线程安全**：明确标记为 `Send + Sync`，支持多线程访问
//! 5. **名称验证**：强制 snake_case 命名规范，防止拼写错误
//!
//! # 使用场景
//!
//! - 标准浅水方程字段（水深、动量、地形）
//! - 示踪剂（温度、盐度、污染物）
//! - 用户自定义湍流模型变量
//! - 多相流扩展字段
//!
//! # 示例
//!
//! ```rust
//! use mh_physics::fields::{FieldRegistry, FieldMeta, FieldType, FieldLocation};
//!
//! // 创建标准浅水字段
//! let registry = FieldRegistry::shallow_water();
//!
//! // 动态注册新字段
//! registry.register(
//!     FieldMeta::cell_scalar("temperature", "°C")
//!         .with_desc("水温")
//! ).unwrap(); // 注册成功返回Ok(())
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 字段类型枚举
///
/// 定义物理场的数据结构类型，用于序列化和类型检查。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// 标量场（如：水深、温度）
    Scalar,
    /// 二维向量场（如：流速、流量）
    Vector2D,
    /// 三维向量场（如：含垂向速度的全三维流场）
    Vector3D,
}

/// 字段存储位置
///
/// 标识数据存储在网格的哪个拓扑元素上。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldLocation {
    /// 单元中心（有限体积法主变量）
    Cell,
    /// 面中心（通量、边界条件）
    Face,
    /// 网格节点（连续Galerkin方法）
    Node,
}

impl Default for FieldLocation {
    fn default() -> Self {
        FieldLocation::Cell
    }
}

/// 注册错误类型
///
/// 字段注册失败的具体原因。
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FieldError {
    /// 无效字段名（非 snake_case 或包含非法字符）
    #[error("Invalid field name '{0}': must be snake_case (a-z, 0-9, _)")]
    InvalidName(String),

    /// 试图覆盖保留字段
    #[error("Cannot override reserved field '{0}'")]
    ReservedField(String),

    /// 字段已存在（大小写敏感）
    #[error("Field '{0}' already exists")]
    DuplicateField(String),
}

/// 保留字段集合（禁止覆盖）
///
/// 浅水方程核心字段，覆盖可能导致求解器崩溃。
const RESERVED_FIELDS: &[&str] = &[
    "water_depth",
    "discharge",
    "bed_elevation",
    "manning_n",
];

/// 验证字段名是否符合 snake_case 规范
///
/// # 规则
/// - 仅允许小写字母 a-z、数字 0-9 和下划线 `_`
/// - 必须以字母开头
/// - 不能以下划线开头或结尾
/// - 不能包含连续下划线
fn is_valid_field_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let mut prev_char = '_'; // 初始值确保不会误判连续下划线

    for (i, ch) in name.chars().enumerate() {
        match ch {
            'a'..='z' | '0'..='9' => {
                // 合法字符
            }
            '_' => {
                // 检查连续下划线
                if prev_char == '_' {
                    return false;
                }
                // 不能以下划线开头
                if i == 0 {
                    return false;
                }
            }
            _ => return false, // 非法字符（大写字母、空格、特殊符号等）
        }
        prev_char = ch;
    }

    // 不能以下划线结尾
    !name.ends_with('_')
}

/// 检查是否为保留字段
fn is_reserved_field(name: &str) -> bool {
    RESERVED_FIELDS.contains(&name)
}

/// 字段元数据
///
/// 包含字段的全部语义信息，用于文档生成、单位检查和边界条件配置。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMeta {
    /// 字段名称（唯一标识符）
    pub name: String,
    /// 数据类型
    pub field_type: FieldType,
    /// 存储位置
    #[serde(default)]
    pub location: FieldLocation,
    /// 物理单位（如：m/s, kg/m³）
    pub unit: String,
    /// 描述文本
    #[serde(default)]
    pub description: String,
    /// 是否为守恒量（质量、动量、能量）
    #[serde(default)]
    pub is_conserved: bool,
    /// 是否需要边界条件
    #[serde(default)]
    pub needs_bc: bool,
}

impl FieldMeta {
    /// 创建单元标量场元数据
    ///
    /// # 参数
    /// - `name`: 字段名称（如"water_depth"）
    /// - `unit`: 物理单位（如"m"）
    ///
    /// # 返回
    /// 返回 FieldMeta 实例，默认 is_conserved=false, needs_bc=false
    #[inline]
    pub fn cell_scalar(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Scalar,
            location: FieldLocation::Cell,
            unit: unit.into(),
            description: String::new(),
            is_conserved: false,
            needs_bc: false,
        }
    }

    /// 创建单元二维向量场元数据
    ///
    /// # 参数
    /// - `name`: 字段名称（如"velocity"）
    /// - `unit`: 物理单位（如"m/s"）
    #[inline]
    pub fn cell_vector2d(name: impl Into<String>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Vector2D,
            location: FieldLocation::Cell,
            unit: unit.into(),
            description: String::new(),
            is_conserved: false,
            needs_bc: false,
        }
    }

    /// 标记为守恒量
    ///
    /// 守恒量会参与质量/动量/能量守恒检查。
    #[inline]
    pub fn conserved(mut self) -> Self {
        self.is_conserved = true;
        self
    }

    /// 标记需要边界条件
    ///
    /// 需要边界条件的字段会在求解器初始化时检查 BC 配置。
    #[inline]
    pub fn with_bc(mut self) -> Self {
        self.needs_bc = true;
        self
    }

    /// 添加描述文本
    #[inline]
    pub fn with_desc(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

/// 字段注册表
///
/// 管理所有物理场的元数据，支持运行时动态注册新字段。
/// 注册表本身不存储数据，只存储元数据用于指导和验证。
///
/// # 线程安全
///
/// 本类型实现 `Send + Sync`，可在多线程环境中安全共享。
/// 内部使用不可变数据结构，注册操作通过 `&mut self` 保证互斥。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldRegistry {
    /// 字段元数据映射（名称 -> 元数据）
    fields: HashMap<String, FieldMeta>,
    /// 注册顺序（保证迭代一致性）
    order: Vec<String>,
}

// 明确标记为线程安全
unsafe impl Send for FieldRegistry {}
unsafe impl Sync for FieldRegistry {}

impl FieldRegistry {
    /// 创建空注册表
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建标准浅水方程字段注册表
    ///
    /// 自动注册以下字段：
    /// - `water_depth`: 水深 [m]（守恒量，需 BC）
    /// - `discharge`: 单宽流量 [m²/s]（守恒量，需 BC）
    /// - `bed_elevation`: 河床高程 [m]
    /// - `manning_n`: 曼宁糙率系数 [s/m^(1/3)]
    ///
    /// # 扩展
    /// 可通过 `register()` 方法追加示踪剂、湍流变量等。
    #[inline]
    pub fn shallow_water() -> Self {
        let mut registry = Self::new();
        
        // 水深 - 守恒量，需要边界条件
        registry.register(
            FieldMeta::cell_scalar("water_depth", "m")
                .conserved()
                .with_bc()
                .with_desc("水深 h")
        ).unwrap();
        
        // 流量（动量） - 守恒量，需要边界条件
        registry.register(
            FieldMeta::cell_vector2d("discharge", "m²/s")
                .conserved()
                .with_bc()
                .with_desc("单宽流量 (qx, qy)")
        ).unwrap();
        
        // 地形
        registry.register(
            FieldMeta::cell_scalar("bed_elevation", "m")
                .with_desc("河床高程 zb")
        ).unwrap();
        
        // 糙率
        registry.register(
            FieldMeta::cell_scalar("manning_n", "s/m^(1/3)")
                .with_desc("曼宁糙率系数 n")
        ).unwrap();
        
        registry
    }

    /// 注册新字段
    ///
    /// # 参数
    /// - `meta`: 字段元数据
    ///
    /// # 返回
    /// - `Ok(())`: 注册成功
    /// - `Err(FieldError)`: 验证失败（命名非法、保留字段冲突等）
    ///
    /// # 验证规则
    /// 1. 字段名必须符合 snake_case 规范
    /// 2. 不能覆盖保留字段（除非保留字段不存在）
    /// 3. 重复注册时更新元数据但保留原顺序
    #[inline]
    pub fn register(&mut self, meta: FieldMeta) -> Result<(), FieldError> {
        let name = meta.name.clone();
        
        // 验证 1：命名规范
        if !is_valid_field_name(&name) {
            return Err(FieldError::InvalidName(name));
        }

        // 验证 2：保留字段冲突
        if is_reserved_field(&name) && self.fields.contains_key(&name) {
            return Err(FieldError::ReservedField(name));
        }

        // 验证 3：重复注册
        if self.fields.contains_key(&name) {
            // 更新元数据但保留原顺序
            *self.fields.get_mut(&name).unwrap() = meta;
            return Ok(());
        }

        // 新字段：追加到注册顺序
        self.order.push(name.clone());
        self.fields.insert(name, meta);
        Ok(())
    }

    /// 批量注册字段
    ///
    /// # 参数
    /// - `fields`: 字段元数据数组
    ///
    /// # 返回
    /// 返回第一个失败的错误，或 Ok(())
    #[inline]
    pub fn register_batch(&mut self, fields: &[FieldMeta]) -> Result<(), FieldError> {
        for meta in fields {
            self.register(meta.clone())?;
        }
        Ok(())
    }

    /// 获取字段元数据
    ///
    /// # 参数
    /// - `name`: 字段名称
    ///
    /// # 返回
    /// - `Some(&FieldMeta)`: 字段存在
    /// - `None`: 字段未注册
    #[inline]
    pub fn get(&self, name: &str) -> Option<&FieldMeta> {
        self.fields.get(name)
    }

    /// 检查字段是否存在
    #[inline]
    pub fn contains(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// 获取所有字段名（按注册顺序）
    ///
    /// # 返回
    /// 返回字段名称切片，顺序与注册顺序一致。
    #[inline]
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// 获取字段数量
    #[inline]
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// 检查注册表是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// 迭代所有守恒量字段
    ///
    /// # 返回
    /// 返回守恒量字段元数据迭代器。
    #[inline]
    pub fn conserved_fields(&self) -> impl Iterator<Item = &FieldMeta> {
        self.order.iter()
            .filter_map(|name| self.fields.get(name))
            .filter(|meta| meta.is_conserved)
    }

    /// 迭代需要边界条件的字段
    #[inline]
    pub fn bc_fields(&self) -> impl Iterator<Item = &FieldMeta> {
        self.order.iter()
            .filter_map(|name| self.fields.get(name))
            .filter(|meta| meta.needs_bc)
    }

    /// 验证注册表完整性
    ///
    /// 检查所有必需字段是否存在。
    ///
    /// # 参数
    /// - `required_fields`: 必需字段名称列表
    ///
    /// # 返回
    /// - `Ok(())`: 所有必需字段都存在
    /// - `Err(missing_fields)`: 返回缺失的字段名列表
    #[inline]
    pub fn validate_required<'a>(&self, required_fields: &[&'a str]) -> Result<(), Vec<&'a str>> {
        let missing: Vec<_> = required_fields
            .iter()
            .filter(|&&name| !self.contains(name))
            .copied()
            .collect();
        
        if missing.is_empty() {
            Ok(())
        } else {
            Err(missing)
        }
    }
}

// ============================================================================
// 测试模块
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_name_validation() {
        // 有效名称
        assert!(is_valid_field_name("water_depth"));
        assert!(is_valid_field_name("temperature_2d"));
        assert!(is_valid_field_name("field123"));
        
        // 无效名称
        assert!(!is_valid_field_name("")); // 空
        assert!(!is_valid_field_name("_temp")); // 下划线开头
        assert!(!is_valid_field_name("temp_")); // 下划线结尾
        assert!(!is_valid_field_name("water depth")); // 空格
        assert!(!is_valid_field_name("WaterDepth")); // 大写字母
        assert!(!is_valid_field_name("temp__value")); // 连续下划线
        assert!(!is_valid_field_name("temp-value")); // 连字符
        assert!(!is_valid_field_name("temperature°")); // 特殊字符
    }

    #[test]
    fn test_field_meta_creation() {
        let meta = FieldMeta::cell_scalar("water_depth", "m")
            .conserved()
            .with_bc()
            .with_desc("水深");

        assert_eq!(meta.name, "water_depth");
        assert!(meta.is_conserved);
        assert!(meta.needs_bc);
        assert_eq!(meta.field_type, FieldType::Scalar);
        assert_eq!(meta.location, FieldLocation::Cell);
    }

    #[test]
    fn test_registry_shallow_water() {
        let registry = FieldRegistry::shallow_water();

        assert!(registry.contains("water_depth"));
        assert!(registry.contains("discharge"));
        assert!(registry.contains("bed_elevation"));
        assert!(registry.contains("manning_n"));
        assert_eq!(registry.len(), 4);
    }

    #[test]
    fn test_conserved_fields() {
        let registry = FieldRegistry::shallow_water();
        let conserved: Vec<_> = registry.conserved_fields().collect();
        assert_eq!(conserved.len(), 2); // water_depth and discharge
    }

    #[test]
    fn test_bc_fields() {
        let registry = FieldRegistry::shallow_water();
        let bc_fields: Vec<_> = registry.bc_fields().collect();
        assert_eq!(bc_fields.len(), 2); // water_depth and discharge
    }

    #[test]
    fn test_register_new_field() {
        let mut registry = FieldRegistry::new();
        
        // 成功注册
        assert!(registry.register(
            FieldMeta::cell_scalar("temperature", "°C")
                .with_desc("水温")
        ).is_ok());
        
        assert!(registry.contains("temperature"));
        assert_eq!(registry.len(), 1);
        
        // 注册第二个字段
        assert!(registry.register(
            FieldMeta::cell_scalar("salinity", "‰")
                .with_desc("盐度")
        ).is_ok());
        
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.names(), &["temperature", "salinity"]);
    }

    #[test]
    fn test_update_existing_field() {
        let mut registry = FieldRegistry::new();
        
        // 初始注册
        assert!(registry.register(
            FieldMeta::cell_scalar("temperature", "°C")
        ).is_ok());
        
        // 更新已存在字段（保留原顺序）
        assert!(registry.register(
            FieldMeta::cell_scalar("temperature", "K") // 修改单位
                .with_desc("绝对温度")
        ).is_ok());
        
        assert_eq!(registry.len(), 1);
        let meta = registry.get("temperature").unwrap();
        assert_eq!(meta.unit, "K");
        assert_eq!(meta.description, "绝对温度");
        assert_eq!(registry.names(), &["temperature"]); // 顺序未变
    }

    #[test]
    fn test_register_invalid_name() {
        let mut registry = FieldRegistry::new();
        
        // 大写字母
        let result = registry.register(FieldMeta::cell_scalar("Temp", "K"));
        assert!(matches!(result, Err(FieldError::InvalidName(_))));
        
        // 空格
        let result = registry.register(FieldMeta::cell_scalar("temp value", "K"));
        assert!(matches!(result, Err(FieldError::InvalidName(_))));
        
        // 特殊字符
        let result = registry.register(FieldMeta::cell_scalar("temp-value", "K"));
        assert!(matches!(result, Err(FieldError::InvalidName(_))));
    }

    #[test]
    fn test_register_reserved_field() {
        let mut registry = FieldRegistry::shallow_water();
        
        // 试图覆盖保留字段
        let result = registry.register(
            FieldMeta::cell_scalar("water_depth", "ft")
        );
        assert!(matches!(result, Err(FieldError::ReservedField(_))));
        
        // 保留字段数量未变
        assert_eq!(registry.len(), 4);
    }

    #[test]
    fn test_validate_required_fields() {
        let registry = FieldRegistry::shallow_water();
        
        // 所有必需字段都存在
        assert!(registry.validate_required(&[
            "water_depth",
            "discharge",
            "bed_elevation",
            "manning_n",
        ]).is_ok());
        
        // 缺少必需字段
        let result = registry.validate_required(&[
            "water_depth",
            "missing_field",
        ]);
        assert!(matches!(result, Err(missing) if missing == vec!["missing_field"]));
    }

    #[test]
    fn test_thread_safety() {
        // 验证 Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FieldRegistry>();
    }

    #[test]
    fn test_serde_compatibility() {
        // 序列化
        let registry = FieldRegistry::shallow_water();
        let json = serde_json::to_string(&registry).expect("序列化失败");
        
        // 反序列化
        let deserialized: FieldRegistry = serde_json::from_str(&json)
            .expect("反序列化失败");
        
        // 验证字段存在
        assert_eq!(registry.len(), deserialized.len());
        assert!(deserialized.contains("water_depth"));
        assert_eq!(registry.names(), deserialized.names());
    }

    #[test]
    fn test_clone_independence() {
        let mut registry1 = FieldRegistry::shallow_water();
        let mut registry2 = registry1.clone();
        
        // 修改 clone 不影响原注册表
        registry2.register(
            FieldMeta::cell_scalar("temperature", "K")
        ).unwrap();
        
        assert_eq!(registry1.len(), 4); // 原表未变
        assert_eq!(registry2.len(), 5); // 新表增加
        assert!(!registry1.contains("temperature"));
        assert!(registry2.contains("temperature"));
    }

    #[test]
    fn test_empty_registry() {
        let registry = FieldRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.names().len(), 0);
    }

    #[test]
    fn test_batch_register() {
        let mut registry = FieldRegistry::new();
        
        let fields = vec![
            FieldMeta::cell_scalar("temp", "K"),
            FieldMeta::cell_scalar("salinity", "psu"),
            FieldMeta::cell_scalar("sediment", "kg/m3"),
        ];
        
        // 批量注册成功
        assert!(registry.register_batch(&fields).is_ok());
        assert_eq!(registry.len(), 3);
        
        // 批量注册失败（第一个失败后停止）
        let invalid_fields = vec![
            FieldMeta::cell_scalar("valid_field", "unit"),
            FieldMeta::cell_scalar("InvalidField", "unit"), // 名称无效
        ];
        let result = registry.register_batch(&invalid_fields);
        assert!(matches!(result, Err(FieldError::InvalidName(_))));
        assert_eq!(registry.len(), 3); // 未添加任何字段
    }
}