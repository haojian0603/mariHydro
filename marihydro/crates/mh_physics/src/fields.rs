// marihydro\crates\mh_physics\src\fields.rs
//! 动态字段注册系统
//!
//! 为物理计算提供可扩展的字段管理。
//!
//! # 设计说明
//!
//! 允许运行时动态注册新字段，支持：
//! - 用户自定义湍流模型变量
//! - 插件注入的新物理量
//! - 不同模拟场景的不同字段需求

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 字段类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// 标量场（水深、高程等）
    Scalar,
    /// 向量场 2D（流速等）
    Vector2D,
    /// 向量场 3D
    Vector3D,
}

/// 字段位置
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldLocation {
    /// 单元中心
    Cell,
    /// 面中心
    Face,
    /// 节点
    Node,
}

/// 字段元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMeta {
    /// 字段名称
    pub name: String,
    /// 字段类型
    pub field_type: FieldType,
    /// 存储位置
    pub location: FieldLocation,
    /// 单位
    pub unit: String,
    /// 描述
    pub description: String,
    /// 是否为物理守恒量
    pub is_conserved: bool,
    /// 是否需要边界条件
    pub needs_bc: bool,
}

impl FieldMeta {
    /// 创建单元标量场元数据
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

    /// 创建单元向量场元数据
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
    pub fn conserved(mut self) -> Self {
        self.is_conserved = true;
        self
    }

    /// 标记需要边界条件
    pub fn with_bc(mut self) -> Self {
        self.needs_bc = true;
        self
    }

    /// 添加描述
    pub fn with_desc(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

/// 字段注册表
///
/// 管理所有已注册的物理场。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldRegistry {
    /// 所有字段元数据
    fields: HashMap<String, FieldMeta>,
    /// 字段注册顺序
    order: Vec<String>,
}

impl FieldRegistry {
    /// 创建空注册表
    pub fn new() -> Self {
        Self::default()
    }

    /// 创建带有标准浅水方程字段的注册表
    pub fn shallow_water() -> Self {
        let mut registry = Self::new();
        
        // 水深
        registry.register(
            FieldMeta::cell_scalar("water_depth", "m")
                .conserved()
                .with_bc()
                .with_desc("水深 h")
        );
        
        // 流量
        registry.register(
            FieldMeta::cell_vector2d("discharge", "m²/s")
                .conserved()
                .with_bc()
                .with_desc("单宽流量 (qx, qy)")
        );
        
        // 地形
        registry.register(
            FieldMeta::cell_scalar("bed_elevation", "m")
                .with_desc("河床高程 zb")
        );
        
        // 糙率
        registry.register(
            FieldMeta::cell_scalar("manning_n", "s/m^(1/3)")
                .with_desc("曼宁糙率系数 n")
        );
        
        registry
    }

    /// 注册新字段
    pub fn register(&mut self, meta: FieldMeta) {
        let name = meta.name.clone();
        if !self.fields.contains_key(&name) {
            self.order.push(name.clone());
        }
        self.fields.insert(name, meta);
    }

    /// 获取字段元数据
    pub fn get(&self, name: &str) -> Option<&FieldMeta> {
        self.fields.get(name)
    }

    /// 检查字段是否存在
    pub fn contains(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// 获取所有字段名（按注册顺序）
    pub fn names(&self) -> &[String] {
        &self.order
    }

    /// 获取字段数量
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// 迭代所有守恒量字段
    pub fn conserved_fields(&self) -> impl Iterator<Item = &FieldMeta> {
        self.order.iter()
            .filter_map(|name| self.fields.get(name))
            .filter(|meta| meta.is_conserved)
    }

    /// 迭代需要边界条件的字段
    pub fn bc_fields(&self) -> impl Iterator<Item = &FieldMeta> {
        self.order.iter()
            .filter_map(|name| self.fields.get(name))
            .filter(|meta| meta.needs_bc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_meta() {
        let meta = FieldMeta::cell_scalar("water_depth", "m")
            .conserved()
            .with_bc()
            .with_desc("水深");

        assert_eq!(meta.name, "water_depth");
        assert!(meta.is_conserved);
        assert!(meta.needs_bc);
        assert_eq!(meta.field_type, FieldType::Scalar);
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
}
