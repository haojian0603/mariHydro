// marihydro\crates\mh_physics\src\fields.rs
//! 动态字段注册系统
//!
//! 为物理计算提供可扩展的字段管理。
//!
//! # 设计说明
//!
//! å…è®¸è¿è¡Œæ—¶åŠ¨æ€æ³¨å†Œæ–°å­—æ®µï¼Œæ”¯æŒï¼š
//! - ç”¨æˆ·è‡ªå®šä¹‰æ¹æµæ¨¡åž‹å˜é‡
//! - æ’ä»¶æ³¨å…¥çš„æ–°ç‰©ç†é‡
//! - ä¸åŒæ¨¡æ‹Ÿåœºæ™¯çš„ä¸åŒå­—æ®µéœ€æ±‚

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// å­—æ®µç±»åž‹
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// æ ‡é‡åœºï¼ˆæ°´æ·±ã€é«˜ç¨‹ç­‰ï¼‰
    Scalar,
    /// å‘é‡åœº 2Dï¼ˆæµé€Ÿç­‰ï¼‰
    Vector2D,
    /// å‘é‡åœº 3D
    Vector3D,
}

/// å­—æ®µä½ç½®
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldLocation {
    /// å•å…ƒä¸­å¿ƒ
    Cell,
    /// é¢ä¸­å¿ƒ
    Face,
    /// èŠ‚ç‚¹
    Node,
}

/// å­—æ®µå…ƒæ•°æ®
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldMeta {
    /// å­—æ®µåç§°
    pub name: String,
    /// å­—æ®µç±»åž‹
    pub field_type: FieldType,
    /// å­˜å‚¨ä½ç½®
    pub location: FieldLocation,
    /// å•ä½
    pub unit: String,
    /// æè¿°
    pub description: String,
    /// æ˜¯å¦ä¸ºç‰©ç†å®ˆæ’é‡
    pub is_conserved: bool,
    /// æ˜¯å¦éœ€è¦è¾¹ç•Œæ¡ä»¶
    pub needs_bc: bool,
}

impl FieldMeta {
    /// åˆ›å»ºå•å…ƒæ ‡é‡åœºå…ƒæ•°æ®
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

    /// åˆ›å»ºå•å…ƒå‘é‡åœºå…ƒæ•°æ®
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

    /// æ ‡è®°ä¸ºå®ˆæ’é‡
    pub fn conserved(mut self) -> Self {
        self.is_conserved = true;
        self
    }

    /// æ ‡è®°éœ€è¦è¾¹ç•Œæ¡ä»¶
    pub fn with_bc(mut self) -> Self {
        self.needs_bc = true;
        self
    }

    /// æ·»åŠ æè¿°
    pub fn with_desc(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

/// å­—æ®µæ³¨å†Œè¡¨
///
/// ç®¡ç†æ‰€æœ‰å·²æ³¨å†Œçš„ç‰©ç†åœºã€‚
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldRegistry {
    /// æ‰€æœ‰å­—æ®µå…ƒæ•°æ®
    fields: HashMap<String, FieldMeta>,
    /// å­—æ®µæ³¨å†Œé¡ºåº
    order: Vec<String>,
}

impl FieldRegistry {
    /// åˆ›å»ºç©ºæ³¨å†Œè¡¨
    pub fn new() -> Self {
        Self::default()
    }

    /// åˆ›å»ºå¸¦æœ‰æ ‡å‡†æµ…æ°´æ–¹ç¨‹å­—æ®µçš„æ³¨å†Œè¡¨
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
