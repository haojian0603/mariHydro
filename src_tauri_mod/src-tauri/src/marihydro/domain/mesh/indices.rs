// src-tauri/src/marihydro/domain/mesh/indices.rs

//! 网格索引类型
//!
//! **已废弃**: 此模块已迁移至 `core/types/indices.rs`。
//! 请使用 `crate::core::types::{CellIndex, FaceIndex, NodeIndex, CellId, FaceId, NodeId}` 替代。
//!
//! 为保持向后兼容性，本模块仍然可用，但建议迁移到新位置。

#![deprecated(
    since = "0.3.0",
    note = "此模块已迁移至 core/types/indices.rs，请使用 crate::core::types::{CellId, FaceId, NodeId} 替代"
)]

use serde::{Deserialize, Serialize};
use std::fmt;

/// 无效单元标记
pub const INVALID_CELL: usize = usize::MAX;

/// 无效面标记
pub const INVALID_FACE: usize = usize::MAX;

/// 无效节点标记
pub const INVALID_NODE: usize = usize::MAX;

/// 单元索引
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CellId(pub usize);

impl CellId {
    pub const INVALID: Self = Self(INVALID_CELL);

    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn idx(self) -> usize {
        self.0
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_CELL
    }
}

impl From<usize> for CellId {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<CellId> for usize {
    fn from(id: CellId) -> usize {
        id.0
    }
}

impl fmt::Display for CellId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Cell({})", self.0)
        } else {
            write!(f, "Cell(INVALID)")
        }
    }
}

/// 面索引
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FaceId(pub usize);

impl FaceId {
    pub const INVALID: Self = Self(INVALID_FACE);

    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn idx(self) -> usize {
        self.0
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_FACE
    }
}

impl From<usize> for FaceId {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<FaceId> for usize {
    fn from(id: FaceId) -> usize {
        id.0
    }
}

impl fmt::Display for FaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Face({})", self.0)
        } else {
            write!(f, "Face(INVALID)")
        }
    }
}

/// 节点索引
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeId(pub usize);

impl NodeId {
    pub const INVALID: Self = Self(INVALID_NODE);

    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn idx(self) -> usize {
        self.0
    }

    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_NODE
    }
}

impl From<usize> for NodeId {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<NodeId> for usize {
    fn from(id: NodeId) -> usize {
        id.0
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Node({})", self.0)
        } else {
            write!(f, "Node(INVALID)")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_id() {
        let id = CellId::new(10);
        assert!(id.is_valid());
        assert_eq!(id.idx(), 10);

        let invalid = CellId::INVALID;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_conversions() {
        let id: CellId = 42.into();
        let back: usize = id.into();
        assert_eq!(back, 42);
    }
}
