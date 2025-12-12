// crates/mh_core/src/indices.rs

//! 统一索引类型定义
//!
//! 全项目唯一的索引类型定义处，所有其他模块必须从这里引用。
//!
//! # 设计原则
//!
//! 1. **唯一来源**: 所有索引类型只在此处定义
//! 2. **类型安全**: 不同类型的索引不可混用
//! 3. **零开销**: 编译期类型检查，运行时无开销

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

/// 无效索引标记
pub const INVALID_INDEX: usize = usize::MAX;

// ============================================================================
// 基础索引trait
// ============================================================================

/// 索引类型trait
pub trait Index: Copy + Clone + Eq + Hash + fmt::Debug {
    /// 创建新索引
    fn new(idx: usize) -> Self;
    
    /// 获取索引值
    fn get(self) -> usize;
    
    /// 创建无效索引
    fn invalid() -> Self;
    
    /// 检查是否有效
    fn is_valid(self) -> bool;
    
    /// 检查是否无效
    fn is_invalid(self) -> bool {
        !self.is_valid()
    }
}

// ============================================================================
// 宏：生成索引类型
// ============================================================================

macro_rules! define_index {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[repr(transparent)]
        pub struct $name(pub usize);

        impl $name {
            /// 无效索引常量
            pub const INVALID: Self = Self(INVALID_INDEX);

            /// 创建新索引
            #[inline]
            pub const fn new(idx: usize) -> Self {
                Self(idx)
            }

            /// 从usize创建
            #[inline]
            pub const fn from_usize(idx: usize) -> Self {
                Self(idx)
            }

            /// 获取索引值
            #[inline]
            pub const fn get(self) -> usize {
                self.0
            }

            /// 转换为usize
            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0
            }

            /// 转换为u32
            #[inline]
            pub fn as_u32(self) -> u32 {
                self.0 as u32
            }

            /// 从u32创建
            #[inline]
            pub fn from_u32(idx: u32) -> Self {
                Self(idx as usize)
            }

            /// 检查是否有效
            #[inline]
            pub const fn is_valid(self) -> bool {
                self.0 != INVALID_INDEX
            }

            /// 检查是否无效
            #[inline]
            pub const fn is_invalid(self) -> bool {
                self.0 == INVALID_INDEX
            }

            /// 转换为Option
            #[inline]
            pub fn to_option(self) -> Option<usize> {
                if self.is_valid() {
                    Some(self.0)
                } else {
                    None
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::INVALID
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.is_valid() {
                    write!(f, "{}({})", stringify!($name), self.0)
                } else {
                    write!(f, "{}(INVALID)", stringify!($name))
                }
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                if self.is_valid() {
                    write!(f, "{}", self.0)
                } else {
                    write!(f, "INVALID")
                }
            }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(idx: usize) -> Self {
                Self(idx)
            }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(idx: $name) -> usize {
                idx.0
            }
        }

        impl From<u32> for $name {
            #[inline]
            fn from(idx: u32) -> Self {
                Self(idx as usize)
            }
        }

        impl From<$name> for u32 {
            #[inline]
            fn from(idx: $name) -> u32 {
                idx.0 as u32
            }
        }

        impl Index for $name {
            #[inline]
            fn new(idx: usize) -> Self {
                Self(idx)
            }

            #[inline]
            fn get(self) -> usize {
                self.0
            }

            #[inline]
            fn invalid() -> Self {
                Self::INVALID
            }

            #[inline]
            fn is_valid(self) -> bool {
                self.0 != INVALID_INDEX
            }
        }
    };
}

// ============================================================================
// 索引类型定义
// ============================================================================

define_index! {
    /// 单元索引 - 用于索引网格单元
    CellIndex
}

define_index! {
    /// 面索引 - 用于索引网格面/边
    FaceIndex
}

define_index! {
    /// 节点索引 - 用于索引网格节点
    NodeIndex
}

define_index! {
    /// 顶点索引 - 用于半边网格的顶点
    VertexIndex
}

define_index! {
    /// 半边索引 - 用于半边网格的半边
    HalfEdgeIndex
}

define_index! {
    /// 边界索引 - 用于索引边界条件
    BoundaryIndex
}

// ============================================================================
// 便捷构造函数
// ============================================================================

/// 创建单元索引
#[inline]
pub const fn cell(idx: usize) -> CellIndex {
    CellIndex::new(idx)
}

/// 创建面索引
#[inline]
pub const fn face(idx: usize) -> FaceIndex {
    FaceIndex::new(idx)
}

/// 创建节点索引
#[inline]
pub const fn node(idx: usize) -> NodeIndex {
    NodeIndex::new(idx)
}

/// 创建顶点索引
#[inline]
pub const fn vertex(idx: usize) -> VertexIndex {
    VertexIndex::new(idx)
}

/// 创建半边索引
#[inline]
pub const fn halfedge(idx: usize) -> HalfEdgeIndex {
    HalfEdgeIndex::new(idx)
}

/// 创建边界索引
#[inline]
pub const fn boundary(idx: usize) -> BoundaryIndex {
    BoundaryIndex::new(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_index() {
        let idx = CellIndex::new(42);
        assert!(idx.is_valid());
        assert_eq!(idx.get(), 42);
        assert_eq!(idx.as_u32(), 42);
    }

    #[test]
    fn test_invalid_index() {
        let idx = CellIndex::INVALID;
        assert!(idx.is_invalid());
        assert!(!idx.is_valid());
    }

    #[test]
    fn test_index_conversion() {
        let idx: CellIndex = 100usize.into();
        assert_eq!(idx.get(), 100);

        let val: usize = idx.into();
        assert_eq!(val, 100);
    }

    #[test]
    fn test_different_index_types() {
        let cell = CellIndex::new(1);
        let face = FaceIndex::new(1);
        let node = NodeIndex::new(1);

        // 不同类型不能直接比较（编译期检查）
        assert_eq!(cell.get(), face.get());
        assert_eq!(face.get(), node.get());
    }

    #[test]
    fn test_convenience_functions() {
        assert_eq!(cell(5).get(), 5);
        assert_eq!(face(10).get(), 10);
        assert_eq!(node(15).get(), 15);
    }

    #[test]
    fn test_serde() {
        let idx = CellIndex::new(42);
        let json = serde_json::to_string(&idx).unwrap();
        let parsed: CellIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(idx, parsed);
    }
}
