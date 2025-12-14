// crates/mh_runtime/src/indices.rs

//! 公共计算索引 - 无代际验证
//!
//! 提供类型安全的索引类型，用于网格单元、面、节点等的引用。
//! 这些索引是轻量级的，不包含代际验证（代际验证在 arena_ext 模块中提供）。
//!
//! # 设计原则
//!
//! 1. **类型安全**: 不同类型的索引不可混用（CellIndex ≠ FaceIndex）
//! 2. **零开销**: 编译期类型检查，运行时与 usize 完全相同
//! 3. **无代际**: 索引仅包含位置信息，不包含代际验证
//!
//! # 示例
//!
//! ```rust
//! use mh_runtime::indices::{CellIndex, FaceIndex, cell, face};
//!
//! let c = CellIndex::new(0);
//! let f = face(5);
//!
//! assert!(c.is_valid());
//! assert_eq!(f.get(), 5);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;

/// 无效索引标记
pub const INVALID_INDEX: usize = usize::MAX;

// =============================================================================
// 索引 Trait
// =============================================================================

/// 索引类型 Trait
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
    #[inline]
    fn is_invalid(self) -> bool {
        !self.is_valid()
    }
}

// =============================================================================
// 宏：生成索引类型
// =============================================================================

macro_rules! define_index {
    ($(#[$meta:meta])* $name:ident, $doc:literal) => {
        #[doc = $doc]
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

            /// 从 usize 创建
            #[inline]
            pub const fn from_usize(idx: usize) -> Self {
                Self(idx)
            }

            /// 获取索引值
            #[inline]
            pub const fn get(self) -> usize {
                self.0
            }

            /// 转换为 usize
            #[inline]
            pub const fn as_usize(self) -> usize {
                self.0
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
        }

        impl Index for $name {
            #[inline]
            fn new(idx: usize) -> Self { Self(idx) }
            
            #[inline]
            fn get(self) -> usize { self.0 }
            
            #[inline]
            fn invalid() -> Self { Self::INVALID }
            
            #[inline]
            fn is_valid(self) -> bool { self.0 != INVALID_INDEX }
        }

        impl From<usize> for $name {
            #[inline]
            fn from(idx: usize) -> Self { Self::new(idx) }
        }

        impl From<$name> for usize {
            #[inline]
            fn from(idx: $name) -> usize { idx.get() }
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

        impl Default for $name {
            fn default() -> Self { Self::INVALID }
        }
    };
}

// =============================================================================
// 索引类型定义
// =============================================================================

define_index!(CellIndex, "单元索引");
define_index!(FaceIndex, "面/边索引");
define_index!(NodeIndex, "节点索引");
define_index!(EdgeIndex, "边索引（用于非结构化网格）");
define_index!(VertexIndex, "顶点索引（用于半边网格）");
define_index!(HalfEdgeIndex, "半边索引");
define_index!(BoundaryIndex, "边界索引");
define_index!(LayerIndex, "层索引（垂向分层）");

// =============================================================================
// 便捷构造函数
// =============================================================================

/// 创建单元索引
#[inline]
pub const fn cell(idx: usize) -> CellIndex { CellIndex::new(idx) }

/// 创建面索引
#[inline]
pub const fn face(idx: usize) -> FaceIndex { FaceIndex::new(idx) }

/// 创建节点索引
#[inline]
pub const fn node(idx: usize) -> NodeIndex { NodeIndex::new(idx) }

/// 创建边索引
#[inline]
pub const fn edge(idx: usize) -> EdgeIndex { EdgeIndex::new(idx) }

/// 创建顶点索引
#[inline]
pub const fn vertex(idx: usize) -> VertexIndex { VertexIndex::new(idx) }

/// 创建半边索引
#[inline]
pub const fn half_edge(idx: usize) -> HalfEdgeIndex { HalfEdgeIndex::new(idx) }

/// 创建边界索引
#[inline]
pub const fn boundary(idx: usize) -> BoundaryIndex { BoundaryIndex::new(idx) }

/// 创建层索引
#[inline]
pub const fn layer(idx: usize) -> LayerIndex { LayerIndex::new(idx) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_index() {
        let idx = CellIndex::new(42);
        assert!(idx.is_valid());
        assert_eq!(idx.get(), 42);
        
        let invalid = CellIndex::INVALID;
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_type_safety() {
        let c = cell(0);
        let f = face(0);
        
        // 类型安全：不同索引类型不相等
        // 这会编译错误：assert_ne!(c, f);
        assert_eq!(c.get(), f.get()); // 但值可以比较
    }

    #[test]
    fn test_from_usize() {
        let idx: CellIndex = 10.into();
        assert_eq!(idx.get(), 10);
        
        let val: usize = idx.into();
        assert_eq!(val, 10);
    }
}
