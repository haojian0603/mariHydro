// marihydro\crates\mh_foundation\src/index.rs

//! 强类型索引系统
//!
//! 使用泛型 `Idx<T>` 实现带代际验证的类型安全索引。
//!
//! # 设计目标
//!
//! 1. **类型安全**: 编译期区分不同类型的索引（Cell/Face/Node等）
//! 2. **零开销**: 在release模式下与 usize 完全相同的性能
//! 3. **悬垂检测**: 通过代际(generation)检测已删除元素的访问
//! 4. **简洁API**: 提供类型别名和便捷方法
//!
//! # 示例
//!
//! ```
//! use mh_foundation::index::{Idx, CellIndex, FaceIndex};
//!
//! let cell_idx = CellIndex::new(0, 1);
//! assert!(cell_idx.is_valid());
//! assert_eq!(cell_idx.index(), 0);
//! assert_eq!(cell_idx.generation(), 1);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

/// 无效索引标记
pub const INVALID_INDEX: u32 = u32::MAX;

/// 无效代际标记
pub const INVALID_GENERATION: u32 = 0;

// ============================================================================
// 标记类型 (Phantom Types)
// ============================================================================

/// 单元索引标记
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellTag;

/// 面/边索引标记
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceTag;

/// 节点索引标记
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeTag;

/// 顶点索引标记 (用于半边网格)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexTag;

/// 半边索引标记
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HalfEdgeTag;

/// 边界索引标记
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundaryTag;

// ============================================================================
// 泛型索引类型
// ============================================================================

/// 带代际验证的泛型索引
///
/// 使用 Phantom Type `T` 区分不同类型的索引，避免误用。
/// `generation` 字段用于检测悬垂引用。
///
/// # 内存布局
///
/// 使用两个 u32 字段，总共 8 字节：
/// - `index`: 实际索引值 (0 到 2^32-2)
/// - `generation`: 代际号 (1 起始，0 表示无效)
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct Idx<T> {
    /// 索引值
    index: u32,
    /// 代际号 (用于检测悬垂引用)
    generation: u32,
    /// 类型标记
    #[serde(skip)]
    _marker: PhantomData<fn() -> T>,
}

// 手动实现 Copy 和 Clone，因为 PhantomData<T> 的 Copy 需要 T: Copy
impl<T> Copy for Idx<T> {}

impl<T> Clone for Idx<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Idx<T> {
    /// 无效索引常量
    pub const INVALID: Self = Self {
        index: INVALID_INDEX,
        generation: INVALID_GENERATION,
        _marker: PhantomData,
    };

    /// 创建新索引
    ///
    /// # 参数
    /// - `index`: 索引值
    /// - `generation`: 代际号 (应 >= 1)
    #[inline]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }

    /// 从 usize 创建（代际默认为1）
    #[inline]
    pub fn from_usize(index: usize) -> Self {
        Self::new(index as u32, 1)
    }

    /// 仅用于索引位置已知有效的情况，不带代际检查
    #[inline]
    pub const fn from_raw(index: u32) -> Self {
        Self::new(index, 1)
    }

    /// 获取索引值
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// 获取索引值（usize）
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.index as usize
    }

    /// 获取代际号
    #[inline]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// 判断索引是否有效
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.index != INVALID_INDEX && self.generation != INVALID_GENERATION
    }

    /// 判断索引是否无效
    #[inline]
    pub const fn is_invalid(self) -> bool {
        !self.is_valid()
    }

    /// 创建下一代索引（用于重用slot）
    #[inline]
    pub fn next_generation(self) -> Self {
        let next_gen = self.generation.wrapping_add(1);
        Self::new(self.index, if next_gen == 0 { 1 } else { next_gen })
    }

    /// 检查代际是否匹配
    #[inline]
    pub const fn matches_generation(self, generation: u32) -> bool {
        self.generation == generation
    }

    /// 转换为 `Option<usize>`
    #[inline]
    pub fn to_option(self) -> Option<usize> {
        if self.is_valid() {
            Some(self.as_usize())
        } else {
            None
        }
    }
}

// ============================================================================
// Trait 实现
// ============================================================================

impl<T> Default for Idx<T> {
    fn default() -> Self {
        Self::INVALID
    }
}

impl<T> PartialEq for Idx<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Idx<T> {}

impl<T> PartialOrd for Idx<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Idx<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.generation.cmp(&other.generation) {
            std::cmp::Ordering::Equal => self.index.cmp(&other.index),
            other => other,
        }
    }
}

impl<T> Hash for Idx<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<T> fmt::Debug for Idx<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Idx({}@{})", self.index, self.generation)
        } else {
            write!(f, "Idx(INVALID)")
        }
    }
}

impl<T> fmt::Display for Idx<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "{}", self.index)
        } else {
            write!(f, "INVALID")
        }
    }
}

// usize 转换
impl<T> From<usize> for Idx<T> {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl<T> From<Idx<T>> for usize {
    #[inline]
    fn from(idx: Idx<T>) -> usize {
        idx.as_usize()
    }
}

// u32 转换
impl<T> From<u32> for Idx<T> {
    #[inline]
    fn from(index: u32) -> Self {
        Self::from_raw(index)
    }
}

impl<T> From<Idx<T>> for u32 {
    #[inline]
    fn from(idx: Idx<T>) -> u32 {
        idx.index()
    }
}

// Option 转换
impl<T> From<Option<usize>> for Idx<T> {
    fn from(opt: Option<usize>) -> Self {
        match opt {
            Some(i) => Self::from_usize(i),
            None => Self::INVALID,
        }
    }
}

impl<T> From<Idx<T>> for Option<usize> {
    fn from(idx: Idx<T>) -> Self {
        idx.to_option()
    }
}

// ============================================================================
// 类型别名
// ============================================================================

/// 单元索引
pub type CellIndex = Idx<CellTag>;

/// 面/边索引
pub type FaceIndex = Idx<FaceTag>;

/// 节点索引
pub type NodeIndex = Idx<NodeTag>;

/// 顶点索引 (半边网格)
pub type VertexIndex = Idx<VertexTag>;

/// 半边索引
pub type HalfEdgeIndex = Idx<HalfEdgeTag>;

/// 边界索引
pub type BoundaryIndex = Idx<BoundaryTag>;

// ============================================================================
// 便捷函数
// ============================================================================

/// 创建单元索引
#[inline]
pub const fn cell(index: u32) -> CellIndex {
    CellIndex::from_raw(index)
}

/// 创建面索引
#[inline]
pub const fn face(index: u32) -> FaceIndex {
    FaceIndex::from_raw(index)
}

/// 创建节点索引
#[inline]
pub const fn node(index: u32) -> NodeIndex {
    NodeIndex::from_raw(index)
}

/// 创建顶点索引
#[inline]
pub const fn vertex(index: u32) -> VertexIndex {
    VertexIndex::from_raw(index)
}

/// 创建半边索引
#[inline]
pub const fn halfedge(index: u32) -> HalfEdgeIndex {
    HalfEdgeIndex::from_raw(index)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idx_creation() {
        let idx = CellIndex::new(10, 1);
        assert_eq!(idx.index(), 10);
        assert_eq!(idx.generation(), 1);
        assert!(idx.is_valid());
    }

    #[test]
    fn test_idx_invalid() {
        let idx = CellIndex::INVALID;
        assert!(!idx.is_valid());
        assert!(idx.is_invalid());
    }

    #[test]
    fn test_idx_from_usize() {
        let idx: CellIndex = 42usize.into();
        assert_eq!(idx.index(), 42);
        assert_eq!(idx.generation(), 1);
    }

    #[test]
    fn test_idx_to_usize() {
        let idx = CellIndex::new(100, 2);
        let val: usize = idx.into();
        assert_eq!(val, 100);
    }

    #[test]
    fn test_idx_equality() {
        let a = CellIndex::new(1, 1);
        let b = CellIndex::new(1, 1);
        let c = CellIndex::new(1, 2);
        let d = CellIndex::new(2, 1);

        assert_eq!(a, b);
        assert_ne!(a, c); // 不同代际
        assert_ne!(a, d); // 不同索引
    }

    #[test]
    fn test_type_safety() {
        let cell_idx = CellIndex::new(0, 1);
        let face_idx = FaceIndex::new(0, 1);
        
        // 编译时类型检查：下面的代码如果取消注释会编译失败
        // let _: CellIndex = face_idx;
        
        // 但可以比较索引值
        assert_eq!(cell_idx.index(), face_idx.index());
    }

    #[test]
    fn test_next_generation() {
        let idx = CellIndex::new(5, 1);
        let next = idx.next_generation();
        assert_eq!(next.index(), 5);
        assert_eq!(next.generation(), 2);
    }

    #[test]
    fn test_generation_match() {
        let idx = CellIndex::new(5, 3);
        assert!(idx.matches_generation(3));
        assert!(!idx.matches_generation(2));
    }

    #[test]
    fn test_to_option() {
        let valid = CellIndex::new(10, 1);
        assert_eq!(valid.to_option(), Some(10));

        let invalid = CellIndex::INVALID;
        assert_eq!(invalid.to_option(), None);
    }

    #[test]
    fn test_display() {
        let valid = CellIndex::new(42, 1);
        assert_eq!(format!("{}", valid), "42");

        let invalid = CellIndex::INVALID;
        assert_eq!(format!("{}", invalid), "INVALID");
    }

    #[test]
    fn test_debug() {
        let valid = CellIndex::new(42, 3);
        assert_eq!(format!("{:?}", valid), "Idx(42@3)");

        let invalid = CellIndex::INVALID;
        assert_eq!(format!("{:?}", invalid), "Idx(INVALID)");
    }

    #[test]
    fn test_ordering() {
        let a = CellIndex::new(1, 1);
        let b = CellIndex::new(2, 1);
        let c = CellIndex::new(1, 2);

        assert!(a < b); // 同代际，比较索引
        assert!(a < c); // 不同代际，先比较代际
    }

    #[test]
    fn test_convenience_functions() {
        let c = cell(0);
        let f = face(1);
        let n = node(2);
        let v = vertex(3);
        let h = halfedge(4);

        assert_eq!(c.index(), 0);
        assert_eq!(f.index(), 1);
        assert_eq!(n.index(), 2);
        assert_eq!(v.index(), 3);
        assert_eq!(h.index(), 4);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        
        let mut set = HashSet::new();
        set.insert(CellIndex::new(1, 1));
        set.insert(CellIndex::new(2, 1));
        set.insert(CellIndex::new(1, 2));
        
        assert_eq!(set.len(), 3);
        assert!(set.contains(&CellIndex::new(1, 1)));
    }

    #[test]
    fn test_serialization() {
        let idx = CellIndex::new(42, 3);
        let json = serde_json::to_string(&idx).unwrap();
        let deserialized: CellIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(idx, deserialized);
    }
}
