// marihydro\crates\mh_foundation\src/index.rs

//! 强类型索引系统
//!
//! 使用泛型 `Idx<T>` 实现类型安全的轻量级索引。
//!
//! # 设计目标
//!
//! 1. **类型安全**: 编译期区分不同类型的索引（Cell/Face/Node等）
//! 2. **零开销**: 与 u32 完全相同的内存布局和性能
//! 3. **简洁API**: 提供类型别名和便捷方法
//!
//! # 注意
//!
//! 此模块的 `Idx<T>` 是**无代际验证**的轻量级索引。
//! 如需悬垂引用检测，请使用 `mh_runtime::SafeIdx`。
//!
//! # 示例
//!
//! ```
//! use mh_foundation::index::{Idx, CellIndex, FaceIndex};
//!
//! let cell_idx = CellIndex::new(0);
//! assert!(cell_idx.is_valid());
//! assert_eq!(cell_idx.index(), 0);
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

/// 无效索引标记
pub const INVALID_INDEX: u32 = u32::MAX;

/// 无效代际标记
#[deprecated(note = "Generation moved to mh_runtime::SafeIdx")]
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

/// 轻量级泛型索引（4 字节）
///
/// 使用 Phantom Type `T` 区分不同类型的索引，避免误用。
///
/// # 内存布局
///
/// 与 u32 完全相同的内存布局（4 字节）。
/// 使用 `#[repr(transparent)]` 保证零开销抽象。
///
/// # 注意
///
/// 此类型不带代际验证。如需悬垂引用检测，请使用 `mh_runtime::SafeIdx`。
#[derive(Serialize, Deserialize)]
#[repr(transparent)]
pub struct Idx<T> {
    /// 索引值
    index: u32,
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
        _marker: PhantomData,
    };

    /// 创建新索引
    ///
    /// # 参数
    /// - `index`: 索引值
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self {
            index,
            _marker: PhantomData,
        }
    }

    /// 兼容旧 API：创建带 generation 参数的索引
    ///
    /// generation 参数会被忽略。
    #[deprecated(note = "Use Idx::new(index) instead. Generation moved to mh_runtime::SafeIdx")]
    #[inline]
    pub const fn new_with_generation(index: u32, _generation: u32) -> Self {
        Self::new(index)
    }

    /// 从 usize 创建
    #[inline]
    pub fn from_usize(index: usize) -> Self {
        Self::new(index as u32)
    }

    /// 从原始 u32 创建
    #[inline]
    pub const fn from_raw(index: u32) -> Self {
        Self::new(index)
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

    /// 获取代际号（兼容旧 API，始终返回 1）
    #[deprecated(note = "Generation moved to mh_runtime::SafeIdx")]
    #[inline]
    pub const fn generation(self) -> u32 {
        1
    }

    /// 判断索引是否有效
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.index != INVALID_INDEX
    }

    /// 判断索引是否无效
    #[inline]
    pub const fn is_invalid(self) -> bool {
        self.index == INVALID_INDEX
    }

    /// 创建下一代索引（兼容旧 API，返回自身）
    #[deprecated(note = "Generation moved to mh_runtime::SafeIdx")]
    #[inline]
    pub fn next_generation(self) -> Self {
        self
    }

    /// 检查代际是否匹配（兼容旧 API，始终返回 true）
    #[deprecated(note = "Generation moved to mh_runtime::SafeIdx")]
    #[inline]
    pub const fn matches_generation(self, _generation: u32) -> bool {
        true
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
        self.index == other.index
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
        self.index.cmp(&other.index)
    }
}

impl<T> Hash for Idx<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> fmt::Debug for Idx<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Idx({})", self.index)
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
        let idx = CellIndex::new(10);
        assert_eq!(idx.index(), 10);
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
        assert!(idx.is_valid());
    }

    #[test]
    fn test_idx_to_usize() {
        let idx = CellIndex::new(100);
        let val: usize = idx.into();
        assert_eq!(val, 100);
    }

    #[test]
    fn test_idx_equality() {
        let a = CellIndex::new(1);
        let b = CellIndex::new(1);
        let c = CellIndex::new(2);

        assert_eq!(a, b); // 相同索引
        assert_ne!(a, c); // 不同索引
    }

    #[test]
    fn test_type_safety() {
        let cell_idx = CellIndex::new(0);
        let face_idx = FaceIndex::new(0);
        
        // 编译时类型检查：下面的代码如果取消注释会编译失败
        // let _: CellIndex = face_idx;
        
        // 但可以比较索引值
        assert_eq!(cell_idx.index(), face_idx.index());
    }

    #[test]
    fn test_idx_size() {
        // 确保 Idx<T> 与 u32 大小相同（4 字节）
        assert_eq!(std::mem::size_of::<CellIndex>(), 4);
        assert_eq!(std::mem::size_of::<FaceIndex>(), 4);
        assert_eq!(std::mem::size_of::<NodeIndex>(), 4);
    }

    #[test]
    fn test_to_option() {
        let valid = CellIndex::new(10);
        assert_eq!(valid.to_option(), Some(10));

        let invalid = CellIndex::INVALID;
        assert_eq!(invalid.to_option(), None);
    }

    #[test]
    fn test_display() {
        let valid = CellIndex::new(42);
        assert_eq!(format!("{}", valid), "42");

        let invalid = CellIndex::INVALID;
        assert_eq!(format!("{}", invalid), "INVALID");
    }

    #[test]
    fn test_debug() {
        let valid = CellIndex::new(42);
        assert_eq!(format!("{:?}", valid), "Idx(42)");

        let invalid = CellIndex::INVALID;
        assert_eq!(format!("{:?}", invalid), "Idx(INVALID)");
    }

    #[test]
    fn test_ordering() {
        let a = CellIndex::new(1);
        let b = CellIndex::new(2);
        let c = CellIndex::new(3);

        assert!(a < b);
        assert!(b < c);
        assert!(a < c);
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
        set.insert(CellIndex::new(1));
        set.insert(CellIndex::new(2));
        set.insert(CellIndex::new(3));
        
        assert_eq!(set.len(), 3);
        assert!(set.contains(&CellIndex::new(1)));
        assert!(set.contains(&CellIndex::new(2)));
    }

    #[test]
    fn test_serialization() {
        let idx = CellIndex::new(42);
        let json = serde_json::to_string(&idx).unwrap();
        let deserialized: CellIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(idx, deserialized);
        assert_eq!(idx.index(), 42);
    }
}

// ============================================================================
// 轻量级索引类型（用于 FrozenMesh 和计算路径）
// ============================================================================

/// 无效索引常量（用于哨兵值）
pub const INVALID_IDX: u32 = u32::MAX;

/// 轻量级单元索引（4 字节，用于计算路径）
/// 
/// 比 `CellIndex` 更轻量，不带代际验证。
/// 适用于 FrozenMesh 等静态拓扑结构。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct CellIdx(pub u32);

impl CellIdx {
    /// 无效索引哨兵值
    pub const INVALID: Self = Self(INVALID_IDX);
    
    /// 创建新索引
    #[inline]
    pub const fn new(idx: u32) -> Self { Self(idx) }
    
    /// 从 usize 创建
    #[inline]
    pub fn from_usize(idx: usize) -> Self { Self(idx as u32) }
    
    /// 是否有效
    #[inline]
    pub const fn is_valid(self) -> bool { self.0 != INVALID_IDX }
    
    /// 是否无效
    #[inline]
    pub const fn is_invalid(self) -> bool { self.0 == INVALID_IDX }
    
    /// 转换为 usize
    #[inline]
    pub const fn as_usize(self) -> usize { self.0 as usize }
    
    /// 获取原始值
    #[inline]
    pub const fn raw(self) -> u32 { self.0 }
    
    /// 转换为 Option（无效则返回 None）
    #[inline]
    pub fn to_option(self) -> Option<usize> {
        if self.is_valid() { Some(self.as_usize()) } else { None }
    }
}

impl From<u32> for CellIdx {
    #[inline] fn from(v: u32) -> Self { Self(v) }
}

impl From<usize> for CellIdx {
    #[inline] fn from(v: usize) -> Self { Self(v as u32) }
}

/// 轻量级面索引（4 字节，用于计算路径）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct FaceIdx(pub u32);

impl FaceIdx {
    /// 无效索引哨兵值
    pub const INVALID: Self = Self(INVALID_IDX);
    
    /// 创建新索引
    #[inline]
    pub const fn new(idx: u32) -> Self { Self(idx) }
    
    /// 从 usize 创建
    #[inline]
    pub fn from_usize(idx: usize) -> Self { Self(idx as u32) }
    
    /// 是否有效
    #[inline]
    pub const fn is_valid(self) -> bool { self.0 != INVALID_IDX }
    
    /// 是否无效
    #[inline]
    pub const fn is_invalid(self) -> bool { self.0 == INVALID_IDX }
    
    /// 转换为 usize
    #[inline]
    pub const fn as_usize(self) -> usize { self.0 as usize }
    
    /// 获取原始值
    #[inline]
    pub const fn raw(self) -> u32 { self.0 }
    
    /// 转换为 Option
    #[inline]
    pub fn to_option(self) -> Option<usize> {
        if self.is_valid() { Some(self.as_usize()) } else { None }
    }
}

impl From<u32> for FaceIdx {
    #[inline] fn from(v: u32) -> Self { Self(v) }
}

impl From<usize> for FaceIdx {
    #[inline] fn from(v: usize) -> Self { Self(v as u32) }
}

/// 轻量级节点索引（4 字节，用于计算路径）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct NodeIdx(pub u32);

impl NodeIdx {
    /// 无效索引哨兵值
    pub const INVALID: Self = Self(INVALID_IDX);
    
    /// 创建新索引
    #[inline]
    pub const fn new(idx: u32) -> Self { Self(idx) }
    
    /// 从 usize 创建
    #[inline]
    pub fn from_usize(idx: usize) -> Self { Self(idx as u32) }
    
    /// 是否有效
    #[inline]
    pub const fn is_valid(self) -> bool { self.0 != INVALID_IDX }
    
    /// 是否无效
    #[inline]
    pub const fn is_invalid(self) -> bool { self.0 == INVALID_IDX }
    
    /// 转换为 usize
    #[inline]
    pub const fn as_usize(self) -> usize { self.0 as usize }
    
    /// 获取原始值
    #[inline]
    pub const fn raw(self) -> u32 { self.0 }
    
    /// 转换为 Option
    #[inline]
    pub fn to_option(self) -> Option<usize> {
        if self.is_valid() { Some(self.as_usize()) } else { None }
    }
}

impl From<u32> for NodeIdx {
    #[inline] fn from(v: u32) -> Self { Self(v) }
}

impl From<usize> for NodeIdx {
    #[inline] fn from(v: usize) -> Self { Self(v as u32) }
}

/// 边界条件 ID（4 字节）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(transparent)]
pub struct BoundaryId(pub u32);

impl BoundaryId {
    /// 无效边界 ID（非边界面）
    pub const INVALID: Self = Self(INVALID_IDX);
    
    /// 创建新边界 ID
    #[inline]
    pub const fn new(id: u32) -> Self { Self(id) }
    
    /// 是否有效边界
    #[inline]
    pub const fn is_valid(self) -> bool { self.0 != INVALID_IDX }
    
    /// 获取原始值
    #[inline]
    pub const fn raw(self) -> u32 { self.0 }
}

impl From<u32> for BoundaryId {
    #[inline] fn from(v: u32) -> Self { Self(v) }
}

#[cfg(test)]
mod simple_idx_tests {
    use super::*;
    
    #[test]
    fn test_cell_idx() {
        let idx = CellIdx::new(10);
        assert!(idx.is_valid());
        assert_eq!(idx.as_usize(), 10);
        assert_eq!(idx.to_option(), Some(10));
        
        let invalid = CellIdx::INVALID;
        assert!(invalid.is_invalid());
        assert_eq!(invalid.to_option(), None);
    }
    
    #[test]
    fn test_face_idx() {
        let idx = FaceIdx::from_usize(42);
        assert!(idx.is_valid());
        assert_eq!(idx.raw(), 42);
    }
    
    #[test]
    fn test_node_idx() {
        let idx: NodeIdx = 100u32.into();
        assert!(idx.is_valid());
        assert_eq!(idx.as_usize(), 100);
    }
    
    #[test]
    fn test_boundary_id() {
        let bid = BoundaryId::new(0);
        assert!(bid.is_valid());
        
        let invalid = BoundaryId::INVALID;
        assert!(!invalid.is_valid());
    }
}