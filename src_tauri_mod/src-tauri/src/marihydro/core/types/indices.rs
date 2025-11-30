//! 索引类型定义
//!
//! 使用 `#[repr(transparent)]` 保证与 usize 布局相同，
//! 实现零开销的类型安全抽象。
//!
//! # 设计目标
//!
//! 1. 类型安全：编译期区分 Cell/Face/Node 索引
//! 2. 零开销：与 usize 完全相同的内存布局
//! 3. 安全转换：提供安全的切片重解释函数
//!
//! # 与 domain/mesh/indices.rs 的关系
//!
//! 本模块是新的统一索引类型定义。`domain/mesh/indices.rs` 中的
//! `CellId`、`FaceId`、`NodeId` 已被标记为 deprecated，
//! 建议使用本模块中的类型。

use serde::{Deserialize, Serialize};
use std::fmt;

/// 无效索引标记
pub const INVALID_INDEX: usize = usize::MAX;

// ============================================================
// 面索引类型
// ============================================================

/// 面索引 - 用于索引网格面
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FaceIndex(pub usize);

/// 面 ID - 内部使用，与 FaceIndex 布局相同
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct FaceId(pub usize);

impl FaceIndex {
    /// 无效面索引
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的面索引
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断索引是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }

    /// 从 usize 创建
    #[inline]
    pub const fn from_usize(idx: usize) -> Self {
        Self(idx)
    }
}

impl FaceId {
    /// 无效面 ID
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的面 ID
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断 ID 是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn idx(self) -> usize {
        self.0
    }
}

// ============================================================
// 单元索引类型
// ============================================================

/// 单元索引 - 用于索引网格单元
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CellIndex(pub usize);

/// 单元 ID - 内部使用，与 CellIndex 布局相同
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct CellId(pub usize);

impl CellIndex {
    /// 无效单元索引
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的单元索引
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断索引是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }

    /// 从 usize 创建
    #[inline]
    pub const fn from_usize(idx: usize) -> Self {
        Self(idx)
    }
}

impl CellId {
    /// 无效单元 ID
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的单元 ID
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断 ID 是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn idx(self) -> usize {
        self.0
    }
}

// ============================================================
// 节点索引类型
// ============================================================

/// 节点索引 - 用于索引网格节点
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeIndex(pub usize);

/// 节点 ID - 内部使用，与 NodeIndex 布局相同
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct NodeId(pub usize);

impl NodeIndex {
    /// 无效节点索引
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的节点索引
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断索引是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }

    /// 从 usize 创建
    #[inline]
    pub const fn from_usize(idx: usize) -> Self {
        Self(idx)
    }
}

impl NodeId {
    /// 无效节点 ID
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的节点 ID
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断 ID 是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn idx(self) -> usize {
        self.0
    }
}

// ============================================================
// 边界索引类型
// ============================================================

/// 边界索引 - 用于索引边界条件
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BoundaryIndex(pub usize);

impl BoundaryIndex {
    /// 无效边界索引
    pub const INVALID: Self = Self(INVALID_INDEX);

    /// 创建新的边界索引
    #[inline]
    pub const fn new(idx: usize) -> Self {
        Self(idx)
    }

    /// 判断索引是否有效
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 != INVALID_INDEX
    }

    /// 获取内部 usize 值
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }

    /// 从 usize 创建
    #[inline]
    pub const fn from_usize(idx: usize) -> Self {
        Self(idx)
    }
}

// ============================================================
// 编译时断言：确保类型布局兼容
// ============================================================

const _: () = {
    assert!(std::mem::size_of::<FaceId>() == std::mem::size_of::<FaceIndex>());
    assert!(std::mem::align_of::<FaceId>() == std::mem::align_of::<FaceIndex>());
    assert!(std::mem::size_of::<CellId>() == std::mem::size_of::<CellIndex>());
    assert!(std::mem::align_of::<CellId>() == std::mem::align_of::<CellIndex>());
    assert!(std::mem::size_of::<NodeId>() == std::mem::size_of::<NodeIndex>());
    assert!(std::mem::align_of::<NodeId>() == std::mem::align_of::<NodeIndex>());
    assert!(std::mem::size_of::<FaceIndex>() == std::mem::size_of::<usize>());
    assert!(std::mem::size_of::<CellIndex>() == std::mem::size_of::<usize>());
    assert!(std::mem::size_of::<NodeIndex>() == std::mem::size_of::<usize>());
};

// ============================================================
// 类型间转换
// ============================================================

// FaceId <-> FaceIndex
impl From<FaceId> for FaceIndex {
    #[inline]
    fn from(id: FaceId) -> Self {
        FaceIndex(id.0)
    }
}

impl From<FaceIndex> for FaceId {
    #[inline]
    fn from(idx: FaceIndex) -> Self {
        FaceId(idx.0)
    }
}

// CellId <-> CellIndex
impl From<CellId> for CellIndex {
    #[inline]
    fn from(id: CellId) -> Self {
        CellIndex(id.0)
    }
}

impl From<CellIndex> for CellId {
    #[inline]
    fn from(idx: CellIndex) -> Self {
        CellId(idx.0)
    }
}

// NodeId <-> NodeIndex
impl From<NodeId> for NodeIndex {
    #[inline]
    fn from(id: NodeId) -> Self {
        NodeIndex(id.0)
    }
}

impl From<NodeIndex> for NodeId {
    #[inline]
    fn from(idx: NodeIndex) -> Self {
        NodeId(idx.0)
    }
}

// usize 转换
impl From<usize> for FaceIndex {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<FaceIndex> for usize {
    #[inline]
    fn from(idx: FaceIndex) -> usize {
        idx.0
    }
}

impl From<usize> for FaceId {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<FaceId> for usize {
    #[inline]
    fn from(id: FaceId) -> usize {
        id.0
    }
}

impl From<usize> for CellIndex {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<CellIndex> for usize {
    #[inline]
    fn from(idx: CellIndex) -> usize {
        idx.0
    }
}

impl From<usize> for CellId {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<CellId> for usize {
    #[inline]
    fn from(id: CellId) -> usize {
        id.0
    }
}

impl From<usize> for NodeIndex {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<NodeIndex> for usize {
    #[inline]
    fn from(idx: NodeIndex) -> usize {
        idx.0
    }
}

impl From<usize> for NodeId {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<NodeId> for usize {
    #[inline]
    fn from(id: NodeId) -> usize {
        id.0
    }
}

impl From<usize> for BoundaryIndex {
    #[inline]
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<BoundaryIndex> for usize {
    #[inline]
    fn from(idx: BoundaryIndex) -> usize {
        idx.0
    }
}

// ============================================================
// Display 实现
// ============================================================

impl fmt::Display for FaceIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Face({})", self.0)
        } else {
            write!(f, "Face(INVALID)")
        }
    }
}

impl fmt::Display for FaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "FaceId({})", self.0)
        } else {
            write!(f, "FaceId(INVALID)")
        }
    }
}

impl fmt::Display for CellIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Cell({})", self.0)
        } else {
            write!(f, "Cell(INVALID)")
        }
    }
}

impl fmt::Display for CellId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "CellId({})", self.0)
        } else {
            write!(f, "CellId(INVALID)")
        }
    }
}

impl fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Node({})", self.0)
        } else {
            write!(f, "Node(INVALID)")
        }
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "NodeId({})", self.0)
        } else {
            write!(f, "NodeId(INVALID)")
        }
    }
}

impl fmt::Display for BoundaryIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "Boundary({})", self.0)
        } else {
            write!(f, "Boundary(INVALID)")
        }
    }
}

// ============================================================
// Default 实现
// ============================================================

impl Default for FaceIndex {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Default for FaceId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Default for CellIndex {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Default for CellId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Default for NodeIndex {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::INVALID
    }
}

impl Default for BoundaryIndex {
    fn default() -> Self {
        Self::INVALID
    }
}

// ============================================================
// 安全的切片重解释函数
// ============================================================

/// 将 FaceId 切片重解释为 FaceIndex 切片
///
/// # Safety
///
/// 由于使用了 `#[repr(transparent)]`，`FaceId` 和 `FaceIndex` 布局相同，
/// 此转换是安全的零开销操作。
#[inline]
pub fn reinterpret_face_ids(ids: &[FaceId]) -> &[FaceIndex] {
    // SAFETY: FaceId 和 FaceIndex 都是 #[repr(transparent)] 包装 usize，
    // 编译时断言已验证布局相同
    unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const FaceIndex, ids.len()) }
}

/// 将 FaceIndex 切片重解释为 FaceId 切片
#[inline]
pub fn reinterpret_face_indices(indices: &[FaceIndex]) -> &[FaceId] {
    unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const FaceId, indices.len()) }
}

/// 将 CellId 切片重解释为 CellIndex 切片
#[inline]
pub fn reinterpret_cell_ids(ids: &[CellId]) -> &[CellIndex] {
    unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const CellIndex, ids.len()) }
}

/// 将 CellIndex 切片重解释为 CellId 切片
#[inline]
pub fn reinterpret_cell_indices(indices: &[CellIndex]) -> &[CellId] {
    unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const CellId, indices.len()) }
}

/// 将 NodeId 切片重解释为 NodeIndex 切片
#[inline]
pub fn reinterpret_node_ids(ids: &[NodeId]) -> &[NodeIndex] {
    unsafe { std::slice::from_raw_parts(ids.as_ptr() as *const NodeIndex, ids.len()) }
}

/// 将 NodeIndex 切片重解释为 NodeId 切片
#[inline]
pub fn reinterpret_node_indices(indices: &[NodeIndex]) -> &[NodeId] {
    unsafe { std::slice::from_raw_parts(indices.as_ptr() as *const NodeId, indices.len()) }
}

/// 将 usize 切片重解释为 CellIndex 切片
#[inline]
pub fn usize_slice_as_cell_indices(slice: &[usize]) -> &[CellIndex] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const CellIndex, slice.len()) }
}

/// 将 usize 切片重解释为 FaceIndex 切片
#[inline]
pub fn usize_slice_as_face_indices(slice: &[usize]) -> &[FaceIndex] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const FaceIndex, slice.len()) }
}

/// 将 usize 切片重解释为 NodeIndex 切片
#[inline]
pub fn usize_slice_as_node_indices(slice: &[usize]) -> &[NodeIndex] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const NodeIndex, slice.len()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_face_index_validity() {
        let valid = FaceIndex::new(10);
        assert!(valid.is_valid());
        assert_eq!(valid.get(), 10);

        let invalid = FaceIndex::INVALID;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_cell_index_validity() {
        let valid = CellIndex::new(42);
        assert!(valid.is_valid());
        assert_eq!(valid.get(), 42);

        let invalid = CellIndex::INVALID;
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_conversions() {
        let id: CellId = 42.into();
        let idx: CellIndex = id.into();
        let back: usize = idx.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn test_reinterpret_slices() {
        let ids = vec![FaceId(0), FaceId(1), FaceId(2)];
        let indices = reinterpret_face_ids(&ids);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices[0].get(), 0);
        assert_eq!(indices[1].get(), 1);
        assert_eq!(indices[2].get(), 2);
    }

    #[test]
    fn test_size_and_alignment() {
        assert_eq!(std::mem::size_of::<FaceIndex>(), std::mem::size_of::<usize>());
        assert_eq!(std::mem::size_of::<CellIndex>(), std::mem::size_of::<usize>());
        assert_eq!(std::mem::size_of::<NodeIndex>(), std::mem::size_of::<usize>());
    }
}
