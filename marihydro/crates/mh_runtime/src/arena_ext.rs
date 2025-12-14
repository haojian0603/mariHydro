// crates/mh_runtime/src/arena_ext.rs

//! 代际验证内存池（Generation-Checked Arena）
//!
//! 这是 Runtime 层的核心安全机制。代际验证是"内存时间的版本号"，
//! 用于检测悬垂引用（dangling reference）和 use-after-free 问题。
//!
//! # 架构设计
//!
//! - **Layer 1 (Foundation)**: `Idx<Tag>` 只有 index（4字节），`Arena` 是简单内存池
//! - **Layer 2 (Runtime)**: `SafeIdx<Tag>` 带 generation（8字节），`SafeArena` 做代际验证
//!
//! # 核心概念
//!
//! - **Generation（代际）**: 每个槽位被重用时递增的版本号
//! - **SafeIdx**: 包含 (index, generation) 的复合索引
//! - **Stale Index（过期索引）**: 当 arena 中槽位的 generation
//!   与索引的 generation 不匹配时，该索引已过期

use mh_foundation::index::{CellTag, FaceTag, NodeTag, BoundaryTag, HalfEdgeTag, VertexTag, INVALID_INDEX};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

// =============================================================================
// SafeIdx - 带代际验证的索引（Runtime 层）
// =============================================================================

/// 无效代际标记
pub const INVALID_GENERATION: u32 = 0;

/// 带代际验证的安全索引
///
/// 与 Foundation 的 `Idx<Tag>` 不同，此类型包含 generation 字段用于悬垂检测。
///
/// # 内存布局
///
/// 使用两个 u32 字段，总共 8 字节：
/// - `index`: 实际索引值 (0 到 2^32-2)
/// - `generation`: 代际号 (1 起始，0 表示无效)
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct SafeIdx<Tag> {
    /// 索引值
    index: u32,
    /// 代际号 (用于检测悬垂引用)
    generation: u32,
    /// 类型标记
    #[serde(skip)]
    _marker: PhantomData<fn() -> Tag>,
}

impl<Tag> Copy for SafeIdx<Tag> {}

impl<Tag> Clone for SafeIdx<Tag> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Tag> SafeIdx<Tag> {
    /// 无效索引常量
    pub const INVALID: Self = Self {
        index: INVALID_INDEX,
        generation: INVALID_GENERATION,
        _marker: PhantomData,
    };

    /// 创建新索引
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

    /// 从原始 u32 创建（代际默认为1）
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

impl<Tag> Default for SafeIdx<Tag> {
    fn default() -> Self {
        Self::INVALID
    }
}

impl<Tag> PartialEq for SafeIdx<Tag> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<Tag> Eq for SafeIdx<Tag> {}

impl<Tag> PartialOrd for SafeIdx<Tag> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<Tag> Ord for SafeIdx<Tag> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.generation.cmp(&other.generation) {
            std::cmp::Ordering::Equal => self.index.cmp(&other.index),
            other => other,
        }
    }
}

impl<Tag> Hash for SafeIdx<Tag> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

impl<Tag> fmt::Debug for SafeIdx<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "SafeIdx({}@{})", self.index, self.generation)
        } else {
            write!(f, "SafeIdx(INVALID)")
        }
    }
}

impl<Tag> fmt::Display for SafeIdx<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid() {
            write!(f, "{}@{}", self.index, self.generation)
        } else {
            write!(f, "INVALID")
        }
    }
}

// =============================================================================
// 便捷类型别名
// =============================================================================

/// 单元安全索引
pub type SafeCellIndex = SafeIdx<CellTag>;
/// 面/边安全索引
pub type SafeFaceIndex = SafeIdx<FaceTag>;
/// 节点安全索引
pub type SafeNodeIndex = SafeIdx<NodeTag>;
/// 边界安全索引
pub type SafeBoundaryIndex = SafeIdx<BoundaryTag>;
/// 顶点安全索引
pub type SafeVertexIndex = SafeIdx<VertexTag>;
/// 半边安全索引
pub type SafeHalfEdgeIndex = SafeIdx<HalfEdgeTag>;

// 重导出标记类型
pub use mh_foundation::index::{
    CellTag as SafeCellTag,
    FaceTag as SafeFaceTag,
    NodeTag as SafeNodeTag,
    BoundaryTag as SafeBoundaryTag,
};

// =============================================================================
// SafeArena - 带代际验证的内存池（Runtime 层）
// =============================================================================

/// Arena 中的槽位
#[derive(Debug, Clone)]
enum Slot<T> {
    /// 已占用的槽位
    Occupied { value: T, generation: u32 },
    /// 空闲槽位，指向下一个空闲位置，保存最后使用的 generation
    Vacant { next_free: Option<u32>, generation: u32 },
}

/// 带代际验证的安全内存池
///
/// 与 Foundation 的 `Arena` 不同，此类型返回带 generation 的 `SafeIdx`，
/// 并在访问时验证代际是否匹配。
#[derive(Debug, Clone)]
pub struct SafeArena<T, Tag> {
    /// 槽位数组
    slots: Vec<Slot<T>>,
    /// 空闲链表头
    free_head: Option<u32>,
    /// 已占用元素数量
    len: usize,
    /// 类型标记
    _marker: PhantomData<Tag>,
}

impl<T, Tag> Default for SafeArena<T, Tag> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Tag> SafeArena<T, Tag> {
    /// 创建空的 SafeArena
    #[inline]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// 创建指定容量的 SafeArena
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// 返回元素数量
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 判断是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 返回容量
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// 返回槽位总数（包括空闲槽位）
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// 插入元素，返回安全索引
    pub fn insert(&mut self, value: T) -> SafeIdx<Tag> {
        match self.free_head {
            Some(free_idx) => {
                let idx = free_idx as usize;
                match &self.slots[idx] {
                    Slot::Vacant { next_free, generation } => {
                        self.free_head = *next_free;
                        let new_generation = generation.wrapping_add(1);
                        let new_generation = if new_generation == 0 { 1 } else { new_generation };
                        self.slots[idx] = Slot::Occupied { value, generation: new_generation };
                        self.len += 1;
                        SafeIdx::new(free_idx, new_generation)
                    }
                    Slot::Occupied { .. } => {
                        panic!("SafeArena corruption: free list points to occupied slot");
                    }
                }
            }
            None => {
                let idx = self.slots.len() as u32;
                let generation = 1;
                self.slots.push(Slot::Occupied { value, generation });
                self.len += 1;
                SafeIdx::new(idx, generation)
            }
        }
    }

    /// 移除元素，返回被移除的值
    ///
    /// 如果索引无效或代际不匹配，返回 None
    pub fn remove(&mut self, idx: SafeIdx<Tag>) -> Option<T> {
        if !idx.is_valid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        let current_generation = match &self.slots[slot_idx] {
            Slot::Occupied { generation, .. } => {
                // 代际验证
                if *generation != idx.generation() {
                    return None;
                }
                *generation
            }
            Slot::Vacant { .. } => return None,
        };

        let old_slot = std::mem::replace(
            &mut self.slots[slot_idx],
            Slot::Vacant {
                next_free: self.free_head,
                generation: current_generation,
            },
        );

        match old_slot {
            Slot::Occupied { value, .. } => {
                self.free_head = Some(idx.index());
                self.len -= 1;
                Some(value)
            }
            Slot::Vacant { .. } => None,
        }
    }

    /// 获取元素的不可变引用
    ///
    /// 如果索引无效或代际不匹配，返回 None
    #[inline]
    pub fn get(&self, idx: SafeIdx<Tag>) -> Option<&T> {
        if !idx.is_valid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        match &self.slots[slot_idx] {
            Slot::Occupied { value, generation } => {
                if *generation == idx.generation() {
                    Some(value)
                } else {
                    None
                }
            }
            Slot::Vacant { .. } => None,
        }
    }

    /// 获取元素的可变引用
    ///
    /// 如果索引无效或代际不匹配，返回 None
    #[inline]
    pub fn get_mut(&mut self, idx: SafeIdx<Tag>) -> Option<&mut T> {
        if !idx.is_valid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        match &mut self.slots[slot_idx] {
            Slot::Occupied { value, generation } => {
                if *generation == idx.generation() {
                    Some(value)
                } else {
                    None
                }
            }
            Slot::Vacant { .. } => None,
        }
    }

    /// 检查索引是否仍然有效
    #[inline]
    pub fn contains(&self, idx: SafeIdx<Tag>) -> bool {
        self.get(idx).is_some()
    }

    /// 清空所有元素
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.len = 0;
    }

    /// 迭代所有有效元素
    pub fn iter(&self) -> impl Iterator<Item = (SafeIdx<Tag>, &T)> {
        self.slots.iter().enumerate().filter_map(|(idx, slot)| {
            match slot {
                Slot::Occupied { value, generation } => {
                    Some((SafeIdx::new(idx as u32, *generation), value))
                }
                Slot::Vacant { .. } => None,
            }
        })
    }

    /// 可变迭代所有有效元素
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (SafeIdx<Tag>, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(idx, slot)| {
            match slot {
                Slot::Occupied { value, generation } => {
                    Some((SafeIdx::new(idx as u32, *generation), value))
                }
                Slot::Vacant { .. } => None,
            }
        })
    }

    /// 只迭代值
    pub fn values(&self) -> impl Iterator<Item = &T> {
        self.slots.iter().filter_map(|slot| {
            match slot {
                Slot::Occupied { value, .. } => Some(value),
                Slot::Vacant { .. } => None,
            }
        })
    }

    /// 只迭代值（可变）
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.slots.iter_mut().filter_map(|slot| {
            match slot {
                Slot::Occupied { value, .. } => Some(value),
                Slot::Vacant { .. } => None,
            }
        })
    }
}

// =============================================================================
// 辅助函数
// =============================================================================

/// 检查两个索引是否引用同一位置（忽略代际）
#[inline]
pub fn same_slot<Tag>(a: SafeIdx<Tag>, b: SafeIdx<Tag>) -> bool {
    a.index() == b.index()
}

/// 检查索引是否比另一个更新（更高的代际）
#[inline]
pub fn is_newer<Tag>(newer: SafeIdx<Tag>, older: SafeIdx<Tag>) -> bool {
    newer.index() == older.index() && newer.generation() > older.generation()
}

// =============================================================================
// 代际验证错误
// =============================================================================

/// 代际验证错误
#[derive(Debug, Clone)]
pub struct StaleIndexError<Tag> {
    /// 过期的索引
    pub index: SafeIdx<Tag>,
    /// 索引的代际
    pub index_generation: u32,
    /// Arena 槽位的当前代际
    pub current_generation: Option<u32>,
    /// 错误描述
    pub message: String,
}

impl<Tag> fmt::Display for StaleIndexError<Tag> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stale index detected: index {} has generation {}, but slot has generation {:?}. {}",
            self.index.index(),
            self.index_generation,
            self.current_generation,
            self.message
        )
    }
}

impl<Tag: fmt::Debug> std::error::Error for StaleIndexError<Tag> {}

// =============================================================================
// 测试
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_arena_basic() {
        let mut arena: SafeArena<i32, CellTag> = SafeArena::new();
        let idx = arena.insert(42);

        assert!(idx.is_valid());
        assert_eq!(idx.generation(), 1);
        assert_eq!(arena.get(idx), Some(&42));
    }

    #[test]
    fn test_safe_arena_remove() {
        let mut arena: SafeArena<i32, CellTag> = SafeArena::new();
        let idx = arena.insert(42);

        assert_eq!(arena.remove(idx), Some(42));
        assert_eq!(arena.get(idx), None); // 旧索引失效
    }

    #[test]
    fn test_generation_increment() {
        let mut arena: SafeArena<i32, CellTag> = SafeArena::new();
        let idx1 = arena.insert(1);
        let gen1 = idx1.generation();

        arena.remove(idx1);

        let idx2 = arena.insert(2);
        // 复用同一槽位，代际应该递增
        if idx1.index() == idx2.index() {
            assert!(idx2.generation() > gen1);
        }

        // 旧索引不能访问新数据
        assert_eq!(arena.get(idx1), None);
        assert_eq!(arena.get(idx2), Some(&2));
    }

    #[test]
    fn test_stale_index_detection() {
        let mut arena: SafeArena<String, CellTag> = SafeArena::new();
        let idx = arena.insert("hello".to_string());

        // 删除后重新插入
        arena.remove(idx);
        let _new_idx = arena.insert("world".to_string());

        // 旧索引无法访问新数据（代际不匹配）
        assert!(arena.get(idx).is_none());
    }

    #[test]
    fn test_same_slot_helper() {
        let idx1 = SafeIdx::<CellTag>::new(5, 1);
        let idx2 = SafeIdx::<CellTag>::new(5, 2);
        let idx3 = SafeIdx::<CellTag>::new(6, 1);

        assert!(same_slot(idx1, idx2));
        assert!(!same_slot(idx1, idx3));
    }

    #[test]
    fn test_is_newer_helper() {
        let idx1 = SafeIdx::<CellTag>::new(5, 1);
        let idx2 = SafeIdx::<CellTag>::new(5, 2);
        let idx3 = SafeIdx::<CellTag>::new(5, 3);

        assert!(is_newer(idx2, idx1));
        assert!(is_newer(idx3, idx1));
        assert!(!is_newer(idx1, idx2));
    }

    #[test]
    fn test_iter() {
        let mut arena: SafeArena<i32, CellTag> = SafeArena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);

        let values: Vec<_> = arena.values().copied().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }
}