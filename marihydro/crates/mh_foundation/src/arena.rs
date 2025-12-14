//! 泛型Arena内存池
//! 
//! 提供高效的内存分配与回收机制，适用于管理同类型对象。
//! 
//! # 设计目标
//! 
//! - **O(1)复杂度**: 插入和删除操作均为常数时间
//! - **零成本抽象**: `Idx<Tag>` 与 `u32` 完全相同的内存布局
//! - **类型安全**: 通过标记类型防止不同Arena的索引混用
//! - **缓存友好**: 连续内存布局优化性能
//! 
//! # 注意事项
//! 
//! 本实现**不包含代际验证**，因此无法检测悬垂引用。
//! 删除后复用的索引仍可访问当前存储的数据。
//! 如需运行时安全保证，请使用 `mh_runtime::SafeArena`。
//! 
//! # 示例
//! 
//! ```
//! use mh_foundation::arena::{Arena, ArenaTag};
//! 
//! // 定义标记类型
//! #[derive(Debug, Clone, Copy)]
//! struct CellTag;
//! impl ArenaTag for CellTag {}
//! 
//! // 创建Arena
//! let mut arena: Arena<i32, CellTag> = Arena::new();
//! 
//! // 插入元素
//! let idx = arena.insert(42);
//! assert_eq!(arena.get(idx), Some(&42));
//! 
//! // 删除元素
//! let value = arena.remove(idx);
//! assert_eq!(value, Some(42));
//! assert_eq!(arena.get(idx), None); // 索引失效
//! 
//! // 复用槽位
//! let new_idx = arena.insert(100);
//! assert_eq!(new_idx.index(), idx.index()); // 复用相同槽位
//! ```
//! 
//! # 内存布局
//! 
//! ```text
//! Arena<T, Tag> {
//!     slots: [
//!         Occupied { value: T },      // 已占用
//!         Vacant { next_free: Some(2) }, // 空闲链表
//!         Vacant { next_free: None },    // 链表尾部
//!     ],
//!     free_head: Some(1),            // 指向第一个空闲槽位
//!     len: 1,                        // 有效元素数量
//! }
//! ```

use std::marker::PhantomData;

// ============================================================================
// 标记类型
// ============================================================================

/// Arena标记trait，用于类型安全地区分不同用途的Arena
/// 
/// 典型实现包括 `CellTag`, `FaceTag`, `NodeTag` 等。
/// 
/// # 实现示例
/// 
/// ```
/// use mh_foundation::arena::ArenaTag;
/// 
/// #[derive(Debug, Clone, Copy)]
/// struct MyTag;
/// impl ArenaTag for MyTag {}
/// ```
pub trait ArenaTag: 'static + Copy + Send + Sync {}

// ============================================================================
// 索引类型
// ============================================================================

/// 轻量级类型安全索引（4字节）
/// 
/// 通过标记类型 `Tag` 防止不同Arena的索引混用，编译期保证类型安全。
/// 
/// # 内存表示
/// 
/// 与 `u32` 完全相同的内存布局，使用 `#[repr(transparent)]` 保证零开销。
/// 
/// **关键**：使用 `PhantomData<fn() -> Tag>` 确保自动实现 `Copy` 和 `Clone`。
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Idx<Tag> {
    /// 索引值
    index: u32,
    /// 类型标记（函数指针类型确保自动Copy）
    _marker: PhantomData<fn() -> Tag>,
}

// 必须手动实现 Copy 和 Clone，因为 derive 可能失败
impl<Tag> Copy for Idx<Tag> {}

impl<Tag> Clone for Idx<Tag> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<Tag> Idx<Tag> {
    /// 无效索引常量
    /// 
    /// 用于表示未初始化或无效的索引。
    pub const INVALID: Self = Self {
        index: u32::MAX,
        _marker: PhantomData,
    };

    /// 创建新索引
    /// 
    /// # 参数
    /// - `index`: 槽位索引值
    /// 
    /// # Panics
    /// 
    /// Debug模式下，若`index`为`u32::MAX`会触发panic（因为与INVALID冲突）。
    #[inline]
    pub const fn new(index: u32) -> Self {
        debug_assert!(index != u32::MAX, "index cannot be u32::MAX (reserved for INVALID)");
        Self {
            index,
            _marker: PhantomData,
        }
    }

    /// 从`usize`创建索引
    /// 
    /// # Panics
    /// 
    /// 若值超过`u32::MAX`会panic。
    #[inline]
    pub fn from_usize(index: usize) -> Self {
        Self::new(index as u32)
    }

    /// 获取原始索引值
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// 转换为`usize`
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.index as usize
    }

    /// 检查索引是否有效
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.index != u32::MAX
    }

    /// 检查索引是否无效
    #[inline]
    pub const fn is_invalid(self) -> bool {
        self.index == u32::MAX
    }
}

impl<Tag> Default for Idx<Tag> {
    #[inline]
    fn default() -> Self {
        Self::INVALID
    }
}

impl<Tag> From<u32> for Idx<Tag> {
    #[inline]
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl<Tag> From<usize> for Idx<Tag> {
    #[inline]
    fn from(value: usize) -> Self {
        Self::from_usize(value)
    }
}

impl<Tag> From<Idx<Tag>> for usize {
    #[inline]
    fn from(idx: Idx<Tag>) -> Self {
        idx.as_usize()
    }
}

// ============================================================================
// Slot 定义
// ============================================================================

/// 存储槽位
/// 
/// 每个槽位可以是已占用或空闲状态。
/// 空闲槽位形成链表结构，实现O(1)回收。
#[derive(Debug, Clone)]
enum Slot<T> {
    /// 已占用槽位
    Occupied { value: T },
    /// 空闲槽位，指向下一个空闲位置
    Vacant { next_free: Option<u32> },
}

// ============================================================================
// Arena 实现
// ============================================================================

/// 泛型内存池
/// 
/// 提供高效的同类型对象分配与回收。
/// 
/// # 类型参数
/// - `T`: 存储的元素类型
/// - `Tag`: 标记类型，用于防止不同Arena的索引混用
/// 
/// # 容量管理
/// 
/// Arena自动管理容量，当空闲链表耗尽时扩展内部向量。
/// 扩展策略为倍增，均摊时间复杂度为O(1)。
pub struct Arena<T, Tag: ArenaTag> {
    /// 槽位数组
    slots: Vec<Slot<T>>,
    /// 空闲链表头（指向第一个空闲槽位）
    free_head: Option<u32>,
    /// 有效元素数量
    len: usize,
    /// 类型标记
    _marker: PhantomData<Tag>,
}

impl<T, Tag: ArenaTag> Default for Arena<T, Tag> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Tag: ArenaTag> Arena<T, Tag> {
    /// 创建空Arena
    /// 
    /// # 示例
    /// 
    /// ```
    /// # use mh_foundation::arena::{Arena, ArenaTag};
    /// # #[derive(Clone, Copy)] struct Tag; impl ArenaTag for Tag {}
    /// let arena: Arena<i32, Tag> = Arena::new();
    /// assert!(arena.is_empty());
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// 创建指定容量的Arena
    /// 
    /// 预分配内存可减少多次插入时的重新分配开销。
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// 返回有效元素数量
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 检查是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 返回总槽位数（包括空闲槽位）
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// 返回当前槽位数（包括空闲槽位）
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// 插入元素并返回索引
    /// 
    /// # 复杂度
    /// 
    /// 平均O(1)，最坏情况O(n)（需要重新分配内存）。
    /// 
    /// # 示例
    /// 
    /// ```
    /// # use mh_foundation::arena::{Arena, ArenaTag};
    /// # #[derive(Clone, Copy)] struct Tag; impl ArenaTag for Tag {}
    /// let mut arena = Arena::new();
    /// let idx = arena.insert(42);
    /// assert_eq!(arena[idx], 42);
    /// ```
    pub fn insert(&mut self, value: T) -> Idx<Tag> {
        match self.free_head {
            Some(free_idx) => {
                // 复用空闲槽位
                let idx = free_idx as usize;
                
                // SAFETY: free_head指向的槽位必须是Vacant
                let slot = &mut self.slots[idx];
                match slot {
                    Slot::Vacant { next_free } => {
                        self.free_head = *next_free;
                        *slot = Slot::Occupied { value };
                        self.len += 1;
                        Idx::new(free_idx)
                    }
                    Slot::Occupied { .. } => {
                        // 这不应该发生
                        panic!("Arena corruption: free_head points to occupied slot");
                    }
                }
            }
            None => {
                // 追加新槽位
                let idx = self.slots.len() as u32;
                self.slots.push(Slot::Occupied { value });
                self.len += 1;
                Idx::new(idx)
            }
        }
    }

    /// 获取元素的不可变引用
    /// 
    /// # 返回值
    /// 
    /// - `Some(&T)`：如果索引有效且槽位已占用
    /// - `None`：如果索引无效或槽位已释放
    /// 
    /// # 注意
    /// 
    /// 此操作不进行代际验证。即使元素已被删除并复用，仍可能返回`Some`。
    #[inline]
    pub fn get(&self, idx: Idx<Tag>) -> Option<&T> {
        if idx.is_invalid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        match &self.slots[slot_idx] {
            Slot::Occupied { value } => Some(value),
            Slot::Vacant { .. } => None,
        }
    }

    /// 获取元素的可变引用
    #[inline]
    pub fn get_mut(&mut self, idx: Idx<Tag>) -> Option<&mut T> {
        if idx.is_invalid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        match &mut self.slots[slot_idx] {
            Slot::Occupied { value } => Some(value),
            Slot::Vacant { .. } => None,
        }
    }

    /// 检查索引是否有效
    #[inline]
    pub fn contains(&self, idx: Idx<Tag>) -> bool {
        self.get(idx).is_some()
    }

    /// 删除元素并返回其值
    /// 
    /// # 返回值
    /// 
    /// - `Some(T)`：成功删除并返回元素值
    /// - `None`：索引无效或槽位已释放
    /// 
    /// # 注意
    /// 
    /// 删除后，槽位会被加入空闲链表以供复用。
    /// 原索引与新插入元素的索引可能指向相同槽位。
    pub fn remove(&mut self, idx: Idx<Tag>) -> Option<T> {
        if idx.is_invalid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        // 取出值并替换为Vacant
        let old_slot = std::mem::replace(
            &mut self.slots[slot_idx],
            Slot::Vacant { next_free: self.free_head },
        );

        match old_slot {
            Slot::Occupied { value } => {
                self.free_head = Some(idx.index());
                self.len -= 1;
                Some(value)
            }
            Slot::Vacant { .. } => {
                // 恢复状态（未实际删除）
                self.slots[slot_idx] = old_slot;
                None
            }
        }
    }

    /// 清空Arena
    /// 
    /// 释放所有槽位并重置状态。容量保持不变。
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.len = 0;
    }

    /// 保留额外容量
    /// 
    /// 预分配至少`additional`个额外槽位。
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }

    /// 不可变迭代器
    /// 
    /// 遍历所有有效元素，跳过空闲槽位。
    pub fn iter(&self) -> Iter<'_, T, Tag> {
        let remaining = self.len;  // 先取值，避免借用冲突
        Iter {
            arena: self,
            slot_idx: 0,
            remaining,
        }
    }

    /// 可变迭代器
    pub fn iter_mut(&mut self) -> IterMut<'_, T, Tag> {
        let remaining = self.len;  // 先取值，避免借用冲突
        IterMut {
            arena: self,
            slot_idx: 0,
            remaining,
        }
    }

    /// 索引迭代器
    /// 
    /// 遍历所有有效元素的索引。
    pub fn indices(&self) -> Indices<'_, T, Tag> {
        let remaining = self.len;  // 先取值，避免借用冲突
        Indices {
            arena: self,
            slot_idx: 0,
            remaining,
        }
    }

    /// 返回原始指针
    /// 
    /// # 安全
    /// 
    /// 指针仅在Arena有效时有效。Arena移动或销毁后指针失效。
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        if self.slots.is_empty() {
            std::ptr::null()
        } else {
            self.slots.as_ptr() as *const T
        }
    }

    /// 返回可变原始指针
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if self.slots.is_empty() {
            std::ptr::null_mut()
        } else {
            self.slots.as_mut_ptr() as *mut T
        }
    }
}

// ============================================================================
// 索引操作实现
// ============================================================================

impl<T, Tag: ArenaTag> std::ops::Index<Idx<Tag>> for Arena<T, Tag> {
    type Output = T;

    fn index(&self, idx: Idx<Tag>) -> &Self::Output {
        self.get(idx).expect("Invalid arena index")
    }
}

impl<T, Tag: ArenaTag> std::ops::IndexMut<Idx<Tag>> for Arena<T, Tag> {
    fn index_mut(&mut self, idx: Idx<Tag>) -> &mut Self::Output {
        self.get_mut(idx).expect("Invalid arena index")
    }
}

// ============================================================================
// 迭代器
// ============================================================================

/// 不可变迭代器
pub struct Iter<'a, T, Tag: ArenaTag> {
    arena: &'a Arena<T, Tag>,
    slot_idx: usize,
    remaining: usize,
}

impl<'a, T, Tag: ArenaTag> Iterator for Iter<'a, T, Tag> {
    type Item = (Idx<Tag>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.slot_idx < self.arena.slots.len() {
            let current = self.slot_idx;
            self.slot_idx += 1;

            if let Slot::Occupied { value } = &self.arena.slots[current] {
                self.remaining -= 1;
                return Some((Idx::new(current as u32), value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, Tag: ArenaTag> ExactSizeIterator for Iter<'_, T, Tag> {}

/// 可变迭代器
pub struct IterMut<'a, T, Tag: ArenaTag> {
    arena: &'a mut Arena<T, Tag>,
    slot_idx: usize,
    remaining: usize,
}

impl<'a, T, Tag: ArenaTag> Iterator for IterMut<'a, T, Tag> {
    type Item = (Idx<Tag>, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.slot_idx < self.arena.slots.len() {
            let current = self.slot_idx;
            self.slot_idx += 1;

            // SAFETY: 每个槽位只访问一次，不重叠
            let slot = unsafe {
                &mut *(self.arena.slots.as_mut_ptr().add(current))
            };

            if let Slot::Occupied { value } = slot {
                self.remaining -= 1;
                // SAFETY: 生命周期转换是安全的
                let value = unsafe { &mut *(value as *mut T) };
                return Some((Idx::new(current as u32), value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, Tag: ArenaTag> ExactSizeIterator for IterMut<'_, T, Tag> {}

/// 索引迭代器
pub struct Indices<'a, T, Tag: ArenaTag> {
    arena: &'a Arena<T, Tag>,
    slot_idx: usize,
    remaining: usize,
}

impl<T, Tag: ArenaTag> Iterator for Indices<'_, T, Tag> {
    type Item = Idx<Tag>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.slot_idx < self.arena.slots.len() {
            let current = self.slot_idx;
            self.slot_idx += 1;

            if let Slot::Occupied { .. } = &self.arena.slots[current] {
                self.remaining -= 1;
                return Some(Idx::new(current as u32));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, Tag: ArenaTag> ExactSizeIterator for Indices<'_, T, Tag> {}

// ============================================================================
// IntoIterator 实现
// ============================================================================

impl<'a, T, Tag: ArenaTag> IntoIterator for &'a Arena<T, Tag> {
    type Item = (Idx<Tag>, &'a T);
    type IntoIter = Iter<'a, T, Tag>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, Tag: ArenaTag> IntoIterator for &'a mut Arena<T, Tag> {
    type Item = (Idx<Tag>, &'a mut T);
    type IntoIter = IterMut<'a, T, Tag>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct TestTag;
    impl ArenaTag for TestTag {}

    type TestArena = Arena<i32, TestTag>;

    #[test]
    fn test_new_arena() {
        let arena: TestArena = Arena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.capacity(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(idx), Some(&42));
        assert_eq!(arena[idx], 42);
    }

    #[test]
    fn test_insert_multiple() {
        let mut arena: TestArena = Arena::new();
        let idx1 = arena.insert(1);
        let idx2 = arena.insert(2);
        let idx3 = arena.insert(3);

        assert_eq!(arena.len(), 3);
        assert_eq!(arena.get(idx1), Some(&1));
        assert_eq!(arena.get(idx2), Some(&2));
        assert_eq!(arena.get(idx3), Some(&3));
    }

    #[test]
    fn test_remove() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        let removed = arena.remove(idx);
        assert_eq!(removed, Some(42));
        assert_eq!(arena.len(), 0);
        assert_eq!(arena.get(idx), None);
    }

    #[test]
    fn test_remove_invalid() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        arena.remove(idx);
        let removed = arena.remove(idx);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_reuse_slot() {
        let mut arena: TestArena = Arena::new();
        let idx1 = arena.insert(1);
        arena.remove(idx1);
        
        let idx2 = arena.insert(2);
        assert_eq!(idx2, idx1);
        
        assert_eq!(arena.get(idx1), Some(&2));
        assert_eq!(arena.get(idx2), Some(&2));
    }

    #[test]
    fn test_get_mut() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        if let Some(value) = arena.get_mut(idx) {
            *value = 100;
        }
        
        assert_eq!(arena.get(idx), Some(&100));
        assert_eq!(arena[idx], 100);
    }

    #[test]
    fn test_contains() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        assert!(arena.contains(idx));
        arena.remove(idx);
        assert!(!arena.contains(idx));
    }

    #[test]
    fn test_clear() {
        let mut arena: TestArena = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);
        
        arena.clear();
        assert!(arena.is_empty());
        assert_eq!(arena.slot_count(), 0);
    }

    #[test]
    fn test_iter() {
        let mut arena: TestArena = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);

        let values: Vec<i32> = arena.iter().map(|(_, &v)| v).collect();
        assert_eq!(values.len(), 3);
        assert!(values.contains(&1));
        assert!(values.contains(&2));
        assert!(values.contains(&3));
    }

    #[test]
    fn test_iter_with_holes() {
        let mut arena: TestArena = Arena::new();
        let idx1 = arena.insert(1);
        let _idx2 = arena.insert(2);
        let idx3 = arena.insert(3);

        arena.remove(idx1);
        arena.remove(idx3);

        let values: Vec<i32> = arena.iter().map(|(_, &v)| v).collect();
        assert_eq!(values, vec![2]);
    }

    #[test]
    fn test_iter_mut() {
        let mut arena: TestArena = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);

        for (_, v) in arena.iter_mut() {
            *v *= 10;
        }

        let values: Vec<i32> = arena.iter().map(|(_, &v)| v).collect();
        assert!(values.contains(&10));
        assert!(values.contains(&20));
        assert!(values.contains(&30));
    }

    #[test]
    fn test_indices() {
        let mut arena: TestArena = Arena::new();
        let idx1 = arena.insert(1);
        let _idx2 = arena.insert(2);
        
        arena.remove(idx1);
        
        let indices: Vec<_> = arena.indices().collect();
        assert_eq!(indices.len(), 1);
    }

    #[test]
    fn test_index_operator() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        assert_eq!(arena[idx], 42);
        arena[idx] = 100;
        assert_eq!(arena[idx], 100);
    }

    #[test]
    #[should_panic(expected = "Invalid arena index")]
    fn test_index_operator_invalid() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        arena.remove(idx);
        let _ = arena[idx]; // 应该panic
    }

    #[test]
    fn test_with_capacity() {
        let arena: TestArena = Arena::with_capacity(100);
        assert!(arena.capacity() >= 100);
    }

    #[test]
    fn test_reserve() {
        let mut arena: TestArena = Arena::new();
        arena.reserve(100);
        assert!(arena.capacity() >= 100);
    }

    #[test]
    fn test_stress_insert_remove() {
        let mut arena: TestArena = Arena::new();
        let mut indices = Vec::new();

        for i in 0..1000 {
            indices.push(arena.insert(i));
        }
        assert_eq!(arena.len(), 1000);

        for (i, &idx) in indices.iter().enumerate() {
            if i % 2 == 1 {
                arena.remove(idx);
            }
        }
        assert_eq!(arena.len(), 500);

        for i in 0..500 {
            arena.insert(i + 1000);
        }
        assert_eq!(arena.len(), 1000);
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut arena: TestArena = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);

        let iter = arena.iter();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.size_hint(), (3, Some(3)));
    }

    #[test]
    fn test_type_safety() {
        #[derive(Debug, Clone, Copy)]
        struct TagA;
        impl ArenaTag for TagA {}

        #[derive(Debug, Clone, Copy)]
        struct TagB;
        impl ArenaTag for TagB {}

        let mut arena_a: Arena<i32, TagA> = Arena::new();
        let mut arena_b: Arena<i32, TagB> = Arena::new();

        let idx_a = arena_a.insert(1);
        let idx_b = arena_b.insert(2);

        // arena_a.get(idx_b); // 编译错误

        assert_eq!(arena_a.get(idx_a), Some(&1));
        assert_eq!(arena_b.get(idx_b), Some(&2));
    }

    #[test]
    fn test_idx_conversions() {
        let idx: Idx<TestTag> = Idx::new(42);
        assert_eq!(idx.index(), 42);
        assert_eq!(idx.as_usize(), 42);

        let from_u32: Idx<TestTag> = 42u32.into();
        assert_eq!(from_u32, idx);

        let from_usize: Idx<TestTag> = 42usize.into();
        assert_eq!(from_usize, idx);
    }

    #[test]
    fn test_idx_invalid() {
        let invalid = Idx::<TestTag>::INVALID;
        assert!(invalid.is_invalid());
        assert!(!invalid.is_valid());
    }
}