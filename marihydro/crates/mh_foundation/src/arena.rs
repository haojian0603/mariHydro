// marihydro\crates\mh_foundation\src/arena.rs

//! 泛型 Arena 内存池
//!
//! 提供带代际验证的内存池实现，用于高效管理同类型对象。
//!
//! # 设计目标
//!
//! 1. **高效分配**: O(1) 插入和删除
//! 2. **悬垂检测**: 通过代际号检测已删除元素的访问
//! 3. **缓存友好**: 连续内存布局
//! 4. **迭代安全**: 提供安全的迭代器
//!
//! # 示例
//!
//! ```
//! use mh_foundation::arena::Arena;
//! use mh_foundation::index::CellTag;
//!
//! let mut arena: Arena<i32, CellTag> = Arena::new();
//! let idx = arena.insert(42);
//! assert_eq!(arena.get(idx), Some(&42));
//!
//! arena.remove(idx);
//! assert_eq!(arena.get(idx), None);
//! ```

use crate::index::Idx;
use std::marker::PhantomData;

/// Arena 中的槽位
#[derive(Debug, Clone)]
enum Slot<T> {
    /// 已占用的槽位
    Occupied { value: T, generation: u32 },
    /// 空闲槽位，指向下一个空闲位置，保存最后使用的 generation
    Vacant { next_free: Option<u32>, generation: u32 },
}

/// 泛型 Arena 内存池
///
/// # 类型参数
///
/// - `T`: 存储的元素类型
/// - `Tag`: 索引标记类型（用于类型安全）
#[derive(Debug, Clone)]
pub struct Arena<T, Tag> {
    /// 槽位数组
    slots: Vec<Slot<T>>,
    /// 空闲链表头
    free_head: Option<u32>,
    /// 已占用元素数量
    len: usize,
    /// 类型标记
    _marker: PhantomData<Tag>,
}

impl<T, Tag> Default for Arena<T, Tag> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, Tag> Arena<T, Tag> {
    /// 创建空的 Arena
    #[inline]
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_head: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// 创建指定容量的 Arena
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            slots: Vec::with_capacity(capacity),
            free_head: None,
            len: 0,
            _marker: PhantomData,
        }
    }

    /// 返回 Arena 中元素数量
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 判断 Arena 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 返回 Arena 容量
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// 返回槽位总数（包括空闲槽位）
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// 插入元素，返回索引
    pub fn insert(&mut self, value: T) -> Idx<Tag> {
        match self.free_head {
            Some(free_idx) => {
                // 复用空闲槽位
                let idx = free_idx as usize;
                match &self.slots[idx] {
                    Slot::Vacant { next_free, generation } => {
                        self.free_head = *next_free;
                        // 代际递增
                        let new_generation = generation.wrapping_add(1);
                        self.slots[idx] = Slot::Occupied { value, generation: new_generation };
                        self.len += 1;
                        Idx::new(free_idx, new_generation)
                    }
                    Slot::Occupied { .. } => {
                        // 这不应该发生
                        panic!("Arena corruption: free list points to occupied slot");
                    }
                }
            }
            None => {
                // 追加新槽位
                let idx = self.slots.len() as u32;
                let generation = 1;
                self.slots.push(Slot::Occupied { value, generation });
                self.len += 1;
                Idx::new(idx, generation)
            }
        }
    }

    /// 移除元素，返回被移除的值
    pub fn remove(&mut self, idx: Idx<Tag>) -> Option<T> {
        if !idx.is_valid() {
            return None;
        }

        let slot_idx = idx.as_usize();
        if slot_idx >= self.slots.len() {
            return None;
        }

        let current_generation = match &self.slots[slot_idx] {
            Slot::Occupied { generation, .. } => {
                if *generation != idx.generation() {
                    return None;
                }
                *generation
            }
            Slot::Vacant { .. } => return None,
        };

        // 取出值并替换为空闲槽位，保留 generation
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
    #[inline]
    pub fn get(&self, idx: Idx<Tag>) -> Option<&T> {
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
    #[inline]
    pub fn get_mut(&mut self, idx: Idx<Tag>) -> Option<&mut T> {
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

    /// 检查索引是否有效
    #[inline]
    pub fn contains(&self, idx: Idx<Tag>) -> bool {
        self.get(idx).is_some()
    }

    /// 清空 Arena
    pub fn clear(&mut self) {
        self.slots.clear();
        self.free_head = None;
        self.len = 0;
    }

    /// 遍历所有元素（不可变）
    #[inline]
    pub fn iter(&self) -> ArenaIter<'_, T, Tag> {
        ArenaIter {
            slots: &self.slots,
            index: 0,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// 遍历所有元素（可变）
    #[inline]
    pub fn iter_mut(&mut self) -> ArenaIterMut<'_, T, Tag> {
        ArenaIterMut {
            slots: &mut self.slots,
            index: 0,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// 遍历所有索引
    #[inline]
    pub fn indices(&self) -> ArenaIndices<'_, T, Tag> {
        ArenaIndices {
            slots: &self.slots,
            index: 0,
            remaining: self.len,
            _marker: PhantomData,
        }
    }

    /// 通过索引获取元素（不检查代际，仅用于内部优化）
    ///
    /// # Safety
    ///
    /// 调用者必须确保索引有效
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: Idx<Tag>) -> &T {
        match &self.slots[idx.as_usize()] {
            Slot::Occupied { value, .. } => value,
            Slot::Vacant { .. } => std::hint::unreachable_unchecked(),
        }
    }

    /// 通过索引获取元素（不检查代际，仅用于内部优化）
    ///
    /// # Safety
    ///
    /// 调用者必须确保索引有效
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, idx: Idx<Tag>) -> &mut T {
        match &mut self.slots[idx.as_usize()] {
            Slot::Occupied { value, .. } => value,
            Slot::Vacant { .. } => std::hint::unreachable_unchecked(),
        }
    }

    /// 保留额外容量
    pub fn reserve(&mut self, additional: usize) {
        self.slots.reserve(additional);
    }
}

// ============================================================================
// 迭代器
// ============================================================================

/// Arena 不可变迭代器
pub struct ArenaIter<'a, T, Tag> {
    slots: &'a [Slot<T>],
    index: usize,
    remaining: usize,
    _marker: PhantomData<Tag>,
}

impl<'a, T, Tag> Iterator for ArenaIter<'a, T, Tag> {
    type Item = (Idx<Tag>, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.index < self.slots.len() {
            let current_idx = self.index;
            self.index += 1;

            if let Slot::Occupied { value, generation } = &self.slots[current_idx] {
                self.remaining -= 1;
                return Some((Idx::new(current_idx as u32, *generation), value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, Tag> ExactSizeIterator for ArenaIter<'_, T, Tag> {}

/// Arena 可变迭代器
pub struct ArenaIterMut<'a, T, Tag> {
    slots: &'a mut [Slot<T>],
    index: usize,
    remaining: usize,
    _marker: PhantomData<Tag>,
}

impl<'a, T, Tag> Iterator for ArenaIterMut<'a, T, Tag> {
    type Item = (Idx<Tag>, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.index < self.slots.len() {
            let current_idx = self.index;
            self.index += 1;

            // SAFETY: 我们只访问每个元素一次
            let slot = unsafe { &mut *self.slots.as_mut_ptr().add(current_idx) };

            if let Slot::Occupied { value, generation } = slot {
                self.remaining -= 1;
                return Some((Idx::new(current_idx as u32, *generation), value));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, Tag> ExactSizeIterator for ArenaIterMut<'_, T, Tag> {}

/// Arena 索引迭代器
pub struct ArenaIndices<'a, T, Tag> {
    slots: &'a [Slot<T>],
    index: usize,
    remaining: usize,
    _marker: PhantomData<Tag>,
}

impl<T, Tag> Iterator for ArenaIndices<'_, T, Tag> {
    type Item = Idx<Tag>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.remaining > 0 && self.index < self.slots.len() {
            let current_idx = self.index;
            self.index += 1;

            if let Slot::Occupied { generation, .. } = &self.slots[current_idx] {
                self.remaining -= 1;
                return Some(Idx::new(current_idx as u32, *generation));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, Tag> ExactSizeIterator for ArenaIndices<'_, T, Tag> {}

// ============================================================================
// IntoIterator 实现
// ============================================================================

impl<'a, T, Tag> IntoIterator for &'a Arena<T, Tag> {
    type Item = (Idx<Tag>, &'a T);
    type IntoIter = ArenaIter<'a, T, Tag>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, Tag> IntoIterator for &'a mut Arena<T, Tag> {
    type Item = (Idx<Tag>, &'a mut T);
    type IntoIter = ArenaIterMut<'a, T, Tag>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

// ============================================================================
// 索引操作
// ============================================================================

impl<T, Tag> std::ops::Index<Idx<Tag>> for Arena<T, Tag> {
    type Output = T;

    fn index(&self, idx: Idx<Tag>) -> &Self::Output {
        self.get(idx).expect("Invalid arena index")
    }
}

impl<T, Tag> std::ops::IndexMut<Idx<Tag>> for Arena<T, Tag> {
    fn index_mut(&mut self, idx: Idx<Tag>) -> &mut Self::Output {
        self.get_mut(idx).expect("Invalid arena index")
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::CellTag;

    type TestArena = Arena<i32, CellTag>;

    #[test]
    fn test_new_arena() {
        let arena: TestArena = Arena::new();
        assert!(arena.is_empty());
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut arena: TestArena = Arena::new();
        let idx = arena.insert(42);
        
        assert_eq!(arena.len(), 1);
        assert_eq!(arena.get(idx), Some(&42));
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
        // 再次移除应返回 None
        let removed = arena.remove(idx);
        assert_eq!(removed, None);
    }

    #[test]
    fn test_reuse_slot() {
        let mut arena: TestArena = Arena::new();
        let idx1 = arena.insert(1);
        arena.remove(idx1);
        
        // 新插入应复用槽位
        let idx2 = arena.insert(2);
        assert_eq!(idx2.index(), idx1.index());
        
        // 但旧索引不应有效
        assert_eq!(arena.get(idx1), None);
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
        let _ = arena[idx]; // 应该 panic
    }

    #[test]
    fn test_generation_prevents_dangling() {
        let mut arena: TestArena = Arena::new();
        let old_idx = arena.insert(42);
        arena.remove(old_idx);
        
        // 复用槽位
        let new_idx = arena.insert(100);
        
        // 旧索引的代际不匹配，无法访问
        assert!(arena.get(old_idx).is_none());
        
        // 新索引有效
        assert_eq!(arena.get(new_idx), Some(&100));
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut arena: TestArena = Arena::new();
        arena.insert(1);
        arena.insert(2);
        arena.insert(3);

        let iter = arena.iter();
        assert_eq!(iter.len(), 3);
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

        // 插入1000个元素
        for i in 0..1000 {
            indices.push(arena.insert(i));
        }
        assert_eq!(arena.len(), 1000);

        // 删除奇数位置
        for (i, idx) in indices.iter().enumerate() {
            if i % 2 == 1 {
                arena.remove(*idx);
            }
        }
        assert_eq!(arena.len(), 500);

        // 再插入500个
        for i in 0..500 {
            arena.insert(i + 1000);
        }
        assert_eq!(arena.len(), 1000);
    }
}
