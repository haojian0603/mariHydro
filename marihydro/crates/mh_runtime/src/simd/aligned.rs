// crates/mh_runtime/src/simd/aligned.rs

//! 对齐内存分配
//!
//! 提供 SIMD 友好的内存分配，保证：
//! - 64 字节对齐（AVX-512 要求）
//! - 缓存行对齐
//! - 零拷贝视图

#![allow(unsafe_code)]

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use bytemuck::Pod;

/// 对齐要求（字节）
pub const SIMD_ALIGNMENT: usize = 64;

/// 对齐分配的向量
///
/// 保证 64 字节对齐，适用于 SIMD 操作
#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// SAFETY: 只要 T: Send，AlignedVec<T> 也是 Send
unsafe impl<T: Send> Send for AlignedVec<T> {}
// SAFETY: 只要 T: Sync，AlignedVec<T> 也是 Sync
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> AlignedVec<T> {
    /// 对齐要求（字节）
    const ALIGNMENT: usize = SIMD_ALIGNMENT;

    /// 创建空的对齐向量
    pub fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    /// 创建指定容量的对齐向量
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }

        let layout = Self::layout_for(capacity);
        // SAFETY: layout 非零大小
        let ptr = unsafe { alloc::alloc(layout) as *mut T };

        Self {
            ptr: NonNull::new(ptr).expect("内存分配失败"),
            len: 0,
            cap: capacity,
        }
    }

    /// 计算布局
    fn layout_for(capacity: usize) -> Layout {
        let size = std::mem::size_of::<T>() * capacity;
        let align = Self::ALIGNMENT.max(std::mem::align_of::<T>());
        Layout::from_size_align(size, align).expect("无效布局")
    }

    /// 长度
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 容量
    #[inline]
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取指针（用于 SIMD 加载）
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// 获取可变指针
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// 推入元素
    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            self.grow();
        }
        // SAFETY: 已确保有足够容量
        unsafe {
            std::ptr::write(self.ptr.as_ptr().add(self.len), value);
        }
        self.len += 1;
    }

    /// 扩容
    fn grow(&mut self) {
        let new_cap = if self.cap == 0 {
            8
        } else {
            self.cap * 2
        };
        self.reserve(new_cap - self.cap);
    }

    /// 预留额外容量
    pub fn reserve(&mut self, additional: usize) {
        let required = self.len + additional;
        if required <= self.cap {
            return;
        }

        let new_cap = required.max(self.cap * 2);
        let new_layout = Self::layout_for(new_cap);

        let new_ptr = if self.cap == 0 {
            // SAFETY: 新分配
            unsafe { alloc::alloc(new_layout) as *mut T }
        } else {
            let old_layout = Self::layout_for(self.cap);
            // SAFETY: 重新分配
            unsafe {
                alloc::realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_layout.size())
                    as *mut T
            }
        };

        self.ptr = NonNull::new(new_ptr).expect("内存分配失败");
        self.cap = new_cap;
    }

    /// 调整大小
    pub fn resize(&mut self, new_len: usize, value: T)
    where
        T: Clone,
    {
        if new_len > self.cap {
            self.reserve(new_len - self.cap);
        }

        if new_len > self.len {
            for i in self.len..new_len {
                // SAFETY: 已确保有足够容量
                unsafe {
                    std::ptr::write(self.ptr.as_ptr().add(i), value.clone());
                }
            }
        } else if new_len < self.len {
            for i in new_len..self.len {
                // SAFETY: 需要析构多余元素
                unsafe {
                    std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
            }
        }
        self.len = new_len;
    }

    /// 清空
    pub fn clear(&mut self) {
        for i in 0..self.len {
            // SAFETY: 需要析构元素
            unsafe {
                std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
        }
        self.len = 0;
    }

    /// 填充值
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        for i in 0..self.len {
            // SAFETY: 索引有效
            unsafe {
                *self.ptr.as_ptr().add(i) = value.clone();
            }
        }
    }

    /// 从切片创建
    pub fn from_slice(slice: &[T]) -> Self
    where
        T: Clone,
    {
        let mut vec = Self::with_capacity(slice.len());
        for item in slice {
            vec.push(item.clone());
        }
        vec
    }

    /// 检查对齐
    #[inline]
    pub fn is_aligned(&self) -> bool {
        (self.ptr.as_ptr() as usize) % Self::ALIGNMENT == 0
    }
}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.cap > 0 {
            // 析构元素
            for i in 0..self.len {
                // SAFETY: 索引有效
                unsafe {
                    std::ptr::drop_in_place(self.ptr.as_ptr().add(i));
                }
            }
            // 释放内存
            let layout = Self::layout_for(self.cap);
            // SAFETY: 释放之前分配的内存
            unsafe {
                alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        // SAFETY: 长度有效
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> DerefMut for AlignedVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: 长度有效
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

/// 对齐的 f64 缓冲区（常用类型别名）
pub type AlignedF64 = AlignedVec<f64>;

/// 对齐的 f32 缓冲区
pub type AlignedF32 = AlignedVec<f32>;

/// 创建对齐并初始化为零的缓冲区
pub fn aligned_zeros<T: Pod + Default>(len: usize) -> AlignedVec<T> {
    let mut vec = AlignedVec::with_capacity(len);
    vec.resize(len, T::default());
    vec
}

/// 创建对齐并初始化为指定值的缓冲区
pub fn aligned_filled<T: Clone>(len: usize, value: T) -> AlignedVec<T> {
    let mut vec = AlignedVec::with_capacity(len);
    vec.resize(len, value);
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_basic() {
        let mut vec: AlignedVec<f64> = AlignedVec::new();
        assert!(vec.is_empty());

        vec.push(1.0);
        vec.push(2.0);
        vec.push(3.0);

        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[1], 2.0);
        assert_eq!(vec[2], 3.0);
    }

    #[test]
    fn test_alignment() {
        let vec: AlignedVec<f64> = aligned_zeros(100);
        assert!(vec.is_aligned());
        assert_eq!((vec.as_ptr() as usize) % SIMD_ALIGNMENT, 0);
    }

    #[test]
    fn test_resize() {
        let mut vec: AlignedVec<f64> = aligned_zeros(10);
        assert_eq!(vec.len(), 10);

        vec.resize(20, 0.0);
        assert_eq!(vec.len(), 20);

        vec.resize(5, 0.0);
        assert_eq!(vec.len(), 5);
    }

    #[test]
    fn test_fill() {
        let mut vec: AlignedVec<f64> = aligned_zeros(10);
        vec.fill(42.0);

        for &val in vec.iter() {
            assert_eq!(val, 42.0);
        }
    }

    #[test]
    fn test_clone() {
        let original: AlignedVec<f64> = aligned_filled(10, 3.14);
        let cloned = original.clone();

        assert_eq!(original.len(), cloned.len());
        for (a, b) in original.iter().zip(cloned.iter()) {
            assert_eq!(a, b);
        }
    }
}
