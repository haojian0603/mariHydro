//! Memory alignment utilities.
//!
//! Provides a truly aligned AlignedVec backed by std::alloc for SIMD/GPU-friendly access.
//! Includes parallel iterators and Serde support.

use bytemuck::Pod;
use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, realloc, Layout};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// Alignment requirement.
pub trait Alignment: 'static {
    /// Requested byte alignment.
    const ALIGN: usize;
}

/// CPU alignment (64-byte cache line / AVX-512).
#[derive(Debug, Clone, Copy)]
pub struct CpuAlign;
impl Alignment for CpuAlign {
    const ALIGN: usize = 64;
}

/// GPU alignment (256-byte for CUDA coalescing).
#[derive(Debug, Clone, Copy)]
pub struct GpuAlign;
impl Alignment for GpuAlign {
    const ALIGN: usize = 256;
}

/// Default alignment (8-byte).
#[derive(Debug, Clone, Copy)]
pub struct DefaultAlign;
impl Alignment for DefaultAlign {
    const ALIGN: usize = 8;
}

/// 对齐连续缓冲区
#[derive(Debug)]
pub struct AlignedVec<T: Pod + Default, A: Alignment = CpuAlign> {
    ptr: *mut T,
    len: usize,
    capacity: usize,
    _align: PhantomData<A>,
}

unsafe impl<T: Pod + Default + Send, A: Alignment> Send for AlignedVec<T, A> {}
unsafe impl<T: Pod + Default + Sync, A: Alignment> Sync for AlignedVec<T, A> {}

impl<T: Pod + Default, A: Alignment> AlignedVec<T, A> {
    /// Create zero-initialized buffer of length len.
    pub fn zeros(len: usize) -> Self {
        if len == 0 {
            return Self { ptr: std::ptr::null_mut(), len: 0, capacity: 0, _align: PhantomData };
        }

        let layout = Self::layout_for(len);
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        debug_assert_eq!((ptr as usize) % layout.align(), 0, "Alignment guarantee violated");

        Self { ptr, len, capacity: len, _align: PhantomData }
    }

    /// Create empty buffer with specified capacity (zero-filled).
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self { ptr: std::ptr::null_mut(), len: 0, capacity: 0, _align: PhantomData };
        }

        let layout = Self::layout_for(capacity);
        let ptr = unsafe { alloc_zeroed(layout) as *mut T };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }

        debug_assert_eq!((ptr as usize) % layout.align(), 0, "Alignment guarantee violated");

        Self { ptr, len: 0, capacity, _align: PhantomData }
    }

    /// Re-align from an existing Vec.
    pub fn from_vec(vec: Vec<T>) -> Self {
        let len = vec.len();
        let mut aligned = Self::with_capacity(len);
        unsafe {
            std::ptr::copy_nonoverlapping(vec.as_ptr(), aligned.ptr, len);
        }
        std::mem::forget(vec);
        aligned.len = len;
        aligned
    }

    /// 并行只读迭代器
    pub fn par_iter(&self) -> rayon::slice::Iter<'_, T>
    where
        T: Sync,
    {
        self.as_slice().par_iter()
    }

    /// 并行可变迭代器
    pub fn par_iter_mut(&mut self) -> rayon::slice::IterMut<'_, T>
    where
        T: Send + Sync,
    {
        self.as_mut_slice().par_iter_mut()
    }

    /// Parallel fill.
    pub fn par_fill(&mut self, value: T)
    where
        T: Copy + Send + Sync,
    {
        self.as_mut_slice().par_iter_mut().for_each(|v| *v = value);
    }

    /// Resize; newly added region is default-initialized.
    pub fn resize(&mut self, new_len: usize) {
        if new_len > self.capacity {
            self.grow(new_len.max(self.capacity * 2));
        }
        if new_len > self.len {
            let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(self.len), new_len - self.len) };
            slice.fill(T::default());
        }
        self.len = new_len;
    }

    /// Push element (grows if needed).
    pub fn push(&mut self, value: T) {
        if self.len == self.capacity {
            self.grow(self.next_capacity());
        }
        unsafe { self.ptr.add(self.len).write(value); }
        self.len += 1;
    }

    /// Pop last element.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        let val = unsafe { self.ptr.add(self.len).read() };
        Some(val)
    }

    /// Clear length while keeping capacity.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Raw pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable raw pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Length.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Empty check.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Immutable slice view.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Mutable slice view.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    /// Convert into Vec.
    pub fn into_vec(self) -> Vec<T> {
        let slice = self.as_slice().to_vec();
        std::mem::forget(self);
        slice
    }

    #[inline]
    fn layout_for(capacity: usize) -> Layout {
        Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            A::ALIGN.max(std::mem::align_of::<T>()),
        )
        .expect("Invalid layout")
    }

    fn next_capacity(&self) -> usize {
        if self.capacity == 0 { 4 } else { self.capacity * 2 }
    }

    fn grow(&mut self, new_cap: usize) {
        let new_layout = Self::layout_for(new_cap);
        if self.capacity == 0 {
            let ptr = unsafe { alloc_zeroed(new_layout) as *mut T };
            if ptr.is_null() {
                handle_alloc_error(new_layout);
            }
            self.ptr = ptr;
            self.capacity = new_cap;
            return;
        }

        let old_layout = Self::layout_for(self.capacity);
        let new_ptr = unsafe { realloc(self.ptr as *mut u8, old_layout, new_layout.size()) as *mut T };
        if new_ptr.is_null() {
            handle_alloc_error(new_layout);
        }
        self.ptr = new_ptr;

        if new_cap > self.capacity {
            let added = new_cap - self.capacity;
            let start = self.capacity;
            let slice = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), added) };
            slice.fill(T::default());
        }
        self.capacity = new_cap;
    }
}

impl<T: Pod + Default, A: Alignment> Deref for AlignedVec<T, A> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: Pod + Default, A: Alignment> DerefMut for AlignedVec<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<T: Pod + Default + Clone, A: Alignment> Clone for AlignedVec<T, A> {
    fn clone(&self) -> Self {
        let mut new_vec = Self::with_capacity(self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, new_vec.ptr, self.len);
        }
        new_vec.len = self.len;
        new_vec
    }
}

impl<T: Pod + Default, A: Alignment> Default for AlignedVec<T, A> {
    fn default() -> Self {
        Self { ptr: std::ptr::null_mut(), len: 0, capacity: 0, _align: PhantomData }
    }
}

impl<T: Pod + Default, A: Alignment> Drop for AlignedVec<T, A> {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.capacity == 0 {
            return;
        }

        if std::mem::needs_drop::<T>() {
            for i in 0..self.len {
                unsafe { std::ptr::drop_in_place(self.ptr.add(i)) };
            }
        }

        let layout = Self::layout_for(self.capacity);
        unsafe { dealloc(self.ptr as *mut u8, layout) };
    }
}

impl<T: Pod + Default, A: Alignment> FromIterator<T> for AlignedVec<T, A> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();
        Self::from_vec(vec)
    }
}

impl<T: Pod + Default + Serialize, A: Alignment> Serialize for AlignedVec<T, A> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.as_slice().serialize(serializer)
    }
}

impl<'de, T: Pod + Default + Deserialize<'de>, A: Alignment> Deserialize<'de> for AlignedVec<T, A> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec = Vec::<T>::deserialize(deserializer)?;
        Ok(Self::from_vec(vec))
    }
}

/// Preallocation strategy (for ghost/MPI buffers).
pub trait PreallocStrategy {
    /// Given logical size, return actual allocation size.
    fn allocate_size(logical_size: usize) -> usize;
}

/// No preallocation.
pub struct NoPrealloc;
impl PreallocStrategy for NoPrealloc {
    fn allocate_size(logical_size: usize) -> usize {
        logical_size
    }
}

/// MPI ghost-cell preallocation (extra 5%, min 32).
pub struct GhostPrealloc;
impl PreallocStrategy for GhostPrealloc {
    fn allocate_size(logical_size: usize) -> usize {
        let extra = (logical_size / 20).max(32);
        logical_size + extra
    }
}

/// AMR preallocation (extra 25%).
pub struct AmrPrealloc;
impl PreallocStrategy for AmrPrealloc {
    fn allocate_size(logical_size: usize) -> usize {
        logical_size + logical_size / 4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec_basic() {
        let mut vec: AlignedVec<f64, CpuAlign> = AlignedVec::zeros(10);
        assert_eq!(vec.len(), 10);
        vec[0] = 1.5;
        assert!((vec[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_aligned_vec_push_pop() {
        let mut vec: AlignedVec<i32, DefaultAlign> = AlignedVec::default();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.len(), 3);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.len(), 2);
    }

    #[test]
    fn test_aligned_vec_from_iter() {
        let vec: AlignedVec<i32, CpuAlign> = (0..5).collect();
        assert_eq!(vec.len(), 5);
        assert_eq!(vec[0], 0);
        assert_eq!(vec[4], 4);
    }

    #[test]
    fn test_prealloc_strategies() {
        assert_eq!(NoPrealloc::allocate_size(100), 100);
        assert_eq!(GhostPrealloc::allocate_size(100), 132);
        assert_eq!(GhostPrealloc::allocate_size(10), 42);
        assert_eq!(GhostPrealloc::allocate_size(1000), 1050);
        assert_eq!(AmrPrealloc::allocate_size(100), 125);
    }

    #[test]
    fn test_clone() {
        let mut v1: AlignedVec<f64, CpuAlign> = AlignedVec::zeros(5);
        v1[0] = 3.14;
        let v2 = v1.clone();
        assert_eq!(v1.len(), v2.len());
        assert!((v2[0] - 3.14).abs() < 1e-12);
    }

    #[test]
    fn test_aligned_vec_alignment() {
        let vec: AlignedVec<f64, CpuAlign> = AlignedVec::zeros(100);
        assert_eq!((vec.as_ptr() as usize) % 64, 0);

        let vec_gpu: AlignedVec<f64, GpuAlign> = AlignedVec::zeros(100);
        assert_eq!((vec_gpu.as_ptr() as usize) % 256, 0);
    }

    #[test]
    fn test_aligned_vec_serde_roundtrip() {
        let mut v: AlignedVec<f64, CpuAlign> = AlignedVec::zeros(3);
        v[0] = 1.0;
        v[1] = 2.0;
        v[2] = 3.5;

        let json = serde_json::to_string(&v).unwrap();
        let de: AlignedVec<f64, CpuAlign> = serde_json::from_str(&json).unwrap();
        assert_eq!(de.len(), 3);
        assert!((de[2] - 3.5).abs() < 1e-12);
    }
}
