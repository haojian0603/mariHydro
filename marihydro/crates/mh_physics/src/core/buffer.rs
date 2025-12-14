//! marihydro\crates\mh_physics\src\core\buffer.rs
//! AlignedVec 包装器与 DeviceBuffer 实现
//!
//! 本模块提供 `AlignedBuffer<T>` 作为 `mh_foundation::AlignedVec<T>` 的 newtype 包装，
//! 从而满足孤儿规则，可以为本地类型实现外部 trait。
//!
//! # 设计说明
//!
//! 由于 `DeviceBuffer` trait 定义在 `mh_runtime`，`AlignedVec` 定义在 `mh_foundation`，
//! 两者都是"外部"类型，无法直接实现。通过 newtype 包装绕过限制。

use bytemuck::Pod;
use mh_runtime::DeviceBuffer;
use mh_foundation::memory::AlignedVec;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// 对齐缓冲区 - AlignedVec 的 newtype 包装
/// 
/// 实现 `DeviceBuffer` trait，使其可以在泛型计算中使用。
#[derive(Debug, Clone)]
pub struct AlignedBuffer<T: Pod + Default> {
    inner: AlignedVec<T>,
}

impl<T: Pod + Default> AlignedBuffer<T> {
    /// 创建指定长度的零初始化缓冲区
    pub fn zeros(len: usize) -> Self {
        Self {
            inner: AlignedVec::zeros(len),
        }
    }
    
    /// 从 AlignedVec 创建
    pub fn from_aligned_vec(inner: AlignedVec<T>) -> Self {
        Self { inner }
    }
    
    /// 转换为 AlignedVec
    pub fn into_inner(self) -> AlignedVec<T> {
        self.inner
    }
    
    /// 获取内部 AlignedVec 引用
    pub fn inner(&self) -> &AlignedVec<T> {
        &self.inner
    }
    
    /// 获取内部 AlignedVec 可变引用
    pub fn inner_mut(&mut self) -> &mut AlignedVec<T> {
        &mut self.inner
    }
}

impl<T: Pod + Default> Deref for AlignedBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.inner.as_slice()
    }
}

impl<T: Pod + Default> DerefMut for AlignedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut_slice()
    }
}

impl<T: Pod + Default> Index<usize> for AlignedBuffer<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.inner.as_slice()[index]
    }
}

impl<T: Pod + Default> IndexMut<usize> for AlignedBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner.as_mut_slice()[index]
    }
}

impl<T: Pod + Clone + Default + Send + Sync> DeviceBuffer<T> for AlignedBuffer<T> {
    fn len(&self) -> usize {
        self.inner.len()
    }
    
    fn as_slice(&self) -> &[T] {
        self.inner.as_slice()
    }
    
    fn as_slice_mut(&mut self) -> &mut [T] {
        self.inner.as_mut_slice()
    }
    
    fn fill(&mut self, value: T) {
        self.inner.as_mut_slice().fill(value);
    }
    
    fn copy_from_slice(&mut self, src: &[T]) {
        // 如果长度不匹配，先调整大小
        if self.inner.len() != src.len() {
            self.inner.resize(src.len());
        }
        self.inner.as_mut_slice().copy_from_slice(src);
    }
    
    fn copy_to_slice(&self, dst: &mut [T]) {
        dst.copy_from_slice(self.inner.as_slice());
    }
    
    fn copy_to_vec(&self) -> Vec<T> {
        self.inner.as_slice().to_vec()
    }
    
    fn resize(&mut self, new_len: usize, _value: T) {
        self.inner.resize(new_len);
    }
    
    fn clear(&mut self) {
        self.inner.resize(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer() {
        let mut buf: AlignedBuffer<f64> = AlignedBuffer::zeros(10);
        buf.fill(1.0);
        assert_eq!(buf[0], 1.0);
        assert_eq!(DeviceBuffer::len(&buf), 10);
    }
}
