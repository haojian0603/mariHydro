// crates/mh_core/src/buffer.rs

//! 设备缓冲区抽象
//!
//! 提供CPU/GPU统一的缓冲区接口。

use bytemuck::Pod;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// 设备缓冲区trait
///
/// 抽象CPU Vec和GPU缓冲区的统一接口。
pub trait DeviceBuffer<T: Pod + Send + Sync>: 
    Clone + Send + Sync + Index<usize, Output = T> + IndexMut<usize>
{
    /// 缓冲区长度
    fn len(&self) -> usize;

    /// 是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 获取切片引用
    fn as_slice(&self) -> &[T];

    /// 获取可变切片引用
    fn as_slice_mut(&mut self) -> &mut [T];

    /// 填充值
    fn fill(&mut self, value: T);

    /// 从切片复制
    fn copy_from_slice(&mut self, src: &[T]);

    /// 复制到切片
    fn copy_to_slice(&self, dst: &mut [T]);

    /// 调整大小
    fn resize(&mut self, new_len: usize, value: T);

    /// 清空
    fn clear(&mut self);
}

// ============================================================================
// Vec<T> 实现 DeviceBuffer
// ============================================================================

impl<T: Pod + Clone + Send + Sync> DeviceBuffer<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn as_slice(&self) -> &[T] {
        self.as_ref()
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [T] {
        self.as_mut()
    }

    #[inline]
    fn fill(&mut self, value: T) {
        self.iter_mut().for_each(|x| *x = value);
    }

    #[inline]
    fn copy_from_slice(&mut self, src: &[T]) {
        self.as_mut_slice().copy_from_slice(src);
    }

    #[inline]
    fn copy_to_slice(&self, dst: &mut [T]) {
        dst.copy_from_slice(self.as_slice());
    }

    #[inline]
    fn resize(&mut self, new_len: usize, value: T) {
        Vec::resize(self, new_len, value);
    }

    #[inline]
    fn clear(&mut self) {
        Vec::clear(self);
    }
}

/// CPU缓冲区包装器（用于类型区分）
#[derive(Debug, Clone)]
pub struct CpuBuffer<T> {
    data: Vec<T>,
}

impl<T: Clone> CpuBuffer<T> {
    /// 创建新缓冲区
    pub fn new(len: usize, init: T) -> Self {
        Self {
            data: vec![init; len],
        }
    }

    /// 从Vec创建
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// 转换为Vec
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }
}

impl<T> Deref for CpuBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for CpuBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> Index<usize> for CpuBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for CpuBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Pod + Clone + Send + Sync> DeviceBuffer<T> for CpuBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    fn fill(&mut self, value: T) {
        self.data.fill(value);
    }

    fn copy_from_slice(&mut self, src: &[T]) {
        self.data.copy_from_slice(src);
    }

    fn copy_to_slice(&self, dst: &mut [T]) {
        dst.copy_from_slice(&self.data);
    }

    fn resize(&mut self, new_len: usize, value: T) {
        self.data.resize(new_len, value);
    }

    fn clear(&mut self) {
        self.data.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_as_device_buffer() {
        let mut buf: Vec<f64> = vec![0.0; 10];
        buf.fill(1.0);
        assert_eq!(buf[0], 1.0);
        assert_eq!(buf.len(), 10);
    }

    #[test]
    fn test_cpu_buffer() {
        let mut buf = CpuBuffer::new(5, 0.0f32);
        buf[0] = 1.0;
        assert_eq!(buf[0], 1.0);
        assert_eq!(buf.len(), 5);
    }
}
