// crates/mh_runtime/src/buffer.rs

//! DeviceBuffer - 设备缓冲区抽象
//!
//! 提供统一的缓冲区接口，支持 CPU 向量和未来的 GPU 缓冲区。

use bytemuck::Pod;
use std::ops::{Index, IndexMut};

/// 设备缓冲区 Trait
///
/// 抽象不同计算设备上的内存缓冲区，提供统一的访问接口。
/// CPU 实现使用 `Vec<T>`，GPU 实现可使用 CUDA/Metal 缓冲区。
pub trait DeviceBuffer<T: Pod + Clone + Send + Sync>:
    Clone + Send + Sync + Index<usize, Output = T> + IndexMut<usize>
{
    /// 返回缓冲区长度
    fn len(&self) -> usize;
    
    /// 检查是否为空
    fn is_empty(&self) -> bool { 
        self.len() == 0 
    }
    
    /// 获取只读切片（仅 CPU 缓冲区有效）
    fn as_slice(&self) -> &[T];
    
    /// 获取可变切片（仅 CPU 缓冲区有效）
    fn as_slice_mut(&mut self) -> &mut [T];
    
    /// 用指定值填充
    fn fill(&mut self, value: T);
    
    /// 从切片复制数据
    fn copy_from_slice(&mut self, src: &[T]);
    
    /// 复制到目标切片
    fn copy_to_slice(&self, dst: &mut [T]) {
        dst.copy_from_slice(self.as_slice());
    }
    
    /// 复制到新 Vec
    fn copy_to_vec(&self) -> Vec<T>;
    
    /// 调整大小
    fn resize(&mut self, new_len: usize, value: T);
    
    /// 清空缓冲区
    fn clear(&mut self);
    
    /// 尝试获取只读切片（GPU 缓冲区可能返回 None）
    fn try_as_slice(&self) -> Option<&[T]> {
        Some(self.as_slice())
    }
    
    /// 尝试获取可变切片（GPU 缓冲区可能返回 None）
    fn try_as_slice_mut(&mut self) -> Option<&mut [T]> {
        Some(self.as_slice_mut())
    }
    
    /// 获取指定范围（可选，默认实现）
    fn get_range(&self, start: usize, end: usize) -> Option<&[T]> {
        let slice = self.as_slice();
        if end <= slice.len() && start <= end {
            Some(&slice[start..end])
        } else {
            None
        }
    }
}

// =============================================================================
// Vec<T> 实现
// =============================================================================

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
    
    fn copy_from_slice(&mut self, src: &[T]) {
        Vec::clear(self);
        self.extend_from_slice(src);
    }
    
    #[inline]
    fn copy_to_vec(&self) -> Vec<T> {
        self.clone()
    }
    
    fn resize(&mut self, new_len: usize, value: T) {
        Vec::resize(self, new_len, value);
    }
    
    fn clear(&mut self) {
        Vec::clear(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_buffer() {
        let mut buf: Vec<f64> = vec![0.0; 10];
        assert_eq!(buf.len(), 10);
        assert!(!buf.is_empty());
        
        buf.fill(1.0);
        assert!(buf.iter().all(|&x| x == 1.0));
        
        buf[5] = 2.0;
        assert_eq!(buf[5], 2.0);
    }

    #[test]
    fn test_copy_from_slice() {
        let mut buf: Vec<f32> = vec![0.0; 5];
        buf.copy_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 1.0);
        assert_eq!(buf[2], 3.0);
    }
}
