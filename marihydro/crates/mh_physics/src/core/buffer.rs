//! marihydro\crates\mh_physics\src\core\buffer.rs
//! 设备缓冲区抽象
//!
//! 提供 CPU/GPU 统一的缓冲区接口。

use bytemuck::Pod;
use mh_foundation::memory::AlignedVec;

/// 设备缓冲区接口
pub trait DeviceBuffer<T: Pod>: Clone + Send + Sync + Sized {
    /// 缓冲区长度
    fn len(&self) -> usize;
    
    /// 是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// 从 Host 切片复制数据
    fn copy_from_slice(&mut self, src: &[T]);
    
    /// 复制到 Host Vec
    fn copy_to_vec(&self) -> Vec<T>;
    
    /// 获取只读切片（仅 CPU 有效，GPU 返回 None）
    fn as_slice(&self) -> Option<&[T]>;
    
    /// 获取可变切片（仅 CPU 有效，GPU 返回 None）
    fn as_slice_mut(&mut self) -> Option<&mut [T]>;
    
    /// 填充值
    fn fill(&mut self, value: T);
}

/// Vec<T> 作为 CPU 缓冲区
impl<T: Pod + Clone + Send + Sync> DeviceBuffer<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    
    fn copy_from_slice(&mut self, src: &[T]) {
        self.clear();
        self.extend_from_slice(src);
    }
    
    fn copy_to_vec(&self) -> Vec<T> {
        self.clone()
    }
    
    fn as_slice(&self) -> Option<&[T]> {
        Some(self.as_ref())
    }
    
    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        Some(self.as_mut())
    }
    
    fn fill(&mut self, value: T) {
        self.iter_mut().for_each(|x| *x = value);
    }
}

/// AlignedVec 适配器（复用现有类型）
impl<T: Pod + Clone + Default + Send + Sync> DeviceBuffer<T> for AlignedVec<T> {
    fn len(&self) -> usize {
        AlignedVec::len(self)
    }
    
    fn copy_from_slice(&mut self, src: &[T]) {
        self.as_mut_slice().copy_from_slice(src);
    }
    
    fn copy_to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
    
    fn as_slice(&self) -> Option<&[T]> {
        Some(AlignedVec::as_slice(self))
    }
    
    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        Some(AlignedVec::as_mut_slice(self))
    }
    
    fn fill(&mut self, value: T) {
        // 使用切片的 fill 方法
        self.as_mut_slice().fill(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_buffer() {
        let mut buf: Vec<f64> = vec![0.0; 10];
        buf.fill(1.0);
        assert_eq!(buf[0], 1.0);
        assert_eq!(DeviceBuffer::len(&buf), 10);
    }
}
