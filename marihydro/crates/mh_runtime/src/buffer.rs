// crates/mh_runtime/src/buffer.rs

//! DeviceBuffer - 设备缓冲区抽象
//! 
//! 提供统一的缓冲区接口，支持 CPU 向量和未来的 GPU 缓冲区。
//!
//! # GPU 缓冲区管理
//!
//! 为未来的 GPU 支持预留接口：
//! - `GpuBuffer`: GPU 缓冲区 trait
//! - `BufferPool`: 缓冲区池管理
//! - `TransferQueue`: CPU/GPU 数据传输队列

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

// ============================================================
// GPU 缓冲区管理接口（预留）
// ============================================================

/// GPU 缓冲区状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferState {
    /// 数据在 CPU 上
    Host,
    /// 数据在 GPU 上
    Device,
    /// CPU 和 GPU 数据同步
    Synced,
    /// 数据在传输中
    InTransfer,
}

/// 缓冲区使用模式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferUsage {
    /// 只读
    ReadOnly,
    /// 只写
    WriteOnly,
    /// 读写
    ReadWrite,
    /// 仅 GPU 使用
    DeviceOnly,
}

/// GPU 缓冲区描述符（预留接口）
#[derive(Debug, Clone)]
pub struct GpuBufferDescriptor {
    /// 缓冲区大小（字节）
    pub size_bytes: usize,
    /// 使用模式
    pub usage: BufferUsage,
    /// 名称（用于调试）
    pub name: String,
    /// 是否需要 CPU 可访问的映射
    pub host_visible: bool,
}

impl Default for GpuBufferDescriptor {
    fn default() -> Self {
        Self {
            size_bytes: 0,
            usage: BufferUsage::ReadWrite,
            name: String::new(),
            host_visible: true,
        }
    }
}

impl GpuBufferDescriptor {
    /// 创建新描述符
    pub fn new<T>(len: usize, usage: BufferUsage) -> Self {
        Self {
            size_bytes: len * std::mem::size_of::<T>(),
            usage,
            name: String::new(),
            host_visible: usage != BufferUsage::DeviceOnly,
        }
    }

    /// 设置名称
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

/// 缓冲区池配置
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// 最大缓冲区数量
    pub max_buffers: usize,
    /// 最大总内存（字节）
    pub max_memory_bytes: usize,
    /// 缓冲区对齐要求
    pub alignment: usize,
    /// 是否启用缓冲区重用
    pub enable_reuse: bool,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_buffers: 1000,
            max_memory_bytes: 1024 * 1024 * 1024, // 1 GB
            alignment: 256,
            enable_reuse: true,
        }
    }
}

/// CPU 缓冲区池（用于减少内存分配）
pub struct CpuBufferPool<T: Pod + Clone + Send + Sync> {
    /// 可重用的缓冲区
    free_buffers: std::sync::Mutex<Vec<Vec<T>>>,
    /// 活跃缓冲区数量
    active_count: std::sync::atomic::AtomicUsize,
    /// 配置
    config: BufferPoolConfig,
}

impl<T: Pod + Clone + Send + Sync> CpuBufferPool<T> {
    /// 创建新的缓冲区池
    pub fn new(config: BufferPoolConfig) -> Self {
        Self {
            free_buffers: std::sync::Mutex::new(Vec::new()),
            active_count: std::sync::atomic::AtomicUsize::new(0),
            config,
        }
    }

    /// 从池中获取缓冲区
    pub fn acquire(&self, len: usize, initial_value: T) -> PooledBuffer<'_, T> {
        let buffer = {
            let mut free = self.free_buffers.lock().unwrap_or_else(|poisoned| {
                eprintln!("[mh_runtime::buffer] BufferPool mutex poisoned, recovering");
                poisoned.into_inner()
            });
            // 尝试找到大小合适的缓冲区
            if self.config.enable_reuse {
                if let Some(pos) = free.iter().position(|b| b.capacity() >= len) {
                    let mut buf = free.swap_remove(pos);
                    buf.clear();
                    buf.resize(len, initial_value);
                    Some(buf)
                } else {
                    None
                }
            } else {
                None
            }
        };

        let buffer = buffer.unwrap_or_else(|| vec![initial_value; len]);
        self.active_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        PooledBuffer {
            buffer: Some(buffer),
            pool: self,
        }
    }

    /// 归还缓冲区
    fn release(&self, buffer: Vec<T>) {
        self.active_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        
        if self.config.enable_reuse {
            let mut free = self.free_buffers.lock().unwrap_or_else(|poisoned| {
                eprintln!("[mh_runtime::buffer] BufferPool mutex poisoned, recovering");
                poisoned.into_inner()
            });
            if free.len() < self.config.max_buffers {
                free.push(buffer);
            }
        }
    }

    /// 获取活跃缓冲区数量
    pub fn active_count(&self) -> usize {
        self.active_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// 获取空闲缓冲区数量
    pub fn free_count(&self) -> usize {
        self.free_buffers.lock().unwrap_or_else(|poisoned| {
            poisoned.into_inner()
        }).len()
    }

    /// 清除所有空闲缓冲区
    pub fn clear_free(&self) {
        self.free_buffers.lock().unwrap_or_else(|poisoned| {
            poisoned.into_inner()
        }).clear();
    }
}

impl<T: Pod + Clone + Send + Sync> Default for CpuBufferPool<T> {
    fn default() -> Self {
        Self::new(BufferPoolConfig::default())
    }
}

/// 池化的缓冲区（RAII）
pub struct PooledBuffer<'a, T: Pod + Clone + Send + Sync> {
    buffer: Option<Vec<T>>,
    pool: &'a CpuBufferPool<T>,
}

impl<'a, T: Pod + Clone + Send + Sync> PooledBuffer<'a, T> {
    /// 获取切片
    #[allow(clippy::unwrap_used)]
    pub fn as_slice(&self) -> &[T] {
        // Safety: buffer 只有在 Drop 时才会被设置为 None
        self.buffer.as_ref().expect("buffer should always be Some").as_slice()
    }

    /// 获取可变切片
    #[allow(clippy::unwrap_used)]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        // Safety: buffer 只有在 Drop 时才会被设置为 None
        self.buffer.as_mut().expect("buffer should always be Some").as_mut_slice()
    }

    /// 获取长度
    #[allow(clippy::unwrap_used)]
    pub fn len(&self) -> usize {
        // Safety: buffer 只有在 Drop 时才会被设置为 None
        self.buffer.as_ref().expect("buffer should always be Some").len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T: Pod + Clone + Send + Sync> Drop for PooledBuffer<'a, T> {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

impl<'a, T: Pod + Clone + Send + Sync> std::ops::Deref for PooledBuffer<'a, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T: Pod + Clone + Send + Sync> std::ops::DerefMut for PooledBuffer<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_slice_mut()
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

    #[test]
    fn test_buffer_pool() {
        let pool = CpuBufferPool::<f64>::default();
        
        // 获取缓冲区
        {
            let buf = pool.acquire(100, 0.0);
            assert_eq!(buf.len(), 100);
            assert_eq!(pool.active_count(), 1);
        }
        
        // 缓冲区归还后
        assert_eq!(pool.active_count(), 0);
        assert_eq!(pool.free_count(), 1);
        
        // 重用缓冲区
        {
            let buf = pool.acquire(50, 1.0);
            assert_eq!(buf.len(), 50);
            assert_eq!(pool.free_count(), 0);
        }
    }

    #[test]
    fn test_gpu_buffer_descriptor() {
        let desc = GpuBufferDescriptor::new::<f64>(1000, BufferUsage::ReadWrite)
            .with_name("test_buffer");
        
        assert_eq!(desc.size_bytes, 1000 * std::mem::size_of::<f64>());
        assert_eq!(desc.name, "test_buffer");
        assert!(desc.host_visible);
    }
}