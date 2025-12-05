// crates/mh_physics/src/gpu/buffer.rs

//! GPU 缓冲区管理
//!
//! 提供类型安全的 GPU 缓冲区包装器。

use std::marker::PhantomData;
use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};

/// GPU 缓冲区用途
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBufferUsage {
    /// 存储缓冲区（可读写）
    Storage,
    /// 只读存储
    StorageReadOnly,
    /// 统一缓冲区（小型常量数据）
    Uniform,
    /// 暂存缓冲区（CPU-GPU传输）
    Staging,
    /// 顶点缓冲区
    Vertex,
    /// 索引缓冲区
    Index,
}

impl GpuBufferUsage {
    /// 转换为 wgpu BufferUsages
    pub fn to_wgpu_usage(self) -> BufferUsages {
        match self {
            Self::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            Self::StorageReadOnly => BufferUsages::STORAGE | BufferUsages::COPY_DST,
            Self::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            Self::Staging => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            Self::Vertex => BufferUsages::VERTEX | BufferUsages::COPY_DST,
            Self::Index => BufferUsages::INDEX | BufferUsages::COPY_DST,
        }
    }
}

/// 类型化的 GPU 缓冲区
pub struct TypedBuffer<T> {
    /// 底层 wgpu 缓冲区
    buffer: Buffer,
    /// 元素数量
    len: usize,
    /// 缓冲区用途
    usage: GpuBufferUsage,
    /// 类型标记
    _marker: PhantomData<T>,
}

impl<T: bytemuck::Pod> TypedBuffer<T> {
    /// 创建新的缓冲区
    pub fn new(device: &Device, len: usize, usage: GpuBufferUsage, label: Option<&str>) -> Self {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let buffer = device.create_buffer(&BufferDescriptor {
            label,
            size,
            usage: usage.to_wgpu_usage(),
            mapped_at_creation: false,
        });

        Self {
            buffer,
            len,
            usage,
            _marker: PhantomData,
        }
    }

    /// 从数据创建缓冲区
    pub fn from_data(
        device: &Device,
        data: &[T],
        usage: GpuBufferUsage,
        label: Option<&str>,
    ) -> Self {
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(data),
                usage: usage.to_wgpu_usage(),
            },
        );

        Self {
            buffer,
            len: data.len(),
            usage,
            _marker: PhantomData,
        }
    }

    /// 上传数据到缓冲区
    pub fn write(&self, queue: &Queue, data: &[T]) {
        assert!(data.len() <= self.len, "Data exceeds buffer capacity");
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// 上传部分数据
    pub fn write_at(&self, queue: &Queue, offset: usize, data: &[T]) {
        assert!(offset + data.len() <= self.len, "Data exceeds buffer capacity");
        let byte_offset = (offset * std::mem::size_of::<T>()) as u64;
        queue.write_buffer(&self.buffer, byte_offset, bytemuck::cast_slice(data));
    }

    /// 获取底层缓冲区引用
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// 获取元素数量
    pub fn len(&self) -> usize {
        self.len
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取字节大小
    pub fn size_bytes(&self) -> u64 {
        (self.len * std::mem::size_of::<T>()) as u64
    }

    /// 获取用途
    pub fn usage(&self) -> GpuBufferUsage {
        self.usage
    }

    /// 创建绑定组条目
    pub fn as_entire_binding(&self) -> wgpu::BindingResource {
        self.buffer.as_entire_binding()
    }
}

/// GPU 缓冲区池
/// 
/// 管理多个相关的缓冲区，用于求解器的状态存储。
pub struct BufferPool {
    /// 存储缓冲区列表
    buffers: Vec<Buffer>,
    /// 缓冲区标签
    labels: Vec<String>,
    /// 每个缓冲区的大小
    sizes: Vec<u64>,
}

impl BufferPool {
    /// 创建新的缓冲区池
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            labels: Vec::new(),
            sizes: Vec::new(),
        }
    }

    /// 添加缓冲区
    pub fn add_buffer(&mut self, device: &Device, size: u64, usage: BufferUsages, label: &str) -> usize {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        
        let index = self.buffers.len();
        self.buffers.push(buffer);
        self.labels.push(label.to_string());
        self.sizes.push(size);
        index
    }

    /// 获取缓冲区
    pub fn get(&self, index: usize) -> Option<&Buffer> {
        self.buffers.get(index)
    }

    /// 获取缓冲区数量
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// 获取总内存使用量
    pub fn total_memory(&self) -> u64 {
        self.sizes.iter().sum()
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

/// 双缓冲区用于乒乓式计算
pub struct DoubleBuffer<T> {
    /// 前缓冲区
    front: TypedBuffer<T>,
    /// 后缓冲区
    back: TypedBuffer<T>,
    /// 当前活动的是前缓冲区
    front_active: bool,
}

impl<T: bytemuck::Pod> DoubleBuffer<T> {
    /// 创建双缓冲区
    pub fn new(device: &Device, len: usize, usage: GpuBufferUsage, label: &str) -> Self {
        let front = TypedBuffer::new(device, len, usage, Some(&format!("{}_front", label)));
        let back = TypedBuffer::new(device, len, usage, Some(&format!("{}_back", label)));
        
        Self {
            front,
            back,
            front_active: true,
        }
    }

    /// 获取当前读取缓冲区
    pub fn read_buffer(&self) -> &TypedBuffer<T> {
        if self.front_active { &self.front } else { &self.back }
    }

    /// 获取当前写入缓冲区
    pub fn write_buffer(&self) -> &TypedBuffer<T> {
        if self.front_active { &self.back } else { &self.front }
    }

    /// 交换缓冲区
    pub fn swap(&mut self) {
        self.front_active = !self.front_active;
    }

    /// 获取元素数量
    pub fn len(&self) -> usize {
        self.front.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.front.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_usage() {
        let storage = GpuBufferUsage::Storage.to_wgpu_usage();
        assert!(storage.contains(BufferUsages::STORAGE));
        assert!(storage.contains(BufferUsages::COPY_DST));
        
        let uniform = GpuBufferUsage::Uniform.to_wgpu_usage();
        assert!(uniform.contains(BufferUsages::UNIFORM));
    }

    #[test]
    fn test_buffer_pool_new() {
        let pool = BufferPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.total_memory(), 0);
    }
}
