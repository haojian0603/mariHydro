// src-tauri/src/marihydro/core/compute/wgpu_buffer.rs

//! wgpu GPU缓冲区实现
//!
//! 提供类型安全的GPU缓冲区封装

use std::marker::PhantomData;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Device, Queue,
};

use super::buffer::BufferUsage;
use crate::marihydro::core::error::{MhError, MhResult};

/// wgpu缓冲区封装
pub struct WgpuBuffer<T: Pod + Zeroable> {
    /// GPU缓冲区
    buffer: Buffer,
    /// 用于readback的staging缓冲区
    staging: Option<Buffer>,
    /// 元素数量
    len: usize,
    /// 缓冲区用途
    usage: BufferUsage,
    /// 设备引用
    device: Arc<Device>,
    /// 队列引用
    queue: Arc<Queue>,
    /// 类型标记
    _marker: PhantomData<T>,
}

impl<T: Pod + Zeroable> WgpuBuffer<T> {
    /// 创建新的GPU缓冲区
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        len: usize,
        usage: BufferUsage,
    ) -> MhResult<Self> {
        let byte_size = len * std::mem::size_of::<T>();
        
        if byte_size == 0 {
            return Err(MhError::invalid_input("Buffer size cannot be zero"));
        }

        let wgpu_usage = Self::translate_usage(usage);
        
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("WgpuBuffer"),
            size: byte_size as u64,
            usage: wgpu_usage,
            mapped_at_creation: false,
        });

        // 如果需要读取，创建staging缓冲区
        let staging = if usage == BufferUsage::WriteOnly || usage == BufferUsage::ReadWrite {
            Some(device.create_buffer(&BufferDescriptor {
                label: Some("WgpuBuffer Staging"),
                size: byte_size as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        Ok(Self {
            buffer,
            staging,
            len,
            usage,
            device,
            queue,
            _marker: PhantomData,
        })
    }

    /// 从CPU数据创建缓冲区
    pub fn from_slice(
        device: Arc<Device>,
        queue: Arc<Queue>,
        data: &[T],
        usage: BufferUsage,
    ) -> MhResult<Self> {
        let mut buffer = Self::new(Arc::clone(&device), Arc::clone(&queue), data.len(), usage)?;
        buffer.upload(data)?;
        Ok(buffer)
    }

    /// 上传数据到GPU
    pub fn upload(&mut self, data: &[T]) -> MhResult<()> {
        if data.len() != self.len {
            return Err(MhError::InvalidInput(format!(
                "Data length {} doesn't match buffer length {}",
                data.len(),
                self.len
            )));
        }

        self.queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    /// 从GPU下载数据
    pub fn download(&self, data: &mut [T]) -> MhResult<()> {
        if data.len() != self.len {
            return Err(MhError::InvalidInput(format!(
                "Data length {} doesn't match buffer length {}",
                data.len(),
                self.len
            )));
        }

        let staging = self.staging.as_ref().ok_or_else(|| {
            MhError::ComputeError("Buffer not configured for reading")
        })?;

        // 复制到staging缓冲区
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Download encoder"),
        });
        
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            staging,
            0,
            (self.len * std::mem::size_of::<T>()) as u64,
        );
        
        self.queue.submit(Some(encoder.finish()));

        // 映射并读取
        let buffer_slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.device.poll(wgpu::Maintain::Wait);
        
        rx.recv()
            .map_err(|e| MhError::ComputeError(format!("Failed to receive map result: {}", e)))?
            .map_err(|e| MhError::ComputeError(format!("Failed to map buffer: {:?}", e)))?;

        {
            let mapped = buffer_slice.get_mapped_range();
            let src: &[T] = bytemuck::cast_slice(&mapped);
            data.copy_from_slice(src);
        }
        
        staging.unmap();
        Ok(())
    }

    /// 获取元素数量
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取字节大小
    #[inline]
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// 获取底层wgpu缓冲区
    pub fn raw_buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// 创建绑定组条目
    pub fn as_binding(&self) -> wgpu::BindingResource {
        self.buffer.as_entire_binding()
    }

    /// 转换用途标志
    fn translate_usage(usage: BufferUsage) -> BufferUsages {
        match usage {
            BufferUsage::ReadOnly => BufferUsages::STORAGE | BufferUsages::COPY_DST,
            BufferUsage::WriteOnly => BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            BufferUsage::ReadWrite => {
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
            }
            BufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        }
    }
}

/// GPU缓冲区管理器
pub struct BufferManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    /// 已分配的总字节数
    allocated_bytes: usize,
    /// 最大允许内存
    max_memory_bytes: usize,
}

impl BufferManager {
    /// 创建缓冲区管理器
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, max_memory_bytes: usize) -> Self {
        Self {
            device,
            queue,
            allocated_bytes: 0,
            max_memory_bytes,
        }
    }

    /// 分配缓冲区
    pub fn allocate<T: Pod + Zeroable>(
        &mut self,
        len: usize,
        usage: BufferUsage,
    ) -> MhResult<WgpuBuffer<T>> {
        let byte_size = len * std::mem::size_of::<T>();
        
        if self.allocated_bytes + byte_size > self.max_memory_bytes {
            return Err(MhError::ComputeError(format!(
                "Insufficient GPU memory: requested {} bytes, available {} bytes",
                byte_size,
                self.max_memory_bytes - self.allocated_bytes
            )));
        }

        let buffer = WgpuBuffer::new(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            len,
            usage,
        )?;

        self.allocated_bytes += byte_size;
        Ok(buffer)
    }

    /// 从数据分配缓冲区
    pub fn allocate_from<T: Pod + Zeroable>(
        &mut self,
        data: &[T],
        usage: BufferUsage,
    ) -> MhResult<WgpuBuffer<T>> {
        let byte_size = data.len() * std::mem::size_of::<T>();
        
        if self.allocated_bytes + byte_size > self.max_memory_bytes {
            return Err(MhError::ComputeError(format!(
                "Insufficient GPU memory: requested {} bytes, available {} bytes",
                byte_size,
                self.max_memory_bytes - self.allocated_bytes
            )));
        }

        let buffer = WgpuBuffer::from_slice(
            Arc::clone(&self.device),
            Arc::clone(&self.queue),
            data,
            usage,
        )?;

        self.allocated_bytes += byte_size;
        Ok(buffer)
    }

    /// 获取已分配内存
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// 获取可用内存
    pub fn available_bytes(&self) -> usize {
        self.max_memory_bytes.saturating_sub(self.allocated_bytes)
    }
}

/// 双缓冲区（用于ping-pong更新）
pub struct DoubleBuffer<T: Pod + Zeroable> {
    front: WgpuBuffer<T>,
    back: WgpuBuffer<T>,
    front_is_current: bool,
}

impl<T: Pod + Zeroable> DoubleBuffer<T> {
    /// 创建双缓冲区
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        len: usize,
    ) -> MhResult<Self> {
        let front = WgpuBuffer::new(
            Arc::clone(&device),
            Arc::clone(&queue),
            len,
            BufferUsage::ReadWrite,
        )?;
        
        let back = WgpuBuffer::new(
            device,
            queue,
            len,
            BufferUsage::ReadWrite,
        )?;

        Ok(Self {
            front,
            back,
            front_is_current: true,
        })
    }

    /// 获取当前缓冲区（只读）
    pub fn current(&self) -> &WgpuBuffer<T> {
        if self.front_is_current {
            &self.front
        } else {
            &self.back
        }
    }

    /// 获取下一个缓冲区（写入）
    pub fn next(&self) -> &WgpuBuffer<T> {
        if self.front_is_current {
            &self.back
        } else {
            &self.front
        }
    }

    /// 交换缓冲区
    pub fn swap(&mut self) {
        self.front_is_current = !self.front_is_current;
    }

    /// 获取元素数量
    pub fn len(&self) -> usize {
        self.front.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.front.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 注意：这些测试需要实际GPU
    
    #[test]
    fn test_buffer_usage_translation() {
        let usage = WgpuBuffer::<f32>::translate_usage(BufferUsage::ReadOnly);
        assert!(usage.contains(BufferUsages::STORAGE));
        assert!(usage.contains(BufferUsages::COPY_DST));
    }
}
