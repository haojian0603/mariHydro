//! GPU缓冲区抽象
//!
//! 定义GPU缓冲区的通用接口。

use crate::marihydro::core::error::MhResult;

#[cfg(feature = "gpu")]
use bytemuck::Pod;

/// 缓冲区用途
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// 只读（CPU -> GPU）
    ReadOnly,
    /// 只写（GPU -> CPU）
    WriteOnly,
    /// 读写（GPU内部）
    ReadWrite,
    /// 存储缓冲区（用于compute shader）
    Storage,
    /// Uniform缓冲区（常量数据）
    Uniform,
    /// 顶点缓冲区（网格几何，一次性上传）
    Vertex,
    /// 索引缓冲区（拓扑数据）
    Index,
}

impl BufferUsage {
    /// 是否可从CPU写入
    pub fn is_cpu_writable(&self) -> bool {
        matches!(
            self,
            BufferUsage::ReadOnly | BufferUsage::ReadWrite | BufferUsage::Uniform
        )
    }

    /// 是否可从CPU读取
    pub fn is_cpu_readable(&self) -> bool {
        matches!(self, BufferUsage::WriteOnly | BufferUsage::ReadWrite)
    }

    /// 是否为GPU存储类型
    pub fn is_storage(&self) -> bool {
        matches!(
            self,
            BufferUsage::Storage | BufferUsage::ReadWrite | BufferUsage::ReadOnly
        )
    }
}

/// GPU缓冲区抽象trait
///
/// 提供与GPU缓冲区交互的统一接口。
/// 不同后端（wgpu、CUDA等）实现此trait。
#[cfg(feature = "gpu")]
pub trait GpuBuffer<T: Pod + Send + Sync>: Send + Sync + Sized {
    /// 创建指定大小的缓冲区
    ///
    /// # 参数
    /// - `size`: 元素数量
    /// - `usage`: 缓冲区用途
    fn new(size: usize, usage: BufferUsage) -> MhResult<Self>;

    /// 从CPU数据创建缓冲区
    ///
    /// # 参数
    /// - `data`: CPU端数据切片
    /// - `usage`: 缓冲区用途
    fn from_slice(data: &[T], usage: BufferUsage) -> MhResult<Self>;

    /// 上传数据到GPU
    ///
    /// # 参数
    /// - `data`: 要上传的数据
    ///
    /// # 错误
    /// - 如果数据大小与缓冲区不匹配
    /// - 如果缓冲区不可写
    fn upload(&self, data: &[T]) -> MhResult<()>;

    /// 从GPU下载数据
    ///
    /// # 参数
    /// - `data`: 接收数据的缓冲区
    ///
    /// # 错误
    /// - 如果数据大小与缓冲区不匹配
    /// - 如果缓冲区不可读
    fn download(&self, data: &mut [T]) -> MhResult<()>;

    /// 获取缓冲区大小（元素数）
    fn len(&self) -> usize;

    /// 检查缓冲区是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 获取缓冲区大小（字节）
    fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    /// 获取缓冲区用途
    fn usage(&self) -> BufferUsage;
}

/// CPU端模拟缓冲区（用于测试和CPU后端）
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct CpuBuffer<T> {
    data: Vec<T>,
    usage: BufferUsage,
}

#[cfg(feature = "gpu")]
impl<T: Pod + Send + Sync + Clone + Default> GpuBuffer<T> for CpuBuffer<T> {
    fn new(size: usize, usage: BufferUsage) -> MhResult<Self> {
        Ok(Self {
            data: vec![T::default(); size],
            usage,
        })
    }

    fn from_slice(data: &[T], usage: BufferUsage) -> MhResult<Self> {
        Ok(Self {
            data: data.to_vec(),
            usage,
        })
    }

    fn upload(&self, _data: &[T]) -> MhResult<()> {
        // CPU缓冲区不需要真正上传
        Ok(())
    }

    fn download(&self, data: &mut [T]) -> MhResult<()> {
        if data.len() != self.data.len() {
            return Err(crate::marihydro::core::error::MhError::size_mismatch(
                "buffer download",
                self.data.len(),
                data.len(),
            ));
        }
        data.clone_from_slice(&self.data);
        Ok(())
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn usage(&self) -> BufferUsage {
        self.usage
    }
}

#[cfg(feature = "gpu")]
impl<T> CpuBuffer<T> {
    /// 获取内部数据的不可变引用
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// 获取内部数据的可变引用
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_buffer_creation() {
        let buf: CpuBuffer<f64> = CpuBuffer::new(100, BufferUsage::Storage).unwrap();
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.size_bytes(), 800);
    }

    #[test]
    fn test_cpu_buffer_from_slice() {
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let buf: CpuBuffer<f64> = CpuBuffer::from_slice(&data, BufferUsage::ReadOnly).unwrap();
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.data(), &data);
    }

    #[test]
    fn test_buffer_usage() {
        assert!(BufferUsage::ReadOnly.is_cpu_writable());
        assert!(!BufferUsage::ReadOnly.is_cpu_readable());
        assert!(BufferUsage::WriteOnly.is_cpu_readable());
        assert!(!BufferUsage::WriteOnly.is_cpu_writable());
    }
}
