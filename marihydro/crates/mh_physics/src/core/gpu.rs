// marihydro\crates\mh_physics\src\core\gpu.rs
//! GPU 后端（骨架实现）
//!
//! 预留 CUDA 后端支持，当前仅提供接口定义。
//! 实际 GPU 实现将在未来阶段完成。

use super::buffer::DeviceBuffer;
use mh_core::Scalar;
use bytemuck::Pod;
use std::marker::PhantomData;

/// CUDA 后端占位符
/// 
/// 这是一个占位结构，用于定义 GPU 后端接口。
/// 实际实现需要在启用 `cuda` feature 时完成。
#[derive(Debug, Clone)]
pub struct CudaBackendPlaceholder<S: Scalar> {
    _marker: PhantomData<S>,
}

impl<S: Scalar> CudaBackendPlaceholder<S> {
    /// 创建 CUDA 后端（占位）
    pub fn new(_device_id: usize) -> Result<Self, CudaError> {
        Err(CudaError("CUDA backend not implemented yet".into()))
    }
}

/// GPU 缓冲区占位符
#[derive(Debug, Clone)]
pub struct GpuBuffer<T: Pod> {
    len: usize,
    _marker: PhantomData<T>,
}

// 手动实现 Send 和 Sync（GPU 缓冲区是安全的）
unsafe impl<T: Pod> Send for GpuBuffer<T> {}
unsafe impl<T: Pod> Sync for GpuBuffer<T> {}

impl<T: Pod + Clone + Default + Send + Sync> DeviceBuffer<T> for GpuBuffer<T> {
    fn len(&self) -> usize {
        self.len
    }
    
    fn copy_from_slice(&mut self, _src: &[T]) {
        unimplemented!("GPU buffer not implemented")
    }
    
    fn copy_to_vec(&self) -> Vec<T> {
        unimplemented!("GPU buffer not implemented")
    }
    
    fn as_slice(&self) -> Option<&[T]> {
        None // GPU 缓冲区无法直接访问
    }
    
    fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        None // GPU 缓冲区无法直接访问
    }
    
    fn fill(&mut self, _value: T) {
        unimplemented!("GPU buffer not implemented")
    }
}

/// CUDA 错误类型
#[derive(Debug)]
pub struct CudaError(pub String);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA error: {}", self.0)
    }
}

impl std::error::Error for CudaError {}

/// GPU 设备信息
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// 设备 ID
    pub id: usize,
    /// 设备名称
    pub name: String,
    /// 显存大小（字节）
    pub memory_bytes: usize,
    /// 计算能力
    pub compute_capability: (u32, u32),
}

/// 查询可用 GPU 设备
pub fn available_gpus() -> Vec<GpuDeviceInfo> {
    // 占位实现
    Vec::new()
}

/// 检查是否有可用 GPU
pub fn has_cuda() -> bool {
    false
}
