// marihydro\crates\mh_physics\src\core\gpu.rs
//! GPU 后端（骨架实现）
//!
//! 预留 CUDA 后端支持，当前仅提供接口定义。
//! 实际 GPU 实现将在未来阶段完成。
//!
//! # 说明
//!
//! 当前 GPU 后端为占位实现，提供安全的 CPU 回退以避免运行期崩溃。

use mh_runtime::DeviceBuffer;
use mh_runtime::RuntimeScalar as Scalar;
use bytemuck::Pod;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

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
    data: Vec<T>,
}

impl<T: Pod + Default + Clone> GpuBuffer<T> {
    /// 创建 CPU 回退缓冲区
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![T::default(); len],
        }
    }
}

impl<T: Pod> Index<usize> for GpuBuffer<T> {
    type Output = T;
    fn index(&self, _index: usize) -> &Self::Output {
        &self.data[_index]
    }
}

impl<T: Pod> IndexMut<usize> for GpuBuffer<T> {
    fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
        &mut self.data[_index]
    }
}

impl<T: Pod + Clone + Default + Send + Sync> DeviceBuffer<T> for GpuBuffer<T> {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn copy_from_slice(&mut self, _src: &[T]) {
        self.data.copy_from_slice(_src)
    }
    
    fn copy_to_vec(&self) -> Vec<T> {
        self.data.clone()
    }
    
    fn copy_to_slice(&self, _dst: &mut [T]) {
        _dst.copy_from_slice(&self.data)
    }
    
    fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    fn fill(&mut self, _value: T) {
        self.data.fill(_value)
    }
    
    fn resize(&mut self, _new_len: usize, _value: T) {
        self.data.resize(_new_len, _value)
    }
    
    fn clear(&mut self) {
        self.data.clear()
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
