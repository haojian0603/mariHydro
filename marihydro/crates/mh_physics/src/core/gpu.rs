// marihydro\crates\mh_physics\src\core\gpu.rs
//! GPU 后端契约层。
//!
//! `ungpu` 分支不提供可用的 GPU 运行时。该模块只保留显式错误类型和
//! 兼容性的占位结构，避免把“可用 GPU”伪装成运行时能力。

use mh_runtime::DeviceBuffer;
use mh_runtime::RuntimeScalar as Scalar;
use bytemuck::Pod;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// CUDA 后端占位结构。
///
/// 该类型仅用于保留上层契约。当前分支不提供真实 GPU 后端，因此构造
/// 始终返回错误。
#[derive(Debug, Clone)]
pub struct CudaBackendPlaceholder<S: Scalar> {
    device_id: usize,
    device_name: String,
    _marker: PhantomData<S>,
}

impl<S: Scalar> CudaBackendPlaceholder<S> {
    /// 创建 CUDA 后端占位实例。
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        let _ = device_id;
        Err(CudaError(
            "CUDA backend is unavailable in the ungpu branch".to_string(),
        ))
    }

    /// 获取设备 ID
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// 获取设备名称
    pub fn device_name(&self) -> &str {
        &self.device_name
    }
}

/// GPU 缓冲区占位类型（内部使用 CPU 内存）。
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
    
    fn copy_from_slice(&mut self, src: &[T]) {
        self.data.copy_from_slice(src)
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

/// 查询可用 GPU 设备。
///
/// `ungpu` 分支不声明任何 GPU 设备。
pub fn available_gpus() -> Vec<GpuDeviceInfo> {
    Vec::new()
}

/// 检查是否有可用 GPU。
pub fn has_cuda() -> bool {
    false
}
