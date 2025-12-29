// marihydro\crates\mh_physics\src\core\gpu.rs
//! GPU 后端（骨架实现）
//!
//! 预留 CUDA 后端支持，当前仅提供接口定义。
//! 实际 GPU 实现将在未来阶段完成。
//!
//! # Safety
//!
//! 本模块使用 unsafe 代码实现 Send/Sync traits，经过审查确保线程安全。
#![allow(unsafe_code)]

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
    len: usize,
    _marker: PhantomData<T>,
}

// SAFETY: GpuBuffer<T> 的线程安全性分析
//
// 1. 当前实现说明：
//    - 这是一个占位符类型，尚未实现实际的 GPU 功能
//    - 所有操作方法都会 panic (unimplemented!)
//
// 2. 内部状态：
//    - len: usize - 缓冲区长度（纯数据）
//    - _marker: PhantomData<T> - 零大小类型标记
//
// 3. Send 安全性 (T: Pod 时)：
//    - PhantomData<T> 不包含实际数据
//    - len 是 Copy 类型
//    - 当未来实现 GPU 功能时，CUDA/GPU 缓冲区句柄
//      通常是线程安全的（GPU 驱动处理同步）
//
// 4. Sync 安全性 (T: Pod 时)：
//    - 当前实现不包含可变状态
//    - 未来实现 GPU 功能时，需要确保：
//      a) GPU 内存访问通过驱动同步
//      b) 主机端访问通过适当的复制操作
//
// 5. Pod 约束保证：
//    - T: Pod 确保数据可以安全地在主机和设备间复制
//    - 不包含指针或需要特殊处理的资源
//
// TODO: 实现实际 GPU 功能时需要重新审查这些保证
unsafe impl<T: Pod> Send for GpuBuffer<T> {}
unsafe impl<T: Pod> Sync for GpuBuffer<T> {}

impl<T: Pod> Index<usize> for GpuBuffer<T> {
    type Output = T;
    fn index(&self, _index: usize) -> &Self::Output {
        unimplemented!("GPU buffer direct indexing not supported, use copy_to_vec first")
    }
}

impl<T: Pod> IndexMut<usize> for GpuBuffer<T> {
    fn index_mut(&mut self, _index: usize) -> &mut Self::Output {
        unimplemented!("GPU buffer direct indexing not supported, use copy_to_vec first")
    }
}

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
    
    fn copy_to_slice(&self, _dst: &mut [T]) {
        unimplemented!("GPU buffer not implemented")
    }
    
    fn as_slice(&self) -> &[T] {
        panic!("Cannot access GPU buffer as slice directly")
    }
    
    fn as_slice_mut(&mut self) -> &mut [T] {
        panic!("Cannot access GPU buffer as slice directly")
    }
    
    fn fill(&mut self, _value: T) {
        unimplemented!("GPU buffer not implemented")
    }
    
    fn resize(&mut self, _new_len: usize, _value: T) {
        unimplemented!("GPU buffer not implemented")
    }
    
    fn clear(&mut self) {
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
