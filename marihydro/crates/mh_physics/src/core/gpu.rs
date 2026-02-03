// marihydro\crates\mh_physics\src\core\gpu.rs
//! GPU 后端（模拟实现）
//!
//! 预留 CUDA 后端支持，当前提供可运行的模拟实现。
//! 模拟设备通过环境变量声明，运行时可安全回退到 CPU。
//!
//! # 说明
//!
//! 当前模块不依赖真实 CUDA 运行时，便于在无 GPU 环境下完成流程验证。

use mh_runtime::DeviceBuffer;
use mh_runtime::RuntimeScalar as Scalar;
use bytemuck::Pod;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// CUDA 后端模拟结构
/// 
/// 使用环境变量模拟设备列表，便于在无 CUDA 环境下测试接口流程。
#[derive(Debug, Clone)]
pub struct CudaBackendPlaceholder<S: Scalar> {
    device_id: usize,
    device_name: String,
    _marker: PhantomData<S>,
}

impl<S: Scalar> CudaBackendPlaceholder<S> {
    /// 创建 CUDA 后端（模拟）
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        let device = available_gpus()
            .into_iter()
            .find(|d| d.id == device_id)
            .ok_or_else(|| CudaError(format!("CUDA 设备不可用: {}", device_id)))?;

        Ok(Self {
            device_id,
            device_name: device.name,
            _marker: PhantomData,
        })
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

/// GPU 缓冲区（CPU 回退实现）
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
///
/// 通过环境变量模拟设备列表：
/// - `CUDA_VISIBLE_DEVICES`: 逗号分隔的设备 ID
/// - `CUDA_DEVICE_COUNT`: 设备数量（如 2）
pub fn available_gpus() -> Vec<GpuDeviceInfo> {
    let mut devices = Vec::new();
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        if !visible.trim().is_empty() && visible.trim() != "NoDevFiles" {
            let ids: Vec<usize> = visible
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            for id in ids {
                devices.push(GpuDeviceInfo {
                    id,
                    name: format!("CUDA GPU {} (simulated)", id),
                    memory_bytes: 0,
                    compute_capability: (0, 0),
                });
            }
        }
    }

    if devices.is_empty() {
        if let Some(count) = std::env::var("CUDA_DEVICE_COUNT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            for id in 0..count {
                devices.push(GpuDeviceInfo {
                    id,
                    name: format!("CUDA GPU {} (simulated)", id),
                    memory_bytes: 0,
                    compute_capability: (0, 0),
                });
            }
        }
    }

    devices
}

/// 检查是否有可用 GPU
pub fn has_cuda() -> bool {
    !available_gpus().is_empty()
}
