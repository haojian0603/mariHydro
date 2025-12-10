# Phase 6: GPU 准备

## 目标

完成 CUDA 接入准备，设计 HybridBackend。

## 时间：第 8 周

## 前置依赖

- Phase 0-2 完成（Backend 实例方法、泛型化）

## 任务清单

### 6.1 CudaBackend 骨架

**目标**：创建 CUDA 后端骨架，使用 feature gate 控制。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 扩展 | `core/gpu.rs` | CudaBackend 定义 |
| 修改 | `Cargo.toml` | 添加 cuda feature |

#### 关键代码

**mh_physics/Cargo.toml（添加）**

```toml
[features]
default = []
cuda = ["cudarc"]

[dependencies]
cudarc = { version = "0.10", optional = true }
```

**mh_physics/src/core/gpu.rs（扩展）**

```rust
// mh_physics/src/core/gpu.rs
//! GPU 后端支持

use super::backend::{Backend, MemoryLocation};
use super::scalar::Scalar;
use thiserror::Error;

/// CUDA 错误
#[derive(Error, Debug)]
pub enum CudaError {
    #[error("CUDA driver error: {0}")]
    DriverError(String),
    
    #[error("CUDA memory error: {0}")]
    MemoryError(String),
    
    #[error("CUDA kernel error: {0}")]
    KernelError(String),
    
    #[error("No CUDA device available")]
    NoDevice,
}

/// GPU 设备信息
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub total_memory: usize,
    pub multiprocessor_count: u32,
}

/// 检查 CUDA 是否可用
pub fn has_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// 获取可用 GPU 列表
pub fn available_gpus() -> Vec<GpuDeviceInfo> {
    #[cfg(feature = "cuda")]
    {
        let mut gpus = Vec::new();
        let count = cudarc::driver::result::device::get_count().unwrap_or(0);
        for i in 0..count {
            if let Ok(device) = cudarc::driver::CudaDevice::new(i) {
                gpus.push(GpuDeviceInfo {
                    name: format!("GPU {}", i),
                    compute_capability: (0, 0), // TODO: 获取实际值
                    total_memory: 0,
                    multiprocessor_count: 0,
                });
            }
        }
        gpus
    }
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
}

// ============================================================
// CUDA Backend 实现
// ============================================================

#[cfg(feature = "cuda")]
mod cuda_impl {
    use super::*;
    use cudarc::driver::{CudaDevice, CudaSlice, CudaStream, DevicePtr};
    use std::sync::Arc;
    use std::marker::PhantomData;
    use bytemuck::Pod;
    
    /// CUDA 后端
    pub struct CudaBackend<S: Scalar> {
        device: Arc<CudaDevice>,
        stream: CudaStream,
        _marker: PhantomData<S>,
    }
    
    impl<S: Scalar> CudaBackend<S> {
        /// 创建新的 CUDA 后端
        pub fn new(device_id: usize) -> Result<Self, CudaError> {
            let device = CudaDevice::new(device_id)
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
            let stream = device.fork_default_stream()
                .map_err(|e| CudaError::DriverError(e.to_string()))?;
            
            Ok(Self {
                device,
                stream,
                _marker: PhantomData,
            })
        }
        
        /// 获取设备引用
        pub fn device(&self) -> &Arc<CudaDevice> {
            &self.device
        }
    }
    
    impl<S: Scalar> Clone for CudaBackend<S> {
        fn clone(&self) -> Self {
            Self {
                device: self.device.clone(),
                stream: self.device.fork_default_stream().unwrap(),
                _marker: PhantomData,
            }
        }
    }
    
    impl<S: Scalar> std::fmt::Debug for CudaBackend<S> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "CudaBackend<{}>", S::type_name())
        }
    }
    
    impl<S: Scalar> Backend for CudaBackend<S> {
        type Scalar = S;
        type Buffer<T: Pod + Send + Sync> = CudaSlice<T>;
        
        fn name(&self) -> &'static str {
            if std::mem::size_of::<S>() == 4 { "CUDA-f32" } else { "CUDA-f64" }
        }
        
        fn memory_location(&self) -> MemoryLocation {
            MemoryLocation::Device(0)
        }
        
        fn alloc<T: Pod + Clone + Default + Send + Sync>(&self, len: usize) -> CudaSlice<T> {
            let host = vec![T::default(); len];
            self.device.htod_sync_copy(&host).unwrap()
        }
        
        fn alloc_init<T: Pod + Clone + Send + Sync>(&self, len: usize, init: T) -> CudaSlice<T> {
            let host = vec![init; len];
            self.device.htod_sync_copy(&host).unwrap()
        }
        
        fn alloc_uninit<T: Pod + Send + Sync>(&self, len: usize) -> CudaSlice<T> {
            unsafe { self.device.alloc(len).unwrap() }
        }
        
        fn synchronize(&self) {
            self.stream.synchronize().unwrap();
        }
        
        fn axpy(&self, alpha: S, x: &CudaSlice<S>, y: &mut CudaSlice<S>) {
            // TODO: 实现 CUDA kernel
            // 临时：下载到 CPU 计算后上传
            let x_host = self.device.dtoh_sync_copy(x).unwrap();
            let mut y_host = self.device.dtoh_sync_copy(y).unwrap();
            
            for (yi, xi) in y_host.iter_mut().zip(x_host.iter()) {
                *yi = *yi + alpha * *xi;
            }
            
            self.device.htod_sync_copy_into(&y_host, y).unwrap();
        }
        
        fn dot(&self, x: &CudaSlice<S>, y: &CudaSlice<S>) -> S {
            // TODO: 实现 CUDA kernel
            let x_host = self.device.dtoh_sync_copy(x).unwrap();
            let y_host = self.device.dtoh_sync_copy(y).unwrap();
            
            x_host.iter().zip(y_host.iter())
                .fold(S::ZERO, |acc, (&xi, &yi)| acc + xi * yi)
        }
        
        fn copy(&self, src: &CudaSlice<S>, dst: &mut CudaSlice<S>) {
            // TODO: 使用 cudaMemcpy
            let host = self.device.dtoh_sync_copy(src).unwrap();
            self.device.htod_sync_copy_into(&host, dst).unwrap();
        }
        
        fn reduce_max(&self, x: &CudaSlice<S>) -> S {
            // TODO: 实现 CUDA reduction kernel
            let host = self.device.dtoh_sync_copy(x).unwrap();
            host.iter().cloned().fold(S::from_f64(f64::NEG_INFINITY), S::max)
        }
        
        fn reduce_sum(&self, x: &CudaSlice<S>) -> S {
            // TODO: 实现 CUDA reduction kernel
            let host = self.device.dtoh_sync_copy(x).unwrap();
            host.iter().cloned().fold(S::ZERO, |a, b| a + b)
        }
        
        fn scale(&self, alpha: S, x: &mut CudaSlice<S>) {
            // TODO: 实现 CUDA kernel
            let mut host = self.device.dtoh_sync_copy(x).unwrap();
            for xi in host.iter_mut() {
                *xi = *xi * alpha;
            }
            self.device.htod_sync_copy_into(&host, x).unwrap();
        }
        
        fn apply_elementwise<F>(&self, f: F, x: &mut CudaSlice<S>)
        where
            F: Fn(S) -> S + Send + Sync
        {
            let mut host = self.device.dtoh_sync_copy(x).unwrap();
            for xi in host.iter_mut() {
                *xi = f(*xi);
            }
            self.device.htod_sync_copy_into(&host, x).unwrap();
        }
        
        fn enforce_positivity(&self, x: &mut CudaSlice<S>, min_val: S) {
            self.apply_elementwise(|v| v.max(min_val), x);
        }
        
        fn fill(&self, x: &mut CudaSlice<S>, value: S) {
            let host = vec![value; x.len()];
            self.device.htod_sync_copy_into(&host, x).unwrap();
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::CudaBackend;
```

---

### 6.2 Kernel 接口规范

**目标**：定义 GPU Kernel 接口规范。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 扩展 | `core/kernel.rs` | Kernel trait 和规范 |

#### 关键代码

```rust
// mh_physics/src/core/kernel.rs
//! GPU Kernel 接口规范

/// Kernel 优先级
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelPriority {
    /// P0: 核心算子，必须实现
    Core,
    /// P1: 重要算子，建议实现
    Important,
    /// P2: 优化算子，可选实现
    Optional,
}

/// 数据传输策略
#[derive(Debug, Clone, Copy)]
pub enum TransferPolicy {
    /// 同步传输
    Sync,
    /// 异步传输
    Async,
    /// 零拷贝（统一内存）
    ZeroCopy,
}

/// Kernel 规范
#[derive(Debug, Clone)]
pub struct KernelSpec {
    pub name: &'static str,
    pub priority: KernelPriority,
    pub description: &'static str,
}

/// 核心 Kernel 清单
pub const CORE_KERNELS: &[KernelSpec] = &[
    // P0: BLAS Level 1
    KernelSpec {
        name: "axpy",
        priority: KernelPriority::Core,
        description: "y = alpha * x + y",
    },
    KernelSpec {
        name: "dot",
        priority: KernelPriority::Core,
        description: "dot product",
    },
    KernelSpec {
        name: "scale",
        priority: KernelPriority::Core,
        description: "x = alpha * x",
    },
    KernelSpec {
        name: "reduce_max",
        priority: KernelPriority::Core,
        description: "max reduction",
    },
    KernelSpec {
        name: "reduce_sum",
        priority: KernelPriority::Core,
        description: "sum reduction",
    },
    
    // P1: 物理算子
    KernelSpec {
        name: "flux_compute",
        priority: KernelPriority::Important,
        description: "Riemann flux computation",
    },
    KernelSpec {
        name: "state_update",
        priority: KernelPriority::Important,
        description: "state += dt * rhs",
    },
    KernelSpec {
        name: "gradient_compute",
        priority: KernelPriority::Important,
        description: "Green-Gauss gradient",
    },
    
    // P2: 优化算子
    KernelSpec {
        name: "source_batch",
        priority: KernelPriority::Optional,
        description: "batch source term computation",
    },
    KernelSpec {
        name: "pcg_spmv",
        priority: KernelPriority::Optional,
        description: "sparse matrix-vector product",
    },
];

/// GPU Kernel trait
pub trait GpuKernel {
    /// Kernel 名称
    fn name(&self) -> &'static str;
    
    /// Grid 配置
    fn grid_config(&self, n: usize) -> (u32, u32, u32);
    
    /// Block 配置
    fn block_config(&self) -> (u32, u32, u32) {
        (256, 1, 1)  // 默认 256 线程/block
    }
    
    /// 共享内存大小
    fn shared_memory_size(&self) -> usize {
        0
    }
}
```

---

### 6.3 HybridBackend 设计

**目标**：设计混合 CPU/GPU 后端。

#### 改动文件

| 操作 | 文件 | 说明 |
|------|------|------|
| 新建 | `core/hybrid.rs` | 混合后端 |

#### 关键代码

```rust
// mh_physics/src/core/hybrid.rs
//! 混合 CPU/GPU 后端

use super::backend::{Backend, MemoryLocation};
use super::cpu_backend::CpuBackend;
use super::scalar::Scalar;
use bytemuck::Pod;

#[cfg(feature = "cuda")]
use super::gpu::CudaBackend;

/// 混合后端策略
#[derive(Debug, Clone, Copy)]
pub enum HybridStrategy {
    /// 仅 CPU
    CpuOnly,
    /// 仅 GPU
    GpuOnly,
    /// 自动选择（基于问题规模）
    Auto { threshold: usize },
    /// 手动指定
    Manual { gpu_ratio: f64 },
}

/// 混合缓冲区
pub enum HybridBuffer<S: Scalar> {
    Cpu(Vec<S>),
    #[cfg(feature = "cuda")]
    Gpu(cudarc::driver::CudaSlice<S>),
}

impl<S: Scalar> HybridBuffer<S> {
    pub fn as_slice(&self) -> &[S] {
        match self {
            HybridBuffer::Cpu(v) => v.as_slice(),
            #[cfg(feature = "cuda")]
            HybridBuffer::Gpu(_) => panic!("Cannot get slice from GPU buffer"),
        }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [S] {
        match self {
            HybridBuffer::Cpu(v) => v.as_mut_slice(),
            #[cfg(feature = "cuda")]
            HybridBuffer::Gpu(_) => panic!("Cannot get mut slice from GPU buffer"),
        }
    }
    
    pub fn len(&self) -> usize {
        match self {
            HybridBuffer::Cpu(v) => v.len(),
            #[cfg(feature = "cuda")]
            HybridBuffer::Gpu(s) => s.len(),
        }
    }
}

/// 混合后端
pub struct HybridBackend<S: Scalar> {
    cpu: CpuBackend<S>,
    #[cfg(feature = "cuda")]
    gpu: Option<CudaBackend<S>>,
    strategy: HybridStrategy,
}

impl<S: Scalar> HybridBackend<S> {
    pub fn new(strategy: HybridStrategy) -> Self {
        Self {
            cpu: CpuBackend::new(),
            #[cfg(feature = "cuda")]
            gpu: if matches!(strategy, HybridStrategy::CpuOnly) {
                None
            } else {
                CudaBackend::new(0).ok()
            },
            strategy,
        }
    }
    
    /// 是否使用 GPU
    fn use_gpu(&self, n: usize) -> bool {
        #[cfg(feature = "cuda")]
        {
            if self.gpu.is_none() {
                return false;
            }
            
            match self.strategy {
                HybridStrategy::CpuOnly => false,
                HybridStrategy::GpuOnly => true,
                HybridStrategy::Auto { threshold } => n >= threshold,
                HybridStrategy::Manual { gpu_ratio } => gpu_ratio > 0.5,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

impl<S: Scalar> Clone for HybridBackend<S> {
    fn clone(&self) -> Self {
        Self {
            cpu: self.cpu.clone(),
            #[cfg(feature = "cuda")]
            gpu: self.gpu.clone(),
            strategy: self.strategy,
        }
    }
}

impl<S: Scalar> std::fmt::Debug for HybridBackend<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HybridBackend<{}>({:?})", S::type_name(), self.strategy)
    }
}

// 注意：完整的 Backend 实现需要处理 HybridBuffer
// 这里仅提供骨架设计
```

---

## 验收标准

1. ✅ `CudaBackend` 骨架定义完整
2. ✅ Feature gate 正确控制 CUDA 依赖
3. ✅ Kernel 规范文档完整
4. ✅ `HybridBackend` 设计完成
5. ✅ CPU fallback 正常工作

## 测试用例

```rust
#[test]
fn test_cuda_feature_gate() {
    // 无 cuda feature 时应该编译通过
    assert!(!has_cuda() || cfg!(feature = "cuda"));
}

#[test]
#[cfg(feature = "cuda")]
fn test_cuda_backend_creation() {
    if has_cuda() {
        let backend = CudaBackend::<f64>::new(0);
        assert!(backend.is_ok());
    }
}

#[test]
fn test_hybrid_cpu_fallback() {
    let backend = HybridBackend::<f64>::new(HybridStrategy::CpuOnly);
    let x = backend.alloc_init(100, 1.0);
    assert_eq!(x.len(), 100);
}
```

## Kernel 实现路线图

| 阶段 | Kernel | 优先级 | 预计时间 |
|------|--------|--------|----------|
| Phase 7 | axpy, dot, scale | P0 | 1 天 |
| Phase 7 | reduce_max, reduce_sum | P0 | 1 天 |
| 后续 | flux_compute | P1 | 3 天 |
| 后续 | state_update | P1 | 1 天 |
| 后续 | gradient_compute | P1 | 2 天 |
| 后续 | pcg_spmv | P2 | 3 天 |
