//! 核心抽象层
//!
//! 提供计算后端、缓冲区、标量类型等基础抽象。
//!
//! # 模块结构
//!
//! - [`scalar`]: 标量类型抽象 (f32/f64)
//! - [`buffer`]: 设备缓冲区抽象
//! - [`backend`]: 计算后端抽象 (CPU/GPU)
//! - [`dimension`]: 维度标记 (2D/3D)
//! - [`kernel`]: GPU Kernel 接口规范
//! - [`gpu`]: GPU 后端占位符
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::core::{Backend, CpuBackend, Scalar};
//!
//! // 创建 f64 精度的 CPU 后端实例
//! let backend = CpuBackend::<f64>::new();
//!
//! // 使用实例方法分配和操作缓冲区
//! let x = backend.alloc_init(100, 1.0);
//! let mut y = backend.alloc_init(100, 2.0);
//! backend.axpy(0.5, &x, &mut y);
//! assert!((y[0] - 2.5).abs() < 1e-10);
//! ```

pub mod scalar;
pub mod buffer;
pub mod backend;
pub mod dimension;
pub mod kernel;
pub mod gpu;

// 重导出常用类型
pub use scalar::Scalar;
pub use buffer::DeviceBuffer;
pub use backend::{Backend, CpuBackend, DefaultBackend, MemoryLocation};
pub use dimension::{Dimension, D2, D3};
pub use kernel::{KernelSpec, KernelPriority, TransferPolicy, CORE_KERNELS};
pub use gpu::{CudaError, GpuDeviceInfo, available_gpus, has_cuda};
