//! 核心抽象层
//!
//! 提供计算后端、缓冲区、标量类型等基础抽象。
//!
//! # 模块结构
//!
//! - [`buffer`]: AlignedVec 的 DeviceBuffer 实现（扩展 mh_runtime）
//! - [`dimension`]: 维度标记 (2D/3D)
//! - [`kernel`]: GPU Kernel 接口规范
//! - [`gpu`]: GPU 后端占位符
//!
//! # 设计原则
//!
//! 所有核心抽象（Backend, DeviceBuffer, RuntimeScalar）统一定义在 mh_runtime，
//! 本模块仅提供维度标记、GPU扩展、AlignedVec支持等 physics 层专用类型。
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::core::{Backend, CpuBackend};
//! use mh_runtime::RuntimeScalar as Scalar;
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

// AlignedVec 的 DeviceBuffer 实现
pub mod buffer;
pub mod dimension;
pub mod kernel;
pub mod gpu;

// 从 mh_runtime 重导出核心抽象（Single Source of Truth）
pub use mh_runtime::DeviceBuffer;
pub use mh_runtime::{Backend, CpuBackend, MemoryLocation};

// DefaultBackend 需要单独定义
pub type DefaultBackend = CpuBackend<f64>;

// AlignedBuffer - AlignedVec 的 DeviceBuffer 适配器
pub use buffer::AlignedBuffer;

// 本模块专有类型
pub use dimension::{Dimension, D2, D3};
pub use kernel::{KernelSpec, KernelPriority, TransferPolicy, CORE_KERNELS};
pub use gpu::{CudaError, GpuDeviceInfo, available_gpus, has_cuda};
