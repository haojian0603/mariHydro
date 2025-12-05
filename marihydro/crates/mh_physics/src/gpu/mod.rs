// crates/mh_physics/src/gpu/mod.rs

//! GPU加速模块
//!
//! 提供基于 wgpu 的 GPU 计算加速功能。
//!
//! # 功能
//!
//! - 通量计算 GPU 加速
//! - 时间积分 GPU 加速
//! - 大规模并行网格计算
//!
//! # 架构
//!
//! ```text
//! +------------------+
//! |   GpuContext     |  <- 设备管理和资源分配
//! +------------------+
//!          |
//!          v
//! +------------------+
//! | CompiledPipeline |  <- 计算管线配置
//! +------------------+
//!          |
//!          v
//! +------------------+
//! |   TypedBuffer    |  <- 类型安全缓冲区
//! +------------------+
//! ```
//!
//! # 模块结构
//!
//! - `capabilities` - 设备能力检测
//! - `backend` - 计算后端抽象 (CPU/GPU)
//! - `buffer` - GPU缓冲区管理
//! - `pipeline` - 计算管线
//! - `solver` - GPU求解器
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::gpu::{GpuSolver, GpuSolverConfig, DeviceCapabilities};
//!
//! // 创建GPU求解器
//! let config = GpuSolverConfig::default();
//! let solver = GpuSolver::new(config)?;
//!
//! // 检查设备能力
//! let caps = solver.capabilities();
//! println!("Device: {}", caps.device_name);
//! println!("Memory: {} MB", caps.memory.total_mb());
//! ```

pub mod backend;
pub mod bind_groups;
pub mod buffer;
pub mod capabilities;
pub mod mesh;
pub mod pipeline;
pub mod shaders;
pub mod solver;
pub mod state;
pub mod wgpu_backend;

// 核心类型重导出
pub use backend::{ComputeBackend, ComputeOperation, CpuBackend, PerformanceEstimate};
pub use bind_groups::{BindGroupLayouts, ColoringData, GpuComputeParams, ParamsBuffer};
pub use buffer::{BufferPool, DoubleBuffer, GpuBufferUsage, TypedBuffer};
pub use capabilities::{ComputeFeatures, DeviceCapabilities, DeviceType, MemoryInfo};
pub use mesh::{GpuCellPod, GpuFacePod, GpuMeshData, GpuMeshTopology, ToGpuMesh};
pub use pipeline::{
    BindingConfig, BindingConfigType, CompiledPipeline, ComputePipelineConfig, PipelineCache,
};
pub use solver::{GpuContext, GpuError, GpuSolver, GpuSolverConfig, GpuStats};
pub use state::{GpuStateArrays, GpuWorkspace};
pub use wgpu_backend::WgpuBackend;
