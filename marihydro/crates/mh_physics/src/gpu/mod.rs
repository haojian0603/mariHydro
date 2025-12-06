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
//! - 混合调度 (CPU/GPU 自动切换)
//! - 性能监控和瓶颈分析
//! - 数据验证和质量保证
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
//!          |
//!          v
//! +------------------+     +------------------+
//! | HybridScheduler  | <-> |   GpuProfiler    |
//! +------------------+     +------------------+
//!          |
//!          v
//! +------------------+
//! |   GpuValidator   |  <- 结果验证
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
//! - `hybrid_scheduler` - CPU/GPU 混合调度
//! - `profiler` - 性能监控和分析
//! - `validator` - 数据验证
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::gpu::{GpuSolver, GpuSolverConfig, DeviceCapabilities};
//! use mh_physics::gpu::hybrid_scheduler::{HybridScheduler, HybridSchedulerConfig};
//! use mh_physics::gpu::profiler::{GpuProfiler, ProfilerConfig};
//!
//! // 创建GPU求解器
//! let config = GpuSolverConfig::default();
//! let solver = GpuSolver::new(config)?;
//!
//! // 创建混合调度器
//! let scheduler = HybridScheduler::new(
//!     HybridSchedulerConfig::default(),
//!     Some(solver.capabilities().clone())
//! );
//!
//! // 获取调度决策
//! let decision = scheduler.decide(50_000, 100 * 1024 * 1024);
//! println!("使用后端: {:?}", decision.backend);
//! ```

pub mod backend;
pub mod bind_groups;
pub mod buffer;
pub mod capabilities;
pub mod hybrid_scheduler;
pub mod mesh;
pub mod pipeline;
pub mod profiler;
pub mod shaders;
pub mod solver;
pub mod state;
pub mod validator;
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

// 新增模块重导出
pub use hybrid_scheduler::{
    BackendStats, BackendType, DecisionReason, ExecutionResult, GpuTask, HybridScheduler,
    HybridSchedulerConfig, SchedulerStats, SchedulingDecision,
};
pub use profiler::{
    BottleneckAnalysis, BottleneckType, GpuProfiler, MetricType, ProfilerConfig, ScopeStats,
};
pub use validator::{
    GpuValidator, ValidationConfig, ValidationError, ValidationErrorType, ValidationResult,
    ValidationStats,
};
