//! GPU计算抽象层
//!
//! 提供统一的CPU/GPU计算后端抽象，支持：
//! - 计算后端trait定义
//! - 设备能力检测
//! - GPU缓冲区抽象
//! - 可选的wgpu后端实现
//!
//! # Feature Flags
//!
//! - `gpu`: 启用wgpu GPU后端支持
//!
//! # 使用示例
//!
//! ```rust,ignore
//! use marihydro::core::compute::{ComputeBackend, DeviceCapabilities};
//!
//! // 获取默认后端
//! let backend = get_default_backend();
//! println!("Using backend: {}", backend.name());
//! println!("Supports f64: {}", backend.capabilities().supports_f64);
//! ```

pub mod backend;
pub mod buffer;
pub mod capabilities;
pub mod gpu_boundary;
pub mod gpu_coloring;
#[cfg(feature = "gpu")]
pub mod gpu_mesh;
#[cfg(feature = "gpu")]
pub mod gpu_state;
pub mod hybrid_scheduler;
#[cfg(feature = "gpu")]
pub mod mesh_converter;
pub mod profiler;
pub mod transfer;
pub mod validator;

#[cfg(feature = "gpu")]
pub mod gpu_solver;

#[cfg(feature = "gpu")]
pub mod pipeline;

#[cfg(feature = "gpu")]
pub mod wgpu_backend;

#[cfg(feature = "gpu")]
pub mod wgpu_buffer;

// 重导出核心类型
pub use backend::{ComputeBackend, ComputeOperation, PerformanceEstimate};
pub use buffer::BufferUsage;
#[cfg(feature = "gpu")]
pub use buffer::GpuBuffer;
pub use capabilities::{ComputeFeatures, DeviceCapabilities, DeviceType, MemoryInfo};
pub use gpu_coloring::{GpuColoring, GreedyColoring};
#[cfg(feature = "gpu")]
pub use gpu_mesh::{
    GpuCellGeometry, GpuCellPod, GpuFaceGeometry, GpuFacePod, GpuFluxPod, GpuMeshData,
    GpuMeshTopology, GpuStatePod, ToGpuMesh, GPU_INVALID_CELL,
};
#[cfg(feature = "gpu")]
pub use gpu_state::{
    GpuFaceReconstruction, GpuFluxArrays, GpuGradientArrays, GpuSourceArrays, GpuStateArrays,
    GpuVelocityGradients, GpuWorkspace,
};
#[cfg(feature = "gpu")]
pub use mesh_converter::MeshConverter;
pub use transfer::{CpuStateCache, SyncPolicy, TransferDirection, TransferManager, TransferStats};
pub use hybrid_scheduler::{DeviceSelection, HybridConfig, HybridScheduler, HybridStrategy};
pub use gpu_boundary::{
    BoundaryFace, BoundaryType, BoundaryValue, GpuBoundaryManager, TidalBoundary, TidalConstituent,
};
pub use profiler::{CounterSummary, GpuCounters, GpuProfiler, GpuTimer, MemoryStats};
pub use validator::{
    ConservationResult, GpuValidator, NumberValidityResult, PhysicalConstraintResult,
    ValidationReport, ValidationResult,
};

#[cfg(feature = "gpu")]
pub use gpu_solver::{GpuSolver, GpuSolverConfig, GpuStats};

#[cfg(feature = "gpu")]
pub use pipeline::{PipelineCache, PipelineId};

#[cfg(feature = "gpu")]
pub use wgpu_backend::WgpuBackend;

#[cfg(feature = "gpu")]
pub use wgpu_buffer::{BufferManager, DoubleBuffer, WgpuBuffer};

use crate::marihydro::core::error::MhResult;

/// CPU后端（始终可用）
pub struct CpuBackend {
    capabilities: DeviceCapabilities,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    /// 创建CPU后端
    pub fn new() -> Self {
        Self {
            capabilities: DeviceCapabilities::cpu_default(),
        }
    }
}

impl ComputeBackend for CpuBackend {
    fn name(&self) -> &'static str {
        "CPU (rayon)"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }

    fn estimate_performance(
        &self,
        op: ComputeOperation,
        problem_size: usize,
    ) -> PerformanceEstimate {
        // CPU性能估计（基于问题规模）
        let base_time = match op {
            ComputeOperation::Gradient => 0.001,      // 1μs per cell
            ComputeOperation::Limiter => 0.0005,      // 0.5μs per cell
            ComputeOperation::Riemann => 0.002,       // 2μs per face
            ComputeOperation::FluxAccumulate => 0.001, // 1μs per face
            ComputeOperation::SourceTerms => 0.001,   // 1μs per cell
            ComputeOperation::TimeIntegrate => 0.0005, // 0.5μs per cell
            ComputeOperation::Reduction => 0.0001,    // 0.1μs per element
        };

        let threads = self.capabilities.max_threads().max(1);
        let estimated_time_ms = (base_time * problem_size as f64) / threads as f64;

        PerformanceEstimate {
            estimated_time_ms,
            memory_required: problem_size * 8, // 假设f64
            recommended: problem_size < 100_000, // CPU对小问题更优
        }
    }

    fn synchronize(&self) -> MhResult<()> {
        // CPU不需要同步
        Ok(())
    }
}

/// 获取默认计算后端
///
/// 如果启用了GPU feature且GPU可用，返回GPU后端；
/// 否则返回CPU后端。
pub fn get_default_backend() -> Box<dyn ComputeBackend> {
    #[cfg(feature = "gpu")]
    {
        if let Ok(Some(gpu)) = wgpu_backend::WgpuBackend::new_blocking() {
            return Box::new(gpu);
        }
    }

    Box::new(CpuBackend::new())
}

/// 检查GPU是否可用
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        wgpu_backend::WgpuBackend::new_blocking()
            .map(|opt| opt.is_some())
            .unwrap_or(false)
    }

    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}
