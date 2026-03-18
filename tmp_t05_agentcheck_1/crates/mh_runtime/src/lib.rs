// crates/mh_runtime/src/lib.rs

//! MariHydro Runtime Layer (Layer 2)
//!
//! 运行时抽象层，提供计算后端、标量类型、设备缓冲区等核心抽象。
//!
//! # 模块
//!
//! - [`scalar`]: RuntimeScalar trait（密封，仅f32/f64可实现）
//! - [`backend`]: Backend trait和CpuBackend实现
//! - [`buffer`]: DeviceBuffer trait设备缓冲区抽象
//! - [`indices`]: 公共计算索引（无代际验证）
//! - [`tolerance`]: 泛型容差配置
//! - [`arena_ext`]: SafeArena带代际验证的安全内存池
//! - [`error`]: 运行时错误类型
//!
//! # 层级架构
//!
//! ```text
//! Layer 4: mh_config   → Precision, SolverConfig, DynSolver
//! Layer 3: mh_physics  → ShallowWaterSolver<B: Backend, S: SourceTermGeneric<B>>
//! Layer 2: mh_runtime  → Backend, RuntimeScalar, DeviceBuffer (本层)
//! Layer 1: mh_foundation → Arena, Dimension, AlignedVec
//! ```
//!
//! # 设计原则
//!
//! 1. **密封Trait**: RuntimeScalar只有f32/f64实现
//! 2. **零成本抽象**: 编译期单态化，运行时无开销
//! 3. **无代际索引**: indices模块的索引类型不包含代际验证
//! 4. **可选代际**: 需要代际验证时使用arena_ext::SafeArena

// 使用 workspace 统一的 lint 规则
#![warn(missing_docs)]

/// 层级标识
pub const LAYER: u8 = 2;

#[cfg(feature = "layer-guard")]
compile_error!("mh_runtime 禁止在 Layer 1 或更低层使用");

pub mod scalar;
pub mod backend;
pub mod buffer;
pub mod indices;
pub mod metrics;
pub mod numerics; 
pub mod numa;
pub mod tolerance;
pub mod arena_ext;
pub mod error;
pub mod simd;
pub mod soa_layout;

// 核心类型导出
pub use scalar::{RuntimeScalar, AtomicScalar};
pub use backend::{Backend, CpuBackend, MemoryLocation, Vector2D};
pub use buffer::{
    DeviceBuffer, BufferState, BufferUsage, GpuBufferDescriptor,
    BufferPoolConfig, CpuBufferPool, PooledBuffer,
};
pub use indices::{
    CellIndex, FaceIndex, NodeIndex, EdgeIndex, BoundaryIndex, LayerIndex, VertexIndex,
    HalfEdgeIndex, INVALID_INDEX,
    from_foundation_cell, from_foundation_face, from_foundation_node,
    to_foundation_cell, to_foundation_face, to_foundation_node,
};
pub use tolerance::Tolerance;
pub use arena_ext::{
    SafeArena, SafeIdx, SafeCellIndex, SafeFaceIndex, SafeNodeIndex, SafeBoundaryIndex,
    SafeVertexIndex, SafeHalfEdgeIndex,
    StaleIndexError, INVALID_GENERATION,
    same_slot, is_newer,
};
pub use metrics::{MetricsCollector, MetricsSnapshot, Timer};
pub use numerics::KahanSum;
pub use error::RuntimeError;

/// Prelude 模块
pub mod prelude {
    //! 常用类型预导入
    // 核心类型
    pub use crate::{
        Backend, CpuBackend, RuntimeScalar, DeviceBuffer, MemoryLocation, Vector2D,
        CellIndex, FaceIndex, NodeIndex, EdgeIndex, VertexIndex, HalfEdgeIndex, BoundaryIndex,
        INVALID_INDEX,
        Tolerance, RuntimeError,
    };

    // 数学运算
    pub use num_traits::{Float, FromPrimitive, ToPrimitive, NumAssign};

    // 原子操作
    pub use std::sync::atomic::Ordering;

    // 缓冲区相关
    pub use crate::buffer::{GpuBufferDescriptor, BufferPoolConfig, CpuBufferPool, PooledBuffer};

    // 标量常数
    pub use crate::scalar::{AtomicScalar, AtomicF32, AtomicF64};
}