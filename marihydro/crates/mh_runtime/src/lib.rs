// crates/mh_runtime/src/lib.rs

//! MariHydro Runtime Layer (Layer 2)
//!
//! 运行时抽象层，提供计算后端、标量类型、设备缓冲区等核心抽象。
//!
//! # 模块概览
//!
//! - [`scalar`]: RuntimeScalar trait（密封，仅 f32/f64 可实现）
//! - [`backend`]: Backend trait 和 CpuBackend 实现
//! - [`buffer`]: DeviceBuffer trait 设备缓冲区抽象
//! - [`indices`]: 公共计算索引（无代际验证）
//! - [`tolerance`]: 泛型容差配置
//! - [`arena_ext`]: SafeArena 带代际验证的安全内存池
//! - [`error`]: 运行时错误类型
//!
//! # 层级架构
//!
//! ```text
//! Layer 4: mh_config   ─> Precision, SolverConfig, DynSolver
//! Layer 3: mh_physics  ─> ShallowWaterSolver<B: Backend>
//! Layer 2: mh_runtime  ─> Backend, RuntimeScalar, DeviceBuffer (本层)
//! Layer 1: mh_foundation ─> Arena, Dimension, AlignedVec
//! ```
//!
//! # 设计原则
//!
//! 1. **密封 Trait**: RuntimeScalar 只有 f32/f64 实现
//! 2. **零成本抽象**: 编译期单态化，运行时无开销
//! 3. **无代际索引**: indices 模块的索引类型不包含代际验证
//! 4. **可选代际**: 需要代际验证时使用 arena_ext::SafeArena

#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "layer-guard")]
compile_error!("mh_runtime 禁止在 Layer 1 或更低层使用");

pub mod scalar;
pub mod backend;
pub mod buffer;
pub mod indices;
pub mod tolerance;
pub mod arena_ext;
pub mod error;

/// 层级标识
pub const LAYER: u8 = 2;

// 重导出核心类型
pub use scalar::RuntimeScalar;
pub use backend::{Backend, CpuBackend, MemoryLocation};
pub use buffer::DeviceBuffer;
pub use indices::{CellIndex, FaceIndex, NodeIndex, EdgeIndex, BoundaryIndex, LayerIndex, VertexIndex, HalfEdgeIndex};
pub use indices::INVALID_INDEX;
pub use tolerance::Tolerance;
pub use arena_ext::{
    SafeArena, SafeIdx, SafeCellIndex, SafeFaceIndex, SafeNodeIndex, SafeBoundaryIndex,
    SafeVertexIndex, SafeHalfEdgeIndex,
    StaleIndexError, INVALID_GENERATION,
    same_slot, is_newer,
};
pub use error::RuntimeError;

/// SafeIndex 是 SafeIdx 的别名（向后兼容）
pub type SafeIndex<Tag> = arena_ext::SafeIdx<Tag>;

/// Prelude 模块
pub mod prelude {
    //! 常用类型预导入
    pub use crate::{
        RuntimeScalar, Backend, CpuBackend, DeviceBuffer,
        CellIndex, FaceIndex, NodeIndex, EdgeIndex,
        Tolerance, RuntimeError,
    };
}