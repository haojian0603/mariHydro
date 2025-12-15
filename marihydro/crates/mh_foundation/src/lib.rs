// marihydro\crates\mh_foundation\src/lib.rs

//! MariHydro Foundation Layer
//!
//! 零依赖基础层，提供整个项目的基础抽象。

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod arena;
pub mod dimension;
pub mod error;
// 已删除: pub mod float;  // 迁移到 mh_runtime::RuntimeScalar
pub mod index;
pub mod memory;
pub mod metrics;
pub mod validation;

// 重导出核心类型（仅限基础层）
pub use arena::Arena;
pub use dimension::{D2, D3, D3Dynamic, Dimension, DimensionExt};

// ✅ 修复：使用正确的类型名
pub use error::{MhError, MhResult};

// 已删除 float 相关导出
// pub use float::{SafeF64, KahanSum};

// ✅ 仅导出轻量级索引类型
pub use index::{
    CellIdx, FaceIdx, NodeIdx, BoundaryId, INVALID_IDX,
};

pub use memory::{AlignedVec, Alignment, CpuAlign, GpuAlign};

// ✅ 修复：导出 ArenaTag trait
pub use arena::ArenaTag;

/// Prelude 模块，包含常用类型
pub mod prelude {
    pub use crate::arena::Arena;
    pub use crate::dimension::{D2, D3, D3Dynamic, Dimension, DimensionExt};
    pub use crate::error::{MhError, MhResult};
    
    // 已删除 float 引用
    // pub use crate::float::{SafeF64, safe_div, safe_sqrt};
    
    // ✅ 仅保留轻量级索引
    pub use crate::index::{
        CellIdx, FaceIdx, NodeIdx, BoundaryId, INVALID_IDX,
    };
    
    pub use crate::memory::{AlignedVec, CpuAlign, GpuAlign};
    pub use crate::validation::{ValidationReport, ValidationError, ValidationWarning};
    // 宏在 lib.rs 中定义，prelude 自动包含
}

/// 层级标识（Layer 1）
pub const LAYER: u8 = 1;

/// 验证宏（定义在 lib.rs，避免重复）
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $err:expr) => {
        if !$cond {
            return Err($err);
        }
    };
}

/// 验证 Option（定义在 lib.rs，避免重复）
#[macro_export]
macro_rules! require {
    ($opt:expr, $err:expr) => {
        match $opt {
            Some(v) => v,
            None => return Err($err),
        }
    };
}