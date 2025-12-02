// marihydro\crates\mh_foundation\src/lib.rs

//! MariHydro Foundation Layer
//!
//! 零依赖基础层，提供整个项目的基础抽象。
//!
//! # 模块概览
//!
//! - [`index`]: 强类型索引系统，带代际验证
//! - [`arena`]: 泛型 Arena 内存池
//! - [`float`]: 安全浮点数类型和数值常量
//! - [`error`]: 统一错误类型
//! - [`validation`]: 运行时验证工具
//!
//! # 设计原则
//!
//! 1. **零外部依赖**: 仅依赖 serde 和 thiserror
//! 2. **类型安全**: 编译期防止索引误用
//! 3. **零开销抽象**: release 模式下最小化运行时开销
//! 4. **悬垂检测**: 通过代际机制检测已删除元素的访问
//!
//! # 示例
//!
//! ```
//! use mh_foundation::{
//!     index::{CellIndex, Idx},
//!     arena::Arena,
//!     float::SafeF64,
//!     error::{MhError, MhResult},
//! };
//!
//! // 创建 Arena 并插入元素
//! use mh_foundation::index::CellTag;
//! let mut arena: Arena<f64, CellTag> = Arena::new();
//! let idx = arena.insert(42.0);
//!
//! // 安全浮点数运算
//! let x = SafeF64::new(1.0).unwrap();
//! let y = SafeF64::new(2.0).unwrap();
//! let z = x + y;
//! assert_eq!(z.get(), 3.0);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod arena;
pub mod error;
pub mod float;
pub mod index;
pub mod validation;

// 重导出常用类型
pub use arena::Arena;
pub use error::{MhError, MhResult};
pub use float::SafeF64;
pub use index::{
    BoundaryIndex, CellIndex, FaceIndex, HalfEdgeIndex, Idx, NodeIndex, VertexIndex,
};

/// Prelude 模块，包含常用类型
pub mod prelude {
    pub use crate::arena::Arena;
    pub use crate::error::{MhError, MhResult};
    pub use crate::float::{SafeF64, safe_div, safe_sqrt};
    pub use crate::index::{
        BoundaryIndex, CellIndex, FaceIndex, HalfEdgeIndex, Idx, NodeIndex, VertexIndex,
        cell, face, halfedge, node, vertex,
    };
    pub use crate::validation::{ValidationReport, ValidationError, ValidationWarning};
    pub use crate::{ensure, require};
}
