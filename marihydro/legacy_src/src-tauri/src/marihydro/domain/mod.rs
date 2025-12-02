// src-tauri/src/marihydro/domain/mod.rs

//! 领域模型层
//!
//! 包含网格、状态、边界条件、插值器等核心领域对象。
//!
//! # 层级约束
//!
//! - 本模块属于 Layer 2 (领域层)
//! - 仅依赖 core 层
//! - 禁止依赖 physics, forcing, io, infra, workflow

pub mod boundary;
pub mod feature;
pub mod interpolator;
pub mod mesh;
pub mod state;

// 重导出常用类型
pub use boundary::{BoundaryCondition, BoundaryKind, BoundaryManager};
pub use mesh::{MeshBuilder, MeshStatistics, UnstructuredMesh};
pub use state::{ShallowWaterState, StateView, StateViewMut};
