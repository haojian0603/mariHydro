// src-tauri/src/marihydro/domain/mesh/mod.rs

//! 非结构化网格模块

pub mod builder;
pub mod coloring;
pub mod topology;
pub mod unstructured;

pub use builder::MeshBuilder;
pub use coloring::{ColoringStats, MeshColoring};
pub use topology::MeshTopologyExt;
pub use unstructured::{CellFaces, MeshStatistics, UnstructuredMesh};

// 从 core::types 重导出索引类型（向后兼容）
pub use crate::marihydro::core::types::{CellId, FaceId, NodeId, INVALID_INDEX};
/// 兼容旧的 INVALID_CELL 常量
pub const INVALID_CELL: usize = crate::marihydro::core::types::INVALID_INDEX;
/// 兼容旧的 INVALID_FACE 常量
pub const INVALID_FACE: usize = crate::marihydro::core::types::INVALID_INDEX;
