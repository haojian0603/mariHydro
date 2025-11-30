// src-tauri/src/marihydro/domain/mesh/mod.rs

//! 非结构化网格模块

pub mod builder;
pub mod indices;
pub mod topology;
pub mod unstructured;

pub use builder::MeshBuilder;
pub use indices::{CellId, FaceId, NodeId, INVALID_CELL, INVALID_FACE};
pub use topology::MeshTopologyExt;
pub use unstructured::{CellFaces, MeshStatistics, UnstructuredMesh};
