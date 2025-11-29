// src-tauri/src/marihydro/domain/mesh/mod.rs

pub mod indices;
pub mod unstructured;

pub use indices::{CellId, FaceId, NodeId, INVALID_CELL};
pub use unstructured::{BoundaryKind, CellFaces, UnstructuredMesh};
