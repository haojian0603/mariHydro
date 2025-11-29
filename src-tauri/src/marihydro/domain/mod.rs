// src-tauri/src/marihydro/domain/mod.rs

pub mod boundary;
pub mod feature;
pub mod geometry_mapper;
pub mod interpolator;
pub mod state;

pub mod mesh {
    pub mod indices;
    pub mod unstructured;
}

pub use mesh::indices::{CellId, FaceId, NodeId};
pub use mesh::unstructured::UnstructuredMesh;
pub use state::ConservedState;
