// src-tauri/src/marihydro/domain/mod.rs

pub mod boundary;
pub mod feature;
pub mod interpolator;
pub mod rasterizer;
pub mod state;

// ✅ 非结构化网格子模块
pub mod mesh {
    pub mod indices;
    pub mod unstructured;
}

// ✅ 重新导出常用类型
pub use mesh::indices::{CellId, FaceId, NodeId};
pub use mesh::unstructured::UnstructuredMesh;
pub use state::ConservedState;
