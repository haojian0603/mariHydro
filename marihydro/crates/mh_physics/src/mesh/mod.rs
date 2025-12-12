//! crates/mh_physics/src/mesh/mod.rs
//! 网格抽象层
//!
//! 提供结构化和非结构化网格的统一接口。

pub mod topology;
pub mod unstructured;
pub mod structured;

pub use topology::{MeshKind, MeshTopology, FaceInfo, MeshGeometry};
pub use unstructured::UnstructuredMeshAdapter;
pub use structured::StructuredMesh;
