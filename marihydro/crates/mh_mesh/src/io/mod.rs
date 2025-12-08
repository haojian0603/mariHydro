// marihydro\crates\mh_mesh\src/io/mod.rs

//! 网格 IO 模块
//!
//! 提供各种网格格式的读写支持。
//!
//! - GMSH (.msh)
//! - GeoJSON
//! - MHB (自定义二进制格式)

pub mod fields;
pub mod geojson;
pub mod gmsh;
pub mod mhb;

pub use fields::{DataType, Compression, FieldDescriptor, FieldIndex};
pub use gmsh::{BoundaryKind, GmshLoader, GmshMeshData, GmshWriter};
pub use mhb::{MhbHeader, MhbReader, MhbWriter};
