// marihydro\crates\mh_mesh\src/io/mod.rs

//! 网格 IO 模块
//!
//! 提供各种网格格式的读写支持。
//! 其中 MHB 主链要求计数字段、偏移字段和标量字段都按显式契约读写，
//! 结构不完整、数值非有限或平台不可表示时直接报错，不做默认值回退。
//!
//! MHB_SCOPE: 模块级导出保留 MHB 的显式计数、偏移和有限标量语义；非法结构与非法数值必须立即失败。
//!
//! - GMSH (.msh)
//! - GeoJSON
//! - MHB (自定义二进制格式)

pub mod fields;
pub mod geojson;
pub mod gmsh;
pub mod mhb;

pub use fields::{Compression, DataType, FieldDescriptor, FieldIndex};
pub use gmsh::{BoundaryKind, GmshLoader, GmshMeshData, GmshWriter};
pub use mhb::{load_mhb, save_mhb, MhbHeader, MhbReader, MhbWriter};
