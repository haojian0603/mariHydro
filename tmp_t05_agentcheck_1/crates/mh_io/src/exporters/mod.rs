// crates/mh_io/src/exporters/mod.rs

//! 数据导出模块
//!
//! 提供导出各种格式的功能。
//!
//! # VTU 导出
//!
//! VTU (VTK Unstructured Grid) 格式用于 ParaView 可视化。
//!
//! ```rust,ignore
//! use mh_io::exporters::{VtuExporter, SimpleState};
//!
//! let exporter = VtuExporter::new().h_dry(1e-6);
//! exporter.export("output.vtu", &mesh, &state, 0.0)?;
//! ```

pub mod shapefile;
pub mod vtu;

// 重导出
pub use vtu::{
    SimpleState, StateWithScalars, VtuCellType, VtuError, VtuExportConfig, VtuExporter, VtuMesh,
    VtuState, VtuStateExt,
};