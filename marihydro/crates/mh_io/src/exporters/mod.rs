// crates/mh_io/src/exporters/mod.rs

//! 数据导出模块
//!
//! 提供导出各种格式的功能。
//!
//! # VTU 导出
//!
//! VTU (VTK Unstructured Grid) 格式用于 ParaView 可视化。
//! 额外标量字段一旦缺失、越界或实现者内部失败，主链必须显式报错。
//! 状态数组长度不一致时，构造器必须立即报错，不能把错形状状态带到导出阶段。
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
