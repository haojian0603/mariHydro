// src-tauri/src/marihydro/geo/mod.rs
//! 地理空间处理模块
//! 
//! 提供纯 Rust 实现的坐标投影转换，不依赖外部 C 库（如 PROJ）

pub mod crs;
pub mod projection;
pub mod transform;

pub use crs::{Crs, CrsDefinition, CrsStrategy, ResolvedCrs};
pub use projection::{Projection, ProjectionType};
pub use transform::{conversions, AffineTransform, GeoTransformer};
