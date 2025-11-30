// src-tauri/src/marihydro/geo/mod.rs
pub mod crs;
pub mod transform;

pub use crs::{CrsStrategy, ResolvedCrs};
pub use transform::GeoTransformer;
