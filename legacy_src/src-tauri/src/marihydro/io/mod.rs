// src-tauri/src/marihydro/io/mod.rs
pub mod drivers;
pub mod exporters;
pub mod inspector;
pub mod loaders;
pub mod pipeline;
pub mod traits;

pub use exporters::VtuExporter;
pub use loaders::GmshLoader;
pub use pipeline::{IoPipeline, MeshSnapshot, OutputRequest, StateSnapshot};
pub use traits::{GeoTransform, RasterDriver, RasterMetadata};
