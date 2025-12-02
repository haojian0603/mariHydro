// src-tauri/src/marihydro/io/inspector/dto.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub file_type: FileType,
    pub size_bytes: u64,
    pub metadata: FileMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FileType { Mesh, Raster, NetCdf, Csv, Json, Unknown }

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FileMetadata {
    pub crs_wkt: Option<String>,
    pub extent: Option<[f64; 4]>,
    pub dimensions: Option<[usize; 3]>,
    pub variables: Option<Vec<String>>,
    pub time_range: Option<[String; 2]>,
}

impl FileInfo {
    pub fn unknown(path: &str) -> Self {
        Self { path: path.into(), file_type: FileType::Unknown, size_bytes: 0, metadata: FileMetadata::default() }
    }
}
