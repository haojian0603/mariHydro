// src-tauri/src/marihydro/io/inspector/file_inspector.rs
use super::dto::{FileInfo, FileType, FileMetadata};
use crate::marihydro::core::error::{MhError, MhResult};
use std::path::Path;
use std::fs;

pub struct FileInspector;

impl FileInspector {
    pub fn inspect(path: &str) -> MhResult<FileInfo> {
        let p = Path::new(path);
        if !p.exists() { return Err(MhError::io(format!("File not found: {}", path))); }
        let meta = fs::metadata(p).map_err(|e| MhError::io(e.to_string()))?;
        let size = meta.len();
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let file_type = match ext.as_str() {
            "msh" | "geo" => FileType::Mesh,
            "tif" | "tiff" | "asc" | "dem" => FileType::Raster,
            "nc" | "nc4" => FileType::NetCdf,
            "csv" | "txt" => FileType::Csv,
            "json" => FileType::Json,
            _ => FileType::Unknown,
        };
        Ok(FileInfo { path: path.into(), file_type, size_bytes: size, metadata: FileMetadata::default() })
    }

    pub fn get_extension(path: &str) -> Option<String> {
        Path::new(path).extension().and_then(|e| e.to_str()).map(|s| s.to_lowercase())
    }
}
