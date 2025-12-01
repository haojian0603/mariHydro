// src-tauri/src/marihydro/forcing/providers/mod.rs
pub mod river;
pub mod tide;
pub mod wind;

// 使用显式导出避免 glob re-export 歧义
pub use river::provider::*;
pub use tide::provider::*;
// wind 模块已经在 wind/mod.rs 中定义了显式导出，直接导出模块本身
pub use wind::{
    UniformWindProvider, WindFrame, WindProviderConfig,
    WindReaderConfig, WindReaderFactory, WindFormat, FormatConfig,
    CsvWindReader, CsvWindConfig, CsvTimeFormat, ColumnRef,
    ExcelWindReader, ExcelWindConfig, ExcelTimeFormat,
    TextWindReader, TextWindConfig, TimeFormat,
};
