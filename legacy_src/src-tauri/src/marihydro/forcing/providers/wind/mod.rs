// src-tauri/src/marihydro/forcing/providers/wind/mod.rs
pub mod csv_reader;
pub mod excel_reader;
pub mod factory;
pub mod grib;
pub mod netcdf;
pub mod provider;
pub mod text_reader;

pub use csv_reader::{ColumnRef, CsvTimeFormat, CsvWindConfig, CsvWindReader};
pub use excel_reader::{ExcelTimeFormat, ExcelWindConfig, ExcelWindReader};
pub use factory::{FormatConfig, WindFormat, WindReaderConfig, WindReaderFactory};
pub use provider::{UniformWindProvider, WindFrame, WindProviderConfig};
pub use text_reader::{TextWindConfig, TextWindReader, TimeFormat};
