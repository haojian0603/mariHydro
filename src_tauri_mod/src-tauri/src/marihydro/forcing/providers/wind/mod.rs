// src-tauri/src/marihydro/forcing/providers/wind/mod.rs
pub mod grib;
pub mod netcdf;
pub mod provider;

pub use provider::{WindProviderConfig, WindFrame, UniformWindProvider};
