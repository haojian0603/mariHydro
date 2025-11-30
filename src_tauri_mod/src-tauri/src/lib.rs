// src-tauri/src/lib.rs
pub mod commands;
pub mod marihydro;

pub use marihydro::core::error::{MhError, MhResult};
pub use marihydro::infra::init_logging;
