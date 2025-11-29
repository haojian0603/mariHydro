// src-tauri/src/lib.rs

pub mod commands;
pub mod marihydro;

pub use marihydro::infra::error::{MhError, MhResult};
pub use marihydro::infra::logger::init_logging;

pub fn init_logging() {
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();
}
