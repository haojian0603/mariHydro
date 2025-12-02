// src-tauri/src/marihydro/infra/db/mod.rs
pub mod sqlite;
pub use sqlite::{Database, ProjectRecord, JobRecord, JobStatus};
