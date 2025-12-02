// src-tauri/src/marihydro/mod.rs

//! MariHydro 水动力数值模拟核心库

pub mod core;
pub mod domain;
pub mod forcing;
pub mod geo;
pub mod infra;
pub mod io;
pub mod physics;
pub mod workflow;

// 核心类型导出
pub use core::error::{MhError, MhResult};
pub use core::types::{CellIndex, FaceIndex, NumericalParams};

// 网格和求解器
pub use domain::mesh::UnstructuredMesh;
pub use physics::engine::{ImprovedSolver, SolverConfig, SolverV2Builder};

// 工作流管理
pub use workflow::WorkflowManagerV2;
