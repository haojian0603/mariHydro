// crates/mh_physics/src/builder/mod.rs

//! 求解器构建模块
//!
//! 提供从无泛型配置到泛型引擎的桥梁。
//!
//! # 架构设计
//!
//! ```text
//! App Layer (无泛型)
//!     │
//!     ▼
//! SolverConfig ─────> SolverBuilder
//!     │                    │
//!     │                    ▼ (精度分发)
//!     │               ┌────────────────┐
//!     │               │ Precision::F32 │──> ShallowWaterSolver<CpuBackend<f32>>
//!     │               │ Precision::F64 │──> ShallowWaterSolver<CpuBackend<f64>>
//!     │               └────────────────┘
//!     │                    │
//!     ▼                    ▼
//! DynSolver trait <── Box<dyn DynSolver>
//! ```

pub mod dyn_solver;
pub mod config;
pub mod solver_builder;

pub use dyn_solver::{DynSolver, DynState, DynStepResult};
pub use config::SolverConfig;
pub use solver_builder::{SolverBuilder, BuildError};