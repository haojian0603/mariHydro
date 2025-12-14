// crates/mh_config/src/lib.rs

//! MariHydro Config Layer (Layer 4)
//!
//! 配置层，提供精度选择、求解器配置和运行时多态接口。
//! 本层完全无泛型，使用 `Precision` 枚举进行运行时精度分发。
//!
//! # 模块概览
//!
//! - [`precision`]: Precision 枚举（F32/F64）
//! - [`solver_config`]: SolverConfig 求解器配置（全 f64）
//! - [`dyn_solver`]: DynSolver trait 运行时多态接口
//! - [`error`]: 配置错误类型
//!
//! # 层级架构
//!
//! ```text
//! Layer 5: mh_cli      ─> uses SolverConfig, DynSolver
//! Layer 4: mh_config   ─> Precision, SolverConfig, DynSolver (本层)
//! Layer 3: mh_physics  ─> impl DynSolver for Solver<B>
//! Layer 2: mh_runtime  ─> Backend, RuntimeScalar
//! Layer 1: mh_foundation
//! ```
//!
//! # 设计原则
//!
//! 1. **无泛型**: 本层所有类型都不包含泛型参数
//! 2. **全 f64 配置**: SolverConfig 中所有数值使用 f64
//! 3. **运行时分发**: 通过 Precision 枚举选择 f32/f64
//! 4. **Trait 对象**: 通过 `Box<dyn DynSolver>` 实现多态

#![warn(missing_docs)]
#![warn(clippy::all)]

#[cfg(feature = "layer-guard")]
compile_error!("mh_config 禁止在 Layer 4 以下使用");

pub mod precision;
pub mod solver_config;
pub mod dyn_solver;
pub mod error;

/// 层级标识
pub const LAYER: u8 = 4;

// 重导出核心类型
pub use precision::Precision;
pub use solver_config::{SolverConfig, MeshConfig, PhysicsConfig, OutputConfig};
pub use dyn_solver::{DynSolver, GridInfo, MetricsSnapshot, SolverError};
pub use error::ConfigError;
