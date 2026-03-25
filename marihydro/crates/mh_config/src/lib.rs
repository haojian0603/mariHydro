// crates/mh_config/src/lib.rs

//! MariHydro Config Layer (Layer 4)
//!
//! 配置层提供运行前需要的无泛型配置结构，以及一个可选的运行时求解器 trait 合约。
//! 当前主线 CLI 和 workflow 使用真实的 `Layer3Config -> ShallowWaterSolver` 路径；
//! `DynSolver` 仅作为独立接口定义存在，不能在文档里被描述成“仓库已接入的主运行路径”。
//!
//! # 模块概览
//!
//! - [`precision`]: Precision 枚举（F32/F64）
//! - [`solver_config`]: SolverConfig 求解器配置（全 f64）
//! - [`dyn_solver`]: DynSolver trait 定义与辅助快照类型
//! - [`error`]: 配置错误类型
//!
//! # 层级架构
//!
//! ```text
//! Layer 5: mh_cli / mh_workflow ─> uses SolverConfig and real solver dispatch
//! Layer 4: mh_config            ─> Precision, SolverConfig, optional DynSolver contract
//! Layer 3: mh_physics           ─> Layer3Config, ShallowWaterSolver<B, S>
//! Layer 2: mh_runtime           ─> Backend, RuntimeScalar
//! Layer 1: mh_foundation
//! ```
//!
//! # 设计原则
//!
//! 1. **无泛型配置**: 本层对外配置类型不包含泛型参数
//! 2. **全 f64 配置值**: SolverConfig 中所有数值使用 f64 存储
//! 3. **运行时精度分发**: 通过 Precision 枚举选择 f32/f64
//! 4. **禁止虚假承诺**: 没有真实接线的运行时多态路径不得写成“已支持主链功能”

#![warn(missing_docs)]

#[cfg(feature = "layer-guard")]
compile_error!("mh_config 禁止在 Layer 4 以下使用");

pub mod precision;
pub mod solver_config;
pub mod dyn_solver;
pub mod error;
pub mod hot_reload;

/// 层级标识
pub const LAYER: u8 = 4;

pub use dyn_solver::{DynSolver, GridInfo, MetricsSnapshot, SolverError};
pub use error::ConfigError;
pub use hot_reload::{
    ConfigUpdates, ConfigValue, ConfigWatcher, HotReloadConfig, HotReloadError, HotReloadResult,
    HotReloadable,
};
pub use precision::Precision;
pub use solver_config::{MeshConfig, OutputConfig, PhysicsConfig, SolverConfig};
