// src-tauri/src/marihydro/core/traits/mod.rs

//! 核心抽象 trait 定义
//!
//! 本模块定义了 MariHydro 系统的所有核心抽象接口。
//! 这些 trait 遵循依赖倒置原则：高层模块定义接口，低层模块实现接口。
//!
//! # 层级约束
//!
//! - 本模块属于 Layer 1 (核心层)
//! - 禁止依赖 domain, physics, forcing, io, infra, workflow 等上层模块
//! - 仅依赖 core::types 和 core::error

pub mod flux;
pub mod forcing;
pub mod gradient;
pub mod interpolator;
pub mod limiter;
pub mod mesh;
pub mod repository;
pub mod source;
pub mod state;

// 重导出常用 trait
pub use flux::FluxScheme;
pub use forcing::{ForcingData, RiverProvider, TideProvider, WindProvider};
pub use gradient::{GradientComputer, ScalarGradient, VectorGradient};
pub use interpolator::{SpatialInterpolator, TemporalInterpolator};
pub use limiter::GradientLimiter;
pub use mesh::{CellGeometry, FaceGeometry, MeshAccess, MeshTopology};
pub use repository::{Repository, SimulationRepository};
pub use source::SourceTerm;
pub use state::{ConservedState, StateAccess, StateAccessMut};
