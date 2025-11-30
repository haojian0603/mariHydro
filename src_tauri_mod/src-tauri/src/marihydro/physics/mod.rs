// src-tauri/src/marihydro/physics/mod.rs

//! 物理计算层
//!
//! 包含数值计算核心、通量格式、源项计算等。
//!
//! # 层级约束
//!
//! - 本模块属于 Layer 3 (物理计算层)
//! - 依赖 core 和 domain 层
//! - 禁止依赖 forcing, io, infra, workflow

pub mod numerics;
pub mod schemes;
pub mod sediment;
pub mod sources;
pub mod waves;

// 重导出常用类型
pub use numerics::gradient::{
    GreenGaussGradient, LeastSquaresGradient, ScalarGradientStorage, VectorGradientStorage,
};
pub use numerics::limiter::{BarthJespersenLimiter, VenkatakrishnanLimiter};
pub use schemes::{
    DryWetHandler, HllcSolver, HydrostaticReconstruction, InterfaceFlux, WetDryState,
};
