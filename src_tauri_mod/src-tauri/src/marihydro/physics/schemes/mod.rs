// src-tauri/src/marihydro/physics/schemes/mod.rs

//! 通量格式模块
//!
//! 包含数值通量计算、干湿处理、静水重构等核心数值方法。

// 废弃的模块
#[deprecated(since = "0.3.0", note = "请使用 wetting_drying 模块替代")]
pub mod dry_wet;

#[deprecated(since = "0.3.0", note = "请使用 riemann 模块替代")]
pub mod hllc;

pub mod flux_utils;
pub mod hydrostatic;
pub mod riemann;
pub mod wetting_drying;

// 废弃的重导出（保持向后兼容）
#[allow(deprecated)]
pub use dry_wet::{DryWetHandler, WetDryFaceReconstruction, WetDryState};

#[allow(deprecated)]
pub use hllc::HllcSolver;

pub use flux_utils::{InterfaceFlux, RotatedFlux, RotatedState};
pub use hydrostatic::{BedSlopeSource, HydrostaticFaceState, HydrostaticReconstruction};

// 新的干湿处理导出
pub use wetting_drying::{
    DryWetFlux, FaceReconstruction, MomentumCorrector, MomentumCorrectionMethod,
    SmoothingType, TransitionFunction, WetState, WettingDryingHandler,
};

// 新的黎曼求解器导出
pub use riemann::{
    AdaptiveSolver, InterfaceState, RiemannFlux, RiemannSolver, RusanovSolver,
    SolverCapabilities, SolverSelector, SolverType,
};
// 新的 HLLC 使用 riemann 模块的版本
pub use riemann::HllcSolver as NewHllcSolver;
