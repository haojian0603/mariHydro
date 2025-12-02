// src-tauri/src/marihydro/physics/schemes/mod.rs

//! 通量格式模块
//!
//! 包含数值通量计算、干湿处理、静水重构等核心数值方法。

pub mod flux_utils;
pub mod hydrostatic;
pub mod riemann;
pub mod wetting_drying;

pub use flux_utils::{InterfaceFlux, RotatedFlux, RotatedState};
pub use hydrostatic::{BedSlopeSource, HydrostaticFaceState, HydrostaticReconstruction};

// 干湿处理导出
pub use wetting_drying::{
    DryWetFlux, FaceReconstruction, MomentumCorrector, MomentumCorrectionMethod,
    SmoothingType, TransitionFunction, WetState, WettingDryingHandler,
};

// 黎曼求解器导出
pub use riemann::{
    AdaptiveSolver, HllcSolver, InterfaceState, RiemannFlux, RiemannSolver, RusanovSolver,
    SolverCapabilities, SolverSelector, SolverType,
};

