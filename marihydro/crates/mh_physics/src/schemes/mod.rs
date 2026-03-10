// marihydro\crates\mh_physics\src/schemes/mod.rs

//! 鏁板€兼牸寮忔ā鍧?
//!
//! 鎻愪緵娴呮按鏂圭▼姹傝В鎵€闇€鐨勬暟鍊兼牸寮忥紝鍖呮嫭锛?
//! - 榛庢浖姹傝В鍣?(HLLC)
//! - 骞叉箍澶勭悊
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/physics/schemes 杩佺Щ锛屼繚鎸佺畻娉曚笉鍙樸€?

pub mod riemann;
pub mod wetting_drying;

// 閲嶅鍑哄父鐢ㄧ被鍨?
pub use riemann::{HllcSolver, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
pub use wetting_drying::{WetState, WettingDryingConfig, WettingDryingHandler};
