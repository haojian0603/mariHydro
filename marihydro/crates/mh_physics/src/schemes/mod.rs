// marihydro\crates\mh_physics\src/schemes/mod.rs

//! 数值格式模块
//!
//! 提供浅水方程求解所需的数值格式，包括：
//! - 黎曼求解器 (HLLC)
//! - 干湿处理

pub mod riemann;
pub mod wetting_drying;

// 重导出常用类型
pub use riemann::{HllcSolver, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
pub use wetting_drying::{WetState, WettingDryingConfig, WettingDryingHandler};
