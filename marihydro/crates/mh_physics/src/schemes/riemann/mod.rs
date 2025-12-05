// crates/mh_physics/src/schemes/riemann/mod.rs

//! 黎曼求解器模块
//!
//! 提供浅水方程的近似黎曼求解器：
//! - HLLC: 高精度求解器，正确处理接触间断
//! - Rusanov: 简单鲁棒的求解器
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/schemes/riemann 迁移，保持算法不变。

mod hllc;
mod traits;

pub use hllc::HllcSolver;
pub use traits::{RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
