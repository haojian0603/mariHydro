// crates/mh_physics/src/riemann/mod.rs

//! Riemann 求解器模块
//!
//! 提供浅水方程的数值通量计算

pub mod hllc;
pub mod hlle;
pub mod roe;

pub use hllc::{HllcSolver, HllSolver, RiemannSolverTrait};
pub use hlle::HlleSolver;
pub use roe::RoeSolver;
