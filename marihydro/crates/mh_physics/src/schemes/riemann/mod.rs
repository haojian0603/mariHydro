// crates/mh_physics/src/schemes/riemann/mod.rs

//! 黎曼求解器模块
//!
//! 提供浅水方程的近似黎曼求解器：
//!
//! - [`HllcSolver`]: 高精度求解器，正确处理接触间断
//! - [`RusanovSolver`]: 简单鲁棒的求解器，GPU 友好
//! - [`AdaptiveSolver`]: 自适应求解器，自动选择最优方法
//!
//! # 求解器选择指南
//!
//! | 求解器 | 精度 | 稳定性 | 计算成本 | 适用场景 |
//! |--------|-----|--------|---------|---------|
//! | HLLC | 高 | 中 | 高 | 平滑流、接触间断 |
//! | Rusanov | 低 | 高 | 低 | 强间断、GPU计算 |
//! | Adaptive | 高 | 高 | 中-高 | 通用、干湿过渡 |
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::schemes::riemann::{AdaptiveSolver, RiemannSolver};
//!
//! let solver = AdaptiveSolver::new(&params, 9.81);
//! let flux = solver.solve(h_l, h_r, vel_l, vel_r, normal)?;
//! ```

mod adaptive;
mod hllc;
mod rusanov;
mod traits;

// 核心类型
pub use traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};

// 求解器实现
pub use hllc::HllcSolver;
pub use rusanov::{create_robust_rusanov_solver, create_rusanov_solver, RusanovConfig, RusanovSolver};
pub use adaptive::{
    create_accuracy_adaptive_solver, create_adaptive_solver, create_conservative_adaptive_solver,
    AdaptiveConfig, AdaptiveSolver, AdaptiveStats, SolverChoice,
};

