// crates/mh_physics/src/schemes/riemann/mod.rs

//! 黎曼求解器模块
//!
//! 提供浅水方程的近似黎曼求解器：
//!
//! - [`HllcSolver`]: 高精度求解器，正确处理接触间断
//! - [`RusanovSolver`]: 简单鲁棒的求解器，GPU 友好
//! - [`AdaptiveSolver`]: 自适应求解器，自动选择最优方法
//! - [`HlleSolver`]: HLLE 求解器，强间断更稳定
//! - [`RoeSolver`]: Roe 求解器，接触间断分辨率高
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

use mh_runtime::Backend;

mod adaptive;
mod central;
mod hllc;
mod hlle;
mod roe;
mod rusanov;
mod traits;
pub mod batch;

// 核心类型（泛型化）
pub use traits::{
    RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams,
};

// 中心差分求解器
pub use central::CentralSolver;

// HLLC 求解器
pub use hllc::HllcSolver;

// HLLE 求解器
pub use hlle::{create_hlle_solver, HlleSolver};

// Roe 求解器
pub use roe::{create_roe_solver, RoeSolver};

// Rusanov 求解器
pub use rusanov::{
    create_robust_rusanov_solver, create_rusanov_solver, 
    RusanovConfig, RusanovSolver,
};

// 自适应求解器
pub use adaptive::{
    create_adaptive_solver, create_conservative_adaptive_solver,
    AdaptiveConfig, AdaptiveSolver,
    AdaptiveStats, SolverChoice,
};

// 批量求解器接口
pub use batch::{
    BatchRiemannSolver, BatchCellStates, BatchNormals, BatchFluxes,
};

/// 黎曼求解器枚举（静态分发）
#[derive(Clone)]
pub enum RiemannSolverAny<B: Backend> {
    Hllc(HllcSolver<B>),
    Roe(RoeSolver<B>),
    Rusanov(RusanovSolver<B>),
    Central(CentralSolver<B>),
}

impl<B: Backend> RiemannSolver for RiemannSolverAny<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

    fn name(&self) -> &'static str {
        match self {
            Self::Hllc(solver) => solver.name(),
            Self::Roe(solver) => solver.name(),
            Self::Rusanov(solver) => solver.name(),
            Self::Central(solver) => solver.name(),
        }
    }

    fn capabilities(&self) -> SolverCapabilities {
        match self {
            Self::Hllc(solver) => solver.capabilities(),
            Self::Roe(solver) => solver.capabilities(),
            Self::Rusanov(solver) => solver.capabilities(),
            Self::Central(solver) => solver.capabilities(),
        }
    }

    fn solve(
        &self,
        h_left: Self::Scalar,
        h_right: Self::Scalar,
        vel_left: Self::Vector2D,
        vel_right: Self::Vector2D,
        normal: Self::Vector2D,
    ) -> Result<RiemannFlux<Self::Scalar>, RiemannError> {
        match self {
            Self::Hllc(solver) => solver.solve(h_left, h_right, vel_left, vel_right, normal),
            Self::Roe(solver) => solver.solve(h_left, h_right, vel_left, vel_right, normal),
            Self::Rusanov(solver) => solver.solve(h_left, h_right, vel_left, vel_right, normal),
            Self::Central(solver) => solver.solve(h_left, h_right, vel_left, vel_right, normal),
        }
    }

    fn gravity(&self) -> Self::Scalar {
        match self {
            Self::Hllc(solver) => solver.gravity(),
            Self::Roe(solver) => solver.gravity(),
            Self::Rusanov(solver) => solver.gravity(),
            Self::Central(solver) => solver.gravity(),
        }
    }

    fn dry_threshold(&self) -> Self::Scalar {
        match self {
            Self::Hllc(solver) => solver.dry_threshold(),
            Self::Roe(solver) => solver.dry_threshold(),
            Self::Rusanov(solver) => solver.dry_threshold(),
            Self::Central(solver) => solver.dry_threshold(),
        }
    }
}