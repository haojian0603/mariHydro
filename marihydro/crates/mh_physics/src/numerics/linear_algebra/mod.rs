// crates/mh_physics/src/numerics/linear_algebra/mod.rs

//! 稀疏线性代数模块
//!
//! 提供隐式求解所需的稀疏矩阵和迭代求解器：
//!
//! # 子模块
//!
//! - [`csr`]: 压缩稀疏行（CSR）矩阵格式
//! - [`vector_ops`]: 向量运算（BLAS Level 1）
//! - [`preconditioner`]: 预条件器（Jacobi, ILU0）
//! - [`solver`]: 迭代求解器（CG, PCG, BiCGStab）
//!
//! # 主要类型
//!
//! ## 矩阵类型
//!
//! - [`CsrMatrix`]: CSR 格式稀疏矩阵
//! - [`CsrBuilder`]: 矩阵构建器
//!
//! ## 向量运算
//!
//! - [`dot`]: 点积
//! - [`axpy`]: y = a*x + y
//! - [`norm2`]: 二范数
//!
//! ## 预条件器
//!
//! - [`Preconditioner`]: 预条件器 trait
//! - [`JacobiPreconditioner`]: Jacobi 预条件器
//! - [`IdentityPreconditioner`]: 恒等预条件器
//!
//! ## 求解器
//!
//! - [`IterativeSolver`]: 迭代求解器 trait
//! - [`ConjugateGradient`]: 共轭梯度法
//! - [`PcgSolver`]: 预条件共轭梯度法
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::{
//!     CsrMatrix, CsrBuilder, PcgSolver, JacobiPreconditioner,
//! };
//!
//! // 构建矩阵
//! let mut builder = CsrBuilder::new(n_cells);
//! for i in 0..n_cells {
//!     builder.set(i, i, diag[i]);
//!     for &j in neighbors[i].iter() {
//!         builder.set(i, j, off_diag[i][j]);
//!     }
//! }
//! let matrix = builder.build();
//!
//! // 求解 Ax = b
//! let precond = JacobiPreconditioner::from_matrix(&matrix);
//! let mut solver = PcgSolver::new(1e-8, 100);
//! let result = solver.solve(&matrix, &b, &mut x, &precond);
//! ```
//!
//! # 设计原则
//!
//! 1. **性能优先**：使用 AlignedVec 和 SIMD 友好的数据布局
//! 2. **内存效率**：CSR 格式最小化存储
//! 3. **可扩展性**：通过 trait 支持多种预条件器和求解器
//! 4. **数值稳定**：残差监控和收敛判断

pub mod csr;
pub mod preconditioner;
pub mod solver;
pub mod vector_ops;

// CSR 矩阵
pub use csr::{CsrBuilder, CsrMatrix, CsrPattern};

// 向量运算
pub use vector_ops::{axpy, copy, dot, fill, norm2, scale, xpay};

// 预条件器
pub use preconditioner::{IdentityPreconditioner, JacobiPreconditioner, Preconditioner};

// 求解器
pub use solver::{
    BiCgStabSolver, ConjugateGradient, IterativeSolver, PcgSolver, SolverConfig, SolverResult,
    SolverStatus,
};
