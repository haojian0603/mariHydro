// crates/mh_physics/src/numerics/mod.rs

//! 数值方法模块
//!
//! 包含：
//! - gradient/ - 梯度计算 (Green-Gauss, Least-Squares)
//! - limiter/ - 梯度限制器 (Barth-Jespersen, Venkatakrishnan, Minmod)
//! - reconstruction/ - MUSCL 二阶重构
//! - linear_algebra/ - 稀疏线性代数 (CSR, PCG, BiCGStab)
//! - discretization/ - 有限体积离散化 (拓扑, 组装, 回代)
//! - operators/ - 数值算子 (扩散等)

pub mod discretization;
pub mod gradient;
pub mod limiter;
pub mod linear_algebra;
pub mod operators;
pub mod reconstruction;

pub use gradient::{
    FaceInterpolation, GradientMethod, GreenGaussConfig, GreenGaussGradient, LeastSquaresConfig,
    LeastSquaresGradient, ScalarGradientStorage, VectorGradientStorage,
};

pub use limiter::{
    create_limiter, BarthJespersen, LimiterContext, Minmod, NoLimiter, SlopeLimiter,
    Venkatakrishnan,
};

pub use reconstruction::{GradientType, MusclConfig, MusclReconstructor, ReconstructedState, Reconstructor};

// 稀疏线性代数
pub use linear_algebra::{
    // CSR 矩阵
    CsrBuilder,
    CsrMatrix,
    CsrPattern,
    // 向量运算
    axpy,
    copy,
    dot,
    fill,
    norm2,
    scale,
    xpay,
    // 预条件器
    IdentityPreconditioner,
    JacobiPreconditioner,
    Preconditioner,
    // 求解器
    BiCgStabSolver,
    ConjugateGradient,
    IterativeSolver,
    PcgSolver,
    SolverConfig,
    SolverResult,
    SolverStatus,
};

// 离散化
pub use discretization::{
    // 拓扑
    CellFaceTopology,
    FaceInfo,
    NeighborInfo,
    // 组装器
    AssemblerConfig,
    ImplicitMomentumAssembler,
    PressureMatrixAssembler,
    // 回代
    DepthCorrector,
    VelocityCorrector,
};
