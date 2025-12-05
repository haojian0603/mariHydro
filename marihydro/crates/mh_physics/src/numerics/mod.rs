// crates/mh_physics/src/numerics/mod.rs

//! 数值方法模块
//!
//! 包含：
//! - gradient/ - 梯度计算 (Green-Gauss, Least-Squares)
//! - limiter/ - 梯度限制器 (Barth-Jespersen, Venkatakrishnan, Minmod)
//! - reconstruction/ - MUSCL 二阶重构

pub mod gradient;
pub mod limiter;
pub mod reconstruction;

pub use gradient::{
    GradientMethod,
    ScalarGradientStorage,
    VectorGradientStorage,
    GreenGaussGradient,
    GreenGaussConfig,
    FaceInterpolation,
    LeastSquaresGradient,
    LeastSquaresConfig,
};

pub use limiter::{
    SlopeLimiter,
    LimiterContext,
    NoLimiter,
    BarthJespersen,
    Venkatakrishnan,
    Minmod,
    create_limiter,
};

pub use reconstruction::{
    Reconstructor,
    ReconstructedState,
    MusclReconstructor,
    MusclConfig,
    GradientType,
};
