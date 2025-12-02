// src-tauri/src/marihydro/physics/numerics/gradient/mod.rs

//! 梯度计算模块

pub mod green_gauss;
pub mod least_squares;
pub mod traits;

pub use green_gauss::GreenGaussGradient;
pub use least_squares::LeastSquaresGradient;
pub use traits::{
    FieldValue, GradientContribution, GradientMethod, ScalarGradientStorage, VectorGradientStorage,
};
