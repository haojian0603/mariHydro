//! 3D 湍流模型
//!
//! **注意**: k-ε 模型是 3D RANS 模型，
//! 不应直接用于 2D 浅水方程。2D 场景请使用 `shallow_2d::turbulence`。

// 重导出 3D 湍流模型
pub use super::k_epsilon;

// 便捷类型别名
pub use super::k_epsilon::{KEpsilonModel, KEpsilonParams};
