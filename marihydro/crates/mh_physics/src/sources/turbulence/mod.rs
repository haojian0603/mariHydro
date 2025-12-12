// crates/mh_physics/src/sources/turbulence/mod.rs

//! 湍流闭合模型
//!
//! 提供浅水方程的湍流建模：
//!
//! - [`smagorinsky`]: 2D 亚格子尺度模型（Smagorinsky, 1963）
//!
//! # 维度适用性
//!
//! | 模型 | 2D 浅水 | 说明 |
//! |------|---------|------|
//! | Smagorinsky | ✓ | 适用于深度平均湍流 |
//!
//! **注意**: k-ε 模型已移除，因为浅水方程是深度平均方程，
//! 直接应用 3D 湍流模型在物理上是不正确的。
//!
//! # 使用指南
//!
//! ## 2D 浅水模拟
//!
//! 对于 2D 浅水方程，湍流效应主要通过：
//! 1. **底部摩擦**（主导因素，见 `friction` 模块）
//! 2. **水平涡粘性**（Smagorinsky 或常数涡粘性）
//!
//! ```ignore
//! use mh_physics::sources::turbulence::{SmagorinskySolver, TurbulenceModel};
//!
//! // 推荐：常数涡粘性（0.1-10 m²/s）
//! let solver = SmagorinskySolver::new(n_cells, TurbulenceModel::constant(1.0));
//! ```
//!
//! # 物理说明
//!
//! 深度平均后的湍流效应已通过以下方式隐式处理：
//!
//! - 底部剪切应力（Manning/Chezy 摩擦）
//! - 水平涡粘性扩散（如需要）

mod smagorinsky;
pub mod traits;

// ==================== Smagorinsky (2D) ====================
pub use smagorinsky::{
    TurbulenceModel, TurbulenceConfig, SmagorinskySolver,
};

// ==================== 通用 trait ====================
pub use traits::{TurbulenceClosure, VelocityGradient};
