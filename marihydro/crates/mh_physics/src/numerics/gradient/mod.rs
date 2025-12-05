// crates/mh_physics/src/numerics/gradient/mod.rs

//! 梯度计算模块
//!
//! 提供多种梯度计算方法：
//! - Green-Gauss 梯度 (面积分法)
//! - 最小二乘梯度 (带 SVD 回退)
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/numerics/gradient 迁移，保持算法不变。
//! 适配 PhysicsMesh 接口。

mod traits;
mod green_gauss;
mod least_squares;

pub use traits::*;
pub use green_gauss::*;
pub use least_squares::*;
