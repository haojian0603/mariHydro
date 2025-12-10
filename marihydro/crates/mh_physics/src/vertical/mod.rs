//! 垂向剖面模块
//!
//! 提供 2D→3D 扩展所需的垂向结构：
//! - σ坐标系统 (`sigma`)
//! - 垂向速度计算 (`velocity`)
//! - 垂向混合 (`mixing`)
//! - 分层状态 (`state`)
//! - 垂向剖面恢复 (`profile`) - 2.5D 扩展
//!
//! # 设计理念
//!
//! 本模块提供可选的垂向分层功能，允许浅水模型扩展为准3D模式。
//! 默认情况下，`ShallowWaterState::vertical` 为 `None`，保持2D模式。
//!
//! # σ坐标系统
//!
//! 使用地形跟随坐标 σ = (z - η) / h，其中：
//! - σ = 0 在水面
//! - σ = -1 在底床
//!
//! # 示例
//!
//! ```ignore
//! use mh_physics::vertical::{SigmaCoordinate, LayeredState};
//!
//! let sigma = SigmaCoordinate::uniform(10); // 10层均匀分布
//! let layered = LayeredState::new(n_cells, &sigma);
//! ```

pub mod sigma;
pub mod velocity;
pub mod mixing;
pub mod state;
pub mod profile;

pub use sigma::{SigmaCoordinate, SigmaDistribution};
pub use velocity::VerticalVelocity;
pub use mixing::{VerticalMixing, VerticalMixingModel};
pub use state::{LayeredScalar, LayeredState};
pub use profile::{ProfileRestorer, VerticalProfile, ProfileMethod};
