// crates/mh_physics/src/schemes/wetting_drying/mod.rs

//! 干湿处理模块
//!
//! 提供浅水方程的干湿处理功能：
//! - 干湿状态判定
//! - 状态修正
//! - 通量限制
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/schemes/wetting_drying 迁移，保持算法不变。

mod handler;

pub use handler::{WetState, WettingDryingConfig, WettingDryingHandler};
