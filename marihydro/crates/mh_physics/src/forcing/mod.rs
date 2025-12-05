// crates/mh_physics/src/forcing/mod.rs

//! 外力模块
//!
//! 提供时变外力数据的提供者和管理器：
//! - 风场提供者 (WindProvider)
//! - 潮汐提供者 (TideProvider)
//! - 河流入流提供者 (RiverProvider)
//!
//! # 设计思路
//!
//! 所有数据提供者遵循统一模式：
//! 1. 支持恒定值、时间序列、周期性数据
//! 2. 提供时间插值
//! 3. 缓存最近计算值以提高性能
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::forcing::{WindProvider, TideProvider, RiverProvider};
//!
//! // 创建风场
//! let wind = WindProvider::constant(10.0, 225.0); // 西南风 10m/s
//!
//! // 创建潮汐
//! let tide = TideProvider::semidiurnal(0.0, 1.5, 0.0, 0.5, 30.0);
//!
//! // 创建河流入流
//! let river = RiverProvider::flood_wave(50.0, 500.0, 3600.0, 7200.0);
//! ```

pub mod wind;
pub mod tide;
pub mod river;

// 风场导出
pub use wind::{WindProvider, WindData, SpatialWindProvider};

// 潮汐导出
pub use tide::{
    TideProvider, TideData, TideBoundary,
    TidalConstituent, tidal_constituents,
};

// 河流导出
pub use river::{RiverProvider, RiverData, RiverSystem, RiverEntry};
