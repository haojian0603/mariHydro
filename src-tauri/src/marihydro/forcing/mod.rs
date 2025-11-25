//! 强迫场管理 (Forcing Management)
//!
//! 本模块负责外部驱动项的加载、插值与时序更新，包括：
//! - 表面风应力 (Surface Wind Stress)
//! - 气压梯度 (Atmospheric Pressure Gradient)
//! - 河流流量 (River Discharge) - 稀疏存储
//! - 天文潮 (Astronomical Tide)

pub mod context;
pub mod manager;
pub mod river;
pub mod tide;
pub mod wind;
