// src-tauri/src/marihydro/physics/waves/mod.rs
//! 波浪模块
//!
//! 实现波浪-水流耦合的辐射应力计算，用于近岸水动力模拟。

pub mod bottom_friction;
pub mod radiation_stress;

pub use bottom_friction::{
    WaveBottomFriction, WaveBottomFrictionModel, WaveOrbitalVelocity,
};
pub use radiation_stress::{
    RadiationStress, WaveField, WaveParameters, WaveSource,
};
