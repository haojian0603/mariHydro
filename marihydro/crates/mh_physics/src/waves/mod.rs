// crates/mh_physics/src/waves/mod.rs

//! 波浪模块
//!
//! 提供波浪相关的物理模型，包括：
//! - 辐射应力计算 (`radiation_stress`)
//! - 波浪底摩擦 (`bottom_friction`)

pub mod bottom_friction;
pub mod radiation_stress;

pub use bottom_friction::{
    WaveBottomFriction, WaveBottomFrictionConfig, WaveBottomFrictionModel,
    WaveCurrentInteraction, WaveOrbitalVelocity,
};
pub use radiation_stress::{
    compute_wavenumber_and_n, RadiationStressCalculator, RadiationStressTensor,
    WaveField, WaveParameters, WaveSource,
};
