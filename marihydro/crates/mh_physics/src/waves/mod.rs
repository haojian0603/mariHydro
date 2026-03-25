// crates/mh_physics/src/waves/mod.rs

//! 波浪模块
//!
//! 提供波浪相关的物理模型，包括：
//! - 辐射应力计算 (`radiation_stress`)
//! - 波浪底摩擦 (`bottom_friction`)
//! - 谱波诊断与显式谱构造 (`spectral`)
//!
//! `spectral` 当前保留两类真实能力：
//! - JONSWAP 频率谱构造
//! - 显式余弦幂方向散布
//!
//! 它不宣称实现了完整的方向 JONSWAP 闭合或成熟谱波模式。

pub mod bottom_friction;
pub mod radiation_stress;
pub mod spectral;

pub use bottom_friction::{
    WaveBottomFriction, WaveBottomFrictionConfig, WaveBottomFrictionModel,
    WaveCurrentInteraction, WaveOrbitalVelocity,
};
pub use radiation_stress::{
    compute_wavenumber_and_n, RadiationStressCalculatorGeneric,
    RadiationStressTensorGeneric, WaveFieldGeneric, WaveFieldSnapshot,
    WaveParametersGeneric, WaveSourceGeneric,
};
pub use spectral::{SpectralConfig, SpectralWaveSolver, WaveSpectrum, WaveFieldParams};
