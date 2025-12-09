//! 2D 浅水方程专用源项
//!
//! 仅适用于 2D 浅水方程，从根模块重导出

// 重导出 2D 专用源项
pub use super::atmosphere;
pub use super::diffusion;
pub use super::vegetation;
pub use super::wave_forcing;
pub use super::turbulence;

// 便捷类型别名
pub use super::atmosphere::{
    WindStressConfig, PressureGradientConfig, DragCoefficientMethod,
    wind_drag_coefficient_lp81, wind_drag_coefficient_wu82,
};
pub use super::diffusion::{DiffusionSolver, DiffusionConfig, DiffusionBC};
pub use super::vegetation::{VegetationConfig, VegetationType};
pub use super::wave_forcing::{WaveForcing, WaveForcingConfig};
pub use super::turbulence::{TurbulenceConfig, TurbulenceModel, VelocityGradient};
