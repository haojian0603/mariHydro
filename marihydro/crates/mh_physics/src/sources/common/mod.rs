//! 通用源项模块
//!
//! 2D/3D 共用的源项，从根模块重导出

// 重导出通用源项
pub use super::friction;
pub use super::coriolis;
pub use super::inflow;
pub use super::implicit;
pub use super::structures;

// 便捷类型别名
pub use super::friction::{ManningFrictionConfig, ChezyFrictionConfig, FrictionCalculator};
pub use super::coriolis::{CoriolisConfig, CoriolisSource, EARTH_ANGULAR_VELOCITY};
pub use super::inflow::{InflowConfig, InflowType};
pub use super::implicit::{ImplicitMethod, ImplicitConfig, ImplicitMomentumDecay, DampingCoefficient};
pub use super::structures::{BridgePierDrag, WeirFlow};
