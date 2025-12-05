// crates/mh_physics/src/sediment/mod.rs

//! 泥沙输运模块
//!
//! 提供泥沙输运相关的物理模型，包括：
//! - 泥沙物理属性 (`properties`)
//! - 推移质输沙公式 (`bed_load`)
//! - 床面演变求解 (`bed_evolution`)

pub mod bed_evolution;
pub mod bed_load;
pub mod properties;

pub use bed_evolution::{BedEvolutionSolver, BedEvolutionStats, ExnerConfig};
pub use bed_load::{BedLoadFormula, BedLoadTransport, Einstein, MeyerPeterMuller, VanRijn};
pub use properties::{SedimentClass, SedimentProperties, SedimentType};
