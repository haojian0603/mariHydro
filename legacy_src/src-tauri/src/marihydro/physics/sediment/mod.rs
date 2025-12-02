// src-tauri/src/marihydro/physics/sediment/mod.rs
//! 泥沙输运模块
//! 
//! 包含推移质输沙、悬移质输沙和床面演变

pub mod bed_evolution;
pub mod bed_load;
pub mod properties;

pub use bed_evolution::{BedEvolutionSolver, ExnerConfig};
pub use bed_load::{BedLoadFormula, BedLoadTransport, VanRijn, MeyerPeterMuller};
pub use properties::{SedimentClass, SedimentProperties};
