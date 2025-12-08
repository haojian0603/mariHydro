// crates/mh_physics/src/sediment/mod.rs

//! 泥沙输运模块
//!
//! 提供泥沙输运相关的物理模型，包括：
//! - 泥沙物理属性 (`properties`)
//! - 推移质输沙公式 (`bed_load`, `formulas`)
//! - 床面演变求解 (`bed_evolution`, `morphology`)
//!
//! # 模块结构
//!
//! - `properties`: 泥沙物理属性（粒径、密度、沉降速度等）
//! - `bed_load`: 传统推移质输沙计算
//! - `formulas`: 基于 trait 的推移质公式库（MPM, Van Rijn, Einstein, Engelund-Hansen）
//! - `bed_evolution`: 传统床面演变求解器
//! - `morphology`: 新版河床演变求解器（带崩塌处理）

pub mod bed_evolution;
pub mod bed_load;
pub mod formulas;
pub mod morphology;
pub mod properties;

// 传统导出（向后兼容）
pub use bed_evolution::{BedEvolutionSolver, BedEvolutionStats, ExnerConfig};
pub use bed_load::{BedLoadFormula, BedLoadTransport, Einstein, MeyerPeterMuller, VanRijn};
pub use properties::{SedimentClass, SedimentProperties, SedimentType};

// 新版导出
pub use formulas::{
    EinsteinFormula, EngelundHansenFormula, MeyerPeterMullerFormula, TransportFormula,
    VanRijn1984Formula,
};
pub use morphology::{MorphodynamicsSolver, MorphologyConfig, MorphologyStats};
