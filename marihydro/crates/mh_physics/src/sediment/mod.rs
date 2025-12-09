// crates/mh_physics/src/sediment/mod.rs

//! 泥沙输运模块
//!
//! 提供泥沙输运相关的物理模型，包括：
//! - 泥沙物理属性 (`properties`)
//! - 推移质输沙 (`bed_load`)
//! - 悬移质输沙 (`suspended`)
//! - 床面演变 (`morphology`) - **推荐**
//!
//! # 模块结构
//!
//! - `properties`: 泥沙物理属性（粒径、密度、沉降速度等）
//! - `bed_load`: 推移质输沙公式和计算器
//! - `suspended`: 悬移质输沙（沉降、再悬浮、输运）
//! - `formulas`: 输沙公式库（MPM, Van Rijn, Einstein, Engelund-Hansen）
//! - `morphology`: 河床演变求解器（Exner 方程）- **推荐使用**
//! - `bed_evolution`: 旧版床面演变（已废弃）

// 旧版模块（保留向后兼容）
#[path = "bed_load.rs"]
mod bed_load_legacy;
#[deprecated(since = "0.3.0", note = "使用 morphology 模块替代")]
pub mod bed_evolution;
pub mod formulas;
pub mod morphology;
pub mod properties;

// 新版子模块
#[path = "bed_load/mod.rs"]
pub mod bed_load_new;
pub mod suspended;

// 传统导出（向后兼容，已废弃）
#[deprecated(since = "0.3.0", note = "使用 MorphodynamicsSolver 和 MorphologyConfig 替代")]
pub use bed_evolution::{BedEvolutionSolver, BedEvolutionStats, ExnerConfig};
pub use bed_load_legacy::{BedLoadFormula, BedLoadTransport, Einstein, MeyerPeterMuller, VanRijn};
pub use properties::{SedimentClass, SedimentProperties, SedimentType};

// 推荐导出
pub use formulas::{
    EinsteinFormula, EngelundHansenFormula, MeyerPeterMullerFormula, TransportFormula,
    VanRijn1984Formula,
};
pub use morphology::{MorphodynamicsSolver, MorphologyConfig, MorphologyStats};

// 悬移质导出
pub use suspended::{
    ErosionFormula, GarciaParker, ResuspensionSource, SettlingFormula, SettlingVelocity,
    SmithMcLean, SuspendedTransport,
};
pub use suspended::{DietrichSettling, StokesSettling, VanRijnSettling};

