// crates/mh_physics/src/sediment/mod.rs

//! 泥沙输运模块
//!
//! 提供泥沙输运相关的物理模型，包括：
//! - 泥沙物理属性 (`properties`)
//! - 推移质输沙公式库 (`formulas`)
//! - 悬移质输沙 (`suspended`)
//! - 床面演变 (`morphology`)
//! - 2.5D 泥沙输运 (`transport_2_5d`)
//! - 泥沙系统管理器 (`manager`)
//!
//! # 设计原则
//!
//! 1. **单轨泛型**: 所有接口基于 `RuntimeScalar` 泛型，无 Legacy f64 别名
//! 2. **Backend 感知**: 输运计算器使用 Backend 抽象支持 CPU/GPU
//! 3. **强类型索引**: 所有单元/面索引使用 `CellIndex`/`FaceIndex`
//!
//! # 模块结构
//!
//! - `properties`: 泥沙物理属性（粒径、密度、沉降速度等）
//! - `formulas`: 输沙公式库（MPM, Van Rijn, Einstein, Engelund-Hansen）
//! - `suspended`: 悬移质输沙（沉降、再悬浮、输运）
//! - `morphology`: 河床演变求解器（Exner 方程）
//! - `transport_2_5d`: 2.5D 垂向分层泥沙输运
//! - `manager`: 泥沙系统统一管理器
//! - `exchange`: 泥沙交换
//! - `shear_stress`: 剪切应力计算

pub mod formulas;
pub mod manager;
pub mod morphology;
pub mod properties;
pub mod transport_2_5d;
pub mod exchange;
pub mod shear_stress;
pub mod suspended;

// ============================================================
// 泛型接口导出（推荐使用）
// ============================================================

// 泥沙属性
pub use properties::{SedimentClass, SedimentProperties, SedimentType};

// 输沙公式（全部泛型化）
pub use formulas::{
    EinsteinFormula, EngelundHansenFormula, MeyerPeterMullerFormula, TransportFormula,
    VanRijn1984Formula, available_formulas, get_formula_f64, get_formula_f32,
};

// 形态动力学
pub use morphology::{MorphodynamicsSolver, MorphologyConfig, MorphologyStats};

// 悬移质
pub use suspended::{
    ErosionFormula, GarciaParker, ResuspensionSource, SettlingFormula, SettlingVelocity,
    SmithMcLean, SuspendedTransport,
};
pub use suspended::{DietrichSettling, StokesSettling, VanRijnSettling};

// 泥沙管理器
pub use manager::{
    SedimentManagerGeneric, SedimentStateGeneric, SedimentConfigGeneric,
    SedimentError, SedimentFluxStats,
};

// 剪切应力
pub use shear_stress::{
    ShearStress, ShearStressCalculator, ManningCoeff,
    shields_parameter, critical_shields,
};

// 交换
pub use exchange::{SedimentExchange, ExchangeParams};
