// crates/mh_physics/src/sources/mod.rs

//! 婧愰」妯″潡
//!
//! 鎻愪緵娴呮按鏂圭▼鍜屼笁缁存ā鍨嬬殑鍚勭鐗╃悊婧愰」锛?
//!
//! # 閫氱敤婧愰」锛?D/3D锛?
//!
//! - 鎽╂摝婧愰」 ([`friction`]): Manning, Chezy 搴曞簥鎽╂摝
//! - 绉戞皬鍔?([`coriolis`]): 鍦扮悆鑷浆鏁堝簲
//! - 鍏ユ祦/鍑烘祦 ([`inflow`]): 娌虫祦鍏ユ祦銆侀檷闆ㄣ€佽捀鍙?
//! - 闅愬紡澶勭悊 ([`implicit`]): 鍒氭€ф簮椤圭殑闅愬紡鏃堕棿绉垎
//!
//! # 2D 涓撶敤婧愰」
//!
//! - 澶ф皵寮鸿揩 ([`atmosphere`]): 椋庡簲鍔涖€佹皵鍘嬫搴?
//! - 妞嶈闃诲姏 ([`vegetation`]): 鍒氭€?鏌旀€ф琚?
//! - 娉㈡氮椹卞姩 ([`wave_forcing`]): 杈愬皠搴斿姏姊害
//!
//! # 婀嶆祦妯″瀷 ([`turbulence`])
//!
//! - Smagorinsky 浜氭牸瀛愭ā鍨嬶紙2D锛?
//! - k-蔚 妯″瀷锛?D锛?
//!
//! # 姘村伐缁撴瀯 ([`structures`])
//!
//! - 妗ュⅸ銆佸牥绛変簹缃戞牸缁撴瀯
//!
//! # 妯″潡缁撴瀯 (v0.4+)
//!
//! ```text
//! sources/
//! 鈹溾攢鈹€ traits.rs           # SourceTerm trait 瀹氫箟
//! 鈹溾攢鈹€ friction.rs         # 鎽╂摝婧愰」
//! 鈹溾攢鈹€ coriolis.rs         # 绉戞皬鍔?
//! 鈹溾攢鈹€ implicit.rs         # 闅愬紡澶勭悊
//! 鈹溾攢鈹€ inflow.rs           # 鍏ユ祦/鍑烘祦
//! 鈹溾攢鈹€ atmosphere.rs       # 澶ф皵寮鸿揩锛?D锛?
//! 鈹溾攢鈹€ vegetation.rs       # 妞嶈闃诲姏锛?D锛?
//! 鈹溾攢鈹€ wave_forcing.rs     # 娉㈡氮椹卞姩锛?D锛?
//! 鈹溾攢鈹€ turbulence/         # 婀嶆祦妯″瀷瀛愭ā鍧?
//! 鈹?  鈹溾攢鈹€ smagorinsky.rs  # 2D Smagorinsky
//! 鈹?  鈹斺攢鈹€ k_epsilon.rs    # 3D k-蔚
//! 鈹斺攢鈹€ structures/         # 姘村伐缁撴瀯
//!     鈹溾攢鈹€ bridge_pier.rs
//!     鈹斺攢鈹€ weir.rs
//! ```
//!
//! # 璁捐
//!
//! 鎵€鏈夋簮椤瑰疄鐜?[`SourceTerm`] trait锛屾彁渚涚粺涓€鐨勮绠楁帴鍙ｏ細
//! - `compute_cell()` - 璁＄畻鍗曚釜鍗曞厓鐨勬簮椤硅础鐚?
//! - `compute_all()` - 鎵归噺璁＄畻鎵€鏈夊崟鍏?
//!
//! # 浣跨敤绀轰緥
//!
//! ```ignore
//! use mh_physics::sources::{ManningFrictionConfig, CoriolisSource};
//! use mh_physics::sources::turbulence::{SmagorinskySolver, TurbulenceModel};
//!
//! // 鍒涘缓 Manning 鎽╂摝
//! let friction = ManningFrictionConfig::new(9.81, n_cells, 0.025);
//!
//! // 鍒涘缓绉戞皬鍔涳紙鍖楃含 30 搴︼級
//! let coriolis = CoriolisSource::from_latitude(30.0);
//!
//! // 鍒涘缓 Smagorinsky 婀嶆祦妯″瀷锛堟帹鑽愪娇鐢ㄥ父鏁版丁绮樻€э級
//! let turb = SmagorinskySolver::new(n_cells, TurbulenceModel::constant(1.0));
//! ```

// ==================== 鏍稿績 trait ====================
pub mod traits;
pub mod registry;

// ==================== 閫氱敤婧愰」 ====================
pub mod friction;
pub mod coriolis;
pub mod implicit;
pub mod inflow;

// ==================== 2D 涓撶敤婧愰」 ====================
pub mod atmosphere;
pub mod vegetation;
pub mod wave_forcing;

// ==================== 婀嶆祦妯″瀷锛堢嫭绔嬪瓙妯″潡锛?====================
pub mod turbulence;

// ==================== 姘村伐缁撴瀯 ====================
pub mod structures;

// ==================== 鏍稿績 trait 瀵煎嚭 ====================
pub use traits::{
    SourceContribution, SourceContext, SourceTerm, SourceHelpers,
    SourceContributionGeneric, SourceContextGeneric, SourceTermGeneric,
    SourceStiffness,
};


pub use registry::SourceRegistry;

// ==================== 鎽╂摝妯″潡瀵煎嚭 ====================
pub use friction::{
    ManningFriction, ManningFrictionConfig,
    ChezyFriction, ChezyFrictionConfig,
    FrictionCalculator,
    ManningFrictionGeneric, ManningFrictionConfigGeneric,
    ChezyFrictionGeneric, ChezyFrictionConfigGeneric,
};

// ==================== 绉戞皬鍔涘鍑?====================
pub use coriolis::{
    CoriolisConfig, CoriolisSource, EARTH_ANGULAR_VELOCITY,
};

// ==================== 闅愬紡澶勭悊瀵煎嚭 ====================
pub use implicit::{
    ImplicitMethod, ImplicitConfig, ImplicitMomentumDecay,
    DampingCoefficient, ManningDamping, ChezyDamping,
};

// ==================== 澶ф皵婧愰」瀵煎嚭 ====================
pub use atmosphere::{
    WindStressConfig, PressureGradientConfig, WindStressSource, PressureGradientSource,
    DragCoefficientMethod,
    wind_drag_coefficient_lp81, wind_drag_coefficient_wu82,
};

// ==================== 婀嶆祦妯″瀷瀵煎嚭 ====================
pub use turbulence::{
    // Smagorinsky (2D)
    TurbulenceModel, TurbulenceConfig, SmagorinskySolver,
    VelocityGradient,
    DEFAULT_SMAGORINSKY_CONSTANT, MIN_EDDY_VISCOSITY, MAX_EDDY_VISCOSITY,
    // 閫氱敤 trait
    TurbulenceClosure,
};

// ==================== 妞嶈闃诲姏瀵煎嚭 ====================
pub use vegetation::{
    VegetationType, VegetationConfig, VegetationSource,
    VegetationImplicit,
};

// ==================== 鍏ユ祦婧愰」瀵煎嚭 ====================
pub use inflow::{
    InflowType, InflowConfig, InflowSource,
    RainfallConfig, RainfallSource,
    EvaporationConfig, EvaporationSource,
};

// ==================== 娉㈡氮椹卞姩婧愰」瀵煎嚭 ====================
pub use wave_forcing::{WaveForcing, WaveForcingConfig};

// ==================== 缁撴瀯鐗╂簮椤瑰鍑?====================
pub use structures::{BridgePierDrag, WeirFlow, WeirType};

// ==================== 鎵╂暎绠楀瓙锛堜粠 numerics 閲嶅鍑猴紝淇濇寔鍏煎锛?====================
/// 鎵╂暎鐩稿叧绫诲瀷锛堝凡杩佺Щ鑷?`numerics::operators::diffusion`锛?
///
/// 涓轰繚鎸佸悜鍚庡吋瀹癸紝浠?numerics 妯″潡閲嶅鍑恒€?
/// 鏂颁唬鐮佸缓璁洿鎺ヤ娇鐢?`mh_physics::numerics::operators::diffusion`銆?
pub use crate::numerics::operators::diffusion::{
    DiffusionBC, DiffusionConfig, DiffusionSolver, DiffusionError,
    VariableDiffusionSolver,
    estimate_stable_dt, required_substeps,
};
