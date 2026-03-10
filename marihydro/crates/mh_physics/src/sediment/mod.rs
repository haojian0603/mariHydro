// crates/mh_physics/src/sediment/mod.rs

//! 娉ユ矙杈撹繍妯″潡
//!
//! 鎻愪緵娉ユ矙杈撹繍鐩稿叧鐨勭墿鐞嗘ā鍨嬶紝鍖呮嫭锛?
//! - 娉ユ矙鐗╃悊灞炴€?(`properties`)
//! - 鎺ㄧЩ璐ㄨ緭娌?(`bed_load`)
//! - 鎮Щ璐ㄨ緭娌?(`suspended`)
//! - 搴婇潰婕斿彉 (`morphology`)
//! - 2.5D 娉ユ矙杈撹繍 (`transport_2_5d`) - 鐙珛鎵╁睍
//! - 娉ユ矙绯荤粺绠＄悊鍣?(`manager`) - 缁熶竴绠＄悊
//!
//! # 妯″潡缁撴瀯
//!
//! - `properties`: 娉ユ矙鐗╃悊灞炴€э紙绮掑緞銆佸瘑搴︺€佹矇闄嶉€熷害绛夛級
//! - `bed_load`: 鎺ㄧЩ璐ㄨ緭娌欏叕寮忓拰璁＄畻鍣?
//! - `suspended`: 鎮Щ璐ㄨ緭娌欙紙娌夐檷銆佸啀鎮诞銆佽緭杩愶級
//! - `formulas`: 杈撴矙鍏紡搴擄紙MPM, Van Rijn, Einstein, Engelund-Hansen锛?
//! - `morphology`: 娌冲簥婕斿彉姹傝В鍣紙Exner 鏂圭▼锛?
//! - `transport_2_5d`: 2.5D 鍨傚悜鍒嗗眰娉ユ矙杈撹繍
//! - `manager`: 娉ユ矙绯荤粺缁熶竴绠＄悊鍣?

// 鏃х増妯″潡锛堜繚鐣欏悜鍚庡吋瀹癸級
#[path = "bed_load.rs"]
mod bed_load_legacy;
pub mod formulas;
pub mod manager;
pub mod morphology;
pub mod properties;
pub mod transport_2_5d;
pub mod exchange;

// 鏂扮増瀛愭ā鍧?
#[path = "bed_load/mod.rs"]
pub mod bed_load_new;
pub mod suspended;

// 浼犵粺瀵煎嚭锛堝悜鍚庡吋瀹癸級
pub use bed_load_legacy::{BedLoadFormula, BedLoadTransport, Einstein, MeyerPeterMuller, VanRijn};
pub use properties::{SedimentClass, SedimentProperties, SedimentType};

// 鎺ㄨ崘瀵煎嚭
pub use formulas::{
    EinsteinFormula, EngelundHansenFormula, MeyerPeterMullerFormula, TransportFormula,
    VanRijn1984Formula,
};
pub use morphology::{MorphodynamicsSolver, MorphologyConfig, MorphologyStats};

// 鎮Щ璐ㄥ鍑?
pub use suspended::{
    ErosionFormula, GarciaParker, ResuspensionSource, SettlingFormula, SettlingVelocity,
    SmithMcLean, SuspendedTransport,
};
pub use suspended::{DietrichSettling, StokesSettling, VanRijnSettling};

// 娉ユ矙绠＄悊鍣ㄥ鍑?
pub use manager::{
    SedimentManagerGeneric, SedimentStateGeneric, SedimentConfigGeneric,
    SedimentError, SedimentFluxStats,
};
pub use exchange::{SedimentExchange, ExchangeParams};
