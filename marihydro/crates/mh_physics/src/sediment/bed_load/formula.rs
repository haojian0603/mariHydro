//! 推移质输沙公式
//!
//! 本模块重导出并统一推移质输沙公式接口。
//! 从 `sediment::formulas` 和 `sediment::bed_load_legacy` 合并。

// 重导出公式 trait 和实现
pub use super::super::formulas::{
    available_formulas, get_formula_f64, get_formula_f32, EinsteinFormula, EngelundHansenFormula,
    MeyerPeterMullerFormula, TransportFormula, VanRijn1984Formula,
};

// 重导出旧版 trait（向后兼容）
pub use super::super::bed_load_legacy::{
    BedLoadFormula, Einstein, MeyerPeterMuller, VanRijn,
};

/// 推移质公式类型别名（向后兼容）
pub type MeyerPeterMullerAlias = MeyerPeterMullerFormula<f64>;
pub type VanRijnAlias = VanRijn1984Formula<f64>;
pub type EngelundHansen = EngelundHansenFormula<f64>;
