//! 鎺ㄧЩ璐ㄨ緭娌欏叕寮?
//!
//! 鏈ā鍧楅噸瀵煎嚭骞剁粺涓€鎺ㄧЩ璐ㄨ緭娌欏叕寮忔帴鍙ｃ€?
//! 浠?`sediment::formulas` 鍜?`sediment::bed_load_legacy` 鍚堝苟銆?

// 閲嶅鍑哄叕寮?trait 鍜屽疄鐜?
pub use super::super::formulas::{
    available_formulas, get_formula, EinsteinFormula, EngelundHansenFormula,
    MeyerPeterMullerFormula, TransportFormula, VanRijn1984Formula,
};

// 閲嶅鍑烘棫鐗?trait锛堝悜鍚庡吋瀹癸級
pub use super::super::bed_load_legacy::{
    BedLoadFormula, Einstein, MeyerPeterMuller, VanRijn,
};

/// 鎺ㄧЩ璐ㄥ叕寮忕被鍨嬪埆鍚嶏紙鍚戝悗鍏煎锛?
pub type MeyerPeterMullerAlias = MeyerPeterMullerFormula;
pub type VanRijnAlias = VanRijn1984Formula;
pub type EngelundHansen = EngelundHansenFormula;
