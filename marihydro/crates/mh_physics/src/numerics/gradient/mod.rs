// crates/mh_physics/src/numerics/gradient/mod.rs

//! 姊害璁＄畻妯″潡
//!
//! 鎻愪緵澶氱姊害璁＄畻鏂规硶锛?
//! - Green-Gauss 姊害 (闈㈢Н鍒嗘硶)
//! - 鏈€灏忎簩涔樻搴?(甯?SVD 鍥為€€)
//!
//! # 杩佺Щ璇存槑
//!
//! 浠?history_src/physics/numerics/gradient 杩佺Щ锛屼繚鎸佺畻娉曚笉鍙樸€?
//! 閫傞厤 PhysicsMesh 鎺ュ彛銆?

mod traits;
mod green_gauss;
mod least_squares;

pub use traits::*;
pub use green_gauss::*;
pub use least_squares::*;
