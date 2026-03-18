// crates/mh_runtime/src/numerics/mod.rs

//! 数值算法库（泛型实现）

pub mod kahan_sum;

pub use kahan_sum::KahanSum;