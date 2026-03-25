// crates/mh_physics/src/legacy_limiters/mod.rs
//
// Legacy note:
// - The main engine path uses `crate::numerics::limiter` and
//   `crate::numerics::reconstruction`.
// - This module remains only as a scalar compatibility shim for older callers.
// - New engine work should not add fresh dependencies on this module.
// - `crate::types::LimiterType` is the configuration entry, and
//   `crate::legacy_limiters::{LimiterType, MusclConfig, MusclReconstructor}`
//   is the only supported compatibility namespace for old scalar callers.
// - The compatibility entity is now physically isolated in this directory.

//! 斜率限制器与重构方法
//!
//! 提供高阶空间重构所需的限制器，支持：
//! - 经典限制器（Minmod, Superbee, Van Leer）
//! - Venkatakrishnan 限制器（非结构网格）
//! - Barth-Jespersen 限制器
//! - MUSCL 重构
//!
//! # 设计原则
//!
//! 1. **TVD 保持**：确保总变差不增
//! 2. **高阶精度**：光滑区域保持二阶
//! 3. **无振荡**：间断附近无伪振荡
//! 4. **数值扩散**：最小化人工扩散

// ============================================================================
mod classic;
mod muscl;
mod unstructured;

pub use classic::*;
pub use muscl::*;
pub use unstructured::*;

#[cfg(test)]
mod tests;
