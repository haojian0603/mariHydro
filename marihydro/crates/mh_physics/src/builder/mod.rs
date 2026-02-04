// crates/mh_physics/src/builder/mod.rs

//! 求解器构建模块（Layer 4 → Layer 3桥梁）
//!
//! 提供从无泛型配置到泛型引擎的桥梁，**禁止在Layer 3直接调用本模块**。
//!
//! # 架构契约
//!
//! ```text
//! // ❌ 错误：Layer 3直接调用Layer 4
//! use mh_physics::builder::SolverConfig;
//! 
//! fn layer3_code() {
//!     let config = SolverConfig::default(); // 编译失败
//! }
//! ```
//!
//! ```text
//! // ✅ 正确：Layer 4 → Builder → Layer 3
//! use mh_physics::{Layer3Config, NoSource, ShallowWaterSolver};
//! use mh_runtime::CpuBackend;
//! 
//! fn layer4_code() {
//!     let config = Layer3Config::default();
//!     let solver = ShallowWaterSolver::<CpuBackend<f64>, NoSource<CpuBackend<f64>>>::new(
//!         mesh,
//!         config,
//!         CpuBackend::<f64>::new(),
//!     );
//! }
//! ```
//!
//! # 转换链
//!
//! ```text
//! SolverConfig (Layer 4, f64) → SolverBuilder → ShallowWaterSolver<B, S> (Layer 3, 泛型)
//! ```
//!
//! # 设计原则
//!
//! ```text
//! App Layer (无泛型)
//!     │
//!     ▼
//! SolverConfig ─────> SolverBuilder
//!     │                    │
//!     │                    ▼ (精度分发)
//!     │               ┌────────────────┐
//!     │               │ Precision::F32 │──> ShallowWaterSolver<CpuBackend<f32>, S>
//!     │               │ Precision::F64 │──> ShallowWaterSolver<CpuBackend<f64>, S>
//!     │               └────────────────┘
//!     │                    │
//!     ▼                    ▼
//! SolverHandle 枚举（静态分发）
//! ```

pub mod dyn_solver;
pub mod config;
pub mod solver_builder;

pub use dyn_solver::{DynState, DynStepResult};
pub use config::SolverConfig;
pub use solver_builder::{SolverBuilder, BuildError, SolverHandle};

// ============================================
// 🔥 编译期架构保护（防止Layer 3滥用）
// ============================================

#[cfg(test)]
#[doc(hidden)]
pub mod __private {
    /// 标记Layer 4专用类型
    pub trait Layer4Only {}
    
    impl Layer4Only for super::SolverConfig {}
    impl Layer4Only for super::SolverBuilder {}
}
