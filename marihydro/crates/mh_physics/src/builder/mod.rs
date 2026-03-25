// crates/mh_physics/src/builder/mod.rs

//! Layer 4 入口辅助模块
//!
//! 本模块当前只保留两类内容：
//! - 面向应用层的无泛型配置结构 `SolverConfig`
//! - 运行时结果快照结构 `DynState` / `DynStepResult`
//!
//! 它不再负责构建假求解器，也不允许重新引入旧的空壳构建入口。

pub mod dyn_solver;
pub mod config;

pub use config::SolverConfig;
pub use dyn_solver::{DynState, DynStepResult};

#[cfg(test)]
#[doc(hidden)]
pub mod __private {
    /// 标记 Layer 4 专用类型
    pub trait Layer4Only {}

    impl Layer4Only for super::SolverConfig {}
}
