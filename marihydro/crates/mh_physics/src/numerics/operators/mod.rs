// crates/mh_physics/src/numerics/operators/mod.rs

//! 数值算子模块
//!
//! 提供通用的数值计算算子：
//! - [`diffusion`]: 标量场扩散算子
//!
//! # 模块说明
//!
//! 此模块包含**数学算子**而非**物理源项**。
//! 扩散算子可以应用于任意标量场（温度、盐度、示踪剂等），
//! 而源项则专门针对浅水方程的动量和质量守恒。
//!
//! # 与源项模块的关系
//!
//! 源项模块 (`sources`) 中的某些功能（如湍流扩散）会调用此模块的算子。
//! 为保持向后兼容，`sources` 模块重导出了扩散相关的类型。

pub mod diffusion;

pub use diffusion::{
    DiffusionBC, DiffusionConfig, DiffusionSolver, DiffusionError,
    VariableDiffusionSolver,
    estimate_stable_dt, required_substeps,
};
