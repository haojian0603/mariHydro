// src-tauri/src/marihydro/core/mod.rs

//! MariHydro 核心抽象层
//!
//! 这个模块提供零依赖的基础抽象，包括：
//! - 错误类型系统
//! - 核心trait定义
//! - 类型级安全包装
//! - 预分配缓冲区管理
//! - 运行时验证工具

pub mod error;
pub mod traits;
pub mod types;
pub mod validation;
pub mod workspace;

// 重导出常用类型
pub use error::{MhError, MhResult};
pub use types::{NumericalParams, PhysicalConstants};
pub use workspace::Workspace;
