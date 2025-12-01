// src-tauri/src/marihydro/core/mod.rs

//! MariHydro 核心抽象层
//!
//! 这个模块提供零依赖的基础抽象，包括：
//! - 错误类型系统
//! - 核心trait定义
//! - 类型级安全包装
//! - 预分配缓冲区管理
//! - 运行时验证工具
//! - 数值安全子系统
//! - 并行策略子系统
//! - 内存管理子系统
//! - 计算后端抽象（CPU/GPU）

pub mod compute;
pub mod error;
pub mod memory;
pub mod numerical;
pub mod parallel;
pub mod traits;
pub mod types;
pub mod validation;

// 重导出常用类型
pub use error::{MhError, MhResult};
pub use memory::{BufferPool, PooledBuffer, Workspace, WorkspaceBuilder};
pub use numerical::{AtomicF64, SafeF64};
pub use parallel::{ParallelConfig, ParallelStrategy, StrategySelector};
pub use types::{NumericalParams, PhysicalConstants};

// 计算后端相关
pub use compute::{ComputeBackend, ComputeOperation, DeviceCapabilities, DeviceType};
