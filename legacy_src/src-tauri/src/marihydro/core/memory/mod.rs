//! 内存管理子系统
//!
//! 本模块提供高效的内存管理功能：
//! - 预分配工作空间
//! - 缓冲池管理
//! - 零分配热路径支持
//!
//! # 设计目标
//!
//! 1. 减少热路径上的内存分配
//! 2. 提供线程安全的缓冲池
//! 3. 支持时间步长间的缓冲区复用

pub mod pool;
pub mod workspace;

// 重导出常用类型
pub use pool::{BufferPool, PooledBuffer};
pub use workspace::{Workspace, WorkspaceBuilder};
