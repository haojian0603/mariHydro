//! 统一存储抽象层
//!
//! 本模块提供工作流存储的抽象接口，支持多种后端实现：
//! - 内存存储 (MemoryStorage): 基于 DashMap 的并发安全内存存储
//! - SQLite 存储 (SqliteStorage): 基于 SQLite 的持久化存储
//!
//! # 设计目标
//!
//! 1. 统一接口：通过 `WorkflowStorage` trait 统一不同后端
//! 2. 并发安全：所有实现都是线程安全的
//! 3. 事务支持：可选的事务支持（通过 `TransactionalStorage`）
//!
//! # 使用示例
//!
//! ```rust
//! use marihydro::infra::storage::{MemoryStorage, WorkflowStorage};
//!
//! let storage = MemoryStorage::new();
//! storage.save_project("project-1", "{...}").unwrap();
//! ```

pub mod memory;
pub mod traits;

// 重导出常用类型
pub use memory::MemoryStorage;
pub use traits::{TransactionContext, TransactionalStorage, WorkflowStorage};
