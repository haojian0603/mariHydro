//! 数值安全子系统
//!
//! 本模块提供安全的数值操作，集中解决所有 NaN/Inf/除零 问题。
//!
//! ## 核心组件
//!
//! - [`SafeF64`]: 安全浮点数类型，保证非NaN非Inf
//! - [`AtomicF64`]: 安全原子浮点数，支持无锁并发操作
//! - [`ValidationResult`]: 数值验证结果
//! - [`safe_div!`]: 安全除法宏
//!
//! ## 设计目标
//!
//! 1. 在编译时或运行时捕获数值异常
//! 2. 提供零开销的安全抽象
//! 3. 统一数值验证接口

pub mod atomic_float;
pub mod constants;
pub mod safe_float;
pub mod validation;

// 重导出常用类型
pub use atomic_float::AtomicF64;
pub use constants::*;
pub use safe_float::SafeF64;
pub use validation::{validate_array, ValidationResult};
