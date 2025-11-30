//! 并行策略子系统
//!
//! 本模块提供统一的并行执行策略管理，解决策略分散问题。
//!
//! # 核心组件
//!
//! - [`ParallelConfig`]: 全局并行配置
//! - [`ParallelStrategy`]: 并行执行策略枚举
//! - [`StrategySelector`]: 策略选择器
//! - [`AdaptiveScheduler`]: 自适应调度器
//!
//! # 使用示例
//!
//! ```rust
//! use marihydro::core::parallel::{ParallelConfig, StrategySelector, parallel_for};
//!
//! let selector = StrategySelector::new();
//! let strategy = selector.for_gradient(10000);
//! parallel_for(&data, strategy, |i, item| {
//!     // 处理每个元素
//! });
//! ```

pub mod adaptive;
pub mod config;
pub mod metrics;
pub mod strategy;

// 重导出常用类型
pub use adaptive::AdaptiveScheduler;
pub use config::ParallelConfig;
pub use metrics::PerfMetrics;
pub use strategy::{parallel_for, parallel_for_mut, ParallelStrategy, StrategySelector};
