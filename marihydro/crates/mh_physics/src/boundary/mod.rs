// crates/mh_physics/src/boundary/mod.rs

//! 边界条件模块
//!
//! 本模块提供水动力学模拟中的边界条件处理功能：
//!
//! # 子模块
//!
//! - [`types`]: 边界条件类型定义
//! - [`manager`]: 边界条件管理器
//! - [`ghost`]: 幽灵状态计算器
//!
//! # 主要类型
//!
//! - [`BoundaryKind`]: 边界类型枚举（固壁、开海、入流、出流等）
//! - [`BoundaryCondition`]: 边界条件配置
//! - [`BoundaryManager`]: 边界条件管理器
//! - [`GhostStateCalculator`]: 幽灵状态计算器
//!
//! # 使用示例
//!
//! ## 创建边界管理器
//!
//! ```ignore
//! use mh_physics::boundary::{BoundaryManager, BoundaryCondition, BoundaryParams};
//! use glam::DVec2;
//!
//! let mut manager = BoundaryManager::new(BoundaryParams::default());
//!
//! // 添加边界条件
//! manager.add_condition(BoundaryCondition::wall("north"));
//! manager.add_condition(BoundaryCondition::open_sea("south"));
//!
//! // 注册边界面
//! manager.register_face(0, 0, DVec2::new(0.0, 1.0), 1.0, "north").unwrap();
//! ```
//!
//! ## 计算幽灵状态
//!
//! ```ignore
//! use mh_physics::boundary::{GhostStateCalculator, BoundaryKind, BoundaryParams};
//! use mh_physics::state::ConservedState;
//! use glam::DVec2;
//!
//! let calculator = GhostStateCalculator::default();
//! let interior = ConservedState::from_primitive(1.0, 0.5, 0.0);
//! let normal = DVec2::new(1.0, 0.0);
//! let z_bed = 0.0; // 底床高程
//!
//! let ghost = calculator.compute_ghost(
//!     interior,
//!     BoundaryKind::Wall,
//!     normal,
//!     None,
//!     z_bed,
//! );
//! ```
//!
//! # 设计原则
//!
//! 1. **分离关注点**：类型定义、管理逻辑、计算逻辑分离
//! 2. **可扩展性**：通过 trait 支持自定义边界数据源
//! 3. **性能优先**：支持批量操作，避免虚函数开销
//! 4. **类型安全**：使用枚举替代整数标志

mod types;
mod manager;
mod ghost;

// 从 types 模块导出
pub use types::{
    BoundaryKind,
    BoundaryCondition,
    ExternalForcing,
    BoundaryParams,
};

// 从 manager 模块导出
pub use manager::{
    BoundaryFaceInfo,
    BoundaryManager,
    BoundaryDataProvider,
    ConstantForcingProvider,
    BoundaryError,
};

// 从 ghost 模块导出
pub use ghost::{
    GhostStateCalculator,
    GhostMomentumMode,
    reflect_velocity,
    decompose_velocity,
};


