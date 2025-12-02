// src-tauri/src/marihydro/core/types/mod.rs

//! 核心类型定义
//!
//! 本模块提供类型级安全保障，包括：
//! - 安全包装类型（FiniteF64, SafeDepth, SafeVelocity）
//! - 类型安全索引（CellIndex, FaceIndex 等）
//! - 数值参数配置
//! - 物理常数
//! - 地理变换
//! - 场类型抽象
//!
//! # 层级约束
//!
//! - 本模块属于 Layer 1 (核心层)
//! - 禁止依赖 domain, physics, forcing, io, infra, workflow 等上层模块
//! - 仅依赖标准库和 glam、serde 等基础 crate

pub mod field_types;
pub mod geo_transform;
pub mod indices;
pub mod numerical_params;
pub mod physical_constants;
pub mod safe_types;

// 重导出常用类型
pub use field_types::{FieldStats, ScalarField, VectorField};
pub use geo_transform::GeoTransform;
pub use indices::{
    reinterpret_cell_ids, reinterpret_cell_indices, reinterpret_face_ids,
    reinterpret_face_indices, reinterpret_node_ids, reinterpret_node_indices,
    usize_slice_as_cell_indices, usize_slice_as_face_indices, usize_slice_as_node_indices,
    BoundaryIndex, CellId, CellIndex, FaceId, FaceIndex, NodeId, NodeIndex, INVALID_INDEX,
};
pub use numerical_params::{NumericalParams, NumericalParamsBuilder, ParamsValidationError};
pub use physical_constants::PhysicalConstants;
pub use safe_types::{FiniteF64, NonFiniteError, SafeDepth, SafeVelocity};
