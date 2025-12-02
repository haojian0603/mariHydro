// marihydro\crates\mh_mesh\src/halfedge/mod.rs

//! 半边网格模块
//!
//! 提供统一的半边数据结构，支持高效的拓扑操作和遍历。
//!
//! # 模块结构
//!
//! - [`mesh`]: 核心数据结构
//! - [`traversal`]: 拓扑遍历迭代器
//! - [`operations`]: 拓扑操作 (split/flip/collapse)
//! - [`validate`]: 拓扑验证

pub mod mesh;
pub mod operations;
pub mod traversal;
pub mod validate;

// 重新导出核心类型
pub use mesh::{Face, HalfEdge, HalfEdgeMesh, Vertex};
pub use operations::TopologyResult;
pub use validate::{ValidationError, ValidationReport};
