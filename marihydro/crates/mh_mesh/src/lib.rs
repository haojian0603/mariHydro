// marihydro\crates\mh_mesh\src/lib.rs

//! MariHydro 网格模块
//!
//! 提供统一的半边网格数据结构，支持高效的拓扑操作和计算。
//!
//! # 核心类型
//!
//! - [`HalfEdgeMesh`]: 可编辑的半边网格
//! - [`FrozenMesh`]: 只读的 SoA 布局网格，用于计算
//!
//! # Trait 抽象
//!
//! - [`MeshAccess`]: 网格只读访问接口
//! - [`MeshTopology`]: 网格拓扑计算接口
//!
//! # 模块结构
//!
//! - [`halfedge`]: 半边网格核心实现
//! - [`frozen`]: 冻结网格
//! - [`traits`]: 网格抽象接口
//! - [`compat`]: 兼容层转换
//! - [`io`]: 网格 IO (GMSH, GeoJSON, MHB)
//!
//! # 示例
//!
//! ```rust
//! use mh_mesh::halfedge::HalfEdgeMesh;
//!
//! // 创建半边网格
//! let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();
//!
//! // 添加顶点
//! let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
//! let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
//! let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);
//!
//! // 添加三角形
//! mesh.add_triangle(v0, v1, v2);
//!
//! // 冻结为只读网格
//! let frozen = mesh.freeze();
//! assert_eq!(frozen.n_cells(), 1);
//! ```

pub mod attributes;
pub mod compat;
pub mod frozen;
pub mod halfedge;
pub mod io;
pub mod quality;
pub mod traits;

// 算法模块（待实现）
pub mod algorithms;

// 新增模块：空间索引和点定位
pub mod spatial_index;
pub mod locator;
pub mod converter;

// 重新导出核心类型
pub use attributes::{
    AttributeStats, AttributeStore, ATTR_BED_ELEVATION, ATTR_DISCHARGE_X, ATTR_DISCHARGE_Y,
    ATTR_MANNING_N, ATTR_VELOCITY_X, ATTR_VELOCITY_Y, ATTR_WATER_DEPTH, ATTR_WATER_SURFACE,
};
pub use frozen::{FrozenMesh, MeshStatistics};
pub use halfedge::{Face, HalfEdge, HalfEdgeMesh, Vertex};

// 重新导出 trait 抽象
pub use traits::{
    CellGeometry, FaceGeometry, MeshAccess, MeshAccessExt, MeshTopology, ValidationReport,
    ValidationStats,
};

// 重新导出新增模块的核心类型
pub use spatial_index::{CellEnvelope, MeshSpatialIndex, SpatialBounds, SpatialIndexData};
pub use locator::{
    CachedLocator, LocateResult, LocateTolerance, LocatorCacheStats, MeshLocator,
};
pub use converter::{MeshStatisticsExt, SimpleMeshData};

