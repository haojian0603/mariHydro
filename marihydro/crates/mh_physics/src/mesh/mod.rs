//! crates/mh_physics/src/mesh/mod.rs
//!
//! # 网格抽象层
//!
//! 提供结构化和非结构化网格的统一 Backend 感知接口。
//!
//! ## 架构说明
//!
//! - **Layer 3**: [`MeshTopology`] trait 完全泛型化，无默认 Backend 类型
//! - **Layer 4**: 具体类型别名在 `mh_physics/src/lib.rs` 中定义
//!
//! ## 模块结构
//!
//! - [`topology`]: 网格拓扑 trait 定义和几何辅助函数
//! - [`unstructured`]: 非结构化网格适配器
//! - [`structured`]: 结构化网格（骨架实现）
//!
//! ## 使用示例
//!
//! ```ignore
//! use mh_physics::mesh::{MeshTopology, UnstructuredMeshAdapter};
//! use mh_runtime::CpuBackend;
//!
//! let backend = CpuBackend::<f64>::new();
//! let adapter = UnstructuredMeshAdapter::from_physics_mesh_with_backend(&backend, mesh)?;
//! ```

pub mod topology;
pub mod unstructured;
pub mod structured;

// Layer 3: 完全泛型化的拓扑类型
pub use topology::{MeshKind, MeshTopology, FaceInfo, MeshGeometry};

// Layer 3: 泛型网格适配器（无默认 Backend）
pub use unstructured::UnstructuredMeshAdapter;
pub use structured::StructuredMesh;
