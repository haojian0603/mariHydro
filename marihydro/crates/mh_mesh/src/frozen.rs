// crates/mh_mesh/src/frozen.rs

//! 冻结网格（FrozenMesh<B>）- 只读计算网格
//!
//! 从 HalfEdgeMesh 导出的只读 SoA（Structure of Arrays）布局网格，用于高性能数值计算。
//! 支持 f32/f64 运行时精度切换，零拷贝序列化，内置空间索引。
//!
//! # 设计要点
//!
//! 1. **SoA 布局**：连续内存数组，提升缓存命中率
//! 2. **泛型精度**：`B: Backend` 支持 f32/f64 切换
//! 3. **不可变性**：冻结后不可修改，线程安全
//! 4. **空间索引**：内置 R-tree 支持高效点定位
//! 5. **零拷贝**：支持 mmap 加载和序列化
//!
//! # 数据分类
//!
//! - **几何数据**：cell_center, node_coords, face_center（保持 f64）
//! - **物理场数据**：cell_area, cell_z_bed, face_length（泛型 B::Scalar）
//! - **索引数据**：cell_node_indices, face_owner（保持 u32/usize）
//!
//! # 使用示例
//!
//! ```rust
//! use mh_mesh::frozen::FrozenMesh;
//! use mh_runtime::CpuBackend;
//!
//! // 创建空网格（f64 精度）
//! let backend = CpuBackend::<f64>::new();
//! let mesh: FrozenMesh<CpuBackend<f64>> = FrozenMesh::empty_with_cells_backend(backend, 100);
//!
//! // 创建空网格（f32 精度，节省内存）
//! let backend_f32 = CpuBackend::<f32>::new();
//! let mesh_f32: FrozenMesh<CpuBackend<f32>> = FrozenMesh::empty_with_cells_backend(backend_f32, 1_000_000);
//!
//! // 计算统计信息
//! let stats = mesh.statistics();
//! println!("网格面积: {:.2}", stats.total_area);
//! ```


use crate::locator::MeshLocator;
use crate::spatial_index::MeshSpatialIndex;
use crate::traits::{MeshAccess, MeshTopology};
use mh_geo::{Point2D, Point3D};
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

/// 冻结网格校验错误
#[derive(Debug, Clone, Error)]
pub enum FrozenMeshError {
    #[error("字段 {field} 长度不匹配: expected={expected}, actual={actual}")]
    LengthMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("偏移数组 {field} 非单调或越界: cell={cell}, start={start}, end={end}, total={total}")]
    OffsetInvalid {
        field: &'static str,
        cell: usize,
        start: usize,
        end: usize,
        total: usize,
    },
    #[error("索引越界: {field}={index}, limit={limit}")]
    IndexOutOfRange {
        field: &'static str,
        index: usize,
        limit: usize,
    },
    #[error("单元面积非正: cell={cell}, area={area}")]
    NonPositiveArea { cell: usize, area: String },
    #[error("面长度非正: face={face}, length={length}")]
    NonPositiveFaceLength { face: usize, length: String },
    #[error("单元节点重复: cell={cell}, node={node}")]
    DuplicateCellNode { cell: usize, node: u32 },
    #[error("几何数据包含非有限值: field={field}, index={index}")]
    NonFiniteValue { field: &'static str, index: usize },
    #[error("索引重复: field={field}, index={index}")]
    DuplicateIndex { field: &'static str, index: usize },
}

/// 冻结网格（泛型版本）
///
/// 从半边网格导出的只读计算网格，支持 f32/f64 精度运行时切换。
/// 采用 SoA 布局优化计算性能，不可修改，线程安全。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "B::Buffer<B::Scalar>: Serialize, B::Scalar: Serialize",
    deserialize = "B::Buffer<B::Scalar>: Deserialize<'de>, B::Scalar: Deserialize<'de>, B: Default"
))]
pub struct FrozenMesh<B: Backend> {
    // ===== 节点数据（几何数据）=====
    /// 节点数量（几何拓扑，固定 usize）
    pub n_nodes: usize,
    /// 节点坐标（3D，几何数据保持 f64）
    pub node_coords: Vec<Point3D>,

    // ===== 单元数据（物理场数据，泛型化）=====
    /// 单元数量
    pub n_cells: usize,
    /// 单元中心坐标（2D，几何数据保持 f64）
    pub cell_center: Vec<Point2D>,
    /// 单元面积（物理场数据，泛型 S）
    pub cell_area: B::Buffer<B::Scalar>,
    /// 单元底床高程（物理场数据，泛型 S）
    pub cell_z_bed: B::Buffer<B::Scalar>,
    /// 单元节点索引（压缩格式，索引数据保持 u32）
    pub cell_node_offsets: Vec<usize>,
    /// 单元节点索引列表
    pub cell_node_indices: Vec<u32>,
    /// 单元面索引（压缩格式）
    pub cell_face_offsets: Vec<usize>,
    /// 单元面索引列表
    pub cell_face_indices: Vec<u32>,
    /// 单元邻居索引（压缩格式）
    pub cell_neighbor_offsets: Vec<usize>,
    /// 单元邻居索引列表（u32::MAX 表示无邻居）
    pub cell_neighbor_indices: Vec<u32>,

    // ===== 面数据（混合数据类型）=====
    /// 面总数
    pub n_faces: usize,
    /// 内部面数量
    pub n_interior_faces: usize,
    /// 面中心坐标（2D，几何数据）
    pub face_center: Vec<Point2D>,
    /// 面法向量（3D，几何数据）
    pub face_normal: Vec<Point3D>,
    /// 面长度（物理场数据，泛型 S）
    pub face_length: B::Buffer<B::Scalar>,
    /// 面左侧高程（物理场数据，泛型 S）
    pub face_z_left: B::Buffer<B::Scalar>,
    /// 面右侧高程（物理场数据，泛型 S）
    pub face_z_right: B::Buffer<B::Scalar>,
    /// 面 owner 单元索引（索引数据）
    pub face_owner: Vec<u32>,
    /// 面 neighbor 单元索引（u32::MAX 表示边界）
    pub face_neighbor: Vec<u32>,
    /// 面到 owner 中心的向量（2D，几何数据）
    pub face_delta_owner: Vec<Point2D>,
    /// 面到 neighbor 中心的向量（2D，几何数据）
    pub face_delta_neighbor: Vec<Point2D>,
    /// owner 到 neighbor 的距离（物理场数据，泛型 S）
    pub face_dist_o2n: B::Buffer<B::Scalar>,

    // ===== 边界数据 =====
    /// 边界面索引列表（索引数据）
    pub boundary_face_indices: Vec<u32>,
    /// 边界名称
    pub boundary_names: Vec<String>,
    /// 面的边界 ID（None 表示内部面）
    pub face_boundary_id: Vec<Option<u32>>,

    // ===== 统计信息（物理场数据，泛型 S）=====
    /// 最小单元尺寸
    pub min_cell_size: B::Scalar,
    /// 最大单元尺寸
    pub max_cell_size: B::Scalar,

    // ===== AMR 预分配字段（Phase 2+）=====
    /// 单元细化级别（0=基础网格）
    #[serde(default)]
    pub cell_refinement_level: Vec<u8>,
    /// 父单元索引（顶层单元指向自身）
    #[serde(default)]
    pub cell_parent: Vec<u32>,
    /// Ghost 单元容量（MPI 边界交换）
    #[serde(default)]
    pub ghost_capacity: usize,

    // ===== ID 映射与排列 =====
    /// 原始单元 ID（从网格文件读取）
    #[serde(default)]
    pub cell_original_id: Vec<u32>,
    /// 原始面 ID（边界标识）
    #[serde(default)]
    pub face_original_id: Vec<u32>,
    /// 单元排列索引（frozen_idx -> 原始索引）
    #[serde(default)]
    pub cell_permutation: Vec<u32>,
    /// 逆排列（原始索引 -> frozen_idx）
    #[serde(default)]
    pub cell_inv_permutation: Vec<u32>,

    /// 计算后端（不序列化）
    #[serde(skip, default)]
    backend: B,
}

impl<B: Backend> FrozenMesh<B> {
    // =========================================================================
    // 构造函数
    // =========================================================================

    /// 创建空的冻结网格（显式后端）
    pub fn empty_with_backend(backend: B) -> Self {
        Self {
            n_nodes: 0,
            node_coords: Vec::new(),
            n_cells: 0,
            cell_center: Vec::new(),
            cell_area: backend.alloc(0),
            cell_z_bed: backend.alloc(0),
            cell_node_offsets: vec![0],
            cell_node_indices: Vec::new(),
            cell_face_offsets: vec![0],
            cell_face_indices: Vec::new(),
            cell_neighbor_offsets: vec![0],
            cell_neighbor_indices: Vec::new(),
            n_faces: 0,
            n_interior_faces: 0,
            face_center: Vec::new(),
            face_normal: Vec::new(),
            face_length: backend.alloc(0),
            face_z_left: backend.alloc(0),
            face_z_right: backend.alloc(0),
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            face_delta_owner: Vec::new(),
            face_delta_neighbor: Vec::new(),
            face_dist_o2n: backend.alloc(0),
            boundary_face_indices: Vec::new(),
            boundary_names: Vec::new(),
            face_boundary_id: Vec::new(),
            min_cell_size: B::Scalar::MAX,
            max_cell_size: B::Scalar::ZERO,
            cell_refinement_level: Vec::new(),
            cell_parent: Vec::new(),
            ghost_capacity: 0,
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
            backend,
        }
    }

    /// 创建带有指定单元数量的空网格（显式后端）
    pub fn empty_with_cells_backend(backend: B, n_cells: usize) -> Self {
        Self {
            n_nodes: 0,
            node_coords: Vec::new(),
            n_cells,
            cell_center: vec![Point2D::new(0.0, 0.0); n_cells],
            cell_area: backend.alloc_init(n_cells, B::Scalar::ONE),
            cell_z_bed: backend.alloc_init(n_cells, B::Scalar::ZERO),
            cell_node_offsets: vec![0; n_cells + 1],
            cell_node_indices: Vec::new(),
            cell_face_offsets: vec![0; n_cells + 1],
            cell_face_indices: Vec::new(),
            cell_neighbor_offsets: vec![0; n_cells + 1],
            cell_neighbor_indices: Vec::new(),
            n_faces: 0,
            n_interior_faces: 0,
            face_center: Vec::new(),
            face_normal: Vec::new(),
            face_length: backend.alloc(0),
            face_z_left: backend.alloc(0),
            face_z_right: backend.alloc(0),
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            face_delta_owner: Vec::new(),
            face_delta_neighbor: Vec::new(),
            face_dist_o2n: backend.alloc(0),
            boundary_face_indices: Vec::new(),
            boundary_names: Vec::new(),
            face_boundary_id: Vec::new(),
            min_cell_size: B::Scalar::ONE,
            max_cell_size: B::Scalar::ONE,
            cell_refinement_level: vec![0; n_cells],
            cell_parent: (0..n_cells as u32).collect(),
            ghost_capacity: 0,
            cell_original_id: (0..n_cells as u32).collect(),
            face_original_id: Vec::new(),
            cell_permutation: (0..n_cells as u32).collect(),
            cell_inv_permutation: (0..n_cells as u32).collect(),
            backend,
        }
    }

    // =========================================================================
    // 基本统计
    // =========================================================================

    /// 节点数量
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// 内部面数量
    #[inline]
    pub fn n_interior_faces(&self) -> usize {
        self.n_interior_faces
    }

    /// 边界面数量
    #[inline]
    pub fn n_boundary_faces(&self) -> usize {
        self.n_faces - self.n_interior_faces
    }

    // =========================================================================
    // 单元数据访问
    // =========================================================================

    /// 单元中心（几何数据，f64）
    #[inline]
    pub fn cell_center(&self, cell: usize) -> Point2D {
        self.cell_center[cell]
    }

    /// 单元面积（物理场数据，泛型 B::Scalar）
    #[inline]
    pub fn cell_area(&self, cell: usize) -> B::Scalar {
        self.cell_area[cell]
    }

    /// 单元底床高程（物理场数据，泛型 B::Scalar）
    #[inline]
    pub fn cell_z_bed(&self, cell: usize) -> B::Scalar {
        self.cell_z_bed[cell]
    }

    /// 单元的节点索引列表
    #[inline]
    pub fn cell_nodes(&self, cell: usize) -> &[u32] {
        let start = self.cell_node_offsets[cell];
        let end = self.cell_node_offsets[cell + 1];
        &self.cell_node_indices[start..end]
    }

    /// 单元的面索引列表
    #[inline]
    pub fn cell_faces(&self, cell: usize) -> &[u32] {
        let start = self.cell_face_offsets[cell];
        let end = self.cell_face_offsets[cell + 1];
        &self.cell_face_indices[start..end]
    }

    /// 单元的邻居索引列表
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> &[u32] {
        let start = self.cell_neighbor_offsets[cell];
        let end = self.cell_neighbor_offsets[cell + 1];
        &self.cell_neighbor_indices[start..end]
    }

    // =========================================================================
    // 面数据访问
    // =========================================================================

    /// 面中心（2D 几何数据，f64）
    #[inline]
    pub fn face_center(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    /// 面法向量（3D 几何数据）
    #[inline]
    pub fn face_normal(&self, face: usize) -> Point3D {
        self.face_normal[face]
    }

    /// 面长度（物理场数据，泛型 B::Scalar）
    #[inline]
    pub fn face_length(&self, face: usize) -> B::Scalar {
        self.face_length[face]
    }

    /// 面左侧高程（物理场数据，泛型 B::Scalar）
    #[inline]
    pub fn face_z_left(&self, face: usize) -> B::Scalar {
        self.face_z_left[face]
    }

    /// 面右侧高程（物理场数据，泛型 B::Scalar）
    #[inline]
    pub fn face_z_right(&self, face: usize) -> B::Scalar {
        self.face_z_right[face]
    }

    /// 面 owner 单元索引
    #[inline]
    pub fn face_owner(&self, face: usize) -> u32 {
        self.face_owner[face]
    }

    /// 面 neighbor 单元索引（边界返回 None）
    #[inline]
    pub fn face_neighbor(&self, face: usize) -> Option<u32> {
        let n = self.face_neighbor[face];
        if n == u32::MAX {
            None
        } else {
            Some(n)
        }
    }

    /// 是否为边界面
    #[inline]
    pub fn is_boundary_face(&self, face: usize) -> bool {
        face >= self.n_interior_faces
    }

    // =========================================================================
    // ID 映射与排列
    // =========================================================================

    /// 获取单元的原始 ID（从网格文件读取）
    #[inline]
    pub fn cell_original_id(&self, cell: usize) -> u32 {
        self.cell_original_id.get(cell).copied().unwrap_or(cell as u32)
    }

    /// 获取面的原始 ID（边界标识）
    #[inline]
    pub fn face_original_id(&self, face: usize) -> u32 {
        self.face_original_id.get(face).copied().unwrap_or(face as u32)
    }

    /// 从 frozen 索引获取原始索引
    #[inline]
    pub fn to_original_cell(&self, frozen_idx: usize) -> u32 {
        self.cell_permutation.get(frozen_idx).copied().unwrap_or(frozen_idx as u32)
    }

    /// 从原始索引获取 frozen 索引
    #[inline]
    pub fn from_original_cell(&self, original_idx: usize) -> u32 {
        self.cell_inv_permutation.get(original_idx).copied().unwrap_or(original_idx as u32)
    }

    /// 设置单元排列（网格重排序后更新映射）
    pub fn set_permutation(&mut self, perm: Vec<u32>) {
        let n = perm.len();
        let mut inv = vec![0u32; n];
        for (frozen, &orig) in perm.iter().enumerate() {
            if (orig as usize) < n {
                inv[orig as usize] = frozen as u32;
            }
        }
        self.cell_permutation = perm;
        self.cell_inv_permutation = inv;
    }

    // =========================================================================
    // 节点数据访问
    // =========================================================================

    /// 节点 3D 坐标
    #[inline]
    pub fn node_coords(&self, node: usize) -> Point3D {
        self.node_coords[node]
    }

    /// 节点 2D 坐标 (x, y)
    #[inline]
    pub fn node_xy(&self, node: usize) -> Point2D {
        self.node_coords[node].xy()
    }

    /// 节点高程 (z)
    #[inline]
    pub fn node_z(&self, node: usize) -> f64 {
        self.node_coords[node].z
    }

    // =========================================================================
    // 范围迭代器
    // =========================================================================

    /// 内部面索引范围（0..n_interior_faces）
    #[inline]
    pub fn interior_faces(&self) -> std::ops::Range<usize> {
        0..self.n_interior_faces
    }

    /// 边界面索引范围（n_interior_faces..n_faces）
    #[inline]
    pub fn boundary_faces(&self) -> std::ops::Range<usize> {
        self.n_interior_faces..self.n_faces
    }

    /// 单元索引范围（0..n_cells）
    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        0..self.n_cells
    }

    /// 节点索引范围（0..n_nodes）
    #[inline]
    pub fn nodes(&self) -> std::ops::Range<usize> {
        0..self.n_nodes
    }

    // =========================================================================
    // 统计与验证
    // =========================================================================

    /// 计算网格统计信息（面积、边长等）
    pub fn statistics(&self) -> MeshStatistics<B> {
        let (mut min_area, mut max_area, mut total_area) = if self.cell_area.is_empty() {
            (B::Scalar::ZERO, B::Scalar::ZERO, B::Scalar::ZERO)
        } else {
            (B::Scalar::MAX, B::Scalar::ZERO, B::Scalar::ZERO)
        };

        for &area in self.cell_area.as_slice() {
            min_area = min_area.min(area);
            max_area = max_area.max(area);
            total_area = total_area + area;
        }

        let (mut min_length, mut max_length) = if self.face_length.is_empty() {
            (B::Scalar::ZERO, B::Scalar::ZERO)
        } else {
            (B::Scalar::MAX, B::Scalar::ZERO)
        };

        for &len in self.face_length.as_slice() {
            min_length = min_length.min(len);
            max_length = max_length.max(len);
        }

        MeshStatistics {
            n_cells: self.n_cells,
            n_faces: self.n_faces,
            n_interior_faces: self.n_interior_faces,
            n_boundary_faces: self.n_faces - self.n_interior_faces,
            n_nodes: self.n_nodes,
            total_area,
            min_cell_area: min_area,
            max_cell_area: max_area,
            min_edge_length: min_length,
            max_edge_length: max_length,
        }
    }

    /// 验证网格完整性
    ///
    /// 检查数组长度和索引有效性，用于调试和测试。
    pub fn validate(&self) -> Result<(), FrozenMeshError> {
        let check_len = |field: &'static str, actual: usize, expected: usize| {
            if actual != expected {
                return Err(FrozenMeshError::LengthMismatch {
                    field,
                    expected,
                    actual,
                });
            }
            Ok(())
        };

        check_len("cell_center", self.cell_center.len(), self.n_cells)?;
        check_len("cell_area", self.cell_area.len(), self.n_cells)?;
        check_len("cell_z_bed", self.cell_z_bed.len(), self.n_cells)?;
        check_len("node_coords", self.node_coords.len(), self.n_nodes)?;
        check_len("face_center", self.face_center.len(), self.n_faces)?;
        check_len("face_normal", self.face_normal.len(), self.n_faces)?;
        check_len("face_length", self.face_length.len(), self.n_faces)?;
        check_len("face_z_left", self.face_z_left.len(), self.n_faces)?;
        check_len("face_z_right", self.face_z_right.len(), self.n_faces)?;
        check_len("face_owner", self.face_owner.len(), self.n_faces)?;
        check_len("face_neighbor", self.face_neighbor.len(), self.n_faces)?;
        check_len("face_delta_owner", self.face_delta_owner.len(), self.n_faces)?;
        check_len("face_delta_neighbor", self.face_delta_neighbor.len(), self.n_faces)?;
        check_len("face_dist_o2n", self.face_dist_o2n.len(), self.n_faces)?;
        check_len("face_boundary_id", self.face_boundary_id.len(), self.n_faces)?;
        check_len("cell_node_offsets", self.cell_node_offsets.len(), self.n_cells + 1)?;
        check_len("cell_face_offsets", self.cell_face_offsets.len(), self.n_cells + 1)?;
        check_len("cell_neighbor_offsets", self.cell_neighbor_offsets.len(), self.n_cells + 1)?;

        if let Some(last) = self.cell_node_offsets.last().copied() {
            check_len("cell_node_indices", self.cell_node_indices.len(), last)?;
        }
        if let Some(last) = self.cell_face_offsets.last().copied() {
            check_len("cell_face_indices", self.cell_face_indices.len(), last)?;
        }
        if let Some(last) = self.cell_neighbor_offsets.last().copied() {
            check_len("cell_neighbor_indices", self.cell_neighbor_indices.len(), last)?;
        }

        for (i, coord) in self.node_coords.iter().enumerate() {
            if !coord.x.is_finite() || !coord.y.is_finite() || !coord.z.is_finite() {
                return Err(FrozenMeshError::NonFiniteValue {
                    field: "node_coords",
                    index: i,
                });
            }
        }

        for (i, coord) in self.face_center.iter().enumerate() {
            if !coord.x.is_finite() || !coord.y.is_finite() {
                return Err(FrozenMeshError::NonFiniteValue {
                    field: "face_center",
                    index: i,
                });
            }
        }

        for (i, normal) in self.face_normal.iter().enumerate() {
            if !normal.x.is_finite() || !normal.y.is_finite() || !normal.z.is_finite() {
                return Err(FrozenMeshError::NonFiniteValue {
                    field: "face_normal",
                    index: i,
                });
            }
        }

        for (i, &area) in self.cell_area.as_slice().iter().enumerate() {
            if area <= B::Scalar::ZERO {
                return Err(FrozenMeshError::NonPositiveArea {
                    cell: i,
                    area: format!("{area:?}"),
                });
            }
        }

        for (i, &len) in self.face_length.as_slice().iter().enumerate() {
            if len <= B::Scalar::ZERO {
                return Err(FrozenMeshError::NonPositiveFaceLength {
                    face: i,
                    length: format!("{len:?}"),
                });
            }
        }

        for (_i, &owner) in self.face_owner.iter().enumerate() {
            if owner as usize >= self.n_cells {
                return Err(FrozenMeshError::IndexOutOfRange {
                    field: "face_owner",
                    index: owner as usize,
                    limit: self.n_cells,
                });
            }
        }

        for (_i, &neighbor) in self.face_neighbor.iter().enumerate() {
            if neighbor != u32::MAX && neighbor as usize >= self.n_cells {
                return Err(FrozenMeshError::IndexOutOfRange {
                    field: "face_neighbor",
                    index: neighbor as usize,
                    limit: self.n_cells,
                });
            }
        }

        let mut seen_boundary_faces = HashSet::new();
        for (i, &face_idx) in self.boundary_face_indices.iter().enumerate() {
            if face_idx as usize >= self.n_faces {
                return Err(FrozenMeshError::IndexOutOfRange {
                    field: "boundary_face_indices",
                    index: face_idx as usize,
                    limit: self.n_faces,
                });
            }
            if !seen_boundary_faces.insert(face_idx) {
                return Err(FrozenMeshError::DuplicateIndex {
                    field: "boundary_face_indices",
                    index: face_idx as usize,
                });
            }
            if self.face_boundary_id[face_idx as usize].is_none() {
                return Err(FrozenMeshError::IndexOutOfRange {
                    field: "face_boundary_id",
                    index: face_idx as usize,
                    limit: self.n_faces,
                });
            }
            let _ = i;
        }

        if !self.boundary_names.is_empty() {
            for (face, id) in self.face_boundary_id.iter().enumerate() {
                if let Some(bid) = id {
                    if *bid as usize >= self.boundary_names.len() {
                        return Err(FrozenMeshError::IndexOutOfRange {
                            field: "boundary_names",
                            index: *bid as usize,
                            limit: self.boundary_names.len(),
                        });
                    }
                }
                let _ = face;
            }
        }

        for cell in 0..self.n_cells {
            let start = self.cell_node_offsets[cell];
            let end = self.cell_node_offsets[cell + 1];
            let total = self.cell_node_indices.len();
            if start > end || end > total {
                return Err(FrozenMeshError::OffsetInvalid {
                    field: "cell_node_offsets",
                    cell,
                    start,
                    end,
                    total,
                });
            }

            let mut seen = HashSet::new();
            for &node in &self.cell_node_indices[start..end] {
                let node_usize = node as usize;
                if node_usize >= self.n_nodes {
                    return Err(FrozenMeshError::IndexOutOfRange {
                        field: "cell_node_indices",
                        index: node_usize,
                        limit: self.n_nodes,
                    });
                }
                if !seen.insert(node) {
                    return Err(FrozenMeshError::DuplicateCellNode { cell, node });
                }
            }
        }

        for cell in 0..self.n_cells {
            let start = self.cell_face_offsets[cell];
            let end = self.cell_face_offsets[cell + 1];
            let total = self.cell_face_indices.len();
            if start > end || end > total {
                return Err(FrozenMeshError::OffsetInvalid {
                    field: "cell_face_offsets",
                    cell,
                    start,
                    end,
                    total,
                });
            }
            for &face in &self.cell_face_indices[start..end] {
                let face_usize = face as usize;
                if face_usize >= self.n_faces {
                    return Err(FrozenMeshError::IndexOutOfRange {
                        field: "cell_face_indices",
                        index: face_usize,
                        limit: self.n_faces,
                    });
                }
            }
        }

        for cell in 0..self.n_cells {
            let start = self.cell_neighbor_offsets[cell];
            let end = self.cell_neighbor_offsets[cell + 1];
            let total = self.cell_neighbor_indices.len();
            if start > end || end > total {
                return Err(FrozenMeshError::OffsetInvalid {
                    field: "cell_neighbor_offsets",
                    cell,
                    start,
                    end,
                    total,
                });
            }
            for &neighbor in &self.cell_neighbor_indices[start..end] {
                if neighbor != u32::MAX && neighbor as usize >= self.n_cells {
                    return Err(FrozenMeshError::IndexOutOfRange {
                        field: "cell_neighbor_indices",
                        index: neighbor as usize,
                        limit: self.n_cells,
                    });
                }
            }
        }

        Ok(())
    }

    /// 计算网格边界框
    ///
    /// 返回 (min_x, min_y) 和 (max_x, max_y)
    pub fn bounding_box(&self) -> (Point2D, Point2D) {
        if self.n_nodes == 0 {
            return (Point2D::new(0.0, 0.0), Point2D::new(0.0, 0.0));
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for &coord in &self.node_coords {
            min_x = min_x.min(coord.x);
            min_y = min_y.min(coord.y);
            max_x = max_x.max(coord.x);
            max_y = max_y.max(coord.y);
        }

        (Point2D::new(min_x, min_y), Point2D::new(max_x, max_y))
    }

    /// 计算网格几何中心（面积加权）
    pub fn centroid(&self) -> Point2D {
        if self.n_cells == 0 {
            return Point2D::new(0.0, 0.0);
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut total_area = 0.0_f64;

        for cell in 0..self.n_cells {
            let center = self.cell_center(cell);
            let area: f64 = self.cell_area(cell).to_f64_lossy();
            sum_x += center.x * area;
            sum_y += center.y * area;
            total_area += area;
        }

        if total_area > 0.0 {
            Point2D::new(sum_x / total_area, sum_y / total_area)
        } else {
            Point2D::new(0.0, 0.0)
        }
    }
}

/// 网格统计信息（泛型版本）
///
/// 记录网格的宏观统计信息，用于日志、监控和自适应计算。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStatistics<B: Backend> {
    pub n_cells: usize,
    pub n_faces: usize,
    pub n_interior_faces: usize,
    pub n_boundary_faces: usize,
    pub n_nodes: usize,
    pub total_area: B::Scalar,
    pub min_cell_area: B::Scalar,
    pub max_cell_area: B::Scalar,
    pub min_edge_length: B::Scalar,
    pub max_edge_length: B::Scalar,
}

impl<B: Backend> std::fmt::Display for MeshStatistics<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 网格统计 ===")?;
        writeln!(f, "单元数: {}", self.n_cells)?;
        writeln!(
            f,
            "面数: {} (内部: {}, 边界: {})",
            self.n_faces, self.n_interior_faces, self.n_boundary_faces
        )?;
        writeln!(f, "节点数: {}", self.n_nodes)?;
        writeln!(f, "总面积: {:.2}", self.total_area)?;
        writeln!(
            f,
            "单元面积范围: [{:.2}, {:.2}]",
            self.min_cell_area,
            self.max_cell_area
        )?;
        writeln!(
            f,
            "边长范围: [{:.2}, {:.2}]",
            self.min_edge_length,
            self.max_edge_length
        )
    }
}

// =========================================================================
// MeshAccess trait 实现
// =========================================================================

impl<B: Backend> MeshAccess<B> for FrozenMesh<B> {
    #[inline]
    fn backend(&self) -> &B {
        &self.backend
    }

    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn n_faces(&self) -> usize {
        self.n_faces
    }

    #[inline]
    fn n_internal_faces(&self) -> usize {
        self.n_interior_faces
    }

    #[inline]
    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn cell_centroid(&self, cell: usize) -> Point2D {
        self.cell_center[cell]
    }

    #[inline]
    fn cell_area(&self, cell: usize) -> B::Scalar {
        self.cell_area[cell]
    }

    #[inline]
    fn face_centroid(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    #[inline]
    fn face_length(&self, face: usize) -> B::Scalar {
        self.face_length[face]
    }

    #[inline]
    fn face_normal(&self, face: usize) -> Point3D {
        self.face_normal[face]
    }

    #[inline]
    fn node_position(&self, node: usize) -> Point3D {
        self.node_coords[node]
    }

    #[inline]
    fn cell_bed_elevation(&self, cell: usize) -> B::Scalar {
        self.cell_z_bed[cell]
    }

    #[inline]
    fn face_owner(&self, face: usize) -> usize {
        self.face_owner[face] as usize
    }

    #[inline]
    fn face_neighbor(&self, face: usize) -> Option<usize> {
        let n = self.face_neighbor[face];
        if n == u32::MAX {
            None
        } else {
            Some(n as usize)
        }
    }

    #[inline]
    fn boundary_id(&self, face: usize) -> Option<usize> {
        self.face_boundary_id.get(face).and_then(|opt| opt.map(|id| id as usize))
    }

    #[inline]
    fn boundary_name(&self, boundary_id: usize) -> Option<&str> {
        self.boundary_names.get(boundary_id).map(|s| s.as_str())
    }

    #[inline]
    fn cell_face_indices(&self, cell: usize) -> &[u32] {
        let start = self.cell_face_offsets[cell];
        let end = self.cell_face_offsets[cell + 1];
        &self.cell_face_indices[start..end]
    }

    #[inline]
    fn cell_neighbor_indices(&self, cell: usize) -> &[u32] {
        let start = self.cell_neighbor_offsets[cell];
        let end = self.cell_neighbor_offsets[cell + 1];
        &self.cell_neighbor_indices[start..end]
    }

    #[inline]
    fn cell_node_indices(&self, cell: usize) -> &[u32] {
        let start = self.cell_node_offsets[cell];
        let end = self.cell_node_offsets[cell + 1];
        &self.cell_node_indices[start..end]
    }

    #[inline]
    fn all_cell_centroids(&self) -> &[Point2D] {
        &self.cell_center
    }

    /// 所有单元面积（后端标量）
    fn all_cell_areas(&self) -> Vec<B::Scalar> {
        self.cell_area.copy_to_vec()
    }

    /// 所有单元底床高程（后端标量）
    fn all_cell_bed_elevations(&self) -> Vec<B::Scalar> {
        self.cell_z_bed.copy_to_vec()
    }

    #[inline]
    fn face_z_left(&self, face: usize) -> B::Scalar {
        self.face_z_left[face]
    }

    #[inline]
    fn face_z_right(&self, face: usize) -> B::Scalar {
        self.face_z_right[face]
    }
}

impl<B: Backend> MeshTopology<B> for FrozenMesh<B> {
    #[inline]
    fn face_o2n_distance(&self, face: usize) -> B::Scalar {
        self.face_dist_o2n[face]
    }

    #[inline]
    fn face_delta_owner(&self, face: usize) -> Point2D {
        self.face_delta_owner[face]
    }

    #[inline]
    fn face_delta_neighbor(&self, face: usize) -> Point2D {
        self.face_delta_neighbor[face]
    }

    #[inline]
    fn min_cell_size(&self) -> B::Scalar {
        self.min_cell_size
    }

    #[inline]
    fn max_cell_size(&self) -> B::Scalar {
        self.max_cell_size
    }
}

// =========================================================================
// 空间查询与几何算法
// =========================================================================

impl<B: Backend> FrozenMesh<B> {
    // =========================================================================
    // 空间索引创建
    // =========================================================================

    /// 创建网格空间索引（用于点定位）
    pub fn create_spatial_index(&self) -> MeshSpatialIndex {
        MeshSpatialIndex::build(self.n_cells, |i| self.get_cell_vertices(i))
    }

    /// 创建网格定位器（支持泛型精度）
    pub fn create_locator(&self) -> MeshLocator<'_, B> {
        MeshLocator::new(self)
    }

    // =========================================================================
    // 单元几何查询
    // =========================================================================

    /// 获取单元的顶点坐标（2D）
    pub fn get_cell_vertices(&self, cell: usize) -> Vec<Point2D> {
        self.cell_nodes(cell)
            .iter()
            .map(|&node_idx| self.node_xy(node_idx as usize))
            .collect()
    }

    /// 获取单元的顶点坐标（3D）
    pub fn get_cell_vertices_3d(&self, cell: usize) -> Vec<Point3D> {
        self.cell_nodes(cell)
            .iter()
            .map(|&node_idx| self.node_coords(node_idx as usize))
            .collect()
    }

    /// 计算点相对于三角形单元的重心坐标
    pub fn compute_barycentric(&self, cell: usize, point: Point2D) -> Option<(f64, f64, f64)> {
        let nodes = self.cell_nodes(cell);
        if nodes.len() != 3 {
            return None;
        }

        let v0 = self.node_xy(nodes[0] as usize);
        let v1 = self.node_xy(nodes[1] as usize);
        let v2 = self.node_xy(nodes[2] as usize);

        let v0v1 = Point2D::new(v1.x - v0.x, v1.y - v0.y);
        let v0v2 = Point2D::new(v2.x - v0.x, v2.y - v0.y);
        let v0p = Point2D::new(point.x - v0.x, point.y - v0.y);

        let dot00 = v0v1.x * v0v1.x + v0v1.y * v0v1.y;
        let dot01 = v0v1.x * v0v2.x + v0v1.y * v0v2.y;
        let dot02 = v0v1.x * v0p.x + v0v1.y * v0p.y;
        let dot11 = v0v2.x * v0v2.x + v0v2.y * v0v2.y;
        let dot12 = v0v2.x * v0p.x + v0v2.y * v0p.y;

        let denom = dot00 * dot11 - dot01 * dot01;
        if denom.abs() < 1e-12 {
            return None;
        }

        let inv_denom = 1.0 / denom;
        let u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
        let v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
        let w = 1.0 - u - v;

        Some((w, u, v))
    }

    /// 判断点是否在单元内部（射线法）
    pub fn point_in_cell(&self, cell: usize, point: Point2D, tolerance: f64) -> bool {
        let vertices = self.get_cell_vertices(cell);
        let n = vertices.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;

        for i in 0..n {
            let j = (i + 1) % n;
            let vi = &vertices[i];
            let vj = &vertices[j];

            if ((vi.y > point.y) != (vj.y > point.y))
                && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x - tolerance)
            {
                inside = !inside;
            }
        }

        inside
    }

    /// 查找距离点最近的边界面
    pub fn find_nearest_boundary_face(&self, point: Point2D) -> Option<(usize, f64)> {
        if self.n_interior_faces >= self.n_faces {
            return None;
        }

        let mut best_face = None;
        let mut best_dist = f64::INFINITY;

        for face in self.boundary_faces() {
            let center = self.face_center(face);
            let dx = center.x - point.x;
            let dy = center.y - point.y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < best_dist {
                best_dist = dist;
                best_face = Some(face);
            }
        }

        best_face.map(|f| (f, best_dist))
    }

    /// 根据边界名称获取面索引列表
    pub fn get_boundary_faces_by_name(&self, boundary_name: &str) -> Vec<usize> {
        let boundary_id = self.boundary_names.iter().position(|name| name == boundary_name);

        match boundary_id {
            Some(id) => self
                .boundary_faces()
                .filter(|&face| {
                    self.face_boundary_id
                        .get(face)
                        .and_then(|opt| *opt)
                        .map(|bid| bid as usize == id)
                        .unwrap_or(false)
                })
                .collect(),
            None => Vec::new(),
        }
    }

    /// 获取单元的所有有效邻居（排除边界和无效索引）
    pub fn get_valid_neighbors(&self, cell: usize) -> Vec<usize> {
        self.cell_neighbors(cell)
            .iter()
            .filter(|&&n| n != u32::MAX)
            .map(|&n| n as usize)
            .collect()
    }

    /// 计算两单元中心距离
    #[inline]
    pub fn cell_distance(&self, cell1: usize, cell2: usize) -> f64 {
        let c1 = self.cell_center(cell1);
        let c2 = self.cell_center(cell2);
        let dx = c2.x - c1.x;
        let dy = c2.y - c1.y;
        (dx * dx + dy * dy).sqrt()
    }
}

// =========================================================================
// 测试模块
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::{Backend, CpuBackend};

    #[test]
    fn test_empty_frozen_mesh() {
        let backend = CpuBackend::<f64>::new();
        let mesh: FrozenMesh<CpuBackend<f64>> = FrozenMesh::empty_with_backend(backend);
        assert_eq!(mesh.n_cells(), 0);
        assert_eq!(mesh.n_faces(), 0);
        assert_eq!(mesh.n_nodes(), 0);
    }

    #[test]
    fn test_frozen_mesh_f32() {
        let backend = CpuBackend::<f32>::new();
        let mesh: FrozenMesh<CpuBackend<f32>> =
            FrozenMesh::empty_with_cells_backend(backend, 5);
        assert_eq!(mesh.n_cells(), 5);
        assert_eq!(mesh.cell_area(0), f32::ONE);
    }

    #[test]
    fn test_validate_empty() {
        let backend = CpuBackend::<f64>::new();
        let mesh: FrozenMesh<CpuBackend<f64>> = FrozenMesh::empty_with_backend(backend);
        assert!(mesh.validate().is_ok());
    }

    #[test]
    fn test_mesh_access_trait() {
        let backend = CpuBackend::<f64>::new();
        let mesh: FrozenMesh<CpuBackend<f64>> =
            FrozenMesh::empty_with_cells_backend(backend, 5);

        fn check_mesh<B: Backend, M: MeshAccess<B>>(m: &M) -> usize {
            m.n_cells()
        }
        
        assert_eq!(check_mesh(&mesh), 5);
    }

    #[test]
    fn test_statistics_f32() {
        let backend = CpuBackend::<f32>::new();
        let mesh: FrozenMesh<CpuBackend<f32>> =
            FrozenMesh::empty_with_cells_backend(backend, 3);
        let stats = mesh.statistics();
        assert_eq!(stats.n_cells, 3);
        assert_eq!(stats.total_area, 3.0_f32);
    }

    #[test]
    fn test_centroid() {
        let backend = CpuBackend::<f64>::new();
        let mesh: FrozenMesh<CpuBackend<f64>> =
            FrozenMesh::empty_with_cells_backend(backend, 2);
        let mesh = FrozenMesh {
            cell_center: vec![Point2D::new(0.0, 0.0), Point2D::new(2.0, 2.0)],
            cell_area: vec![1.0, 1.0],
            ..mesh
        };
        let centroid = mesh.centroid();
        assert_eq!(centroid.x, 1.0);
        assert_eq!(centroid.y, 1.0);
    }
}