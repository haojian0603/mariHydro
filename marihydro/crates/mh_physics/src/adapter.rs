// marihydro\crates\mh_physics\src/adapter.rs

//! 网格适配层
//!
//! 将 FrozenMesh 适配为物理引擎所需的接口格式。
//! 这是新旧网格结构之间的桥梁。
//!
//! # 设计原则
//!
//! 1. **零拷贝**: 尽可能引用原始数据，避免复制
//! 2. **类型转换**: 处理 Point2D/Point3D 与 DVec2 的转换
//! 3. **索引转换**: 处理 u32 与 usize 的转换
//!
//! # 示例
//!
//! ```ignore
//! use mh_mesh::FrozenMesh;
//! use mh_physics::adapter::PhysicsMesh;
//!
//! let frozen = mesh.freeze();
//! let physics_mesh = PhysicsMesh::from_frozen(&frozen);
//!
//! // 使用物理引擎接口
//! let area = physics_mesh.cell_area(0);
//! let normal = physics_mesh.face_normal(0);
//! ```

use glam::DVec2;
use mh_mesh::FrozenMesh;
use std::sync::Arc;

/// 无效单元索引常量
pub const INVALID_CELL: usize = usize::MAX;

/// 物理引擎网格适配器
///
/// 包装 FrozenMesh，提供物理引擎所需的接口。
#[derive(Debug, Clone)]
pub struct PhysicsMesh {
    /// 内部 FrozenMesh 引用
    inner: Arc<FrozenMesh>,
}

impl PhysicsMesh {
    /// 从 FrozenMesh 创建
    pub fn new(frozen: Arc<FrozenMesh>) -> Self {
        Self { inner: frozen }
    }

    /// 从 FrozenMesh 引用创建（会克隆）
    pub fn from_frozen(frozen: &FrozenMesh) -> Self {
        Self {
            inner: Arc::new(frozen.clone()),
        }
    }

    /// 创建空网格（用于测试）
    pub fn empty(n_cells: usize) -> Self {
        Self {
            inner: Arc::new(FrozenMesh::empty_with_cells(n_cells)),
        }
    }

    /// 获取内部 FrozenMesh
    pub fn inner(&self) -> &FrozenMesh {
        &self.inner
    }

    // =========================================================================
    // 基本统计
    // =========================================================================

    /// 节点数量
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.inner.n_nodes
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.inner.n_cells
    }

    /// 面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.inner.n_faces
    }

    /// 内部面数量
    #[inline]
    pub fn n_interior_faces(&self) -> usize {
        self.inner.n_interior_faces
    }

    /// 边界面数量
    #[inline]
    pub fn n_boundary_faces(&self) -> usize {
        self.inner.n_faces - self.inner.n_interior_faces
    }

    // =========================================================================
    // 单元访问
    // =========================================================================

    /// 获取单元中心 (DVec2)
    #[inline]
    pub fn cell_center(&self, cell: usize) -> DVec2 {
        let p = self.inner.cell_center[cell];
        DVec2::new(p.x, p.y)
    }

    /// 获取单元底床高程
    #[inline]
    pub fn cell_z_bed(&self, cell: usize) -> f64 {
        self.inner.cell_z_bed[cell]
    }

    /// 获取单元底床高程数组
    #[inline]
    pub fn cell_z_bed_slice(&self) -> &[f64] {
        &self.inner.cell_z_bed
    }

    /// 安全获取单元面积（带边界检查）
    #[inline]
    pub fn cell_area(&self, cell: usize) -> Option<f64> {
        self.inner.cell_area.get(cell).copied()
    }

    /// 获取单元面积（不安全，无边界检查）
    #[inline]
    pub fn cell_area_unchecked(&self, cell: usize) -> f64 {
        self.inner.cell_area[cell]
    }

    /// 获取单元周长（水力直径计算用）
    ///
    /// 利用 cell_faces 快速计算，复杂度 O(cell_faces_count)
    #[inline]
    pub fn cell_perimeter(&self, cell: usize) -> Option<f64> {
        let faces = self.inner.cell_faces(cell);
        if faces.is_empty() {
            return None;
        }
        
        let perimeter: f64 = faces
            .iter()
            .map(|&face_id| self.inner.face_length[face_id as usize])
            .sum();
        
        if perimeter > 0.0 {
            Some(perimeter)
        } else {
            None
        }
    }

    /// 获取单元的所有面索引
    ///
    /// 返回该单元的所有关联面 ID 列表
    #[inline]
    pub fn cell_faces(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        self.inner.cell_faces(cell).iter().map(|&f| f as usize)
    }

    /// 获取单元的邻居单元索引
    ///
    /// 返回所有与该单元共享面的邻居单元（不包含 INVALID_CELL）
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        self.inner.cell_neighbors(cell).iter().filter_map(|&n| {
            if n == u32::MAX {
                None
            } else {
                Some(n as usize)
            }
        })
    }

    /// 获取单元的节点索引
    #[inline]
    pub fn cell_nodes(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        self.inner.cell_nodes(cell).iter().map(|&n| n as usize)
    }

    // =========================================================================
    // 面访问
    // =========================================================================

    /// 获取面中心 (DVec2)
    #[inline]
    pub fn face_center(&self, face: usize) -> DVec2 {
        let p = self.inner.face_center[face];
        DVec2::new(p.x, p.y)
    }

    /// 获取面法向量 (2D, DVec2)
    ///
    /// 注意：FrozenMesh 存储的是 3D 法向量，这里只返回 xy 分量
    #[inline]
    pub fn face_normal(&self, face: usize) -> DVec2 {
        let n = self.inner.face_normal[face];
        DVec2::new(n.x, n.y)
    }

    /// 获取面法向量 (3D)
    #[inline]
    pub fn face_normal_3d(&self, face: usize) -> (f64, f64, f64) {
        let n = self.inner.face_normal[face];
        (n.x, n.y, n.z)
    }

    /// 获取面长度
    #[inline]
    pub fn face_length(&self, face: usize) -> f64 {
        self.inner.face_length[face]
    }

    /// 获取面 owner 单元索引
    #[inline]
    pub fn face_owner(&self, face: usize) -> usize {
        self.inner.face_owner[face] as usize
    }

    /// 获取面 neighbor 单元索引
    ///
    /// 如果是边界面，返回 None
    #[inline]
    pub fn face_neighbor(&self, face: usize) -> Option<usize> {
        let n = self.inner.face_neighbor[face];
        if n == u32::MAX {
            None
        } else {
            Some(n as usize)
        }
    }

    /// 获取面 neighbor 单元索引（返回原始值）
    ///
    /// 如果是边界面，返回 INVALID_CELL
    #[inline]
    pub fn face_neighbor_raw(&self, face: usize) -> usize {
        let n = self.inner.face_neighbor[face];
        if n == u32::MAX {
            INVALID_CELL
        } else {
            n as usize
        }
    }

    /// 检查 neighbor 是否有效
    #[inline]
    pub fn has_neighbor(&self, face: usize) -> bool {
        self.inner.face_neighbor[face] != u32::MAX
    }

    /// 获取面左侧高程
    #[inline]
    pub fn face_z_left(&self, face: usize) -> f64 {
        self.inner.face_z_left[face]
    }

    /// 获取面右侧高程
    #[inline]
    pub fn face_z_right(&self, face: usize) -> f64 {
        self.inner.face_z_right[face]
    }

    /// 获取面到 owner 的向量
    #[inline]
    pub fn face_delta_owner(&self, face: usize) -> DVec2 {
        let d = self.inner.face_delta_owner[face];
        DVec2::new(d.x, d.y)
    }

    /// 获取面到 neighbor 的向量
    #[inline]
    pub fn face_delta_neighbor(&self, face: usize) -> DVec2 {
        let d = self.inner.face_delta_neighbor[face];
        DVec2::new(d.x, d.y)
    }

    /// 获取 owner 到 neighbor 的距离
    #[inline]
    pub fn face_dist_o2n(&self, face: usize) -> f64 {
        self.inner.face_dist_o2n[face]
    }

    /// 获取面距离（owner 到 neighbor 或 owner 到边界）
    ///
    /// 对于内部面，返回 owner 到 neighbor 的距离。
    /// 对于边界面，返回 owner 到边界的距离（使用 face_dist_o2n）。
    #[inline]
    pub fn face_distance(&self, face: usize) -> Option<f64> {
        let dist = self.inner.face_dist_o2n[face];
        if dist > 1e-14 {
            Some(dist)
        } else {
            None
        }
    }

    /// 判断是否为边界面
    #[inline]
    pub fn is_boundary_face(&self, face: usize) -> bool {
        face >= self.inner.n_interior_faces
    }

    // =========================================================================
    // 节点访问
    // =========================================================================

    /// 获取节点坐标 (2D, DVec2)
    #[inline]
    pub fn node_xy(&self, node: usize) -> DVec2 {
        let p = self.inner.node_coords[node];
        DVec2::new(p.x, p.y)
    }

    /// 获取节点高程
    #[inline]
    pub fn node_z(&self, node: usize) -> f64 {
        self.inner.node_coords[node].z
    }

    // =========================================================================
    // 范围迭代
    // =========================================================================

    /// 内部面索引范围
    #[inline]
    pub fn interior_faces(&self) -> std::ops::Range<usize> {
        0..self.inner.n_interior_faces
    }

    /// 边界面索引范围
    #[inline]
    pub fn boundary_faces(&self) -> std::ops::Range<usize> {
        self.inner.n_interior_faces..self.inner.n_faces
    }

    /// 单元索引范围
    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        0..self.inner.n_cells
    }

    /// 面索引范围
    #[inline]
    pub fn faces(&self) -> std::ops::Range<usize> {
        0..self.inner.n_faces
    }

    // =========================================================================
    // 统计信息
    // =========================================================================

    /// 最小单元尺寸
    #[inline]
    pub fn min_cell_size(&self) -> f64 {
        self.inner.min_cell_size
    }

    /// 最大单元尺寸
    #[inline]
    pub fn max_cell_size(&self) -> f64 {
        self.inner.max_cell_size
    }
}

/// 单元索引类型（用于物理引擎内部）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellIndex(pub usize);

impl CellIndex {
    /// 无效索引
    pub const INVALID: Self = Self(INVALID_CELL);

    /// 是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0 != INVALID_CELL
    }

    /// 是否无效
    #[inline]
    pub fn is_invalid(&self) -> bool {
        self.0 == INVALID_CELL
    }
}

impl From<usize> for CellIndex {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<CellIndex> for usize {
    fn from(idx: CellIndex) -> Self {
        idx.0
    }
}

/// 面索引类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FaceIndex(pub usize);

impl From<usize> for FaceIndex {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<FaceIndex> for usize {
    fn from(idx: FaceIndex) -> Self {
        idx.0
    }
}

/// 节点索引类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIndex(pub usize);

impl From<usize> for NodeIndex {
    fn from(idx: usize) -> Self {
        Self(idx)
    }
}

impl From<NodeIndex> for usize {
    fn from(idx: NodeIndex) -> Self {
        idx.0
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_index() {
        let idx = CellIndex(5);
        assert!(idx.is_valid());
        assert_eq!(idx.0, 5);

        let invalid = CellIndex::INVALID;
        assert!(invalid.is_invalid());
    }

    #[test]
    fn test_physics_mesh_from_empty() {
        let frozen = FrozenMesh::empty();
        let mesh = PhysicsMesh::from_frozen(&frozen);

        assert_eq!(mesh.n_cells(), 0);
        assert_eq!(mesh.n_faces(), 0);
        assert_eq!(mesh.n_nodes(), 0);
    }
}
