// src-tauri/src/marihydro/domain/mesh/unstructured.rs

//! 非结构化网格实现

use glam::DVec2;
use rstar::{RTree, RTreeObject, AABB};
use smallvec::SmallVec;

use super::{CellId, FaceId, NodeId, INVALID_CELL};
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::traits::mesh::{CellGeometry, FaceGeometry, MeshAccess, MeshTopology};
use crate::marihydro::core::types::{BoundaryIndex, CellIndex, FaceIndex, NodeIndex};

/// 单元的面信息（压缩存储）
#[derive(Debug, Clone, Default)]
pub struct CellFaces {
    faces: SmallVec<[FaceId; 6]>,
    owner_mask: u8,
}

impl CellFaces {
    /// 检查在局部索引处是否为 owner
    #[inline]
    pub fn is_owner(&self, local_idx: usize) -> bool {
        debug_assert!(local_idx < 8, "local_idx 必须小于 8");
        (self.owner_mask >> local_idx) & 1 == 1
    }

    /// 设置 owner 标记
    #[inline]
    pub fn set_owner(&mut self, local_idx: usize, is_owner: bool) {
        debug_assert!(local_idx < 8);
        if is_owner {
            self.owner_mask |= 1 << local_idx;
        } else {
            self.owner_mask &= !(1 << local_idx);
        }
    }

    /// 添加面
    #[inline]
    pub fn push(&mut self, face_id: FaceId, is_owner: bool) {
        let local_idx = self.faces.len();
        debug_assert!(local_idx < 8, "单元面数不能超过 8");
        self.faces.push(face_id);
        self.set_owner(local_idx, is_owner);
    }

    /// 面数量
    #[inline]
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// 获取面列表
    #[inline]
    pub fn faces(&self) -> &[FaceId] {
        &self.faces
    }
}

/// R-Tree 空间索引用的单元包络
#[derive(Debug, Clone)]
pub struct CellEnvelope {
    pub cell_id: CellId,
    pub center: DVec2,
    pub aabb: AABB<[f64; 2]>,
}

impl RTreeObject for CellEnvelope {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.aabb
    }
}

impl rstar::PointDistance for CellEnvelope {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.center.x - point[0];
        let dy = self.center.y - point[1];
        dx * dx + dy * dy
    }
}

/// 非结构化网格
///
/// 字段为 pub 以便直接访问，也提供访问器方法和 MeshAccess trait
#[derive(Debug)]
pub struct UnstructuredMesh {
    // ===== 节点数据 =====
    pub n_nodes: usize,
    pub node_xy: Vec<DVec2>,
    pub node_z: Vec<f64>,

    // ===== 单元数据 =====
    pub n_cells: usize,
    pub cell_center: Vec<DVec2>,
    pub cell_area: Vec<f64>,
    pub cell_z_bed: Vec<f64>,
    pub cell_node_ids: Vec<SmallVec<[NodeId; 4]>>,
    pub cell_faces: Vec<CellFaces>,
    pub cell_neighbors: Vec<SmallVec<[CellId; 6]>>,

    // ===== 面数据 =====
    pub n_faces: usize,
    pub n_interior_faces: usize,
    pub face_center: Vec<DVec2>,
    pub face_normal: Vec<DVec2>,
    pub face_length: Vec<f64>,
    pub face_z_left: Vec<f64>,
    pub face_z_right: Vec<f64>,
    pub face_owner: Vec<usize>,
    pub face_neighbor: Vec<usize>,
    pub face_delta_owner: Vec<DVec2>,
    pub face_delta_neighbor: Vec<DVec2>,
    pub face_dist_o2n: Vec<f64>,

    // ===== 边界数据 =====
    pub boundary_face_indices: Vec<usize>,
    pub boundary_names: Vec<String>,
    pub face_boundary_id: Vec<Option<usize>>,

    // ===== 空间索引 =====
    pub spatial_index: RTree<CellEnvelope>,

    // ===== 缓存的统计信息 =====
    pub min_cell_size: f64,
    pub max_cell_size: f64,
}

impl UnstructuredMesh {
    /// 创建空网格（用于 Builder）
    pub(crate) fn empty() -> Self {
        Self {
            n_nodes: 0,
            node_xy: Vec::new(),
            node_z: Vec::new(),
            n_cells: 0,
            cell_center: Vec::new(),
            cell_area: Vec::new(),
            cell_z_bed: Vec::new(),
            cell_node_ids: Vec::new(),
            cell_faces: Vec::new(),
            cell_neighbors: Vec::new(),
            n_faces: 0,
            n_interior_faces: 0,
            face_center: Vec::new(),
            face_normal: Vec::new(),
            face_length: Vec::new(),
            face_z_left: Vec::new(),
            face_z_right: Vec::new(),
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            face_delta_owner: Vec::new(),
            face_delta_neighbor: Vec::new(),
            face_dist_o2n: Vec::new(),
            boundary_face_indices: Vec::new(),
            boundary_names: Vec::new(),
            face_boundary_id: Vec::new(),
            spatial_index: RTree::new(),
            min_cell_size: f64::MAX,
            max_cell_size: 0.0,
        }
    }

    // ===== 基本访问器 =====

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

    /// 单元底床高程
    #[inline]
    pub fn cell_z_bed(&self, idx: usize) -> f64 {
        self.cell_z_bed[idx]
    }

    /// 单元底床高程数组
    #[inline]
    pub fn cell_z_bed_slice(&self) -> &[f64] {
        &self.cell_z_bed
    }

    /// 节点高程
    #[inline]
    pub fn node_z(&self, idx: usize) -> f64 {
        self.node_z[idx]
    }

    /// 面左侧高程
    #[inline]
    pub fn face_z_left(&self, idx: usize) -> f64 {
        self.face_z_left[idx]
    }

    /// 面右侧高程
    #[inline]
    pub fn face_z_right(&self, idx: usize) -> f64 {
        self.face_z_right[idx]
    }

    /// 面到 owner 的向量
    #[inline]
    pub fn face_delta_owner(&self, idx: usize) -> DVec2 {
        self.face_delta_owner[idx]
    }

    /// 面到 neighbor 的向量
    #[inline]
    pub fn face_delta_neighbor(&self, idx: usize) -> DVec2 {
        self.face_delta_neighbor[idx]
    }

    /// 面两侧单元中心距离
    #[inline]
    pub fn face_dist_o2n(&self, idx: usize) -> f64 {
        self.face_dist_o2n[idx]
    }

    /// 单元的面信息
    #[inline]
    pub fn cell_faces_info(&self, idx: usize) -> &CellFaces {
        &self.cell_faces[idx]
    }

    // ===== 范围迭代器 =====

    /// 内部面范围
    #[inline]
    pub fn interior_faces(&self) -> std::ops::Range<usize> {
        0..self.n_interior_faces
    }

    /// 边界面范围
    #[inline]
    pub fn boundary_faces(&self) -> std::ops::Range<usize> {
        self.n_interior_faces..self.n_faces
    }

    /// 单元范围
    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        0..self.n_cells
    }

    /// 判断是否为边界面
    #[inline]
    pub fn is_boundary_face_idx(&self, face_idx: usize) -> bool {
        face_idx >= self.n_interior_faces
    }

    /// 边界面的边界索引
    #[inline]
    pub fn boundary_index_of_face(&self, face_idx: usize) -> usize {
        debug_assert!(face_idx >= self.n_interior_faces);
        face_idx - self.n_interior_faces
    }

    // ===== 空间查询 =====

    /// 查找包含指定点的单元
    pub fn find_cell_containing(&self, point: DVec2) -> Option<CellId> {
        let query_point = [point.x, point.y];

        for envelope in self.spatial_index.locate_all_at_point(&query_point) {
            let cell_id = envelope.cell_id;
            if self.point_in_cell(point, cell_id) {
                return Some(cell_id);
            }
        }

        None
    }

    /// 查找最近的 N 个单元
    pub fn find_nearest_cells(&self, point: DVec2, count: usize) -> Vec<(CellId, f64)> {
        let query_point = [point.x, point.y];
        self.spatial_index
            .nearest_neighbor_iter(&query_point)
            .take(count)
            .map(|env| {
                let dist = (point - env.center).length();
                (env.cell_id, dist)
            })
            .collect()
    }

    /// 判断点是否在单元内（射线法）
    fn point_in_cell(&self, point: DVec2, cell_id: CellId) -> bool {
        let nodes = &self.cell_node_ids[cell_id.idx()];
        let n = nodes.len();

        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let vi = self.node_xy[nodes[i].idx()];
            let vj = self.node_xy[nodes[j].idx()];

            if ((vi.y > point.y) != (vj.y > point.y))
                && (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }

            j = i;
        }

        inside
    }

    // ===== 验证 =====

    /// 验证网格拓扑完整性
    pub fn validate_topology(&self) -> MhResult<()> {
        // 节点数组长度检查
        if self.node_xy.len() != self.n_nodes {
            return Err(MhError::invalid_mesh(format!(
                "节点坐标数组长度 {} != n_nodes {}",
                self.node_xy.len(),
                self.n_nodes
            )));
        }

        if self.node_z.len() != self.n_nodes {
            return Err(MhError::invalid_mesh(format!(
                "节点高程数组长度 {} != n_nodes {}",
                self.node_z.len(),
                self.n_nodes
            )));
        }

        // 单元数组长度检查
        if self.cell_center.len() != self.n_cells {
            return Err(MhError::invalid_mesh(format!(
                "单元中心数组长度 {} != n_cells {}",
                self.cell_center.len(),
                self.n_cells
            )));
        }

        // 面的 owner/neighbor 检查
        for (idx, &owner) in self.face_owner.iter().enumerate() {
            if owner >= self.n_cells {
                return Err(MhError::invalid_mesh(format!(
                    "面 {} 的 owner {} 超出范围",
                    idx, owner
                )));
            }
        }

        for (idx, &neighbor) in self.face_neighbor.iter().enumerate() {
            if neighbor != INVALID_CELL && neighbor >= self.n_cells {
                return Err(MhError::invalid_mesh(format!(
                    "面 {} 的 neighbor {} 超出范围",
                    idx, neighbor
                )));
            }
        }

        // 单元面索引检查
        for (idx, cf) in self.cell_faces.iter().enumerate() {
            for face_id in cf.faces() {
                if face_id.idx() >= self.n_faces {
                    return Err(MhError::invalid_mesh(format!(
                        "单元 {} 引用的面 {} 超出范围",
                        idx,
                        face_id.idx()
                    )));
                }
            }
        }

        Ok(())
    }

    /// 获取网格统计信息
    pub fn statistics(&self) -> MeshStatistics {
        let mut min_area = f64::MAX;
        let mut max_area = f64::MIN;
        let mut total_area = 0.0;

        for &area in &self.cell_area {
            min_area = min_area.min(area);
            max_area = max_area.max(area);
            total_area += area;
        }

        let mut min_length = f64::MAX;
        let mut max_length = f64::MIN;

        for &len in &self.face_length {
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
}

// ===== 实现 MeshAccess trait =====

impl MeshAccess for UnstructuredMesh {
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    fn n_faces(&self) -> usize {
        self.n_faces
    }

    fn n_internal_faces(&self) -> usize {
        self.n_interior_faces
    }

    fn n_boundary_faces(&self) -> usize {
        self.n_faces - self.n_interior_faces
    }

    fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    fn cell_centroid(&self, cell: CellIndex) -> DVec2 {
        self.cell_center[cell.0]
    }

    fn cell_area(&self, cell: CellIndex) -> f64 {
        self.cell_area[cell.0]
    }

    fn face_centroid(&self, face: FaceIndex) -> DVec2 {
        self.face_center[face.0]
    }

    fn face_length(&self, face: FaceIndex) -> f64 {
        self.face_length[face.0]
    }

    fn face_normal(&self, face: FaceIndex) -> DVec2 {
        self.face_normal[face.0]
    }

    fn node_position(&self, node: NodeIndex) -> DVec2 {
        self.node_xy[node.0]
    }

    fn cell_bed_elevation(&self, cell: CellIndex) -> f64 {
        self.cell_z_bed[cell.0]
    }

    fn face_owner(&self, face: FaceIndex) -> CellIndex {
        CellIndex(self.face_owner[face.0])
    }

    fn face_neighbor(&self, face: FaceIndex) -> CellIndex {
        let neighbor = self.face_neighbor[face.0];
        if neighbor == INVALID_CELL {
            CellIndex::INVALID
        } else {
            CellIndex(neighbor)
        }
    }

    fn cell_faces(&self, cell: CellIndex) -> &[FaceIndex] {
        // 注意：这里需要转换类型，但 FaceId 和 FaceIndex 布局相同
        // 安全地重新解释内存
        let faces = self.cell_faces[cell.0].faces();
        // FaceId(usize) 和 FaceIndex(usize) 内存布局相同
        unsafe { std::slice::from_raw_parts(faces.as_ptr() as *const FaceIndex, faces.len()) }
    }

    fn cell_neighbors(&self, cell: CellIndex) -> &[CellIndex] {
        let neighbors = &self.cell_neighbors[cell.0];
        unsafe {
            std::slice::from_raw_parts(neighbors.as_ptr() as *const CellIndex, neighbors.len())
        }
    }

    fn cell_nodes(&self, cell: CellIndex) -> &[NodeIndex] {
        let nodes = &self.cell_node_ids[cell.0];
        unsafe { std::slice::from_raw_parts(nodes.as_ptr() as *const NodeIndex, nodes.len()) }
    }

    fn boundary_id(&self, face: FaceIndex) -> Option<BoundaryIndex> {
        self.face_boundary_id
            .get(face.0)
            .and_then(|&opt| opt.map(BoundaryIndex))
    }

    fn boundary_name(&self, boundary: BoundaryIndex) -> Option<&str> {
        self.boundary_names.get(boundary.0).map(|s| s.as_str())
    }

    fn all_cell_centroids(&self) -> &[DVec2] {
        &self.cell_center
    }

    fn all_cell_areas(&self) -> &[f64] {
        &self.cell_area
    }
}

impl MeshTopology for UnstructuredMesh {
    fn min_cell_size(&self) -> f64 {
        self.min_cell_size
    }

    fn max_cell_size(&self) -> f64 {
        self.max_cell_size
    }
}

/// 网格统计信息
#[derive(Debug, Clone)]
pub struct MeshStatistics {
    pub n_cells: usize,
    pub n_faces: usize,
    pub n_interior_faces: usize,
    pub n_boundary_faces: usize,
    pub n_nodes: usize,
    pub total_area: f64,
    pub min_cell_area: f64,
    pub max_cell_area: f64,
    pub min_edge_length: f64,
    pub max_edge_length: f64,
}

impl std::fmt::Display for MeshStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 网格统计 ===")?;
        writeln!(f, "单元数: {}", self.n_cells)?;
        writeln!(
            f,
            "面数: {} (内部: {}, 边界: {})",
            self.n_faces, self.n_interior_faces, self.n_boundary_faces
        )?;
        writeln!(f, "节点数: {}", self.n_nodes)?;
        writeln!(f, "总面积: {:.2} m²", self.total_area)?;
        writeln!(
            f,
            "单元面积: [{:.2}, {:.2}] m²",
            self.min_cell_area, self.max_cell_area
        )?;
        writeln!(
            f,
            "边长: [{:.2}, {:.2}] m",
            self.min_edge_length, self.max_edge_length
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_faces() {
        let mut cf = CellFaces::default();
        cf.push(FaceId(0), true);
        cf.push(FaceId(1), false);
        cf.push(FaceId(2), true);

        assert!(cf.is_owner(0));
        assert!(!cf.is_owner(1));
        assert!(cf.is_owner(2));
        assert_eq!(cf.len(), 3);
    }
}
