//! 非结构化网格定义

use glam::DVec2;
use rstar::{RTree, RTreeObject, AABB};
use smallvec::SmallVec;

use super::indices::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BoundaryKind {
    Wall = 0,
    OpenSea = 1,
    RiverInflow = 2,
    Outflow = 3,
    Symmetry = 4,
}

impl Default for BoundaryKind {
    fn default() -> Self {
        Self::Wall
    }
}

#[derive(Debug, Clone, Default)]
pub struct CellFaces {
    pub faces: SmallVec<[FaceId; 6]>,
    pub owner_mask: u8,
}

impl CellFaces {
    #[inline(always)]
    pub fn is_owner(&self, local_idx: usize) -> bool {
        debug_assert!(local_idx < 8, "local_idx 必须小于 8");
        (self.owner_mask >> local_idx) & 1 == 1
    }

    #[inline(always)]
    pub fn set_owner(&mut self, local_idx: usize, is_owner: bool) {
        debug_assert!(local_idx < 8);
        if is_owner {
            self.owner_mask |= 1 << local_idx;
        } else {
            self.owner_mask &= !(1 << local_idx);
        }
    }

    #[inline]
    pub fn push(&mut self, face_id: FaceId, is_owner: bool) {
        let local_idx = self.faces.len();
        debug_assert!(local_idx < 8, "单元面数不能超过 8");
        self.faces.push(face_id);
        self.set_owner(local_idx, is_owner);
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.faces.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }
}

#[derive(Debug)]
pub struct UnstructuredMesh {
    pub n_nodes: usize,
    pub node_xy: Vec<DVec2>,
    pub node_z: Vec<f64>,

    pub n_cells: usize,
    pub cell_center: Vec<DVec2>,
    pub cell_area: Vec<f64>,
    pub cell_z_bed: Vec<f64>,
    pub cell_node_ids: Vec<SmallVec<[NodeId; 4]>>,

    pub cell_faces: Vec<CellFaces>,

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

    pub bc_kind: Vec<BoundaryKind>,
    pub bc_value_h: Vec<f64>,
    pub bc_value_q: Vec<f64>,
    pub bc_forcing_id: Vec<Option<usize>>,

    pub spatial_index: RTree<CellEnvelope>,
}

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

impl UnstructuredMesh {
    pub fn new() -> Self {
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
            bc_kind: Vec::new(),
            bc_value_h: Vec::new(),
            bc_value_q: Vec::new(),
            bc_forcing_id: Vec::new(),
            spatial_index: RTree::new(),
        }
    }

    #[inline(always)]
    pub fn is_boundary_face(&self, face_idx: usize) -> bool {
        face_idx >= self.n_interior_faces
    }

    #[inline(always)]
    pub fn boundary_index(&self, face_idx: usize) -> usize {
        debug_assert!(face_idx >= self.n_interior_faces);
        face_idx - self.n_interior_faces
    }

    #[inline]
    pub fn interior_faces(&self) -> std::ops::Range<usize> {
        0..self.n_interior_faces
    }

    #[inline]
    pub fn boundary_faces(&self) -> std::ops::Range<usize> {
        self.n_interior_faces..self.n_faces
    }

    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        0..self.n_cells
    }

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

    pub fn build_spatial_index(&mut self) {
        let envelopes: Vec<CellEnvelope> = (0..self.n_cells)
            .map(|idx| {
                let center = self.cell_center[idx];
                let nodes = &self.cell_node_ids[idx];

                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;

                for &nid in nodes.iter() {
                    let pos = self.node_xy[nid.idx()];
                    min_x = min_x.min(pos.x);
                    min_y = min_y.min(pos.y);
                    max_x = max_x.max(pos.x);
                    max_y = max_y.max(pos.y);
                }

                CellEnvelope {
                    cell_id: CellId(idx),
                    center,
                    aabb: AABB::from_corners([min_x, min_y], [max_x, max_y]),
                }
            })
            .collect();

        self.spatial_index = RTree::bulk_load(envelopes);
    }

    pub fn validate_topology(&self) -> Result<(), String> {
        if self.node_xy.len() != self.n_nodes {
            return Err(format!(
                "节点坐标数组长度 {} != n_nodes {}",
                self.node_xy.len(),
                self.n_nodes
            ));
        }

        if self.node_z.len() != self.n_nodes {
            return Err(format!(
                "节点高程数组长度 {} != n_nodes {}",
                self.node_z.len(),
                self.n_nodes
            ));
        }

        if self.cell_center.len() != self.n_cells {
            return Err(format!(
                "单元中心数组长度 {} != n_cells {}",
                self.cell_center.len(),
                self.n_cells
            ));
        }

        let n_bc = self.n_faces - self.n_interior_faces;
        if self.bc_kind.len() != n_bc {
            return Err(format!(
                "边界条件数组长度 {} != 边界面数 {}",
                self.bc_kind.len(),
                n_bc
            ));
        }

        if self.bc_value_h.len() != n_bc {
            return Err(format!(
                "边界水位数组长度 {} != 边界面数 {}",
                self.bc_value_h.len(),
                n_bc
            ));
        }

        for (idx, cf) in self.cell_faces.iter().enumerate() {
            for &face_id in &cf.faces {
                if face_id.idx() >= self.n_faces {
                    return Err(format!("单元 {} 引用的面 {} 超出范围", idx, face_id.idx()));
                }
            }
        }

        for (idx, &owner) in self.face_owner.iter().enumerate() {
            if owner >= self.n_cells {
                return Err(format!("面 {} 的 owner {} 超出范围", idx, owner));
            }
        }

        for (idx, &neighbor) in self.face_neighbor.iter().enumerate() {
            if neighbor != INVALID_CELL && neighbor >= self.n_cells {
                return Err(format!("面 {} 的 neighbor {} 超出范围", idx, neighbor));
            }
        }

        Ok(())
    }

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

impl Default for UnstructuredMesh {
    fn default() -> Self {
        Self::new()
    }
}

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
    fn test_cell_faces_owner_mask() {
        let mut cf = CellFaces::default();
        cf.push(FaceId(0), true);
        cf.push(FaceId(1), false);
        cf.push(FaceId(2), true);

        assert!(cf.is_owner(0));
        assert!(!cf.is_owner(1));
        assert!(cf.is_owner(2));
        assert_eq!(cf.len(), 3);
    }

    #[test]
    fn test_boundary_index() {
        let mut mesh = UnstructuredMesh::new();
        mesh.n_faces = 100;
        mesh.n_interior_faces = 80;

        assert!(!mesh.is_boundary_face(50));
        assert!(mesh.is_boundary_face(85));
        assert_eq!(mesh.boundary_index(85), 5);
    }

    #[test]
    fn test_point_in_triangle() {
        let mut mesh = UnstructuredMesh::new();
        mesh.n_nodes = 3;
        mesh.node_xy = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(1.0, 0.0),
            DVec2::new(0.5, 1.0),
        ];
        mesh.n_cells = 1;
        mesh.cell_node_ids = vec![smallvec::smallvec![NodeId(0), NodeId(1), NodeId(2)]];

        assert!(mesh.point_in_cell(DVec2::new(0.5, 0.3), CellId(0)));
        assert!(!mesh.point_in_cell(DVec2::new(1.5, 0.5), CellId(0)));
    }
}
