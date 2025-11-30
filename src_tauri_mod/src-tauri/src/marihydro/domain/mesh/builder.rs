// src-tauri/src/marihydro/domain/mesh/builder.rs

//! 网格构建器

use glam::DVec2;
use rstar::{RTree, AABB};
use smallvec::SmallVec;
use std::collections::HashMap;

use super::indices::{CellId, FaceId, NodeId, INVALID_CELL};
use super::unstructured::{CellEnvelope, CellFaces, UnstructuredMesh};
use crate::marihydro::core::error::{MhError, MhResult};

/// 网格构建器
///
/// 使用构建器模式创建网格，确保拓扑完整性
pub struct MeshBuilder {
    // 节点
    nodes: Vec<DVec2>,
    node_z: Vec<f64>,

    // 单元（节点索引列表）
    cells: Vec<Vec<usize>>,

    // 边界定义
    boundary_faces: Vec<BoundaryFaceDefinition>,
    boundary_names: Vec<String>,
}

/// 边界面定义
struct BoundaryFaceDefinition {
    node1: usize,
    node2: usize,
    boundary_id: usize,
}

impl MeshBuilder {
    /// 创建新的构建器
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            node_z: Vec::new(),
            cells: Vec::new(),
            boundary_faces: Vec::new(),
            boundary_names: Vec::new(),
        }
    }

    /// 添加节点
    pub fn add_node(&mut self, x: f64, y: f64, z: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(DVec2::new(x, y));
        self.node_z.push(z);
        idx
    }

    /// 批量添加节点
    pub fn add_nodes(&mut self, coords: &[(f64, f64, f64)]) -> Vec<usize> {
        coords
            .iter()
            .map(|&(x, y, z)| self.add_node(x, y, z))
            .collect()
    }

    /// 添加单元
    pub fn add_cell(&mut self, node_indices: Vec<usize>) -> MhResult<usize> {
        // 验证节点索引
        for &idx in &node_indices {
            if idx >= self.nodes.len() {
                return Err(MhError::invalid_mesh(format!(
                    "单元引用的节点索引 {} 超出范围 (共 {} 个节点)",
                    idx,
                    self.nodes.len()
                )));
            }
        }

        if node_indices.len() < 3 {
            return Err(MhError::invalid_mesh("单元至少需要3个节点"));
        }

        let idx = self.cells.len();
        self.cells.push(node_indices);
        Ok(idx)
    }

    /// 添加边界
    pub fn add_boundary(&mut self, name: &str) -> usize {
        let idx = self.boundary_names.len();
        self.boundary_names.push(name.to_string());
        idx
    }

    /// 标记边界面
    pub fn mark_boundary_edge(
        &mut self,
        node1: usize,
        node2: usize,
        boundary_id: usize,
    ) -> MhResult<()> {
        if node1 >= self.nodes.len() || node2 >= self.nodes.len() {
            return Err(MhError::invalid_mesh("边界边的节点索引超出范围"));
        }
        if boundary_id >= self.boundary_names.len() {
            return Err(MhError::invalid_mesh("边界ID超出范围"));
        }

        self.boundary_faces.push(BoundaryFaceDefinition {
            node1,
            node2,
            boundary_id,
        });

        Ok(())
    }

    /// 构建网格
    pub fn build(self) -> MhResult<UnstructuredMesh> {
        if self.nodes.is_empty() {
            return Err(MhError::invalid_mesh("网格没有节点"));
        }
        if self.cells.is_empty() {
            return Err(MhError::invalid_mesh("网格没有单元"));
        }

        let mut mesh = UnstructuredMesh::empty();

        // 设置节点
        mesh.n_nodes = self.nodes.len();
        mesh.node_xy = self.nodes;
        mesh.node_z = self.node_z;

        // 设置单元基本信息
        mesh.n_cells = self.cells.len();
        mesh.cell_node_ids = self
            .cells
            .iter()
            .map(|nodes| nodes.iter().map(|&i| NodeId(i)).collect())
            .collect();

        // 计算单元几何
        Self::compute_cell_geometry(&mut mesh)?;

        // 构建面拓扑
        Self::build_face_topology(&mut mesh, &self.boundary_faces)?;

        // 设置边界名称
        mesh.boundary_names = self.boundary_names;

        // 计算面几何
        Self::compute_face_geometry(&mut mesh)?;

        // 构建单元邻居列表
        Self::build_cell_neighbors(&mut mesh);

        // 构建空间索引
        Self::build_spatial_index(&mut mesh);

        // 计算尺度范围
        Self::compute_size_range(&mut mesh);

        // 验证拓扑
        mesh.validate_topology()?;

        Ok(mesh)
    }

    /// 计算单元几何属性
    fn compute_cell_geometry(mesh: &mut UnstructuredMesh) -> MhResult<()> {
        mesh.cell_center = Vec::with_capacity(mesh.n_cells);
        mesh.cell_area = Vec::with_capacity(mesh.n_cells);
        mesh.cell_z_bed = Vec::with_capacity(mesh.n_cells);

        for cell_nodes in &mesh.cell_node_ids {
            let n = cell_nodes.len();

            // 计算质心
            let mut cx = 0.0;
            let mut cy = 0.0;
            let mut cz = 0.0;

            for node_id in cell_nodes {
                let pos = mesh.node_xy[node_id.idx()];
                cx += pos.x;
                cy += pos.y;
                cz += mesh.node_z[node_id.idx()];
            }

            mesh.cell_center
                .push(DVec2::new(cx / n as f64, cy / n as f64));
            mesh.cell_z_bed.push(cz / n as f64);

            // 计算面积（Shoelace公式）
            let mut area = 0.0;
            for i in 0..n {
                let j = (i + 1) % n;
                let pi = mesh.node_xy[cell_nodes[i].idx()];
                let pj = mesh.node_xy[cell_nodes[j].idx()];
                area += pi.x * pj.y - pj.x * pi.y;
            }
            mesh.cell_area.push(area.abs() / 2.0);
        }

        Ok(())
    }

    /// 构建面拓扑
    fn build_face_topology(
        mesh: &mut UnstructuredMesh,
        boundary_defs: &[BoundaryFaceDefinition],
    ) -> MhResult<()> {
        // 边到面的映射：(min_node, max_node) -> FaceId
        let mut edge_to_face: HashMap<(usize, usize), FaceId> = HashMap::new();

        // 边界边映射
        let mut boundary_edges: HashMap<(usize, usize), usize> = HashMap::new();
        for def in boundary_defs {
            let key = if def.node1 < def.node2 {
                (def.node1, def.node2)
            } else {
                (def.node2, def.node1)
            };
            boundary_edges.insert(key, def.boundary_id);
        }

        mesh.cell_faces = vec![CellFaces::default(); mesh.n_cells];

        let mut interior_faces = Vec::new();
        let mut boundary_face_list = Vec::new();

        // 遍历所有单元的边
        for (cell_idx, cell_nodes) in mesh.cell_node_ids.iter().enumerate() {
            let n = cell_nodes.len();

            for i in 0..n {
                let j = (i + 1) % n;
                let n1 = cell_nodes[i].idx();
                let n2 = cell_nodes[j].idx();

                let edge_key = if n1 < n2 { (n1, n2) } else { (n2, n1) };

                if let Some(&face_id) = edge_to_face.get(&edge_key) {
                    // 已存在的面，当前单元是 neighbor
                    interior_faces[face_id.idx()].1 = cell_idx;
                    mesh.cell_faces[cell_idx].push(face_id, false);
                } else {
                    // 新面
                    let is_boundary = boundary_edges.contains_key(&edge_key);

                    if is_boundary {
                        let boundary_id = boundary_edges[&edge_key];
                        let face_id = FaceId(interior_faces.len() + boundary_face_list.len());
                        boundary_face_list.push((
                            cell_idx,
                            INVALID_CELL,
                            n1,
                            n2,
                            Some(boundary_id),
                        ));
                        edge_to_face.insert(edge_key, face_id);
                        mesh.cell_faces[cell_idx].push(face_id, true);
                    } else {
                        let face_id = FaceId(interior_faces.len());
                        interior_faces.push((cell_idx, INVALID_CELL, n1, n2));
                        edge_to_face.insert(edge_key, face_id);
                        mesh.cell_faces[cell_idx].push(face_id, true);
                    }
                }
            }
        }

        // 识别未标记的边界面
        for (owner, neighbor, n1, n2) in &interior_faces {
            if *neighbor == INVALID_CELL {
                boundary_face_list.push((*owner, *neighbor, *n1, *n2, None));
            }
        }

        // 移除真正的边界面从内部面列表
        let real_interior: Vec<_> = interior_faces
            .iter()
            .filter(|(_, neighbor, _, _)| *neighbor != INVALID_CELL)
            .cloned()
            .collect();

        mesh.n_interior_faces = real_interior.len();
        mesh.n_faces = real_interior.len() + boundary_face_list.len();

        // 分配面数组
        mesh.face_owner = Vec::with_capacity(mesh.n_faces);
        mesh.face_neighbor = Vec::with_capacity(mesh.n_faces);
        mesh.face_boundary_id = vec![None; mesh.n_faces];

        // 先添加内部面
        for (owner, neighbor, _, _) in &real_interior {
            mesh.face_owner.push(*owner);
            mesh.face_neighbor.push(*neighbor);
        }

        // 再添加边界面
        for (owner, _, _, _, boundary_id) in &boundary_face_list {
            let face_idx = mesh.face_owner.len();
            mesh.face_owner.push(*owner);
            mesh.face_neighbor.push(INVALID_CELL);
            mesh.face_boundary_id[face_idx] = *boundary_id;
        }

        mesh.boundary_face_indices = (mesh.n_interior_faces..mesh.n_faces).collect();

        Ok(())
    }

    /// 计算面几何属性
    fn compute_face_geometry(mesh: &mut UnstructuredMesh) -> MhResult<()> {
        mesh.face_center = vec![DVec2::ZERO; mesh.n_faces];
        mesh.face_normal = vec![DVec2::ZERO; mesh.n_faces];
        mesh.face_length = vec![0.0; mesh.n_faces];
        mesh.face_z_left = vec![0.0; mesh.n_faces];
        mesh.face_z_right = vec![0.0; mesh.n_faces];
        mesh.face_delta_owner = vec![DVec2::ZERO; mesh.n_faces];
        mesh.face_delta_neighbor = vec![DVec2::ZERO; mesh.n_faces];
        mesh.face_dist_o2n = vec![0.0; mesh.n_faces];

        // 需要重建边信息来计算面几何
        // 简化处理：遍历单元面，根据单元节点计算
        for cell_idx in 0..mesh.n_cells {
            let cell_nodes = &mesh.cell_node_ids[cell_idx];
            let n = cell_nodes.len();

            let cf = &mesh.cell_faces[cell_idx];
            for (local_idx, face_id) in cf.faces().iter().enumerate() {
                let face_idx = face_id.idx();

                // 找到对应的边
                // 假设面顺序与边顺序一致
                let i = local_idx;
                let j = (i + 1) % n;
                let n1 = cell_nodes[i].idx();
                let n2 = cell_nodes[j].idx();

                let p1 = mesh.node_xy[n1];
                let p2 = mesh.node_xy[n2];
                let z1 = mesh.node_z[n1];
                let z2 = mesh.node_z[n2];

                // 面中心
                mesh.face_center[face_idx] = (p1 + p2) * 0.5;

                // 面长度
                let edge = p2 - p1;
                mesh.face_length[face_idx] = edge.length();

                // 外法向量（从 owner 指向外）
                let normal = DVec2::new(-edge.y, edge.x).normalize();

                // 确保法向量指向 owner 外侧
                let owner_center = mesh.cell_center[mesh.face_owner[face_idx]];
                let to_face = mesh.face_center[face_idx] - owner_center;

                if normal.dot(to_face) < 0.0 {
                    mesh.face_normal[face_idx] = -normal;
                } else {
                    mesh.face_normal[face_idx] = normal;
                }

                // 面高程
                mesh.face_z_left[face_idx] = (z1 + z2) / 2.0;
                mesh.face_z_right[face_idx] = (z1 + z2) / 2.0;

                // 计算距离和向量
                let owner_idx = mesh.face_owner[face_idx];
                mesh.face_delta_owner[face_idx] =
                    mesh.face_center[face_idx] - mesh.cell_center[owner_idx];

                let neighbor_idx = mesh.face_neighbor[face_idx];
                if neighbor_idx != INVALID_CELL {
                    mesh.face_delta_neighbor[face_idx] =
                        mesh.face_center[face_idx] - mesh.cell_center[neighbor_idx];
                    mesh.face_dist_o2n[face_idx] =
                        (mesh.cell_center[neighbor_idx] - mesh.cell_center[owner_idx]).length();
                }
            }
        }

        Ok(())
    }

    /// 构建单元邻居列表
    fn build_cell_neighbors(mesh: &mut UnstructuredMesh) {
        mesh.cell_neighbors = vec![SmallVec::new(); mesh.n_cells];

        for face_idx in 0..mesh.n_interior_faces {
            let owner = mesh.face_owner[face_idx];
            let neighbor = mesh.face_neighbor[face_idx];

            mesh.cell_neighbors[owner].push(CellId(neighbor));
            mesh.cell_neighbors[neighbor].push(CellId(owner));
        }
    }

    /// 构建空间索引
    fn build_spatial_index(mesh: &mut UnstructuredMesh) {
        let envelopes: Vec<CellEnvelope> = (0..mesh.n_cells)
            .map(|idx| {
                let center = mesh.cell_center[idx];
                let nodes = &mesh.cell_node_ids[idx];

                let mut min_x = f64::MAX;
                let mut min_y = f64::MAX;
                let mut max_x = f64::MIN;
                let mut max_y = f64::MIN;

                for node_id in nodes {
                    let pos = mesh.node_xy[node_id.idx()];
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

        mesh.spatial_index = RTree::bulk_load(envelopes);
    }

    /// 计算尺度范围
    fn compute_size_range(mesh: &mut UnstructuredMesh) {
        mesh.min_cell_size = f64::MAX;
        mesh.max_cell_size = 0.0;

        for &area in &mesh.cell_area {
            let size = area.sqrt();
            mesh.min_cell_size = mesh.min_cell_size.min(size);
            mesh.max_cell_size = mesh.max_cell_size.max(size);
        }
    }
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_triangle() {
        let mut builder = MeshBuilder::new();

        // 单个三角形
        builder.add_node(0.0, 0.0, 0.0);
        builder.add_node(1.0, 0.0, 0.0);
        builder.add_node(0.5, 1.0, 0.0);

        builder.add_cell(vec![0, 1, 2]).unwrap();

        let mesh = builder.build().unwrap();

        assert_eq!(mesh.n_cells(), 1);
        assert_eq!(mesh.n_nodes(), 3);
        assert!(mesh.n_faces() > 0);
    }

    #[test]
    fn test_build_two_triangles() {
        let mut builder = MeshBuilder::new();

        // 两个共享边的三角形
        //   2
        //  /|\
        // 0-+-1
        //  \|/
        //   3
        builder.add_node(0.0, 0.0, 0.0); // 0
        builder.add_node(1.0, 0.0, 0.0); // 1
        builder.add_node(0.5, 0.5, 0.0); // 2
        builder.add_node(0.5, -0.5, 0.0); // 3

        builder.add_cell(vec![0, 1, 2]).unwrap();
        builder.add_cell(vec![0, 3, 1]).unwrap();

        let mesh = builder.build().unwrap();

        assert_eq!(mesh.n_cells(), 2);
        assert_eq!(mesh.n_interior_faces(), 1); // 共享边
    }
}
