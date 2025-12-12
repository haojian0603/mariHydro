// marihydro\crates\mh_mesh\src/compat.rs

//! 兼容层
//!
//! 提供 HalfEdgeMesh 和 FrozenMesh 之间的转换。

use crate::frozen::FrozenMesh;
use crate::halfedge::{HalfEdgeMesh, Vertex};
use mh_foundation::index::{FaceIndex, HalfEdgeIndex, VertexIndex};
use mh_geo::{Point2D, Point3D};
use std::collections::HashMap;

/// 从 HalfEdgeMesh 转换为 FrozenMesh
impl<V: Default, F: Default> HalfEdgeMesh<V, F> {
    /// 冻结网格
    ///
    /// 将半边网格转换为只读的 SoA 布局冻结网格。
    /// 冻结后的网格适合用于物理计算。
    pub fn freeze(&self) -> FrozenMesh {
        // 建立顶点索引映射 (VertexIndex -> 连续usize)
        let mut vertex_map: HashMap<VertexIndex, u32> = HashMap::new();
        let mut node_coords = Vec::new();

        for (v_idx, vertex) in self.vertices() {
            let new_idx = node_coords.len() as u32;
            vertex_map.insert(v_idx, new_idx);
            node_coords.push(vertex.position);
        }

        // 建立面索引映射 (FaceIndex -> 连续usize)
        let mut face_map: HashMap<FaceIndex, u32> = HashMap::new();
        let mut cell_center = Vec::new();
        let mut cell_area = Vec::new();
        let mut cell_z_bed = Vec::new();
        let mut cell_node_offsets = vec![0usize];
        let mut cell_node_indices = Vec::new();
        let mut cell_face_offsets = vec![0usize];
        let mut cell_face_indices = Vec::new();
        let mut cell_neighbor_offsets = vec![0usize];
        let mut cell_neighbor_indices = Vec::new();

        for (f_idx, _face) in self.faces() {
            let new_idx = cell_center.len() as u32;
            face_map.insert(f_idx, new_idx);

            // 计算面中心和面积
            let centroid = self.face_centroid(f_idx).unwrap_or(Point2D::new(0.0, 0.0));
            let area = self.face_area(f_idx);

            cell_center.push(centroid);
            cell_area.push(area);

            // 计算底床高程（所有顶点高程的平均值）
            let vertices: Vec<_> = self.face_vertices(f_idx).collect();
            let z_sum: f64 = vertices
                .iter()
                .filter_map(|&v| self.vertex(v).map(|vert| vert.position.z))
                .sum();
            let z_bed = if !vertices.is_empty() {
                z_sum / vertices.len() as f64
            } else {
                0.0
            };
            cell_z_bed.push(z_bed);

            // 收集单元节点
            for v in &vertices {
                if let Some(&node_idx) = vertex_map.get(v) {
                    cell_node_indices.push(node_idx);
                }
            }
            cell_node_offsets.push(cell_node_indices.len());
        }

        // 收集面信息
        // 首先识别所有边和它们的 owner/neighbor
        let mut edge_info: HashMap<(u32, u32), (u32, Option<u32>)> = HashMap::new();

        for (f_idx, _face) in self.faces() {
            let cell_idx = face_map[&f_idx];

            for he_idx in self.face_halfedges(f_idx) {
                if let Some(he) = self.halfedge(he_idx) {
                    let v0 = vertex_map.get(&he.origin).copied().unwrap_or(u32::MAX);
                    let v1 = self
                        .halfedge_target(he_idx)
                        .and_then(|v| vertex_map.get(&v).copied())
                        .unwrap_or(u32::MAX);

                    if v0 == u32::MAX || v1 == u32::MAX {
                        continue;
                    }

                    // 规范化边键（较小顶点在前）
                    let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                    edge_info
                        .entry(edge_key)
                        .and_modify(|(_owner, neighbor)| {
                            if neighbor.is_none() {
                                *neighbor = Some(cell_idx);
                            }
                        })
                        .or_insert((cell_idx, None));
                }
            }
        }

        // 构建面数据
        let mut face_center_vec = Vec::new();
        let mut face_normal_vec = Vec::new();
        let mut face_length_vec = Vec::new();
        let mut face_z_left_vec = Vec::new();
        let mut face_z_right_vec = Vec::new();
        let mut face_owner_vec = Vec::new();
        let mut face_neighbor_vec = Vec::new();
        let mut face_delta_owner_vec = Vec::new();
        let mut face_delta_neighbor_vec = Vec::new();
        let mut face_dist_o2n_vec = Vec::new();
        let mut boundary_face_indices = Vec::new();
        let mut face_boundary_id = Vec::new();

        // 先添加内部面
        let mut face_idx_counter = 0u32;
        let mut edge_to_face: HashMap<(u32, u32), u32> = HashMap::new();

        // 第一遍：内部面
        for (&edge_key, &(owner, neighbor)) in &edge_info {
            if neighbor.is_some() {
                // 内部面
                let (v0, v1) = edge_key;
                let p0 = node_coords[v0 as usize];
                let p1 = node_coords[v1 as usize];

                let center = Point2D::new((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0);
                let dx = p1.x - p0.x;
                let dy = p1.y - p0.y;
                let dz = p1.z - p0.z;
                let length_2d = (dx * dx + dy * dy).sqrt();
                
                // 3D法向量（考虑Z分量）
                let normal = if length_2d > 1e-14 {
                    // 边向量在XY平面的法向量，Z分量表示坡度
                    Point3D::new(dy / length_2d, -dx / length_2d, (dz / length_2d).atan())
                } else {
                    Point3D::new(0.0, 1.0, 0.0)
                };

                let owner_center = cell_center[owner as usize];
                let neighbor_cell = neighbor.unwrap();
                let neighbor_center = cell_center[neighbor_cell as usize];

                let delta_owner = Point2D::new(center.x - owner_center.x, center.y - owner_center.y);
                let delta_neighbor = Point2D::new(center.x - neighbor_center.x, center.y - neighbor_center.y);
                let dist = ((neighbor_center.x - owner_center.x).powi(2)
                    + (neighbor_center.y - owner_center.y).powi(2))
                .sqrt();

                face_center_vec.push(center);
                face_normal_vec.push(normal);
                face_length_vec.push(length_2d);
                face_z_left_vec.push(cell_z_bed[owner as usize]);
                face_z_right_vec.push(cell_z_bed[neighbor_cell as usize]);
                face_owner_vec.push(owner);
                face_neighbor_vec.push(neighbor_cell);
                face_delta_owner_vec.push(delta_owner);
                face_delta_neighbor_vec.push(delta_neighbor);
                face_dist_o2n_vec.push(dist);
                face_boundary_id.push(None);

                edge_to_face.insert(edge_key, face_idx_counter);
                face_idx_counter += 1;
            }
        }

        let n_interior_faces = face_idx_counter as usize;

        // 第二遍：边界面
        for (&edge_key, &(owner, neighbor)) in &edge_info {
            if neighbor.is_none() {
                // 边界面
                let (v0, v1) = edge_key;
                let p0 = node_coords[v0 as usize];
                let p1 = node_coords[v1 as usize];

                let center = Point2D::new((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0);
                let dx = p1.x - p0.x;
                let dy = p1.y - p0.y;
                let dz = p1.z - p0.z;
                let length_2d = (dx * dx + dy * dy).sqrt();
                
                let normal = if length_2d > 1e-14 {
                    Point3D::new(dy / length_2d, -dx / length_2d, (dz / length_2d).atan())
                } else {
                    Point3D::new(0.0, 1.0, 0.0)
                };

                let owner_center = cell_center[owner as usize];
                let delta_owner = Point2D::new(center.x - owner_center.x, center.y - owner_center.y);

                face_center_vec.push(center);
                face_normal_vec.push(normal);
                face_length_vec.push(length_2d);
                face_z_left_vec.push(cell_z_bed[owner as usize]);
                face_z_right_vec.push(cell_z_bed[owner as usize]); // 边界用相同值
                face_owner_vec.push(owner);
                face_neighbor_vec.push(u32::MAX);
                face_delta_owner_vec.push(delta_owner);
                face_delta_neighbor_vec.push(Point2D::new(0.0, 0.0));
                face_dist_o2n_vec.push(0.0);

                boundary_face_indices.push(face_idx_counter);
                face_boundary_id.push(Some(boundary_face_indices.len() as u32 - 1));

                edge_to_face.insert(edge_key, face_idx_counter);
                face_idx_counter += 1;
            }
        }

        // 现在填充单元面和邻居信息
        for (f_idx, _face) in self.faces() {
            let mut faces_for_cell = Vec::new();
            let mut neighbors_for_cell = Vec::new();

            for he_idx in self.face_halfedges(f_idx) {
                if let Some(he) = self.halfedge(he_idx) {
                    let v0 = vertex_map.get(&he.origin).copied().unwrap_or(u32::MAX);
                    let v1 = self
                        .halfedge_target(he_idx)
                        .and_then(|v| vertex_map.get(&v).copied())
                        .unwrap_or(u32::MAX);

                    if v0 == u32::MAX || v1 == u32::MAX {
                        continue;
                    }

                    let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                    if let Some(&face_idx) = edge_to_face.get(&edge_key) {
                        faces_for_cell.push(face_idx);

                        // 找邻居
                        let neighbor = face_neighbor_vec[face_idx as usize];
                        let owner = face_owner_vec[face_idx as usize];
                        let cell_idx = face_map[&f_idx];

                        if neighbor == u32::MAX {
                            neighbors_for_cell.push(u32::MAX);
                        } else if owner == cell_idx {
                            neighbors_for_cell.push(neighbor);
                        } else {
                            neighbors_for_cell.push(owner);
                        }
                    }
                }
            }

            cell_face_indices.extend(faces_for_cell);
            cell_face_offsets.push(cell_face_indices.len());

            cell_neighbor_indices.extend(neighbors_for_cell);
            cell_neighbor_offsets.push(cell_neighbor_indices.len());
        }

        // 计算统计信息
        let mut min_cell_size = f64::MAX;
        let mut max_cell_size = 0.0f64;

        for &area in &cell_area {
            let size = area.sqrt();
            min_cell_size = min_cell_size.min(size);
            max_cell_size = max_cell_size.max(size);
        }

        // 计算单元数量（在移动前提取）
        let n_cells = cell_center.len();

        FrozenMesh {
            n_nodes: node_coords.len(),
            node_coords,
            n_cells,
            cell_center,
            cell_area,
            cell_z_bed,
            cell_node_offsets,
            cell_node_indices,
            cell_face_offsets,
            cell_face_indices,
            cell_neighbor_offsets,
            cell_neighbor_indices,
            n_faces: face_center_vec.len(),
            n_interior_faces,
            face_center: face_center_vec,
            face_normal: face_normal_vec,
            face_length: face_length_vec,
            face_z_left: face_z_left_vec,
            face_z_right: face_z_right_vec,
            face_owner: face_owner_vec,
            face_neighbor: face_neighbor_vec,
            face_delta_owner: face_delta_owner_vec,
            face_delta_neighbor: face_delta_neighbor_vec,
            face_dist_o2n: face_dist_o2n_vec,
            boundary_face_indices,
            boundary_names: Vec::new(),
            face_boundary_id,
            min_cell_size,
            max_cell_size,
            // AMR 预分配字段
            cell_refinement_level: vec![0; n_cells],
            cell_parent: (0..n_cells as u32).collect(),
            ghost_capacity: 0,
            // ID 映射与排列字段（默认为空）
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
        }

    }

}

/// 从 FrozenMesh 创建 HalfEdgeMesh
impl FrozenMesh {
    /// 转换为半边网格
    ///
    /// 将冻结网格转换回可编辑的半边网格。
    pub fn to_halfedge<V: Default + Clone, F: Default + Clone>(
        &self,
    ) -> HalfEdgeMesh<V, F> {
        let mut mesh = HalfEdgeMesh::with_capacity(self.n_nodes, self.n_faces * 3, self.n_cells);

        // 添加所有顶点
        let mut vertex_indices: Vec<VertexIndex> = Vec::with_capacity(self.n_nodes);
        for i in 0..self.n_nodes {
            let pos = self.node_coords[i];
            let v = mesh.add_vertex(Vertex::with_data(
                pos.x,
                pos.y,
                pos.z,
                V::default(),
            ));
            vertex_indices.push(v);
        }

        // 为每个单元创建面
        for cell in 0..self.n_cells {
            let nodes = self.cell_nodes(cell);
            if nodes.len() < 3 {
                continue;
            }

            let vertices: Vec<_> = nodes
                .iter()
                .map(|&n| vertex_indices[n as usize])
                .collect();

            if vertices.len() == 3 {
                mesh.add_triangle(vertices[0], vertices[1], vertices[2]);
            } else if vertices.len() == 4 {
                mesh.add_quad(vertices[0], vertices[1], vertices[2], vertices[3]);
            }
            // 对于更多顶点的多边形，需要更复杂的处理
        }

        // 配对半边设置 twin
        // 需要遍历所有半边，找到共享边
        let mut edge_map: HashMap<(u32, u32), HalfEdgeIndex> = HashMap::new();
        let mut twin_pairs: Vec<(HalfEdgeIndex, HalfEdgeIndex)> = Vec::new();

        // 第一遍：收集所有半边信息并找出配对
        let he_indices: Vec<_> = mesh.halfedge_indices().collect();
        for he_idx in he_indices {
            if let Some(he) = mesh.halfedge(he_idx) {
                let origin = he.origin.index();
                if let Some(target_idx) = mesh.halfedge_target(he_idx) {
                    let target = target_idx.index();

                    // 查找反向边
                    let key = (target, origin);
                    if let Some(&twin_idx) = edge_map.get(&key) {
                        // 记录配对
                        twin_pairs.push((he_idx, twin_idx));
                    } else {
                        edge_map.insert((origin, target), he_idx);
                    }
                }
            }
        }

        // 第二遍：设置 twin 关系
        for (he_idx, twin_idx) in twin_pairs {
            if let Some(h) = mesh.halfedge_mut(he_idx) {
                h.twin = twin_idx;
            }
            if let Some(h) = mesh.halfedge_mut(twin_idx) {
                h.twin = he_idx;
            }
        }

        mesh.clear_dirty();
        mesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_freeze_empty_mesh() {
        let mesh: HalfEdgeMesh<(), ()> = HalfEdgeMesh::new();
        let frozen = mesh.freeze();

        assert_eq!(frozen.n_cells(), 0);
        assert_eq!(frozen.n_nodes(), 0);
        assert_eq!(frozen.n_faces(), 0);
    }

    #[test]
    fn test_freeze_triangle() {
        let mut mesh: HalfEdgeMesh<(), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        mesh.add_triangle(v0, v1, v2);

        let frozen = mesh.freeze();

        assert_eq!(frozen.n_cells(), 1);
        assert_eq!(frozen.n_nodes(), 3);
        assert_eq!(frozen.n_faces(), 3); // 3条边界边

        // 验证面积
        let area = frozen.cell_area(0);
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip() {
        let mut mesh: HalfEdgeMesh<(), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        mesh.add_triangle(v0, v1, v2);

        // 冻结再转回
        let frozen = mesh.freeze();
        let mesh2: HalfEdgeMesh<(), ()> = frozen.to_halfedge();

        assert_eq!(mesh2.n_vertices(), 3);
        assert_eq!(mesh2.n_faces(), 1);
    }
}
