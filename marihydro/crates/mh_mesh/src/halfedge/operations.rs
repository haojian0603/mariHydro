// marihydro\crates\mh_mesh\src/halfedge/operations.rs

//! 半边网格拓扑操作
//!
//! 提供边分裂、边折叠、边翻转等 O(1) 拓扑操作。

use super::mesh::{HalfEdge, HalfEdgeMesh, Vertex};
use mh_foundation::index::{FaceIndex, HalfEdgeIndex, VertexIndex};

/// 拓扑操作结果
#[derive(Debug, Clone)]
pub enum TopologyResult {
    /// 操作成功
    Success,
    /// 操作失败：无效的输入
    InvalidInput,
    /// 操作失败：违反拓扑约束
    TopologyViolation,
    /// 操作失败：边界边
    BoundaryEdge,
}

impl<V: Default + Clone, F: Default + Clone> HalfEdgeMesh<V, F> {
    /// 分裂边
    ///
    /// 在边的中点插入一个新顶点，将边分成两段。
    /// 支持边界边（没有 twin）和内部边（有 twin）。
    ///
    /// # 参数
    /// - `he`: 要分裂的半边
    ///
    /// # 返回
    /// 新创建的顶点索引
    ///
    /// ```text
    /// Before:        After:
    ///   v0 ──he──> v1    v0 ──he──> v_new ──he2──> v1
    ///   v1 <──tw── v0    v1 <──tw2── v_new <──tw── v0 (如果有 twin)
    /// ```
    pub fn split_edge(&mut self, he: HalfEdgeIndex) -> Option<VertexIndex> {
        let he_data = self.halfedge(he)?.clone();
        
        // 获取边的两个端点
        let v0 = he_data.origin;
        let v1_idx = self.halfedge_target(he)?;
        
        let p0 = self.vertex(v0)?.position;
        let p1 = self.vertex(v1_idx)?.position;

        // 计算新顶点位置（边中点）
        let mid_x = (p0.x + p1.x) / 2.0;
        let mid_y = (p0.y + p1.y) / 2.0;
        let mid_z = (p0.z + p1.z) / 2.0;

        // 创建新顶点
        let v_new = self.add_vertex(Vertex::new(mid_x, mid_y, mid_z));

        // 创建新的半边 he2 (v_new -> v1)
        let he2 = self.add_halfedge(HalfEdge::new(v_new));

        // 设置 he2 (v_new -> v1)
        if let Some(h) = self.halfedge_mut(he2) {
            h.next = he_data.next;
            h.prev = he;
            h.face = he_data.face;
        }

        // 更新原始半边 he
        if let Some(h) = self.halfedge_mut(he) {
            h.next = he2;
            // origin 不变，还是 v0
        }

        // 更新 next 的 prev 指针
        if he_data.next.is_valid() {
            if let Some(h) = self.halfedge_mut(he_data.next) {
                h.prev = he2;
            }
        }

        // 设置新顶点的出发半边
        if let Some(v) = self.vertex_mut(v_new) {
            v.halfedge = he2;
        }

        // 处理 twin（如果存在）
        let twin = he_data.twin;
        if twin.is_valid() {
            if let Some(twin_data) = self.halfedge(twin).cloned() {
                // 创建 tw2 (v1 -> v_new)
                let tw2 = self.add_halfedge(HalfEdge::new(v1_idx));

                // 设置 tw2 (v1 -> v_new)
                if let Some(h) = self.halfedge_mut(tw2) {
                    h.next = twin;
                    h.prev = twin_data.prev;
                    h.face = twin_data.face;
                    h.twin = he2;
                }

                // 设置 he2 的 twin
                if let Some(h) = self.halfedge_mut(he2) {
                    h.twin = tw2;
                }

                // 更新 twin (v_new -> v0)
                if let Some(h) = self.halfedge_mut(twin) {
                    h.origin = v_new;
                    h.prev = tw2;
                }

                // 更新 twin_data.prev 的 next 指针
                if twin_data.prev.is_valid() {
                    if let Some(h) = self.halfedge_mut(twin_data.prev) {
                        h.next = tw2;
                    }
                }

                // 标记 twin 相关面为脏
                if twin_data.face.is_valid() {
                    self.mark_face_dirty(twin_data.face);
                }
            }
        }

        // 标记相关面为脏
        if he_data.face.is_valid() {
            self.mark_face_dirty(he_data.face);
        }

        Some(v_new)
    }

    /// 翻转边
    ///
    /// 仅适用于两个三角形之间的内部边。
    /// 将连接两个三角形对角的边替换为连接另一对角的边。
    ///
    /// ```text
    /// Before:             After:
    ///     v2                  v2
    ///    /|\                 / \
    ///   / | \               /   \
    ///  v0-he->v1    =>     v0    v1
    ///   \ | /               \   /
    ///    \|/                 \ /
    ///     v3                  v3
    /// ```
    pub fn flip_edge(&mut self, he: HalfEdgeIndex) -> TopologyResult {
        // 获取半边数据
        let he_data = match self.halfedge(he) {
            Some(h) => h.clone(),
            None => return TopologyResult::InvalidInput,
        };

        let twin = he_data.twin;
        if twin.is_invalid() {
            return TopologyResult::BoundaryEdge;
        }

        let twin_data = match self.halfedge(twin) {
            Some(h) => h.clone(),
            None => return TopologyResult::InvalidInput,
        };

        // 检查是否为边界边
        if he_data.face.is_invalid() || twin_data.face.is_invalid() {
            return TopologyResult::BoundaryEdge;
        }

        // 检查两边是否都是三角形
        let he_next = match self.halfedge(he_data.next) {
            Some(h) => h.clone(),
            None => return TopologyResult::InvalidInput,
        };
        let he_prev = match self.halfedge(he_data.prev) {
            Some(h) => h.clone(),
            None => return TopologyResult::InvalidInput,
        };
        let tw_next = match self.halfedge(twin_data.next) {
            Some(h) => h.clone(),
            None => return TopologyResult::InvalidInput,
        };
        let tw_prev = match self.halfedge(twin_data.prev) {
            Some(h) => h.clone(),
            None => return TopologyResult::InvalidInput,
        };

        // 确保是三角形 (next.next.next == self)
        if he_next.next != he_data.prev || tw_next.next != twin_data.prev {
            return TopologyResult::TopologyViolation;
        }

        // 获取四个顶点
        let v0 = he_data.origin;
        let v1 = twin_data.origin;
        let v2 = he_next.origin; // he.next 起点 = he 终点的下一个 = 第三个顶点
        let v3 = tw_next.origin;

        // 更新顶点出发边（如果它们指向被修改的边）
        if let Some(v) = self.vertex(v0) {
            if v.halfedge == he {
                if let Some(vm) = self.vertex_mut(v0) {
                    vm.halfedge = tw_prev.prev; // 指向另一条边
                }
            }
        }
        if let Some(v) = self.vertex(v1) {
            if v.halfedge == twin {
                if let Some(vm) = self.vertex_mut(v1) {
                    vm.halfedge = he_prev.prev;
                }
            }
        }

        // 翻转边：he 从 v0->v1 变成 v3->v2
        //        twin 从 v1->v0 变成 v2->v3
        let face0 = he_data.face;
        let face1 = twin_data.face;

        // 更新 he
        if let Some(h) = self.halfedge_mut(he) {
            h.origin = v3;
            h.next = he_prev.prev; // he_prev (原来的prev)
            h.prev = twin_data.next;
        }

        // 更新 twin
        if let Some(h) = self.halfedge_mut(twin) {
            h.origin = v2;
            h.next = tw_prev.prev;
            h.prev = he_data.next;
        }

        // 更新周围半边的链接
        // face0: he -> he_prev -> he_next (变成 he -> tw_next -> he_next 的一部分)
        // 实际上需要重新组织

        // 简化版本：直接重新设置所有链接
        let he_next_idx = he_data.next;
        let he_prev_idx = he_data.prev;
        let tw_next_idx = twin_data.next;
        let tw_prev_idx = twin_data.prev;

        // Face 0 新组成: he (v3->v2), he_next (v2->?), tw_prev (->v3)
        // 但这变复杂了，需要正确处理

        // 更简单的实现：
        // he: v3 -> v2
        // twin: v2 -> v3

        // face0 包含: he, tw_prev, he_next 中的某些边
        // face1 包含: twin, he_prev, tw_next 中的某些边

        // 正确的链接：
        // Face0: tw_next -> he -> he_prev
        // Face1: he_next -> twin -> tw_prev

        if let Some(h) = self.halfedge_mut(tw_next_idx) {
            h.next = he;
            h.face = face0;
        }
        if let Some(h) = self.halfedge_mut(he) {
            h.next = he_prev_idx;
            h.prev = tw_next_idx;
            h.face = face0;
        }
        if let Some(h) = self.halfedge_mut(he_prev_idx) {
            h.prev = he;
            h.next = tw_next_idx;
            h.face = face0;
        }

        if let Some(h) = self.halfedge_mut(he_next_idx) {
            h.next = twin;
            h.face = face1;
        }
        if let Some(h) = self.halfedge_mut(twin) {
            h.next = tw_prev_idx;
            h.prev = he_next_idx;
            h.face = face1;
        }
        if let Some(h) = self.halfedge_mut(tw_prev_idx) {
            h.prev = twin;
            h.next = he_next_idx;
            h.face = face1;
        }

        // 更新面的 halfedge
        if let Some(f) = self.face_mut(face0) {
            f.halfedge = he;
        }
        if let Some(f) = self.face_mut(face1) {
            f.halfedge = twin;
        }

        // 标记为脏
        self.mark_face_dirty(face0);
        self.mark_face_dirty(face1);
        self.mark_vertex_dirty(v0);
        self.mark_vertex_dirty(v1);
        self.mark_vertex_dirty(v2);
        self.mark_vertex_dirty(v3);

        TopologyResult::Success
    }

    /// 折叠边
    ///
    /// 将边收缩到一个顶点，删除边和相邻的退化面。
    ///
    /// # 参数
    /// - `he`: 要折叠的半边
    /// - `keep_origin`: true 保留起点，false 保留终点
    ///
    /// # 返回
    /// 保留的顶点索引
    ///
    /// 注意：这是一个复杂操作，可能破坏网格结构，使用需谨慎。
    pub fn collapse_edge(&mut self, he: HalfEdgeIndex, keep_origin: bool) -> Option<VertexIndex> {
        let he_data = self.halfedge(he)?.clone();
        let twin = he_data.twin;

        let v_keep = if keep_origin {
            he_data.origin
        } else {
            self.halfedge(twin)?.origin
        };
        let v_remove = if keep_origin {
            self.halfedge(twin)?.origin
        } else {
            he_data.origin
        };

        // 收集要删除的面
        let face0 = he_data.face;
        let face1 = self.halfedge(twin).map(|h| h.face).unwrap_or(FaceIndex::INVALID);

        // 将 v_remove 的所有出发边重定向到 v_keep
        let outgoing: Vec<HalfEdgeIndex> = self.vertex_outgoing(v_remove).collect();
        for out_he in outgoing {
            if out_he == he || (twin.is_valid() && out_he == twin) {
                continue; // 跳过要删除的边
            }

            if let Some(h) = self.halfedge_mut(out_he) {
                h.origin = v_keep;
            }

            // 更新入射边的 twin 的终点
            if let Some(h) = self.halfedge(out_he) {
                let twin_of_out = h.twin;
                if twin_of_out.is_valid() {
                    // twin 的终点（即 next 的 origin）应该指向 v_keep
                    // 这已经通过 origin 的更改自动处理
                }
            }
        }

        // 更新 v_keep 的位置为中点（可选）
        if let (Some(p0), Some(p1)) = (
            self.vertex(v_keep).map(|v| v.position),
            self.vertex(v_remove).map(|v| v.position),
        ) {
            if let Some(v) = self.vertex_mut(v_keep) {
                v.position = p0.lerp(&p1, 0.5);
            }
        }

        // 修复边界：绕过删除的边
        // he 的 prev 的 next 应该指向 he 的 next
        if he_data.prev.is_valid() && he_data.next.is_valid() {
            if let Some(h) = self.halfedge_mut(he_data.prev) {
                h.next = he_data.next;
            }
            if let Some(h) = self.halfedge_mut(he_data.next) {
                h.prev = he_data.prev;
            }
        }

        // twin 同样处理
        if let Some(twin_data) = self.halfedge(twin).cloned() {
            if twin_data.prev.is_valid() && twin_data.next.is_valid() {
                if let Some(h) = self.halfedge_mut(twin_data.prev) {
                    h.next = twin_data.next;
                }
                if let Some(h) = self.halfedge_mut(twin_data.next) {
                    h.prev = twin_data.prev;
                }
            }
        }

        // 更新 v_keep 的出发边（确保不指向删除的边）
        if let Some(v) = self.vertex(v_keep) {
            if v.halfedge == he || v.halfedge == twin {
                // 找一条新的出发边
                for out_he in self.vertex_outgoing(v_keep) {
                    if out_he != he && out_he != twin {
                        if let Some(vm) = self.vertex_mut(v_keep) {
                            vm.halfedge = out_he;
                        }
                        break;
                    }
                }
            }
        }

        // 删除元素
        self.remove_halfedge(he);
        if twin.is_valid() {
            self.remove_halfedge(twin);
        }
        self.remove_vertex(v_remove);

        // 如果面变成退化（少于3个顶点），删除
        if face0.is_valid() {
            let count = self.face_vertex_count(face0);
            if count < 3 {
                // 删除面及其半边
                let halfedges: Vec<_> = self.face_halfedges(face0).collect();
                for he_idx in halfedges {
                    self.remove_halfedge(he_idx);
                }
                self.remove_face(face0);
            } else {
                self.mark_face_dirty(face0);
            }
        }

        if face1.is_valid() && face1 != face0 {
            let count = self.face_vertex_count(face1);
            if count < 3 {
                let halfedges: Vec<_> = self.face_halfedges(face1).collect();
                for he_idx in halfedges {
                    self.remove_halfedge(he_idx);
                }
                self.remove_face(face1);
            } else {
                self.mark_face_dirty(face1);
            }
        }

        self.mark_vertex_dirty(v_keep);

        Some(v_keep)
    }

    /// 检查边翻转是否有效
    ///
    /// 只有在两边都是三角形且不会创建非流形边时才有效
    pub fn can_flip_edge(&self, he: HalfEdgeIndex) -> bool {
        let he_data = match self.halfedge(he) {
            Some(h) => h,
            None => return false,
        };

        let twin = he_data.twin;
        if twin.is_invalid() {
            return false;
        }

        let twin_data = match self.halfedge(twin) {
            Some(h) => h,
            None => return false,
        };

        // 检查是否为边界边
        if he_data.face.is_invalid() || twin_data.face.is_invalid() {
            return false;
        }

        // 检查两边是否都是三角形
        let face0_count = self.face_vertex_count(he_data.face);
        let face1_count = self.face_vertex_count(twin_data.face);

        face0_count == 3 && face1_count == 3
    }

    /// 检查边折叠是否安全
    ///
    /// 检查折叠是否会导致拓扑问题
    pub fn can_collapse_edge(&self, he: HalfEdgeIndex) -> bool {
        let he_data = match self.halfedge(he) {
            Some(h) => h,
            None => return false,
        };

        let twin = he_data.twin;
        let v0 = he_data.origin;
        let v1 = if twin.is_valid() {
            match self.halfedge(twin) {
                Some(h) => h.origin,
                None => return false,
            }
        } else {
            return false;
        };

        // 简单检查：确保两个顶点的共享邻居数量合理
        let v0_neighbors: std::collections::HashSet<_> = self.vertex_neighbors(v0).collect();
        let v1_neighbors: std::collections::HashSet<_> = self.vertex_neighbors(v1).collect();

        let common: Vec<_> = v0_neighbors.intersection(&v1_neighbors).collect();

        // 如果是内部边，应该恰好有2个共享邻居（两个三角形的第三个顶点）
        // 如果是边界边，可能有1个
        common.len() <= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mesh::Face;

    /// 创建两个共享边的三角形
    fn create_two_triangles() -> (HalfEdgeMesh<(), ()>, HalfEdgeIndex) {
        let mut mesh = HalfEdgeMesh::new();

        //     v2
        //    /|\
        //   / | \
        //  v0-|-v1
        //   \ | /
        //    \|/
        //     v3

        let v0 = mesh.add_vertex_xyz(0.0, 0.5, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.5, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);
        let v3 = mesh.add_vertex_xyz(0.5, 0.0, 0.0);

        // 上三角形: v0 -> v1 -> v2
        let he0 = mesh.add_halfedge(HalfEdge::new(v0));
        let he1 = mesh.add_halfedge(HalfEdge::new(v1));
        let he2 = mesh.add_halfedge(HalfEdge::new(v2));

        let face0 = mesh.add_face(Face::new(he0));

        if let Some(h) = mesh.halfedge_mut(he0) {
            h.next = he1;
            h.prev = he2;
            h.face = face0;
        }
        if let Some(h) = mesh.halfedge_mut(he1) {
            h.next = he2;
            h.prev = he0;
            h.face = face0;
        }
        if let Some(h) = mesh.halfedge_mut(he2) {
            h.next = he0;
            h.prev = he1;
            h.face = face0;
        }

        // 下三角形: v1 -> v0 -> v3
        let he3 = mesh.add_halfedge(HalfEdge::new(v1));
        let he4 = mesh.add_halfedge(HalfEdge::new(v0));
        let he5 = mesh.add_halfedge(HalfEdge::new(v3));

        let face1 = mesh.add_face(Face::new(he3));

        if let Some(h) = mesh.halfedge_mut(he3) {
            h.next = he4;
            h.prev = he5;
            h.face = face1;
        }
        if let Some(h) = mesh.halfedge_mut(he4) {
            h.next = he5;
            h.prev = he3;
            h.face = face1;
        }
        if let Some(h) = mesh.halfedge_mut(he5) {
            h.next = he3;
            h.prev = he4;
            h.face = face1;
        }

        // 设置 twin 关系: he0 <-> he3
        if let Some(h) = mesh.halfedge_mut(he0) {
            h.twin = he3;
        }
        if let Some(h) = mesh.halfedge_mut(he3) {
            h.twin = he0;
        }

        // 设置顶点出发边
        if let Some(v) = mesh.vertex_mut(v0) {
            v.halfedge = he0;
        }
        if let Some(v) = mesh.vertex_mut(v1) {
            v.halfedge = he1;
        }
        if let Some(v) = mesh.vertex_mut(v2) {
            v.halfedge = he2;
        }
        if let Some(v) = mesh.vertex_mut(v3) {
            v.halfedge = he5;
        }

        (mesh, he0)
    }

    #[test]
    fn test_split_edge() {
        let mut mesh = HalfEdgeMesh::<(), ()>::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(2.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(1.0, 1.0, 0.0);

        mesh.add_triangle(v0, v1, v2);

        let face = mesh.face_indices().next().unwrap();
        let he = mesh.face(face).unwrap().halfedge;

        // 分裂边
        let v_new = mesh.split_edge(he);
        assert!(v_new.is_some());

        let v_new = v_new.unwrap();

        // 验证新顶点位置
        let new_vertex = mesh.vertex(v_new).unwrap();
        assert!((new_vertex.position.x - 1.0).abs() < 1e-10);
        assert!((new_vertex.position.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_can_flip_edge() {
        let (mesh, he) = create_two_triangles();

        assert!(mesh.can_flip_edge(he));
    }
}
