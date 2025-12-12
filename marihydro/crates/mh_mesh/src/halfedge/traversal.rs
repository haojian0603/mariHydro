// marihydro\crates\mh_mesh\src/halfedge/traversal.rs

//! 半边网格拓扑遍历迭代器
//!
//! 提供高效的网格遍历功能。

use super::mesh::HalfEdgeMesh;
use mh_foundation::index::{FaceIndex, HalfEdgeIndex, VertexIndex};
use mh_geo::{Point2D, Point3D};

// ============================================================================
// 面遍历
// ============================================================================

/// 遍历面的所有顶点
pub struct FaceVertexIter<'a, V, F> {
    mesh: &'a HalfEdgeMesh<V, F>,
    start: HalfEdgeIndex,
    current: HalfEdgeIndex,
    done: bool,
}

impl<'a, V, F> FaceVertexIter<'a, V, F> {
    pub(crate) fn new(mesh: &'a HalfEdgeMesh<V, F>, face: FaceIndex) -> Self {
        let start = mesh
            .face(face)
            .map(|f| f.halfedge)
            .unwrap_or(HalfEdgeIndex::INVALID);

        Self {
            mesh,
            start,
            current: start,
            done: start.is_invalid(),
        }
    }
}

impl<'a, V, F> Iterator for FaceVertexIter<'a, V, F> {
    type Item = VertexIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let he = self.mesh.halfedge(self.current)?;
        let vertex = he.origin;

        self.current = he.next;
        if self.current == self.start {
            self.done = true;
        }

        Some(vertex)
    }
}

/// 遍历面的所有半边
pub struct FaceHalfEdgeIter<'a, V, F> {
    mesh: &'a HalfEdgeMesh<V, F>,
    start: HalfEdgeIndex,
    current: HalfEdgeIndex,
    done: bool,
}

impl<'a, V, F> FaceHalfEdgeIter<'a, V, F> {
    pub(crate) fn new(mesh: &'a HalfEdgeMesh<V, F>, face: FaceIndex) -> Self {
        let start = mesh
            .face(face)
            .map(|f| f.halfedge)
            .unwrap_or(HalfEdgeIndex::INVALID);

        Self {
            mesh,
            start,
            current: start,
            done: start.is_invalid(),
        }
    }
}

impl<'a, V, F> Iterator for FaceHalfEdgeIter<'a, V, F> {
    type Item = HalfEdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;
        let he = self.mesh.halfedge(self.current)?;

        self.current = he.next;
        if self.current == self.start {
            self.done = true;
        }

        Some(result)
    }
}

// ============================================================================
// 顶点遍历
// ============================================================================

/// 遍历顶点周围的所有面
pub struct VertexFaceIter<'a, V, F> {
    mesh: &'a HalfEdgeMesh<V, F>,
    start: HalfEdgeIndex,
    current: HalfEdgeIndex,
    done: bool,
}

impl<'a, V, F> VertexFaceIter<'a, V, F> {
    pub(crate) fn new(mesh: &'a HalfEdgeMesh<V, F>, vertex: VertexIndex) -> Self {
        let start = mesh
            .vertex(vertex)
            .map(|v| v.halfedge)
            .unwrap_or(HalfEdgeIndex::INVALID);

        Self {
            mesh,
            start,
            current: start,
            done: start.is_invalid(),
        }
    }
}

impl<'a, V, F> Iterator for VertexFaceIter<'a, V, F> {
    type Item = FaceIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let he = self.mesh.halfedge(self.current)?;
        let face = he.face;

        // 移动到下一个半边: twin -> next
        let twin = self.mesh.halfedge(he.twin)?;
        self.current = twin.next;

        if self.current == self.start {
            self.done = true;
        }

        // 跳过无效面（边界）
        if face.is_valid() {
            Some(face)
        } else {
            self.next()
        }
    }
}

/// 遍历顶点周围的所有出发半边
pub struct VertexOutgoingIter<'a, V, F> {
    mesh: &'a HalfEdgeMesh<V, F>,
    start: HalfEdgeIndex,
    current: HalfEdgeIndex,
    done: bool,
}

impl<'a, V, F> VertexOutgoingIter<'a, V, F> {
    pub(crate) fn new(mesh: &'a HalfEdgeMesh<V, F>, vertex: VertexIndex) -> Self {
        let start = mesh
            .vertex(vertex)
            .map(|v| v.halfedge)
            .unwrap_or(HalfEdgeIndex::INVALID);

        Self {
            mesh,
            start,
            current: start,
            done: start.is_invalid(),
        }
    }
}

impl<'a, V, F> Iterator for VertexOutgoingIter<'a, V, F> {
    type Item = HalfEdgeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current;

        // 移动到下一个出发边: twin -> next
        let he = self.mesh.halfedge(self.current)?;
        if let Some(twin) = self.mesh.halfedge(he.twin) {
            self.current = twin.next;
            if self.current == self.start {
                self.done = true;
            }
        } else {
            self.done = true;
        }

        Some(result)
    }
}

/// 遍历顶点的所有邻居顶点
pub struct VertexNeighborIter<'a, V, F> {
    inner: VertexOutgoingIter<'a, V, F>,
}

impl<'a, V, F> VertexNeighborIter<'a, V, F> {
    pub(crate) fn new(mesh: &'a HalfEdgeMesh<V, F>, vertex: VertexIndex) -> Self {
        Self {
            inner: VertexOutgoingIter::new(mesh, vertex),
        }
    }
}

impl<'a, V, F> Iterator for VertexNeighborIter<'a, V, F> {
    type Item = VertexIndex;

    fn next(&mut self) -> Option<Self::Item> {
        let he_idx = self.inner.next()?;
        self.inner.mesh.halfedge_target(he_idx)
    }
}

// ============================================================================
// HalfEdgeMesh 遍历方法扩展
// ============================================================================

impl<V, F> HalfEdgeMesh<V, F> {
    /// 遍历面的所有顶点
    pub fn face_vertices(&self, face: FaceIndex) -> FaceVertexIter<'_, V, F> {
        FaceVertexIter::new(self, face)
    }

    /// 遍历面的所有半边
    pub fn face_halfedges(&self, face: FaceIndex) -> FaceHalfEdgeIter<'_, V, F> {
        FaceHalfEdgeIter::new(self, face)
    }

    /// 遍历顶点周围的所有面
    pub fn vertex_faces(&self, vertex: VertexIndex) -> VertexFaceIter<'_, V, F> {
        VertexFaceIter::new(self, vertex)
    }

    /// 遍历顶点的所有出发半边
    pub fn vertex_outgoing(&self, vertex: VertexIndex) -> VertexOutgoingIter<'_, V, F> {
        VertexOutgoingIter::new(self, vertex)
    }

    /// 遍历顶点的所有邻居顶点
    pub fn vertex_neighbors(&self, vertex: VertexIndex) -> VertexNeighborIter<'_, V, F> {
        VertexNeighborIter::new(self, vertex)
    }

    /// 获取面的顶点数量
    pub fn face_vertex_count(&self, face: FaceIndex) -> usize {
        self.face_vertices(face).count()
    }

    /// 获取顶点的度 (邻接边数)
    pub fn vertex_degree(&self, vertex: VertexIndex) -> usize {
        self.vertex_outgoing(vertex).count()
    }

    /// 收集面的所有顶点到 Vec
    pub fn face_vertex_vec(&self, face: FaceIndex) -> Vec<VertexIndex> {
        self.face_vertices(face).collect()
    }

    /// 收集面的所有顶点3D位置
    pub fn face_positions_3d(&self, face: FaceIndex) -> Vec<Point3D> {
        self.face_vertices(face)
            .filter_map(|v| self.vertex(v).map(|vert| vert.position))
            .collect()
    }

    /// 收集面的所有顶点2D位置（XY平面投影）
    pub fn face_positions(&self, face: FaceIndex) -> Vec<Point2D> {
        self.face_vertices(face)
            .filter_map(|v| self.vertex(v).map(|vert| vert.position.xy()))
            .collect()
    }

    /// 计算面的2D中心
    pub fn face_centroid(&self, face: FaceIndex) -> Option<Point2D> {
        let positions = self.face_positions(face);
        if positions.is_empty() {
            return None;
        }

        let n = positions.len() as f64;
        let sum_x: f64 = positions.iter().map(|p| p.x).sum();
        let sum_y: f64 = positions.iter().map(|p| p.y).sum();

        Some(Point2D::new(sum_x / n, sum_y / n))
    }

    /// 计算面的3D中心
    pub fn face_centroid_3d(&self, face: FaceIndex) -> Option<Point3D> {
        let positions = self.face_positions_3d(face);
        if positions.is_empty() {
            return None;
        }

        let n = positions.len() as f64;
        let sum_x: f64 = positions.iter().map(|p| p.x).sum();
        let sum_y: f64 = positions.iter().map(|p| p.y).sum();
        let sum_z: f64 = positions.iter().map(|p| p.z).sum();

        Some(Point3D::new(sum_x / n, sum_y / n, sum_z / n))
    }

    /// 计算面的面积 (使用鞋带公式，XY平面投影)
    pub fn face_area(&self, face: FaceIndex) -> f64 {
        let positions = self.face_positions(face);
        if positions.len() < 3 {
            return 0.0;
        }

        let n = positions.len();
        let mut area = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            area += positions[i].x * positions[j].y;
            area -= positions[j].x * positions[i].y;
        }

        (area / 2.0).abs()
    }

    /// 计算面的周长
    pub fn face_perimeter(&self, face: FaceIndex) -> f64 {
        let positions = self.face_positions(face);
        if positions.len() < 2 {
            return 0.0;
        }

        let n = positions.len();
        let mut perimeter = 0.0;

        for i in 0..n {
            let j = (i + 1) % n;
            let dx = positions[j].x - positions[i].x;
            let dy = positions[j].y - positions[i].y;
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        perimeter
    }

    /// 计算半边的2D长度
    pub fn halfedge_length(&self, he: HalfEdgeIndex) -> Option<f64> {
        let origin = self.halfedge(he)?.origin;
        let target = self.halfedge_target(he)?;

        let p0 = self.vertex(origin)?.position;
        let p1 = self.vertex(target)?.position;

        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;

        Some((dx * dx + dy * dy).sqrt())
    }

    /// 计算半边的方向向量 (单位向量, 2D)
    pub fn halfedge_direction(&self, he: HalfEdgeIndex) -> Option<Point2D> {
        let origin = self.halfedge(he)?.origin;
        let target = self.halfedge_target(he)?;

        let p0 = self.vertex(origin)?.position;
        let p1 = self.vertex(target)?.position;

        let dx = p1.x - p0.x;
        let dy = p1.y - p0.y;
        let len = (dx * dx + dy * dy).sqrt();

        if len < 1e-14 {
            None
        } else {
            Some(Point2D::new(dx / len, dy / len))
        }
    }

    /// 计算半边的法向量 (朝向面外)
    pub fn halfedge_normal(&self, he: HalfEdgeIndex) -> Option<Point2D> {
        let dir = self.halfedge_direction(he)?;
        // 顺时针旋转90度得到外法向
        Some(Point2D::new(dir.y, -dir.x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_triangle_mesh() -> HalfEdgeMesh<(), ()> {
        let mut mesh = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        mesh.add_triangle(v0, v1, v2);

        mesh
    }

    #[test]
    fn test_face_vertices() {
        let mesh = create_triangle_mesh();

        let face = mesh.face_indices().next().unwrap();
        let vertices: Vec<_> = mesh.face_vertices(face).collect();

        assert_eq!(vertices.len(), 3);
    }

    #[test]
    fn test_face_halfedges() {
        let mesh = create_triangle_mesh();

        let face = mesh.face_indices().next().unwrap();
        let halfedges: Vec<_> = mesh.face_halfedges(face).collect();

        assert_eq!(halfedges.len(), 3);

        // 验证链接正确
        for i in 0..3 {
            let he = mesh.halfedge(halfedges[i]).unwrap();
            assert_eq!(he.next, halfedges[(i + 1) % 3]);
        }
    }

    #[test]
    fn test_face_vertex_count() {
        let mesh = create_triangle_mesh();

        let face = mesh.face_indices().next().unwrap();
        assert_eq!(mesh.face_vertex_count(face), 3);
    }

    #[test]
    fn test_face_area() {
        let mesh = create_triangle_mesh();

        let face = mesh.face_indices().next().unwrap();
        let area = mesh.face_area(face);

        // 三角形面积 = 0.5 * base * height = 0.5 * 1.0 * 1.0 = 0.5
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_face_centroid() {
        let mesh = create_triangle_mesh();

        let face = mesh.face_indices().next().unwrap();
        let centroid = mesh.face_centroid(face).unwrap();

        assert!((centroid.x - 0.5).abs() < 1e-10);
        assert!((centroid.y - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_halfedge_length() {
        let mesh = create_triangle_mesh();

        let face = mesh.face_indices().next().unwrap();
        let he = mesh.face(face).unwrap().halfedge;

        let length = mesh.halfedge_length(he).unwrap();
        // v0 -> v1 长度应该是 1.0
        assert!((length - 1.0).abs() < 1e-10);
    }
}
