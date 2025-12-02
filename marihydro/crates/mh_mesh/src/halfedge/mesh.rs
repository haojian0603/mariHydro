// marihydro\crates\mh_mesh\src/halfedge/mesh.rs

//! 半边网格核心数据结构
//!
//! 使用半边数据结构实现统一的网格编辑和计算。
//!
//! # 设计要点
//!
//! 1. **半边表示**: 每条边由两个半边表示，支持O(1)邻接查询
//! 2. **Arena存储**: 使用代际索引防止悬垂引用
//! 3. **泛型数据**: 顶点/边/面可附加用户数据
//! 4. **脏标记**: 支持增量计算

use mh_foundation::arena::Arena;
use mh_foundation::index::{FaceIndex, FaceTag, HalfEdgeIndex, HalfEdgeTag, VertexIndex, VertexTag};
use mh_geo::Point3D;
use std::collections::HashSet;

/// 顶点数据
#[derive(Debug, Clone)]
pub struct Vertex<V> {
    /// 顶点位置（3D坐标）
    pub position: Point3D,
    /// 出发的任一半边 (可能为INVALID，表示孤立顶点)
    pub halfedge: HalfEdgeIndex,
    /// 用户附加数据
    pub data: V,
}

impl<V: Default> Vertex<V> {
    /// 创建新顶点
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            position: Point3D::new(x, y, z),
            halfedge: HalfEdgeIndex::INVALID,
            data: V::default(),
        }
    }

    /// 创建带数据的顶点
    pub fn with_data(x: f64, y: f64, z: f64, data: V) -> Self {
        Self {
            position: Point3D::new(x, y, z),
            halfedge: HalfEdgeIndex::INVALID,
            data,
        }
    }
}

/// 半边数据
#[derive(Debug, Clone)]
pub struct HalfEdge<E> {
    /// 起点顶点
    pub origin: VertexIndex,
    /// 对偶半边 (边界时为INVALID)
    pub twin: HalfEdgeIndex,
    /// 下一条半边 (同一面内)
    pub next: HalfEdgeIndex,
    /// 上一条半边 (同一面内)
    pub prev: HalfEdgeIndex,
    /// 所属面 (边界半边时为INVALID)
    pub face: FaceIndex,
    /// 用户附加数据
    pub data: E,
}

impl<E: Default> Default for HalfEdge<E> {
    fn default() -> Self {
        Self {
            origin: VertexIndex::INVALID,
            twin: HalfEdgeIndex::INVALID,
            next: HalfEdgeIndex::INVALID,
            prev: HalfEdgeIndex::INVALID,
            face: FaceIndex::INVALID,
            data: E::default(),
        }
    }
}

impl<E: Default> HalfEdge<E> {
    /// 创建新半边
    pub fn new(origin: VertexIndex) -> Self {
        Self {
            origin,
            twin: HalfEdgeIndex::INVALID,
            next: HalfEdgeIndex::INVALID,
            prev: HalfEdgeIndex::INVALID,
            face: FaceIndex::INVALID,
            data: E::default(),
        }
    }
}

/// 面数据
#[derive(Debug, Clone)]
pub struct Face<F> {
    /// 任一边界半边
    pub halfedge: HalfEdgeIndex,
    /// 用户附加数据
    pub data: F,
}

impl<F: Default> Face<F> {
    /// 创建新面
    pub fn new(halfedge: HalfEdgeIndex) -> Self {
        Self {
            halfedge,
            data: F::default(),
        }
    }

    /// 创建带数据的面
    pub fn with_data(halfedge: HalfEdgeIndex, data: F) -> Self {
        Self { halfedge, data }
    }
}

/// 半边网格
///
/// # 类型参数
///
/// - `V`: 顶点附加数据类型
/// - `E`: 半边附加数据类型
/// - `F`: 面附加数据类型
#[derive(Debug)]
pub struct HalfEdgeMesh<V = (), E = (), F = ()> {
    /// 顶点存储
    vertices: Arena<Vertex<V>, VertexTag>,
    /// 半边存储
    halfedges: Arena<HalfEdge<E>, HalfEdgeTag>,
    /// 面存储
    faces: Arena<Face<F>, FaceTag>,
    /// 脏顶点集合 (需要重新计算的顶点)
    dirty_vertices: HashSet<VertexIndex>,
    /// 脏面集合 (需要重新计算的面)
    dirty_faces: HashSet<FaceIndex>,
}

impl<V, E, F> Default for HalfEdgeMesh<V, E, F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, E, F> HalfEdgeMesh<V, E, F> {
    /// 创建空网格
    pub fn new() -> Self {
        Self {
            vertices: Arena::new(),
            halfedges: Arena::new(),
            faces: Arena::new(),
            dirty_vertices: HashSet::new(),
            dirty_faces: HashSet::new(),
        }
    }

    /// 创建指定容量的网格
    pub fn with_capacity(n_vertices: usize, n_halfedges: usize, n_faces: usize) -> Self {
        Self {
            vertices: Arena::with_capacity(n_vertices),
            halfedges: Arena::with_capacity(n_halfedges),
            faces: Arena::with_capacity(n_faces),
            dirty_vertices: HashSet::new(),
            dirty_faces: HashSet::new(),
        }
    }

    // =========================================================================
    // 基本统计
    // =========================================================================

    /// 顶点数量
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// 半边数量
    #[inline]
    pub fn n_halfedges(&self) -> usize {
        self.halfedges.len()
    }

    /// 面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// 边数量 (半边数/2)
    #[inline]
    pub fn n_edges(&self) -> usize {
        self.halfedges.len() / 2
    }

    /// 网格是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    // =========================================================================
    // 顶点操作
    // =========================================================================

    /// 添加顶点
    pub fn add_vertex(&mut self, vertex: Vertex<V>) -> VertexIndex {
        let idx = self.vertices.insert(vertex);
        self.dirty_vertices.insert(idx);
        idx
    }

    /// 获取顶点 (不可变)
    #[inline]
    pub fn vertex(&self, idx: VertexIndex) -> Option<&Vertex<V>> {
        self.vertices.get(idx)
    }

    /// 获取顶点 (可变)
    #[inline]
    pub fn vertex_mut(&mut self, idx: VertexIndex) -> Option<&mut Vertex<V>> {
        if let Some(v) = self.vertices.get_mut(idx) {
            self.dirty_vertices.insert(idx);
            Some(v)
        } else {
            None
        }
    }

    /// 移除顶点
    pub fn remove_vertex(&mut self, idx: VertexIndex) -> Option<Vertex<V>> {
        self.dirty_vertices.remove(&idx);
        self.vertices.remove(idx)
    }

    /// 检查顶点是否存在
    #[inline]
    pub fn contains_vertex(&self, idx: VertexIndex) -> bool {
        self.vertices.contains(idx)
    }

    /// 遍历所有顶点索引
    pub fn vertex_indices(&self) -> impl Iterator<Item = VertexIndex> + '_ {
        self.vertices.indices()
    }

    /// 遍历所有顶点
    pub fn vertices(&self) -> impl Iterator<Item = (VertexIndex, &Vertex<V>)> + '_ {
        self.vertices.iter()
    }

    // =========================================================================
    // 半边操作
    // =========================================================================

    /// 添加半边
    pub fn add_halfedge(&mut self, halfedge: HalfEdge<E>) -> HalfEdgeIndex {
        self.halfedges.insert(halfedge)
    }

    /// 获取半边 (不可变)
    #[inline]
    pub fn halfedge(&self, idx: HalfEdgeIndex) -> Option<&HalfEdge<E>> {
        self.halfedges.get(idx)
    }

    /// 获取半边 (可变)
    #[inline]
    pub fn halfedge_mut(&mut self, idx: HalfEdgeIndex) -> Option<&mut HalfEdge<E>> {
        self.halfedges.get_mut(idx)
    }

    /// 移除半边
    pub fn remove_halfedge(&mut self, idx: HalfEdgeIndex) -> Option<HalfEdge<E>> {
        self.halfedges.remove(idx)
    }

    /// 检查半边是否存在
    #[inline]
    pub fn contains_halfedge(&self, idx: HalfEdgeIndex) -> bool {
        self.halfedges.contains(idx)
    }

    /// 遍历所有半边索引
    pub fn halfedge_indices(&self) -> impl Iterator<Item = HalfEdgeIndex> + '_ {
        self.halfedges.indices()
    }

    /// 遍历所有半边
    pub fn halfedges(&self) -> impl Iterator<Item = (HalfEdgeIndex, &HalfEdge<E>)> + '_ {
        self.halfedges.iter()
    }

    // =========================================================================
    // 面操作
    // =========================================================================

    /// 添加面
    pub fn add_face(&mut self, face: Face<F>) -> FaceIndex {
        let idx = self.faces.insert(face);
        self.dirty_faces.insert(idx);
        idx
    }

    /// 获取面 (不可变)
    #[inline]
    pub fn face(&self, idx: FaceIndex) -> Option<&Face<F>> {
        self.faces.get(idx)
    }

    /// 获取面 (可变)
    #[inline]
    pub fn face_mut(&mut self, idx: FaceIndex) -> Option<&mut Face<F>> {
        if let Some(f) = self.faces.get_mut(idx) {
            self.dirty_faces.insert(idx);
            Some(f)
        } else {
            None
        }
    }

    /// 移除面
    pub fn remove_face(&mut self, idx: FaceIndex) -> Option<Face<F>> {
        self.dirty_faces.remove(&idx);
        self.faces.remove(idx)
    }

    /// 检查面是否存在
    #[inline]
    pub fn contains_face(&self, idx: FaceIndex) -> bool {
        self.faces.contains(idx)
    }

    /// 遍历所有面索引
    pub fn face_indices(&self) -> impl Iterator<Item = FaceIndex> + '_ {
        self.faces.indices()
    }

    /// 遍历所有面
    pub fn faces(&self) -> impl Iterator<Item = (FaceIndex, &Face<F>)> + '_ {
        self.faces.iter()
    }

    // =========================================================================
    // 拓扑查询辅助
    // =========================================================================

    /// 获取半边的终点顶点
    #[inline]
    pub fn halfedge_target(&self, he: HalfEdgeIndex) -> Option<VertexIndex> {
        self.halfedge(he)
            .and_then(|h| self.halfedge(h.next))
            .map(|h| h.origin)
    }

    /// 判断半边是否为边界
    #[inline]
    pub fn is_boundary_halfedge(&self, he: HalfEdgeIndex) -> bool {
        self.halfedge(he)
            .map(|h| h.face.is_invalid())
            .unwrap_or(false)
    }

    /// 判断顶点是否在边界上
    pub fn is_boundary_vertex(&self, v: VertexIndex) -> bool {
        if let Some(vertex) = self.vertex(v) {
            if vertex.halfedge.is_invalid() {
                return true; // 孤立顶点视为边界
            }

            let start = vertex.halfedge;
            let mut current = start;

            loop {
                if let Some(he) = self.halfedge(current) {
                    if he.face.is_invalid() {
                        return true;
                    }
                    if let Some(twin) = self.halfedge(he.twin) {
                        current = twin.next;
                        if current == start {
                            break;
                        }
                    } else {
                        return true; // 无twin表示边界
                    }
                } else {
                    return true;
                }
            }
        }
        false
    }

    // =========================================================================
    // 脏标记系统
    // =========================================================================

    /// 标记顶点为脏
    pub fn mark_vertex_dirty(&mut self, idx: VertexIndex) {
        self.dirty_vertices.insert(idx);
    }

    /// 标记面为脏
    pub fn mark_face_dirty(&mut self, idx: FaceIndex) {
        self.dirty_faces.insert(idx);
    }

    /// 标记顶点邻接的所有面为脏
    pub fn mark_vertex_faces_dirty(&mut self, v: VertexIndex) {
        // 先收集需要标记为脏的面
        let faces_to_mark: Vec<FaceIndex> = {
            let vertex = match self.vertex(v) {
                Some(vertex) if vertex.halfedge.is_valid() => vertex,
                _ => return,
            };

            let start = vertex.halfedge;
            let mut current = start;
            let mut faces = Vec::new();

            loop {
                if let Some(he) = self.halfedge(current) {
                    if he.face.is_valid() {
                        faces.push(he.face);
                    }
                    if let Some(twin) = self.halfedge(he.twin) {
                        current = twin.next;
                        if current == start {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            faces
        };

        // 然后统一标记
        for face in faces_to_mark {
            self.dirty_faces.insert(face);
        }
    }

    /// 获取脏顶点集合
    pub fn dirty_vertices(&self) -> &HashSet<VertexIndex> {
        &self.dirty_vertices
    }

    /// 获取脏面集合
    pub fn dirty_faces(&self) -> &HashSet<FaceIndex> {
        &self.dirty_faces
    }

    /// 清除所有脏标记
    pub fn clear_dirty(&mut self) {
        self.dirty_vertices.clear();
        self.dirty_faces.clear();
    }

    /// 清除脏顶点标记
    pub fn clear_dirty_vertices(&mut self) {
        self.dirty_vertices.clear();
    }

    /// 清除脏面标记
    pub fn clear_dirty_faces(&mut self) {
        self.dirty_faces.clear();
    }

    /// 获取并清除脏顶点
    pub fn take_dirty_vertices(&mut self) -> HashSet<VertexIndex> {
        std::mem::take(&mut self.dirty_vertices)
    }

    /// 获取并清除脏面
    pub fn take_dirty_faces(&mut self) -> HashSet<FaceIndex> {
        std::mem::take(&mut self.dirty_faces)
    }
}

// ============================================================================
// 高级构建方法
// ============================================================================

impl<V: Default, E: Default, F: Default> HalfEdgeMesh<V, E, F> {
    /// 从顶点列表创建简单顶点
    pub fn add_vertex_xyz(&mut self, x: f64, y: f64, z: f64) -> VertexIndex {
        self.add_vertex(Vertex::new(x, y, z))
    }

    /// 添加三角形面
    ///
    /// 输入三个顶点索引，创建面和对应的半边
    /// 返回新创建的面索引
    pub fn add_triangle(
        &mut self,
        v0: VertexIndex,
        v1: VertexIndex,
        v2: VertexIndex,
    ) -> Option<FaceIndex> {
        // 检查顶点是否存在
        if !self.contains_vertex(v0) || !self.contains_vertex(v1) || !self.contains_vertex(v2) {
            return None;
        }

        // 创建三个半边
        let he0 = self.add_halfedge(HalfEdge::new(v0));
        let he1 = self.add_halfedge(HalfEdge::new(v1));
        let he2 = self.add_halfedge(HalfEdge::new(v2));

        // 创建面
        let face_idx = self.add_face(Face::new(he0));

        // 设置半边的 next/prev 关系
        if let Some(h) = self.halfedge_mut(he0) {
            h.next = he1;
            h.prev = he2;
            h.face = face_idx;
        }
        if let Some(h) = self.halfedge_mut(he1) {
            h.next = he2;
            h.prev = he0;
            h.face = face_idx;
        }
        if let Some(h) = self.halfedge_mut(he2) {
            h.next = he0;
            h.prev = he1;
            h.face = face_idx;
        }

        // 设置顶点的出发半边（如果尚未设置）
        if let Some(v) = self.vertex_mut(v0) {
            if v.halfedge.is_invalid() {
                v.halfedge = he0;
            }
        }
        if let Some(v) = self.vertex_mut(v1) {
            if v.halfedge.is_invalid() {
                v.halfedge = he1;
            }
        }
        if let Some(v) = self.vertex_mut(v2) {
            if v.halfedge.is_invalid() {
                v.halfedge = he2;
            }
        }

        Some(face_idx)
    }

    /// 添加四边形面
    ///
    /// 输入四个顶点索引（逆时针顺序），创建面和对应的半边
    pub fn add_quad(
        &mut self,
        v0: VertexIndex,
        v1: VertexIndex,
        v2: VertexIndex,
        v3: VertexIndex,
    ) -> Option<FaceIndex> {
        // 检查顶点是否存在
        if !self.contains_vertex(v0)
            || !self.contains_vertex(v1)
            || !self.contains_vertex(v2)
            || !self.contains_vertex(v3)
        {
            return None;
        }

        // 创建四个半边
        let he0 = self.add_halfedge(HalfEdge::new(v0));
        let he1 = self.add_halfedge(HalfEdge::new(v1));
        let he2 = self.add_halfedge(HalfEdge::new(v2));
        let he3 = self.add_halfedge(HalfEdge::new(v3));

        // 创建面
        let face_idx = self.add_face(Face::new(he0));

        // 设置半边的 next/prev 关系
        if let Some(h) = self.halfedge_mut(he0) {
            h.next = he1;
            h.prev = he3;
            h.face = face_idx;
        }
        if let Some(h) = self.halfedge_mut(he1) {
            h.next = he2;
            h.prev = he0;
            h.face = face_idx;
        }
        if let Some(h) = self.halfedge_mut(he2) {
            h.next = he3;
            h.prev = he1;
            h.face = face_idx;
        }
        if let Some(h) = self.halfedge_mut(he3) {
            h.next = he0;
            h.prev = he2;
            h.face = face_idx;
        }

        // 设置顶点的出发半边
        for (v, he) in [(v0, he0), (v1, he1), (v2, he2), (v3, he3)] {
            if let Some(vertex) = self.vertex_mut(v) {
                if vertex.halfedge.is_invalid() {
                    vertex.halfedge = he;
                }
            }
        }

        Some(face_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_mesh() {
        let mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();
        assert!(mesh.is_empty());
        assert_eq!(mesh.n_vertices(), 0);
        assert_eq!(mesh.n_halfedges(), 0);
        assert_eq!(mesh.n_faces(), 0);
    }

    #[test]
    fn test_add_vertices() {
        let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        assert_eq!(mesh.n_vertices(), 3);
        assert!(mesh.contains_vertex(v0));
        assert!(mesh.contains_vertex(v1));
        assert!(mesh.contains_vertex(v2));

        let vertex = mesh.vertex(v0).unwrap();
        assert_eq!(vertex.position.x, 0.0);
        assert_eq!(vertex.position.y, 0.0);
    }

    #[test]
    fn test_add_triangle() {
        let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        let face = mesh.add_triangle(v0, v1, v2);
        assert!(face.is_some());

        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.n_halfedges(), 3);

        // 验证半边链接
        let face_he = mesh.face(face.unwrap()).unwrap().halfedge;
        let he0 = mesh.halfedge(face_he).unwrap();
        let he1 = mesh.halfedge(he0.next).unwrap();
        let he2 = mesh.halfedge(he1.next).unwrap();

        assert_eq!(he2.next, face_he);
        assert_eq!(he0.origin, v0);
        assert_eq!(he1.origin, v1);
        assert_eq!(he2.origin, v2);
    }

    #[test]
    fn test_dirty_marks() {
        let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        // 添加顶点会标记为脏
        assert!(mesh.dirty_vertices().contains(&v0));
        assert!(mesh.dirty_vertices().contains(&v1));
        assert!(mesh.dirty_vertices().contains(&v2));

        mesh.clear_dirty_vertices();
        assert!(mesh.dirty_vertices().is_empty());

        // 修改顶点会再次标记
        mesh.vertex_mut(v0);
        assert!(mesh.dirty_vertices().contains(&v0));

        // 添加三角形
        let face = mesh.add_triangle(v0, v1, v2).unwrap();
        assert!(mesh.dirty_faces().contains(&face));
    }

    #[test]
    fn test_remove_vertex() {
        let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);

        assert_eq!(mesh.n_vertices(), 2);

        mesh.remove_vertex(v0);
        assert_eq!(mesh.n_vertices(), 1);
        assert!(!mesh.contains_vertex(v0));
        assert!(mesh.contains_vertex(v1));
    }
}
