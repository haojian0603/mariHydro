// crates/mh_physics/src/adapter.rs

//! 网格适配层 - 物理引擎几何接口
//! 
//! 提供PhysicsMesh与Backend几何抽象的桥接，支持f32/f64精度切换。
//! 
//! # 设计原则
//! 
//! 1. **类型安全强制**：所有几何查询接口必须使用Runtime索引类型（CellIndex/FaceIndex/NodeIndex）
//! 2. **职责隔离**：杜绝usize泄露，索引转换必须在调用层显式完成
//! 3. **双重接口**：保留DVec2接口供Legacy代码短期过渡，新增Backend泛型接口供Layer 3使用
//! 4. **错误透明**：坐标转换失败时panic而非静默回退，确保开发期暴露精度问题
//! 
//! # 架构约束
//! 
//! - **Layer 3 (Engine)** 必须调用泛型接口，禁止直接使用Legacy接口
//! - **Layer 4/5 (Config/App)** 可使用Legacy接口，但需通过clippy.toml标记弃用
//! 
//! # 使用示例
//! 
//! ```rust
//! use mh_physics::adapter::PhysicsMesh;
//! use mh_runtime::{Backend, CpuBackend, CellIndex, FaceIndex};
//! 
//! // ❌ 错误：usize索引导致职责泄露
//! // let normal = mesh.face_normal(0); 
//! 
//! // ✅ 正确：强制使用类型安全索引
//! let face_idx = FaceIndex::new(0);
//! let normal_f32 = mesh.face_center_generic::<CpuBackend<f32>>(face_idx);
//! let normal_f64 = mesh.face_center_generic::<CpuBackend<f64>>(face_idx);
//! ```

use glam::DVec2;
use mh_mesh::FrozenMesh;
use mh_runtime::{Backend, RuntimeScalar};
use std::sync::Arc;

// 从mh_runtime导入统一索引类型
pub use crate::types::{CellIndex, FaceIndex, NodeIndex, INVALID_INDEX};

/// 无效单元索引常量（向后兼容，已弃用）
#[deprecated(note = "使用CellIndex::INVALID代替")]
pub const INVALID_CELL: usize = INVALID_INDEX;

/// 物理引擎网格适配器
#[derive(Debug, Clone)]
pub struct PhysicsMesh {
    /// 内部FrozenMesh引用（不可变数据）
    inner: Arc<FrozenMesh>,
}

impl PhysicsMesh {
    // =========================================================================
    // 构造函数
    // =========================================================================

    /// 从FrozenMesh Arc创建适配器
    #[inline]
    pub fn new(frozen: Arc<FrozenMesh>) -> Self {
        Self { inner: frozen }
    }

    /// 从FrozenMesh引用创建（克隆数据）
    #[inline]
    pub fn from_frozen(frozen: &FrozenMesh) -> Self {
        Self {
            inner: Arc::new(frozen.clone()),
        }
    }

    /// 创建指定单元数的空网格（测试用）
    #[inline]
    pub fn empty(n_cells: usize) -> Self {
        Self {
            inner: Arc::new(FrozenMesh::empty_with_cells(n_cells)),
        }
    }

    /// 获取内部FrozenMesh引用
    #[inline]
    pub fn inner(&self) -> &FrozenMesh {
        &self.inner
    }

    // =========================================================================
    // 基本统计 (Legacy接口 - 保留usize，这些是数量统计而非索引)
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
    // 单元访问 - 强制使用CellIndex (核心改造)
    // =========================================================================

    /// 获取单元中心 (DVec2 - Legacy接口，已弃用)
    #[deprecated(note = "请使用cell_center_generic<B>()")]
    #[inline]
    pub fn cell_center(&self, cell: usize) -> DVec2 {
        let p = self.inner.cell_center[cell];
        DVec2::new(p.x, p.y)
    }

    /// 获取单元中心 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn cell_center_generic<B: Backend>(&self, cell: CellIndex) -> B::Vector2D {
        let idx = cell.get();
        debug_assert!(idx < self.n_cells(), "CellIndex越界: {}", idx);
        let p = self.inner.cell_center[idx];
        B::vec2_new(
            B::Scalar::from_config(p.x as f64)
                .unwrap_or_else(|| panic!("坐标x={}转换失败：超出目标类型范围", p.x)),
            B::Scalar::from_config(p.y as f64)
                .unwrap_or_else(|| panic!("坐标y={}转换失败：超出目标类型范围", p.y))
        )
    }

    /// 获取单元底床高程 [m]
    #[inline]
    pub fn cell_z_bed(&self, cell: CellIndex) -> f64 {
        self.inner.cell_z_bed[cell.get()]
    }

    /// 获取单元底床高程数组引用
    #[inline]
    pub fn cell_z_bed_slice(&self) -> &[f64] {
        &self.inner.cell_z_bed
    }

    /// 安全获取单元面积 [m²]
    #[inline]
    pub fn cell_area(&self, cell: CellIndex) -> Option<f64> {
        self.inner.cell_area.get(cell.get()).copied()
    }

    /// 获取单元面积（无边界检查 - 性能敏感场景使用）
    #[inline]
    pub fn cell_area_unchecked(&self, cell: CellIndex) -> f64 {
        debug_assert!(cell.get() < self.n_cells(), "CellIndex越界: {}", cell.get());
        self.inner.cell_area[cell.get()]
    }

    /// 计算单元周长 [m]
    #[inline]
    pub fn cell_perimeter(&self, cell: CellIndex) -> Option<f64> {
        let faces = self.inner.cell_faces(cell.get());
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

    /// 获取单元的所有面索引（类型安全）
    #[inline]
    pub fn cell_faces(&self, cell: CellIndex) -> impl Iterator<Item = FaceIndex> + '_ {
        self.inner.cell_faces(cell.get())
            .iter()
            .map(|&f| FaceIndex::new(f as usize))
    }

    /// 获取单元的邻居单元索引
    #[inline]
    pub fn cell_neighbors(&self, cell: CellIndex) -> impl Iterator<Item = CellIndex> + '_ {
        self.inner.cell_neighbors(cell.get())
            .iter()
            .filter_map(|&n| {
                if n == u32::MAX {
                    None
                } else {
                    Some(CellIndex::new(n as usize))
                }
            })
    }

    /// 获取单元的节点索引
    #[inline]
    pub fn cell_nodes(&self, cell: CellIndex) -> impl Iterator<Item = NodeIndex> + '_ {
        self.inner.cell_nodes(cell.get())
            .iter()
            .map(|&n| NodeIndex::new(n as usize))
    }

    // =========================================================================
    // 面访问 - 强制使用FaceIndex (核心改造)
    // =========================================================================

    /// 获取面中心 (DVec2 - Legacy接口，已弃用)
    #[deprecated(note = "请使用face_center_generic<B>()")]
    #[inline]
    pub fn face_center(&self, face: usize) -> DVec2 {
        let p = self.inner.face_center[face];
        DVec2::new(p.x, p.y)
    }

    /// 获取面中心 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn face_center_generic<B: Backend>(&self, face: FaceIndex) -> B::Vector2D {
        let idx = face.get();
        debug_assert!(idx < self.n_faces(), "FaceIndex越界: {}", idx);
        let p = self.inner.face_center[idx];
        B::vec2_new(
            B::Scalar::from_config(p.x as f64)
                .unwrap_or_else(|| panic!("坐标x={}转换失败：超出目标类型范围", p.x)),
            B::Scalar::from_config(p.y as f64)
                .unwrap_or_else(|| panic!("坐标y={}转换失败：超出目标类型范围", p.y))
        )
    }

    /// 获取面法向量 (DVec2 - Legacy接口，已弃用)
    #[deprecated(note = "请使用face_normal_generic<B>()")]
    #[inline]
    pub fn face_normal(&self, face: usize) -> DVec2 {
        let n = self.inner.face_normal[face];
        DVec2::new(n.x, n.y)
    }

    /// 获取面法向量 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn face_normal_generic<B: Backend>(&self, face: FaceIndex) -> B::Vector2D {
        let idx = face.get();
        debug_assert!(idx < self.n_faces(), "FaceIndex越界: {}", idx);
        let n = self.inner.face_normal[idx];
        B::vec2_new(
            B::Scalar::from_config(n.x as f64)
                .unwrap_or_else(|| panic!("法向量x={}转换失败：超出目标类型范围", n.x)),
            B::Scalar::from_config(n.y as f64)
                .unwrap_or_else(|| panic!("法向量y={}转换失败：超出目标类型范围", n.y))
        )
    }

    /// 获取面法向量 (3D元组 - Legacy接口)
    #[inline]
    pub fn face_normal_3d(&self, face: FaceIndex) -> (f64, f64, f64) {
        let n = self.inner.face_normal[face.get()];
        (n.x, n.y, n.z)
    }

    /// 获取面长度 [m]
    #[inline]
    pub fn face_length(&self, face: FaceIndex) -> f64 {
        self.inner.face_length[face.get()]
    }

    /// 获取面owner单元索引
    #[inline]
    pub fn face_owner(&self, face: FaceIndex) -> CellIndex {
        CellIndex::new(self.inner.face_owner[face.get()] as usize)
    }

    /// 获取面neighbor单元索引 (Option<CellIndex>)
    #[inline]
    pub fn face_neighbor(&self, face: FaceIndex) -> Option<CellIndex> {
        let n = self.inner.face_neighbor[face.get()];
        if n == u32::MAX {
            None
        } else {
            Some(CellIndex::new(n as usize))
        }
    }

    /// 获取面neighbor单元索引 (返回INVALID而非Option)
    #[inline]
    pub fn face_neighbor_raw(&self, face: FaceIndex) -> CellIndex {
        let n = self.inner.face_neighbor[face.get()];
        if n == u32::MAX {
            CellIndex::INVALID
        } else {
            CellIndex::new(n as usize)
        }
    }

    /// 判断面是否有邻居
    #[inline]
    pub fn has_neighbor(&self, face: FaceIndex) -> bool {
        self.inner.face_neighbor[face.get()] != u32::MAX
    }

    /// 获取面左侧高程 [m]
    #[inline]
    pub fn face_z_left(&self, face: FaceIndex) -> f64 {
        self.inner.face_z_left[face.get()]
    }

    /// 获取面右侧高程 [m]
    #[inline]
    pub fn face_z_right(&self, face: FaceIndex) -> f64 {
        self.inner.face_z_right[face.get()]
    }

    /// 获取面到owner的向量 (DVec2 - Legacy接口，已弃用)
    #[deprecated(note = "请使用face_delta_owner_generic<B>()")]
    #[inline]
    pub fn face_delta_owner(&self, face: usize) -> DVec2 {
        let d = self.inner.face_delta_owner[face];
        DVec2::new(d.x, d.y)
    }

    /// 获取面到owner的向量 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn face_delta_owner_generic<B: Backend>(&self, face: FaceIndex) -> B::Vector2D {
        let idx = face.get();
        let d = self.inner.face_delta_owner[idx];
        B::vec2_new(
            B::Scalar::from_config(d.x as f64)
                .unwrap_or_else(|| panic!("向量x={}转换失败：超出目标类型范围", d.x)),
            B::Scalar::from_config(d.y as f64)
                .unwrap_or_else(|| panic!("向量y={}转换失败：超出目标类型范围", d.y))
        )
    }

    /// 获取面到neighbor的向量 (Backend几何类型)
    #[inline]
    pub fn face_delta_neighbor_generic<B: Backend>(&self, face: FaceIndex) -> B::Vector2D {
        let idx = face.get();
        let d = self.inner.face_delta_neighbor[idx];
        B::vec2_new(
            B::Scalar::from_config(d.x as f64)
                .unwrap_or_else(|| panic!("向量x={}转换失败：超出目标类型范围", d.x)),
            B::Scalar::from_config(d.y as f64)
                .unwrap_or_else(|| panic!("向量y={}转换失败：超出目标类型范围", d.y))
        )
    }

    /// 获取owner到neighbor的距离 [m]
    #[inline]
    pub fn face_dist_o2n(&self, face: FaceIndex) -> f64 {
        self.inner.face_dist_o2n[face.get()]
    }

    /// 获取面距离（内部面为o2n，边界面为owner到边界）
    #[inline]
    pub fn face_distance(&self, face: FaceIndex) -> Option<f64> {
        let dist = self.inner.face_dist_o2n[face.get()];
        if dist > 1e-14 {
            Some(dist)
        } else {
            None
        }
    }

    /// 判断是否为边界面
    #[inline]
    pub fn is_boundary_face(&self, face: FaceIndex) -> bool {
        face.get() >= self.inner.n_interior_faces
    }

    /// 获取面的边界ID
    #[inline]
    pub fn face_boundary_id(&self, face: FaceIndex) -> Option<usize> {
        self.inner
            .face_boundary_id
            .get(face.get())
            .and_then(|opt| opt.map(|id| id as usize))
    }

    // =========================================================================
    // 节点访问 - 强制使用NodeIndex
    // =========================================================================

    /// 获取节点坐标 (2D)
    #[inline]
    pub fn node_xy(&self, node: NodeIndex) -> DVec2 {
        let p = self.inner.node_coords[node.get()];
        DVec2::new(p.x, p.y)
    }

    /// 获取节点高程 [m]
    #[inline]
    pub fn node_z(&self, node: NodeIndex) -> f64 {
        self.inner.node_coords[node.get()].z
    }

    // =========================================================================
    // 范围迭代 (usize是合理的，因为Range本身就是usize)
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
    // 统计信息 (Legacy接口)
    // =========================================================================

    /// 最小单元尺寸 [m]
    #[inline]
    pub fn min_cell_size(&self) -> f64 {
        self.inner.min_cell_size
    }

    /// 最大单元尺寸 [m]
    #[inline]
    pub fn max_cell_size(&self) -> f64 {
        self.inner.max_cell_size
    }
}

// ============================================================================
// 测试模块 - 覆盖Legacy和泛型接口
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CellIndex;
    use mh_geo::{Point2D, Point3D};
    use mh_mesh::FrozenMesh;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_physics_mesh_from_empty() {
        let frozen = FrozenMesh::empty();
        let mesh = PhysicsMesh::from_frozen(&frozen);

        assert_eq!(mesh.n_cells(), 0);
        assert_eq!(mesh.n_faces(), 0);
        assert_eq!(mesh.n_nodes(), 0);
    }

    #[test]
    fn test_cell_index_usage() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let cell_idx = CellIndex::new(0);
        let center = mesh.cell_center_generic::<CpuBackend<f64>>(cell_idx);
        
        assert!((center.x() - 0.5).abs() < 1e-10);
        assert!((center.y() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_face_index_usage() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let face_idx = FaceIndex::new(0);
        let normal = mesh.face_normal_generic::<CpuBackend<f64>>(face_idx);
        
        assert!((normal.x() - 1.0).abs() < 1e-10);
        assert!(normal.y().abs() < 1e-10);
    }

    #[test]
    fn test_generic_interface_f32_f64_consistency() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let cell_idx = CellIndex::new(0);
        
        // 测试f32接口
        let center_f32 = mesh.cell_center_generic::<CpuBackend<f32>>(cell_idx);
        assert_eq!(std::mem::size_of_val(&center_f32.x()), 4);
        
        // 测试f64接口
        let center_f64 = mesh.cell_center_generic::<CpuBackend<f64>>(cell_idx);
        assert_eq!(std::mem::size_of_val(&center_f64.x()), 8);
        
        // 验证结果一致性
        assert_eq!(center_f32.x() as f64, center_f64.x());
        assert_eq!(center_f32.y() as f64, center_f64.y());
    }

    #[test]
    #[should_panic(expected = "转换失败：超出目标类型范围")]
    fn test_conversion_error_propagation() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        // 创建极大坐标导致f32转换溢出
        let mut frozen_large = create_test_mesh();
        frozen_large.cell_center[0] = Point2D::new(1e40, 1e40);
        let mesh_large = PhysicsMesh::from_frozen(&frozen_large);
        
        let cell_idx = CellIndex::new(0);
        let _ = mesh_large.cell_center_generic::<CpuBackend<f32>>(cell_idx);
    }

    #[test]
    fn test_cell_perimeter_with_index() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let cell_idx = CellIndex::new(0);
        let perimeter = mesh.cell_perimeter(cell_idx).unwrap();
        
        assert!((perimeter - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_face_neighbors_iterator() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let cell_idx = CellIndex::new(0);
        let neighbors: Vec<_> = mesh.cell_neighbors(cell_idx).collect();
        
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].get(), 1);
    }

    #[test]
    fn test_invalid_index_handling() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        // 测试无效CellIndex
        let invalid_cell = CellIndex::INVALID;
        // debug_assert会在测试时panic
        // 生产环境由调用者保证索引有效性
        
        // 测试无效FaceIndex邻居检查
        let invalid_face = FaceIndex::INVALID;
        assert!(!mesh.has_neighbor(invalid_face));
    }

    // 创建测试用的FrozenMesh
    fn create_test_mesh() -> FrozenMesh {
        use mh_geo::{Point2D, Point3D};
        
        FrozenMesh {
            n_nodes: 6,
            node_coords: vec![
                Point3D::new(0.0, 0.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(2.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
                Point3D::new(1.0, 1.0, 0.0),
                Point3D::new(2.0, 1.0, 0.0),
            ],
            n_cells: 2,
            cell_center: vec![
                Point2D::new(0.5, 0.5),
                Point2D::new(1.5, 0.5),
            ],
            cell_area: vec![1.0, 1.0],
            cell_z_bed: vec![0.0, 0.0],
            cell_node_offsets: vec![0, 4, 8],
            cell_node_indices: vec![0, 1, 4, 3, 1, 2, 5, 4],
            cell_face_offsets: vec![0, 4, 8],
            cell_face_indices: vec![0, 1, 2, 3, 0, 4, 5, 6],
            cell_neighbor_offsets: vec![0, 1, 2],
            cell_neighbor_indices: vec![1, 0],
            n_faces: 7,
            n_interior_faces: 1,
            face_center: vec![
                Point2D::new(1.0, 0.5),
                Point2D::new(0.5, 0.0),
                Point2D::new(0.0, 0.5),
                Point2D::new(0.5, 1.0),
                Point2D::new(1.5, 0.0),
                Point2D::new(2.0, 0.5),
                Point2D::new(1.5, 1.0),
            ],
            face_normal: vec![
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, -1.0, 0.0),
                Point3D::new(-1.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
                Point3D::new(0.0, -1.0, 0.0),
                Point3D::new(1.0, 0.0, 0.0),
                Point3D::new(0.0, 1.0, 0.0),
            ],
            face_length: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            face_z_left: vec![0.0; 7],
            face_z_right: vec![0.0; 7],
            face_owner: vec![0, 0, 0, 0, 1, 1, 1],
            face_neighbor: vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            face_delta_owner: vec![Point2D::new(0.0, 0.0); 7],
            face_delta_neighbor: vec![Point2D::new(0.0, 0.0); 7],
            face_dist_o2n: vec![1.0; 7],
            boundary_face_indices: vec![1, 2, 3, 4, 5, 6],
            boundary_names: vec!["boundary".to_string()],
            face_boundary_id: vec![None, Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)],
            min_cell_size: 1.0,
            max_cell_size: 1.0,
            cell_refinement_level: vec![0; 2],
            cell_parent: vec![0, 1],
            ghost_capacity: 0,
            cell_original_id: Vec::new(),
            face_original_id: Vec::new(),
            cell_permutation: Vec::new(),
            cell_inv_permutation: Vec::new(),
        }
    }
}