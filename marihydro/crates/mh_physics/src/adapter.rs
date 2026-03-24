// crates/mh_physics/src/adapter.rs

//! 网格适配层 - 物理引擎几何接口
//! 
//! 提供PhysicsMesh与Backend几何抽象的桥接，支持f32/f64精度切换。
//! 
//! # 设计原则
//! 
//! 1. **类型安全强制**：所有几何查询接口必须使用Runtime索引类型（CellIndex/FaceIndex/NodeIndex）
//! 2. **职责隔离**：杜绝usize泄露，索引转换必须在调用层显式完成
//! 3. **Backend 泛型化**：所有几何接口使用 `B::Vector2D` 返回类型
//! 4. **错误透明**：坐标转换失败时panic而非静默回退，确保开发期暴露精度问题
//! 
//! # 架构约束
//! 
//! - **Layer 3 (Engine)** 必须调用泛型接口
//! - DVec2 Legacy 接口已删除
//! 
//! # 使用示例
//! 
//! ```rust,ignore
//! use mh_physics::adapter::PhysicsMesh;
//! use mh_runtime::{CpuBackend, CellIndex, FaceIndex};
//! 
//! let cell_idx = CellIndex::new(0);
//! let normal_f32 = mesh.face_center_generic::<CpuBackend<f32>>(FaceIndex::new(0));
//! let normal_f64 = mesh.face_center_generic::<CpuBackend<f64>>(FaceIndex::new(0));
//! ```

use mh_mesh::FrozenMesh;
use mh_mesh::structured::StructuredMesh as StructuredMesh2D;
use mh_runtime::{Backend, RuntimeScalar};
use std::sync::Arc;
use mh_foundation::MhError;

// 从mh_runtime导入统一索引类型
pub use crate::types::{CellIndex, FaceIndex, NodeIndex, INVALID_INDEX};

/// 物理引擎网格适配器
#[derive(Debug, Clone)]
pub struct PhysicsMesh {
    /// 内部FrozenMesh引用（不可变数据）
    inner: Arc<FrozenMesh>,
}

#[inline]
fn convert_scalar<B: Backend>(value: f64, field: &'static str) -> Result<B::Scalar, MhError> {
    B::Scalar::from_config(value)
        .ok_or_else(|| MhError::invalid_input(format!("{field}转换失败: value={value}")))
}

#[inline]
fn convert_vec2<B: Backend>(
    x: f64,
    y: f64,
    x_field: &'static str,
    y_field: &'static str,
) -> Result<B::Vector2D, MhError> {
    Ok(B::vec2_new(
        convert_scalar::<B>(x, x_field)?,
        convert_scalar::<B>(y, y_field)?,
    ))
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

    /// 从结构化网格创建适配器
    #[inline]
    pub fn from_structured(mesh: &StructuredMesh2D) -> Result<Self, MhError> {
        let frozen = mesh
            .freeze()
            .map_err(|e| MhError::invalid_input(format!("结构化网格冻结失败: {}", e)))?;
        frozen
            .validate()
            .map_err(|e| MhError::invalid_input(format!("结构化网格冻结失败: {}", e)))?;
        Ok(Self::new(Arc::new(frozen)))
    }

    /// 从FrozenMesh引用创建（克隆数据）
    #[inline]
    pub fn from_frozen(frozen: &FrozenMesh) -> Self {
        Self {
            inner: Arc::new(frozen.clone()),
        }
    }

    /// 获取内部FrozenMesh引用
    #[inline]
    pub fn inner(&self) -> &FrozenMesh {
        &self.inner
    }

    // =========================================================================
    // 基本统计（强制显式命名，避免旧接口残留）
    // =========================================================================

    /// 节点数量
    #[inline]
    pub fn node_count(&self) -> usize {
        self.inner.n_nodes
    }


    /// 单元数量
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.inner.n_cells
    }


    /// 面数量
    #[inline]
    pub fn face_count(&self) -> usize {
        self.inner.n_faces
    }


    /// 内部面数量
    #[inline]
    pub fn interior_face_count(&self) -> usize {
        self.inner.n_interior_faces
    }

    /// 边界面数量
    #[inline]
    pub fn boundary_face_count(&self) -> usize {
        self.inner.n_faces - self.inner.n_interior_faces
    }

    /// 单元索引迭代器
    #[inline]
    pub fn cell_indices(&self) -> impl Iterator<Item = CellIndex> + '_ {
        (0..self.cell_count()).map(CellIndex::new)
    }

    /// 面索引迭代器
    #[inline]
    pub fn face_indices(&self) -> impl Iterator<Item = FaceIndex> + '_ {
        (0..self.face_count()).map(FaceIndex::new)
    }

    /// 内部面索引迭代器
    #[inline]
    pub fn interior_face_indices(&self) -> impl Iterator<Item = FaceIndex> + '_ {
        (0..self.interior_face_count()).map(FaceIndex::new)
    }

    /// 边界面索引迭代器
    #[inline]
    pub fn boundary_face_indices(&self) -> impl Iterator<Item = FaceIndex> + '_ {
        (self.interior_face_count()..self.face_count()).map(FaceIndex::new)
    }

    // =========================================================================
    // 单元访问 - 强制使用CellIndex (核心改造)
    // =========================================================================

    /// 获取单元中心 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn cell_center_generic<B: Backend>(&self, cell: CellIndex) -> Result<B::Vector2D, MhError> {
        let idx = cell.get();
        if idx >= self.cell_count() {
            return Err(MhError::index_out_of_bounds("Cell", idx, self.cell_count()));
        }
        let p = self.inner.cell_center[idx];
        convert_vec2::<B>(p.x as f64, p.y as f64, "cell_center.x", "cell_center.y")
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
        debug_assert!(cell.get() < self.cell_count(), "CellIndex越界: {}", cell.get());
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

    /// 获取面中心 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn face_center_generic<B: Backend>(&self, face: FaceIndex) -> Result<B::Vector2D, MhError> {
        let idx = face.get();
        if idx >= self.face_count() {
            return Err(MhError::index_out_of_bounds("Face", idx, self.face_count()));
        }
        let p = self.inner.face_center[idx];
        convert_vec2::<B>(p.x as f64, p.y as f64, "face_center.x", "face_center.y")
    }

    /// 获取面法向量 (Backend几何类型 - Layer 3强制使用)
    #[inline]
    pub fn face_normal_generic<B: Backend>(&self, face: FaceIndex) -> Result<B::Vector2D, MhError> {
        let idx = face.get();
        if idx >= self.face_count() {
            return Err(MhError::index_out_of_bounds("Face", idx, self.face_count()));
        }
        let n = self.inner.face_normal[idx];
        convert_vec2::<B>(n.x as f64, n.y as f64, "face_normal.x", "face_normal.y")
    }

    /// 获取面长度 [m]
    #[inline]
    pub fn face_length(&self, face: FaceIndex) -> f64 {
        self.inner.face_length[face.get()]
    }

    /// 获取面长度（Backend 标量）
    #[inline]
    pub fn face_length_scalar<B: Backend>(&self, face: FaceIndex, backend: &B) -> Result<B::Scalar, MhError> {
        let idx = face.get();
        if idx >= self.face_count() {
            return Err(MhError::index_out_of_bounds("Face", idx, self.face_count()));
        }
        backend
            .try_config_scalar(
                self.inner.face_length[idx],
                "PhysicsMeshAdapter.face_length_scalar",
            )
            .map_err(|err| MhError::invalid_input(format!("面长度转换失败: {err}")))
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
        let idx = face.get();
        if idx >= self.inner.face_neighbor.len() {
            return false;
        }
        self.inner.face_neighbor[idx] != u32::MAX
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

    /// 获取面左侧床面高程 [m]
    #[inline]
    pub fn face_z_left(&self, face: FaceIndex) -> f64 {
        self.inner.face_z_left[face.get()]
    }

    /// 获取面左侧床面高程（Backend 标量）
    #[inline]
    pub fn face_z_left_scalar<B: Backend>(&self, face: FaceIndex, backend: &B) -> Result<B::Scalar, MhError> {
        let idx = face.get();
        if idx >= self.face_count() {
            return Err(MhError::index_out_of_bounds("Face", idx, self.face_count()));
        }
        backend
            .try_config_scalar(
                self.inner.face_z_left[idx],
                "PhysicsMeshAdapter.face_z_left_scalar",
            )
            .map_err(|err| MhError::invalid_input(format!("面左侧高程转换失败: {err}")))
    }

    /// 获取面右侧床面高程 [m]
    #[inline]
    pub fn face_z_right(&self, face: FaceIndex) -> f64 {
        self.inner.face_z_right[face.get()]
    }

    /// 获取面右侧床面高程（Backend 标量）
    #[inline]
    pub fn face_z_right_scalar<B: Backend>(&self, face: FaceIndex, backend: &B) -> Result<B::Scalar, MhError> {
        let idx = face.get();
        if idx >= self.face_count() {
            return Err(MhError::index_out_of_bounds("Face", idx, self.face_count()));
        }
        backend
            .try_config_scalar(
                self.inner.face_z_right[idx],
                "PhysicsMeshAdapter.face_z_right_scalar",
            )
            .map_err(|err| MhError::invalid_input(format!("面右侧高程转换失败: {err}")))
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

    /// 获取节点坐标 (Backend 几何类型)
    #[inline]
    pub fn node_xy_generic<B: Backend>(&self, node: NodeIndex) -> Result<B::Vector2D, MhError> {
        let p = self.inner.node_coords[node.get()];
        convert_vec2::<B>(p.x as f64, p.y as f64, "node_xy.x", "node_xy.y")
    }

    /// 获取节点高程 [m]
    #[inline]
    pub fn node_z(&self, node: NodeIndex) -> f64 {
        self.inner.node_coords[node.get()].z
    }

    // =========================================================================
    // 范围迭代器 (usize是合理的，因为Range本身就是usize)
    // =========================================================================

}

// ============================================================================
// VTU trait实现
// ============================================================================

impl mh_io::exporters::vtu::VtuMesh for PhysicsMesh {
    fn n_nodes(&self) -> usize {
        self.node_count()
    }

    fn n_cells(&self) -> usize {
        self.cell_count()
    }

    fn node_position(&self, idx: usize) -> [f64; 3] {
        let p = &self.inner.node_coords[idx];
        [p.x, p.y, p.z]
    }

    fn cell_nodes(&self, idx: usize) -> Vec<usize> {
        let start = self.inner.cell_node_offsets[idx];
        let end = self.inner.cell_node_offsets[idx + 1];
        self.inner.cell_node_indices[start..end]
            .iter()
            .map(|&n| n as usize)
            .collect()
    }

    fn cell_z_bed(&self, idx: usize) -> f64 {
        self.inner.cell_z_bed[idx]
    }

    fn cell_area(&self, idx: usize) -> f64 {
        self.inner.cell_area[idx]
    }
}

// ============================================================================
// 测试模块 - 覆盖Legacy和泛型接口
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CellIndex;
    use mh_mesh::FrozenMesh;
    use mh_runtime::Vector2D;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_physics_mesh_from_empty() {
        let backend = CpuBackend::<f64>::new();
        let frozen = FrozenMesh::empty_with_backend(backend);
        let mesh = PhysicsMesh::from_frozen(&frozen);

        assert_eq!(mesh.cell_count(), 0);
        assert_eq!(mesh.face_count(), 0);
        assert_eq!(mesh.node_count(), 0);
    }

    #[test]
    fn test_cell_index_usage() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let cell_idx = CellIndex::new(0);
        let center = mesh.cell_center_generic::<CpuBackend<f64>>(cell_idx).unwrap();
        
        assert!((center.x() - 0.5).abs() < 1e-10);
        assert!((center.y() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_face_index_usage() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let face_idx = FaceIndex::new(0);
        let normal = mesh.face_normal_generic::<CpuBackend<f64>>(face_idx).unwrap();
        
        assert!((normal.x() - 1.0).abs() < 1e-10);
        assert!(normal.y().abs() < 1e-10);
    }

    #[test]
    fn test_generic_interface_f32_f64_consistency() {
        let frozen = create_test_mesh();
        let mesh = PhysicsMesh::from_frozen(&frozen);
        
        let cell_idx = CellIndex::new(0);
        
        // 测试f32接口
        let center_f32 = mesh.cell_center_generic::<CpuBackend<f32>>(cell_idx).unwrap();
        assert_eq!(std::mem::size_of_val(&center_f32.x()), 4);
        
        // 测试f64接口
        let center_f64 = mesh.cell_center_generic::<CpuBackend<f64>>(cell_idx).unwrap();
        assert_eq!(std::mem::size_of_val(&center_f64.x()), 8);
        
        // 验证结果一致性
        assert_eq!(center_f32.x() as f64, center_f64.x());
        assert_eq!(center_f32.y() as f64, center_f64.y());
    }

    // 注意：极端坐标值（如 1e40）的溢出检测不是核心功能
    // 实际模拟场景中的坐标值都在合理范围内
    // 如需检测溢出，可以通过外部验证工具在加载数据时检查

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
        assert!(mesh.cell_area(invalid_cell).is_none());
        // debug_assert会在测试时panic
        // 生产环境由调用者保证索引有效性
        
        // 测试无效FaceIndex邻居检查
        let invalid_face = FaceIndex::INVALID;
        assert!(!mesh.has_neighbor(invalid_face));
    }

    // 创建测试用的FrozenMesh
    fn create_test_mesh() -> FrozenMesh {
        use mh_geo::{Point2D, Point3D};
        let backend = CpuBackend::<f64>::new();
        let mut mesh = FrozenMesh::empty_with_cells_backend(backend.clone(), 2);
        mesh.n_nodes = 6;
        mesh.node_coords = vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(2.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(1.0, 1.0, 0.0),
            Point3D::new(2.0, 1.0, 0.0),
        ];
        mesh.n_cells = 2;
        mesh.cell_center = vec![Point2D::new(0.5, 0.5), Point2D::new(1.5, 0.5)];
        let mut cell_area = backend.alloc(2);
        cell_area.copy_from_slice(&[1.0, 1.0]);
        mesh.cell_area = cell_area;
        let mut cell_z_bed = backend.alloc(2);
        cell_z_bed.copy_from_slice(&[0.0, 0.0]);
        mesh.cell_z_bed = cell_z_bed;
        mesh.cell_node_offsets = vec![0, 4, 8];
        mesh.cell_node_indices = vec![0, 1, 4, 3, 1, 2, 5, 4];
        mesh.cell_face_offsets = vec![0, 4, 8];
        mesh.cell_face_indices = vec![0, 1, 2, 3, 0, 4, 5, 6];
        mesh.cell_neighbor_offsets = vec![0, 1, 2];
        mesh.cell_neighbor_indices = vec![1, 0];
        mesh.n_faces = 7;
        mesh.n_interior_faces = 1;
        mesh.face_center = vec![
            Point2D::new(1.0, 0.5),
            Point2D::new(0.5, 0.0),
            Point2D::new(0.0, 0.5),
            Point2D::new(0.5, 1.0),
            Point2D::new(1.5, 0.0),
            Point2D::new(2.0, 0.5),
            Point2D::new(1.5, 1.0),
        ];
        mesh.face_normal = vec![
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, -1.0, 0.0),
            Point3D::new(-1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, -1.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
        ];
        let mut face_length = backend.alloc(7);
        face_length.copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        mesh.face_length = face_length;
        let mut face_z_left = backend.alloc(7);
        face_z_left.copy_from_slice(&[0.0; 7]);
        mesh.face_z_left = face_z_left;
        let mut face_z_right = backend.alloc(7);
        face_z_right.copy_from_slice(&[0.0; 7]);
        mesh.face_z_right = face_z_right;
        mesh.face_owner = vec![0, 0, 0, 0, 1, 1, 1];
        mesh.face_neighbor = vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        mesh.face_delta_owner = vec![Point2D::new(0.0, 0.0); 7];
        mesh.face_delta_neighbor = vec![Point2D::new(0.0, 0.0); 7];
        let mut face_dist_o2n = backend.alloc(7);
        face_dist_o2n.copy_from_slice(&[1.0; 7]);
        mesh.face_dist_o2n = face_dist_o2n;
        mesh.boundary_face_indices = vec![1, 2, 3, 4, 5, 6];
        mesh.boundary_names = vec!["boundary".to_string()];
        mesh.face_boundary_id = vec![None, Some(0), Some(0), Some(0), Some(0), Some(0), Some(0)];
        mesh.min_cell_size = 1.0;
        mesh.max_cell_size = 1.0;
        mesh.cell_refinement_level = vec![0; 2];
        mesh.cell_parent = vec![0, 1];
        mesh.ghost_capacity = 0;
        mesh.cell_original_id = Vec::new();
        mesh.face_original_id = Vec::new();
        mesh.cell_permutation = Vec::new();
        mesh.cell_inv_permutation = Vec::new();

        mesh
    }
}
