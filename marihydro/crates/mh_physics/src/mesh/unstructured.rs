//! 非结构化网格适配器
//!
//! 将现有的 FrozenMesh/PhysicsMesh 适配到 MeshTopology trait。
//!
//! # 设计说明
//!
//! - **无默认 Backend**: 必须显式指定 Backend 类型
//! - **Backend 缓冲区**: 所有数组使用 `B::Buffer<B::Scalar>`
//! - **存储 Backend 实例**: 支持后续缓冲区分配
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::mesh::UnstructuredMeshAdapter;
//! use mh_runtime::CpuBackend;
//!
//! let backend = CpuBackend::<f64>::new();
//! let adapter = UnstructuredMeshAdapter::from_physics_mesh_with_backend(&backend, mesh);
//! ```

use crate::adapter::PhysicsMesh;
use crate::core::{Backend, CpuBackend};
use super::topology::{MeshKind, MeshTopology};
use mh_runtime::{CellIndex, FaceIndex, RuntimeScalar};
use num_traits::FromPrimitive;
use std::sync::Arc;

/// 非结构化网格适配器
///
/// 将 `PhysicsMesh` 适配到 `MeshTopology` trait。
///
/// # 类型参数
///
/// - `B`: 计算后端类型（无默认值，必须显式指定）
pub struct UnstructuredMeshAdapter<B: Backend> {
    /// 原始网格引用
    mesh: Arc<PhysicsMesh>,
    /// 计算后端实例
    backend: B,
    /// 单元面积缓冲区
    cell_areas: B::Buffer<B::Scalar>,
    /// 面长度缓冲区
    face_lengths: B::Buffer<B::Scalar>,
    /// 边界面索引
    boundary_face_indices: Vec<usize>,
    /// 内部面索引
    interior_face_indices: Vec<usize>,
    /// 单元-面映射缓存
    cell_face_map: Vec<Vec<usize>>,
}

impl<B: Backend + Clone> UnstructuredMeshAdapter<B>
where
    B::Scalar: RuntimeScalar,
{
    /// 从 PhysicsMesh 创建适配器（带 Backend）
    ///
    /// # 参数
    ///
    /// - `backend`: 计算后端实例
    /// - `mesh`: 物理网格引用
    pub fn from_physics_mesh_with_backend(backend: &B, mesh: Arc<PhysicsMesh>) -> Self {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();
        
        // 使用 Backend 分配单元面积缓冲区
        let mut cell_areas_vec = Vec::with_capacity(n_cells);
        for i in 0..n_cells {
            let area = mesh.cell_area(CellIndex(i)).unwrap_or(0.0);
            if !area.is_finite() || area <= 0.0 {
                panic!("UnstructuredMeshAdapter: 无效单元面积 cell={i}, area={area}");
            }
            cell_areas_vec.push(B::Scalar::from_f64(area).unwrap_or(B::Scalar::ZERO));
        }
        let cell_areas = {
            let mut buf = backend.alloc(n_cells);
            buf.copy_from_slice(&cell_areas_vec);
            buf
        };
        
        // 使用 Backend 分配面长度缓冲区
        let mut face_lengths_vec = Vec::with_capacity(n_faces);
        for i in 0..n_faces {
            let length = mesh.face_length(FaceIndex(i));
            if !length.is_finite() || length <= 0.0 {
                panic!("UnstructuredMeshAdapter: 无效面长度 face={i}, length={length}");
            }
            face_lengths_vec.push(B::Scalar::from_f64(length).unwrap_or(B::Scalar::ZERO));
        }
        let face_lengths = {
            let mut buf = backend.alloc(n_faces);
            buf.copy_from_slice(&face_lengths_vec);
            buf
        };
        
        // 构建边界和内部面索引列表
        let mut boundary_face_indices = Vec::new();
        let mut interior_face_indices = Vec::new();
        for i in 0..n_faces {
            if mesh.face_neighbor(FaceIndex(i)).is_none() {
                boundary_face_indices.push(i);
            } else {
                interior_face_indices.push(i);
            }
        }
        
        // 构建单元-面映射
        let mut cell_face_map = vec![Vec::new(); n_cells];
        for face in 0..n_faces {
            let owner = mesh.face_owner(FaceIndex(face));
            if owner.get() >= n_cells {
                panic!("UnstructuredMeshAdapter: face owner 越界 face={face}, owner={}", owner.get());
            }
            cell_face_map[owner.get()].push(face);
            if let Some(neighbor) = mesh.face_neighbor(FaceIndex(face)) {
                if neighbor.get() >= n_cells {
                    panic!("UnstructuredMeshAdapter: face neighbor 越界 face={face}, neighbor={}", neighbor.get());
                }
                cell_face_map[neighbor.get()].push(face);
            }
        }
        
        let adapter = Self {
            mesh,
            backend: backend.clone(),
            cell_areas,
            face_lengths,
            boundary_face_indices,
            interior_face_indices,
            cell_face_map,
        };

        if let Err(err) = adapter.validate() {
            panic!("UnstructuredMeshAdapter 校验失败: {err}");
        }

        adapter
    }
    
    /// 获取原始网格引用
    #[inline]
    pub fn inner(&self) -> &PhysicsMesh {
        &self.mesh
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

// Layer 4 便捷方法：仅为 CpuBackend<f64> 提供无 Backend 参数的构造函数
impl UnstructuredMeshAdapter<CpuBackend<f64>> {
    /// 从 PhysicsMesh 创建适配器（默认 f64 精度，Layer 4 便捷方法）
    ///
    /// 此方法仅在 Layer 4 应用层使用，Layer 3 代码应使用 `from_physics_mesh_with_backend`
    pub fn from_physics_mesh(mesh: Arc<PhysicsMesh>) -> Self {
        let backend = CpuBackend::<f64>::new();
        Self::from_physics_mesh_with_backend(&backend, mesh)
    }
}

impl<B: Backend + Clone> MeshTopology<B> for UnstructuredMeshAdapter<B>
where
    B::Scalar: RuntimeScalar,
{
    fn n_cells(&self) -> usize {
        self.mesh.n_cells()
    }
    
    fn n_faces(&self) -> usize {
        self.mesh.n_faces()
    }
    
    fn n_interior_faces(&self) -> usize {
        self.interior_face_indices.len()
    }
    
    fn n_nodes(&self) -> usize {
        self.mesh.n_nodes()
    }
    
    fn cell_center(&self, cell: usize) -> [B::Scalar; 2] {
        let (x, y) = self.mesh.cell_center_tuple(cell);
        [
            B::Scalar::from_f64(x).unwrap_or(B::Scalar::ZERO),
            B::Scalar::from_f64(y).unwrap_or(B::Scalar::ZERO),
        ]
    }
    
    fn cell_area(&self, cell: usize) -> B::Scalar {
        self.cell_areas.get(cell).copied().unwrap_or(B::Scalar::ZERO)
    }
    
    fn face_normal(&self, face: usize) -> [B::Scalar; 2] {
        let (nx, ny) = self.mesh.face_normal_2d_tuple(face);
        [
            B::Scalar::from_f64(nx).unwrap_or(B::Scalar::ZERO),
            B::Scalar::from_f64(ny).unwrap_or(B::Scalar::ZERO),
        ]
    }
    
    fn face_length(&self, face: usize) -> B::Scalar {
        self.face_lengths.get(face).copied().unwrap_or(B::Scalar::ZERO)
    }
    
    fn face_center(&self, face: usize) -> [B::Scalar; 2] {
        let (x, y) = self.mesh.face_center_tuple(face);
        [
            B::Scalar::from_f64(x).unwrap_or(B::Scalar::ZERO),
            B::Scalar::from_f64(y).unwrap_or(B::Scalar::ZERO),
        ]
    }
    
    fn face_owner(&self, face: usize) -> usize {
        self.mesh.face_owner(FaceIndex(face)).get()
    }
    
    fn face_neighbor(&self, face: usize) -> Option<usize> {
        self.mesh.face_neighbor(FaceIndex(face)).map(|c| c.get())
    }
    
    fn cell_faces(&self, cell: usize) -> &[usize] {
        self.cell_face_map.get(cell).map(|v| v.as_slice()).unwrap_or(&[])
    }
    
    fn cell_neighbors(&self, cell: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        if let Some(faces) = self.cell_face_map.get(cell) {
            for &face in faces {
                let owner = self.face_owner(face);
                if owner == cell {
                    if let Some(neighbor) = self.face_neighbor(face) {
                        neighbors.push(neighbor);
                    }
                } else {
                    neighbors.push(owner);
                }
            }
        }
        neighbors
    }
    
    fn boundary_faces(&self) -> &[usize] {
        &self.boundary_face_indices
    }
    
    fn interior_faces(&self) -> &[usize] {
        &self.interior_face_indices
    }
    
    fn mesh_kind(&self) -> MeshKind {
        MeshKind::Unstructured
    }
    
    fn cell_areas_buffer(&self) -> &B::Buffer<B::Scalar> {
        &self.cell_areas
    }
    
    fn face_lengths_buffer(&self) -> &B::Buffer<B::Scalar> {
        &self.face_lengths
    }
}

#[cfg(test)]
mod tests {
    // 需要 PhysicsMesh 的测试数据，暂时跳过
    // #[test]
    // fn test_unstructured_adapter() { ... }
}
