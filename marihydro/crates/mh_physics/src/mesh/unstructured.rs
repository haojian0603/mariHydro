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
//! let adapter = UnstructuredMeshAdapter::from_physics_mesh_with_backend(&backend, mesh)?;
//! ```

use crate::adapter::PhysicsMesh;
use crate::core::Backend;
use super::topology::{MeshKind, MeshTopology, MeshValidationError};
use mh_runtime::prelude::{CellIndex, FaceIndex, RuntimeScalar, Vector2D};
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
    pub fn from_physics_mesh_with_backend(
        backend: &B,
        mesh: Arc<PhysicsMesh>,
    ) -> Result<Self, MeshValidationError> {
        let n_cells = mesh.cell_count();
        let n_faces = mesh.face_count();
        
        // 使用 Backend 分配单元面积缓冲区
        let mut cell_areas_vec = Vec::with_capacity(n_cells);
        for i in 0..n_cells {
            let area = mesh.cell_area(CellIndex(i)).unwrap_or(f64::NAN);
            if !area.is_finite() || area <= 0.0 {
                return Err(MeshValidationError::InvalidCellArea { cell: i, area });
            }
            let scalar = backend.config_scalar(area, "unstructured_mesh.cell_area");
            cell_areas_vec.push(scalar);
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
                return Err(MeshValidationError::InvalidFaceLength { face: i, length });
            }
            let scalar = backend.config_scalar(length, "unstructured_mesh.face_length");
            face_lengths_vec.push(scalar);
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
                return Err(MeshValidationError::FaceOwnerOutOfRange {
                    face,
                    owner: owner.get(),
                    n_cells,
                });
            }
            cell_face_map[owner.get()].push(face);
            if let Some(neighbor) = mesh.face_neighbor(FaceIndex(face)) {
                if neighbor.get() >= n_cells {
                    return Err(MeshValidationError::FaceNeighborOutOfRange {
                        face,
                        neighbor: neighbor.get(),
                        n_cells,
                    });
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

        adapter.validate()?;

        Ok(adapter)
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

// 移除默认 Backend 便捷构造函数，避免遗留接口

impl<B: Backend + Clone> MeshTopology<B> for UnstructuredMeshAdapter<B>
where
    B::Scalar: RuntimeScalar,
{
    fn n_cells(&self) -> usize {
        self.mesh.cell_count()
    }
    
    fn n_faces(&self) -> usize {
        self.mesh.face_count()
    }
    
    fn n_interior_faces(&self) -> usize {
        self.interior_face_indices.len()
    }
    
    fn n_nodes(&self) -> usize {
        self.mesh.node_count()
    }
    
    fn cell_center(&self, cell: usize) -> [B::Scalar; 2] {
        let center = self
            .mesh
            .cell_center_generic::<B>(CellIndex::new(cell))
            .expect("cell_center out of range");
        [
            center.x(),
            center.y(),
        ]
    }
    
    fn cell_area(&self, cell: usize) -> B::Scalar {
        self.cell_areas.get(cell).copied().unwrap_or(B::Scalar::ZERO)
    }
    
    fn face_normal(&self, face: usize) -> [B::Scalar; 2] {
        let normal = self
            .mesh
            .face_normal_generic::<B>(FaceIndex::new(face))
            .expect("face_normal out of range");
        [
            normal.x(),
            normal.y(),
        ]
    }
    
    fn face_length(&self, face: usize) -> B::Scalar {
        self.face_lengths.get(face).copied().unwrap_or(B::Scalar::ZERO)
    }
    
    fn face_center(&self, face: usize) -> [B::Scalar; 2] {
        let center = self
            .mesh
            .face_center_generic::<B>(FaceIndex::new(face))
            .expect("face_center out of range");
        [
            center.x(),
            center.y(),
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
