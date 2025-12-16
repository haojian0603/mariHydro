//! 非结构化网格适配器
//!
//! 将现有的 FrozenMesh/PhysicsMesh 适配到 MeshTopology trait。

use crate::adapter::PhysicsMesh;
use crate::core::{Backend, CpuBackend};
use super::topology::{MeshKind, MeshTopology};
use mh_runtime::{CellIndex, FaceIndex};
use std::sync::Arc;

/// 非结构化网格适配器
pub struct UnstructuredMeshAdapter<B: Backend = CpuBackend<f64>> {
    /// 原始网格引用
    mesh: Arc<PhysicsMesh>,
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

impl UnstructuredMeshAdapter<CpuBackend<f64>> {
    /// 从 PhysicsMesh 创建适配器（默认 f64 精度）
    pub fn from_physics_mesh(mesh: Arc<PhysicsMesh>) -> Self {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();
        
        // 构建单元面积缓冲区
        let mut cell_areas = vec![0.0; n_cells];
        for i in 0..n_cells {
            cell_areas[i] = mesh.cell_area(mh_runtime::CellIndex(i)).unwrap_or(0.0);
        }
        
        // 构建面长度缓冲区
        let mut face_lengths = vec![0.0; n_faces];
        for i in 0..n_faces {
            face_lengths[i] = mesh.face_length(FaceIndex(i));
        }
        
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
            cell_face_map[owner.get()].push(face);
            if let Some(neighbor) = mesh.face_neighbor(FaceIndex(face)) {
                cell_face_map[neighbor.get()].push(face);
            }
        }
        
        Self {
            mesh,
            cell_areas,
            face_lengths,
            boundary_face_indices,
            interior_face_indices,
            cell_face_map,
        }
    }
    
    /// 获取原始网格引用
    pub fn inner(&self) -> &PhysicsMesh {
        &self.mesh
    }
}

impl MeshTopology<CpuBackend<f64>> for UnstructuredMeshAdapter<CpuBackend<f64>> {
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
    
    #[allow(deprecated)]
    fn cell_center(&self, cell: usize) -> [f64; 2] {
        let c = self.mesh.cell_center(cell);
        [c.x, c.y]
    }
    
    fn cell_area(&self, cell: usize) -> f64 {
        self.mesh.cell_area(CellIndex(cell)).unwrap_or(0.0)
    }
    
    #[allow(deprecated)]
    fn face_normal(&self, face: usize) -> [f64; 2] {
        let n = self.mesh.face_normal(face);
        [n.x, n.y]
    }
    
    fn face_length(&self, face: usize) -> f64 {
        self.mesh.face_length(FaceIndex(face))
    }
    
    #[allow(deprecated)]
    fn face_center(&self, face: usize) -> [f64; 2] {
        let c = self.mesh.face_center(face);
        [c.x, c.y]
    }
    
    fn face_owner(&self, face: usize) -> usize {
        self.mesh.face_owner(FaceIndex(face)).get()
    }
    
    fn face_neighbor(&self, face: usize) -> Option<usize> {
        self.mesh.face_neighbor(FaceIndex(face)).map(|c| c.get())
    }
    
    fn cell_faces(&self, cell: usize) -> &[usize] {
        &self.cell_face_map[cell]
    }
    
    fn cell_neighbors(&self, cell: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for &face in &self.cell_face_map[cell] {
            let owner = self.face_owner(face);
            if owner == cell {
                if let Some(neighbor) = self.face_neighbor(face) {
                    neighbors.push(neighbor);
                }
            } else {
                neighbors.push(owner);
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
    
    fn cell_areas_buffer(&self) -> &Vec<f64> {
        &self.cell_areas
    }
    
    fn face_lengths_buffer(&self) -> &Vec<f64> {
        &self.face_lengths
    }
}
