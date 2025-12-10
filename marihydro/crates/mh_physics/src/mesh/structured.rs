//! 结构化网格（骨架实现）
//!
//! 预留结构化网格支持，当前仅提供 trait 实现骨架。

use crate::core::Backend;
use super::topology::{MeshKind, MeshTopology};
use std::marker::PhantomData;

/// 结构化网格（未实现）
#[allow(dead_code)]
pub struct StructuredMesh<B: Backend> {
    nx: usize,
    ny: usize,
    dx: B::Scalar,
    dy: B::Scalar,
    cell_areas: B::Buffer<B::Scalar>,
    face_lengths: B::Buffer<B::Scalar>,
    boundary_faces_cache: Vec<usize>,
    interior_faces_cache: Vec<usize>,
    cell_faces_cache: Vec<Vec<usize>>,
    _marker: PhantomData<B>,
}

impl<B: Backend> StructuredMesh<B> {
    /// 创建结构化网格（未实现）
    pub fn new(_nx: usize, _ny: usize, _dx: f64, _dy: f64) -> Self {
        unimplemented!("StructuredMesh is not yet implemented")
    }
}

impl<B: Backend> MeshTopology<B> for StructuredMesh<B> {
    fn n_cells(&self) -> usize {
        self.nx * self.ny
    }
    
    fn n_faces(&self) -> usize {
        // 内部面 + 边界面
        (self.nx - 1) * self.ny + self.nx * (self.ny - 1) + 2 * (self.nx + self.ny)
    }
    
    fn n_interior_faces(&self) -> usize {
        (self.nx - 1) * self.ny + self.nx * (self.ny - 1)
    }
    
    fn n_nodes(&self) -> usize {
        (self.nx + 1) * (self.ny + 1)
    }
    
    fn cell_center(&self, _cell: usize) -> [B::Scalar; 2] {
        unimplemented!()
    }
    
    fn cell_area(&self, _cell: usize) -> B::Scalar {
        self.dx * self.dy
    }
    
    fn face_normal(&self, _face: usize) -> [B::Scalar; 2] {
        unimplemented!()
    }
    
    fn face_length(&self, _face: usize) -> B::Scalar {
        unimplemented!()
    }
    
    fn face_center(&self, _face: usize) -> [B::Scalar; 2] {
        unimplemented!()
    }
    
    fn face_owner(&self, _face: usize) -> usize {
        unimplemented!()
    }
    
    fn face_neighbor(&self, _face: usize) -> Option<usize> {
        unimplemented!()
    }
    
    fn cell_faces(&self, _cell: usize) -> &[usize] {
        unimplemented!()
    }
    
    fn cell_neighbors(&self, _cell: usize) -> Vec<usize> {
        unimplemented!()
    }
    
    fn boundary_faces(&self) -> &[usize] {
        unimplemented!()
    }
    
    fn interior_faces(&self) -> &[usize] {
        unimplemented!()
    }
    
    fn mesh_kind(&self) -> MeshKind {
        MeshKind::Structured { nx: self.nx, ny: self.ny }
    }
    
    fn cell_areas_buffer(&self) -> &B::Buffer<B::Scalar> {
        &self.cell_areas
    }
    
    fn face_lengths_buffer(&self) -> &B::Buffer<B::Scalar> {
        &self.face_lengths
    }
}
