// src-tauri/src/marihydro/core/compute/mesh_converter.rs

//! CPU网格到GPU网格的转换
//!
//! 将UnstructuredMesh转换为GPU友好的GpuMeshData格式

use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::CellIndex;

use super::gpu_mesh::{
    GpuCellGeometry, GpuFaceGeometry, GpuMeshData, GpuMeshTopology, GPU_INVALID_CELL,
};

/// 网格转换器
pub struct MeshConverter;

impl MeshConverter {
    /// 将任何实现MeshAccess的网格转换为GPU格式
    pub fn convert<M: MeshAccess>(mesh: &M) -> GpuMeshData {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();
        let n_internal = mesh.n_internal_faces();

        // 转换单元几何
        let cells = Self::convert_cell_geometry(mesh, n_cells);

        // 转换面几何
        let faces = Self::convert_face_geometry(mesh, n_faces);

        // 转换拓扑
        let topology = Self::convert_topology(mesh, n_cells, n_faces, n_internal);

        GpuMeshData {
            cells,
            faces,
            topology,
        }
    }

    /// 转换单元几何数据
    fn convert_cell_geometry<M: MeshAccess>(mesh: &M, n_cells: usize) -> GpuCellGeometry {
        let mut geom = GpuCellGeometry::with_capacity(n_cells);

        let centroids = mesh.all_cell_centroids();
        let areas = mesh.all_cell_areas();

        for i in 0..n_cells {
            let centroid = centroids[i];
            let area = areas[i];
            
            // 底床高程需要从状态获取，这里暂设为0
            geom.push(
                centroid.x as f32,
                centroid.y as f32,
                area as f32,
                0.0, // z_bed
            );
        }

        geom
    }

    /// 转换面几何数据
    fn convert_face_geometry<M: MeshAccess>(mesh: &M, n_faces: usize) -> GpuFaceGeometry {
        let mut geom = GpuFaceGeometry::with_capacity(n_faces);

        for f in 0..n_faces {
            let face = crate::marihydro::core::types::FaceIndex(f);
            
            let centroid = mesh.face_centroid(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            
            // 计算到owner/neighbor的距离
            let owner_centroid = mesh.cell_centroid(owner);
            let dist_owner = (centroid - owner_centroid).length() as f32;
            
            let dist_neighbor = if neighbor.is_valid() {
                let neighbor_centroid = mesh.cell_centroid(neighbor);
                (centroid - neighbor_centroid).length() as f32
            } else {
                dist_owner // 边界面使用owner距离
            };

            geom.centroid_x.push(centroid.x as f32);
            geom.centroid_y.push(centroid.y as f32);
            geom.normal_x.push(normal.x as f32);
            geom.normal_y.push(normal.y as f32);
            geom.length.push(length as f32);
            geom.dist_owner.push(dist_owner);
            geom.dist_neighbor.push(dist_neighbor);
        }

        geom
    }

    /// 转换拓扑数据为CSR格式
    fn convert_topology<M: MeshAccess>(
        mesh: &M,
        n_cells: usize,
        n_faces: usize,
        n_internal: usize,
    ) -> GpuMeshTopology {
        let mut topo = GpuMeshTopology::with_capacity(n_cells, n_faces, 6);

        // 面的owner/neighbor
        for f in 0..n_faces {
            let face = crate::marihydro::core::types::FaceIndex(f);
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);

            topo.face_owner.push(owner.0 as u32);
            topo.face_neighbor.push(if neighbor.is_valid() {
                neighbor.0 as u32
            } else {
                GPU_INVALID_CELL
            });

            // 边界ID
            if let Some(bid) = mesh.boundary_id(face) {
                topo.face_boundary_id.push(bid.0 as u32);
            } else {
                topo.face_boundary_id.push(u32::MAX);
            }
        }

        // 构建单元到面的CSR
        topo.cell_faces_offset.push(0);
        for c in 0..n_cells {
            let cell = CellIndex(c);
            let cell_faces = mesh.cell_faces(cell);
            
            for face in cell_faces {
                topo.cell_faces_indices.push(face.0 as u32);
            }
            
            topo.cell_faces_offset.push(topo.cell_faces_indices.len() as u32);
        }

        // 构建单元到邻居的CSR
        topo.cell_neighbors_offset.push(0);
        for c in 0..n_cells {
            let cell = CellIndex(c);
            let neighbors = mesh.cell_neighbors(cell);
            
            for &neighbor in neighbors {
                topo.cell_neighbors_indices.push(neighbor.0 as u32);
            }
            
            topo.cell_neighbors_offset.push(topo.cell_neighbors_indices.len() as u32);
        }

        // 边界面索引列表
        for f in n_internal..n_faces {
            topo.boundary_face_indices.push(f as u32);
        }

        topo.n_cells = n_cells as u32;
        topo.n_faces = n_faces as u32;
        topo.n_internal_faces = n_internal as u32;

        topo
    }
}

/// 用于设置底床高程的扩展trait
pub trait SetBedElevation {
    /// 设置底床高程数据
    fn set_bed_elevation(&mut self, z_bed: &[f64]);
}

impl SetBedElevation for GpuMeshData {
    fn set_bed_elevation(&mut self, z_bed: &[f64]) {
        debug_assert_eq!(z_bed.len(), self.cells.len());
        for (i, &z) in z_bed.iter().enumerate() {
            self.cells.z_bed[i] = z as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 简单的测试网格mock
    struct SimpleMesh {
        n_cells: usize,
        n_faces: usize,
    }

    // 需要完整的MeshAccess实现才能测试
    // 这里仅验证编译

    #[test]
    fn test_gpu_invalid_cell() {
        assert_eq!(GPU_INVALID_CELL, u32::MAX);
    }
}
