// crates/mh_physics/src/gpu/mesh.rs

//! GPU 网格数据结构
//!
//! 将 FrozenMesh 转换为 GPU 可用的缓冲区布局。

use super::buffer::{GpuBufferUsage, TypedBuffer};
use bytemuck::{Pod, Zeroable};
use wgpu::{Device, Queue};

/// GPU 网格拓扑信息
#[derive(Debug, Clone, Copy)]
pub struct GpuMeshTopology {
    /// 单元数量
    pub num_cells: u32,
    /// 面数量
    pub num_faces: u32,
    /// 节点数量
    pub num_nodes: u32,
    /// 内部面数量
    pub num_interior_faces: u32,
    /// 边界面数量
    pub num_boundary_faces: u32,
}

/// GPU 单元几何缓冲区
pub struct GpuCellGeometry {
    /// 单元面积
    pub areas: TypedBuffer<f32>,
    /// 单元中心 X 坐标
    pub centers_x: TypedBuffer<f32>,
    /// 单元中心 Y 坐标
    pub centers_y: TypedBuffer<f32>,
    /// 单元特征长度 (用于 CFL)
    pub char_lengths: TypedBuffer<f32>,
    /// 底床高程
    pub z_bed: TypedBuffer<f32>,
}

/// GPU 面几何缓冲区
pub struct GpuFaceGeometry {
    /// 面中心 X 坐标
    pub centers_x: TypedBuffer<f32>,
    /// 面中心 Y 坐标
    pub centers_y: TypedBuffer<f32>,
    /// 法向量 X 分量
    pub normals_x: TypedBuffer<f32>,
    /// 法向量 Y 分量
    pub normals_y: TypedBuffer<f32>,
    /// 面长度
    pub lengths: TypedBuffer<f32>,
    /// 左侧高程
    pub z_left: TypedBuffer<f32>,
    /// 右侧高程
    pub z_right: TypedBuffer<f32>,
}

/// GPU 面拓扑缓冲区
pub struct GpuFaceTopology {
    /// Owner 单元索引
    pub owners: TypedBuffer<u32>,
    /// Neighbor 单元索引 (u32::MAX 表示边界)
    pub neighbors: TypedBuffer<u32>,
}

/// GPU 单元邻接关系 (CSR 格式)
pub struct GpuCellAdjacency {
    /// 单元-面偏移指针
    pub face_ptr: TypedBuffer<u32>,
    /// 单元-面索引列表
    pub face_idx: TypedBuffer<u32>,
}

/// GPU 网格数据
pub struct GpuMeshData {
    /// 拓扑信息
    pub topology: GpuMeshTopology,
    /// 单元几何
    pub cells: GpuCellGeometry,
    /// 面几何
    pub faces: GpuFaceGeometry,
    /// 面拓扑
    pub face_topo: GpuFaceTopology,
    /// 单元邻接
    pub adjacency: GpuCellAdjacency,
}

/// GPU 单元数据 Pod 类型 (用于一次性传输)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuCellPod {
    /// 单元面积
    pub area: f32,
    /// 中心 X
    pub cx: f32,
    /// 中心 Y
    pub cy: f32,
    /// 特征长度
    pub char_length: f32,
}

/// GPU 面数据 Pod 类型
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuFacePod {
    /// 中心 X
    pub cx: f32,
    /// 中心 Y
    pub cy: f32,
    /// 法向量 X
    pub nx: f32,
    /// 法向量 Y
    pub ny: f32,
    /// 面长度
    pub length: f32,
    /// Owner 单元
    pub owner: u32,
    /// Neighbor 单元
    pub neighbor: u32,
    /// 填充
    pub _pad: u32,
}

impl GpuMeshData {
    /// 从 FrozenMesh 创建 GPU 网格数据
    pub fn from_frozen_mesh(device: &Device, queue: &Queue, mesh: &mh_mesh::FrozenMesh) -> Self {
        let topology = GpuMeshTopology {
            num_cells: mesh.n_cells as u32,
            num_faces: mesh.n_faces as u32,
            num_nodes: mesh.n_nodes as u32,
            num_interior_faces: mesh.n_interior_faces as u32,
            num_boundary_faces: (mesh.n_faces - mesh.n_interior_faces) as u32,
        };

        // 转换单元数据
        let areas: Vec<f32> = mesh.cell_area.iter().map(|&x| x as f32).collect();
        let centers_x: Vec<f32> = mesh.cell_center.iter().map(|p| p.x as f32).collect();
        let centers_y: Vec<f32> = mesh.cell_center.iter().map(|p| p.y as f32).collect();
        let z_bed: Vec<f32> = mesh.cell_z_bed.iter().map(|&x| x as f32).collect();

        // 计算特征长度 (sqrt(area))
        let char_lengths: Vec<f32> = mesh.cell_area.iter().map(|&a| (a as f32).sqrt()).collect();

        let cells = GpuCellGeometry {
            areas: TypedBuffer::from_data(device, &areas, GpuBufferUsage::StorageReadOnly, Some("cell_areas")),
            centers_x: TypedBuffer::from_data(device, &centers_x, GpuBufferUsage::StorageReadOnly, Some("cell_cx")),
            centers_y: TypedBuffer::from_data(device, &centers_y, GpuBufferUsage::StorageReadOnly, Some("cell_cy")),
            char_lengths: TypedBuffer::from_data(device, &char_lengths, GpuBufferUsage::StorageReadOnly, Some("cell_char_length")),
            z_bed: TypedBuffer::from_data(device, &z_bed, GpuBufferUsage::StorageReadOnly, Some("cell_z_bed")),
        };

        // 转换面数据
        let face_centers_x: Vec<f32> = mesh.face_center.iter().map(|p| p.x as f32).collect();
        let face_centers_y: Vec<f32> = mesh.face_center.iter().map(|p| p.y as f32).collect();
        let face_normals_x: Vec<f32> = mesh.face_normal.iter().map(|n| n.x as f32).collect();
        let face_normals_y: Vec<f32> = mesh.face_normal.iter().map(|n| n.y as f32).collect();
        let face_lengths: Vec<f32> = mesh.face_length.iter().map(|&x| x as f32).collect();
        let face_z_left: Vec<f32> = mesh.face_z_left.iter().map(|&x| x as f32).collect();
        let face_z_right: Vec<f32> = mesh.face_z_right.iter().map(|&x| x as f32).collect();

        let faces = GpuFaceGeometry {
            centers_x: TypedBuffer::from_data(device, &face_centers_x, GpuBufferUsage::StorageReadOnly, Some("face_cx")),
            centers_y: TypedBuffer::from_data(device, &face_centers_y, GpuBufferUsage::StorageReadOnly, Some("face_cy")),
            normals_x: TypedBuffer::from_data(device, &face_normals_x, GpuBufferUsage::StorageReadOnly, Some("face_nx")),
            normals_y: TypedBuffer::from_data(device, &face_normals_y, GpuBufferUsage::StorageReadOnly, Some("face_ny")),
            lengths: TypedBuffer::from_data(device, &face_lengths, GpuBufferUsage::StorageReadOnly, Some("face_length")),
            z_left: TypedBuffer::from_data(device, &face_z_left, GpuBufferUsage::StorageReadOnly, Some("face_z_left")),
            z_right: TypedBuffer::from_data(device, &face_z_right, GpuBufferUsage::StorageReadOnly, Some("face_z_right")),
        };

        // 面拓扑
        let face_topo = GpuFaceTopology {
            owners: TypedBuffer::from_data(device, &mesh.face_owner, GpuBufferUsage::StorageReadOnly, Some("face_owner")),
            neighbors: TypedBuffer::from_data(device, &mesh.face_neighbor, GpuBufferUsage::StorageReadOnly, Some("face_neighbor")),
        };

        // 单元邻接 (CSR)
        let face_ptr: Vec<u32> = mesh.cell_face_offsets.iter().map(|&x| x as u32).collect();
        let face_idx: Vec<u32> = mesh.cell_face_indices.clone();

        let adjacency = GpuCellAdjacency {
            face_ptr: TypedBuffer::from_data(device, &face_ptr, GpuBufferUsage::StorageReadOnly, Some("cell_face_ptr")),
            face_idx: TypedBuffer::from_data(device, &face_idx, GpuBufferUsage::StorageReadOnly, Some("cell_face_idx")),
        };

        Self {
            topology,
            cells,
            faces,
            face_topo,
            adjacency,
        }
    }

    /// 获取单元数量
    #[inline]
    pub fn num_cells(&self) -> u32 {
        self.topology.num_cells
    }

    /// 获取面数量
    #[inline]
    pub fn num_faces(&self) -> u32 {
        self.topology.num_faces
    }

    /// 获取内部面数量
    #[inline]
    pub fn num_interior_faces(&self) -> u32 {
        self.topology.num_interior_faces
    }

    /// 获取边界面数量
    #[inline]
    pub fn num_boundary_faces(&self) -> u32 {
        self.topology.num_boundary_faces
    }
}

/// 网格转 GPU trait
pub trait ToGpuMesh {
    /// 转换为 GPU 网格数据
    fn to_gpu(&self, device: &Device, queue: &Queue) -> GpuMeshData;
}

impl ToGpuMesh for mh_mesh::FrozenMesh {
    fn to_gpu(&self, device: &Device, queue: &Queue) -> GpuMeshData {
        GpuMeshData::from_frozen_mesh(device, queue, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_cell_pod_layout() {
        assert_eq!(std::mem::size_of::<GpuCellPod>(), 16);
    }

    #[test]
    fn test_gpu_face_pod_layout() {
        assert_eq!(std::mem::size_of::<GpuFacePod>(), 32);
    }
}
