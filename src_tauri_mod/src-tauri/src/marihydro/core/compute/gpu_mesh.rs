// src-tauri/src/marihydro/core/compute/gpu_mesh.rs

//! GPU友好的网格数据结构
//!
//! 将非结构化网格拓扑转换为CRS（压缩稀疏行）格式，
//! 使其适合GPU并行计算。SmallVec等变长数据结构无法直接
//! 传输到GPU，因此需要使用固定布局的CSR格式。

#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};

/// 无效单元标记（用于边界面的neighbor）
pub const GPU_INVALID_CELL: u32 = u32::MAX;

/// GPU单元几何数据（SoA格式）
///
/// 将DVec2拆分为独立的x/y数组以满足GPU对齐要求
#[derive(Debug, Clone)]
pub struct GpuCellGeometry {
    /// 单元质心x坐标
    pub centroid_x: Vec<f32>,
    /// 单元质心y坐标  
    pub centroid_y: Vec<f32>,
    /// 单元面积
    pub area: Vec<f32>,
    /// 底床高程
    pub z_bed: Vec<f32>,
}

impl GpuCellGeometry {
    /// 创建指定大小的几何数据
    pub fn with_capacity(n_cells: usize) -> Self {
        Self {
            centroid_x: Vec::with_capacity(n_cells),
            centroid_y: Vec::with_capacity(n_cells),
            area: Vec::with_capacity(n_cells),
            z_bed: Vec::with_capacity(n_cells),
        }
    }

    /// 单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.centroid_x.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroid_x.is_empty()
    }

    /// 添加单元几何数据
    pub fn push(&mut self, cx: f32, cy: f32, area: f32, z: f32) {
        self.centroid_x.push(cx);
        self.centroid_y.push(cy);
        self.area.push(area);
        self.z_bed.push(z);
    }
}

/// GPU面几何数据（SoA格式）
#[derive(Debug, Clone)]
pub struct GpuFaceGeometry {
    /// 面中点x坐标
    pub centroid_x: Vec<f32>,
    /// 面中点y坐标
    pub centroid_y: Vec<f32>,
    /// 面外法向量x分量
    pub normal_x: Vec<f32>,
    /// 面外法向量y分量
    pub normal_y: Vec<f32>,
    /// 面长度
    pub length: Vec<f32>,
    /// owner到face中心的距离
    pub dist_owner: Vec<f32>,
    /// neighbor到face中心的距离（边界面设为dist_owner）
    pub dist_neighbor: Vec<f32>,
}

impl GpuFaceGeometry {
    /// 创建指定大小的几何数据
    pub fn with_capacity(n_faces: usize) -> Self {
        Self {
            centroid_x: Vec::with_capacity(n_faces),
            centroid_y: Vec::with_capacity(n_faces),
            normal_x: Vec::with_capacity(n_faces),
            normal_y: Vec::with_capacity(n_faces),
            length: Vec::with_capacity(n_faces),
            dist_owner: Vec::with_capacity(n_faces),
            dist_neighbor: Vec::with_capacity(n_faces),
        }
    }

    /// 面数量
    #[inline]
    pub fn len(&self) -> usize {
        self.centroid_x.len()
    }

    /// 是否为空
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.centroid_x.is_empty()
    }
}

/// GPU网格拓扑（CSR格式）
///
/// 使用压缩稀疏行格式存储变长邻接关系
#[derive(Debug, Clone)]
pub struct GpuMeshTopology {
    // ===== 面的owner/neighbor =====
    /// 面的owner单元索引
    pub face_owner: Vec<u32>,
    /// 面的neighbor单元索引（边界面为GPU_INVALID_CELL）
    pub face_neighbor: Vec<u32>,

    // ===== 单元到面的CSR映射 =====
    /// CSR偏移数组，长度为n_cells+1
    /// cell_faces_offset[i]到cell_faces_offset[i+1]之间的索引
    /// 对应单元i的所有面
    pub cell_faces_offset: Vec<u32>,
    /// CSR索引数组，存储面索引
    pub cell_faces_indices: Vec<u32>,

    // ===== 单元到邻居的CSR映射 =====
    /// CSR偏移数组
    pub cell_neighbors_offset: Vec<u32>,
    /// CSR索引数组，存储邻居单元索引
    pub cell_neighbors_indices: Vec<u32>,

    // ===== 边界信息 =====
    /// 边界面的索引列表
    pub boundary_face_indices: Vec<u32>,
    /// 每个面的边界ID（非边界面为u32::MAX）
    pub face_boundary_id: Vec<u32>,

    // ===== 统计信息 =====
    /// 总单元数
    pub n_cells: u32,
    /// 总面数
    pub n_faces: u32,
    /// 内部面数
    pub n_internal_faces: u32,
}

impl GpuMeshTopology {
    /// 创建空拓扑
    pub fn empty() -> Self {
        Self {
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            cell_faces_offset: vec![0],
            cell_faces_indices: Vec::new(),
            cell_neighbors_offset: vec![0],
            cell_neighbors_indices: Vec::new(),
            boundary_face_indices: Vec::new(),
            face_boundary_id: Vec::new(),
            n_cells: 0,
            n_faces: 0,
            n_internal_faces: 0,
        }
    }

    /// 创建指定大小的拓扑
    pub fn with_capacity(n_cells: usize, n_faces: usize, avg_faces_per_cell: usize) -> Self {
        Self {
            face_owner: Vec::with_capacity(n_faces),
            face_neighbor: Vec::with_capacity(n_faces),
            cell_faces_offset: Vec::with_capacity(n_cells + 1),
            cell_faces_indices: Vec::with_capacity(n_cells * avg_faces_per_cell),
            cell_neighbors_offset: Vec::with_capacity(n_cells + 1),
            cell_neighbors_indices: Vec::with_capacity(n_cells * avg_faces_per_cell),
            boundary_face_indices: Vec::new(),
            face_boundary_id: Vec::with_capacity(n_faces),
            n_cells: n_cells as u32,
            n_faces: n_faces as u32,
            n_internal_faces: 0,
        }
    }

    /// 获取单元的面索引范围
    #[inline]
    pub fn cell_face_range(&self, cell: u32) -> std::ops::Range<usize> {
        let start = self.cell_faces_offset[cell as usize] as usize;
        let end = self.cell_faces_offset[cell as usize + 1] as usize;
        start..end
    }

    /// 获取单元的所有面
    #[inline]
    pub fn cell_faces(&self, cell: u32) -> &[u32] {
        let range = self.cell_face_range(cell);
        &self.cell_faces_indices[range]
    }

    /// 获取单元的邻居索引范围
    #[inline]
    pub fn cell_neighbor_range(&self, cell: u32) -> std::ops::Range<usize> {
        let start = self.cell_neighbors_offset[cell as usize] as usize;
        let end = self.cell_neighbors_offset[cell as usize + 1] as usize;
        start..end
    }

    /// 获取单元的所有邻居
    #[inline]
    pub fn cell_neighbors(&self, cell: u32) -> &[u32] {
        let range = self.cell_neighbor_range(cell);
        &self.cell_neighbors_indices[range]
    }

    /// 验证CSR结构完整性
    pub fn validate(&self) -> Result<(), String> {
        // 检查offset数组长度
        if self.cell_faces_offset.len() != self.n_cells as usize + 1 {
            return Err(format!(
                "cell_faces_offset长度错误: {} != {}",
                self.cell_faces_offset.len(),
                self.n_cells + 1
            ));
        }

        if self.cell_neighbors_offset.len() != self.n_cells as usize + 1 {
            return Err(format!(
                "cell_neighbors_offset长度错误: {} != {}",
                self.cell_neighbors_offset.len(),
                self.n_cells + 1
            ));
        }

        // 检查offset单调递增
        for i in 1..self.cell_faces_offset.len() {
            if self.cell_faces_offset[i] < self.cell_faces_offset[i - 1] {
                return Err(format!(
                    "cell_faces_offset非单调递增: [{}]={} < [{}]={}",
                    i,
                    self.cell_faces_offset[i],
                    i - 1,
                    self.cell_faces_offset[i - 1]
                ));
            }
        }

        // 检查索引范围
        for &face_idx in &self.cell_faces_indices {
            if face_idx >= self.n_faces {
                return Err(format!(
                    "面索引越界: {} >= {}",
                    face_idx, self.n_faces
                ));
            }
        }

        // 检查owner/neighbor
        if self.face_owner.len() != self.n_faces as usize {
            return Err(format!(
                "face_owner长度错误: {} != {}",
                self.face_owner.len(),
                self.n_faces
            ));
        }

        for (i, &owner) in self.face_owner.iter().enumerate() {
            if owner >= self.n_cells {
                return Err(format!(
                    "face_owner[{}]越界: {} >= {}",
                    i, owner, self.n_cells
                ));
            }
        }

        Ok(())
    }
}

/// 完整的GPU网格数据
#[derive(Debug, Clone)]
pub struct GpuMeshData {
    /// 单元几何
    pub cells: GpuCellGeometry,
    /// 面几何
    pub faces: GpuFaceGeometry,
    /// 拓扑结构
    pub topology: GpuMeshTopology,
}

impl GpuMeshData {
    /// 创建空的GPU网格数据
    pub fn empty() -> Self {
        Self {
            cells: GpuCellGeometry::with_capacity(0),
            faces: GpuFaceGeometry::with_capacity(0),
            topology: GpuMeshTopology::empty(),
        }
    }

    /// 估计GPU内存占用（字节）
    pub fn gpu_memory_estimate(&self) -> usize {
        let f32_size = std::mem::size_of::<f32>();
        let u32_size = std::mem::size_of::<u32>();

        // 单元几何: 4 arrays * n_cells * f32
        let cell_geom = 4 * self.cells.len() * f32_size;

        // 面几何: 7 arrays * n_faces * f32
        let face_geom = 7 * self.faces.len() * f32_size;

        // 拓扑
        let topo = (self.topology.face_owner.len()
            + self.topology.face_neighbor.len()
            + self.topology.cell_faces_offset.len()
            + self.topology.cell_faces_indices.len()
            + self.topology.cell_neighbors_offset.len()
            + self.topology.cell_neighbors_indices.len()
            + self.topology.boundary_face_indices.len()
            + self.topology.face_boundary_id.len())
            * u32_size;

        cell_geom + face_geom + topo
    }

    /// 验证数据完整性
    pub fn validate(&self) -> Result<(), String> {
        // 验证拓扑
        self.topology.validate()?;

        // 验证几何数据长度
        if self.cells.len() != self.topology.n_cells as usize {
            return Err(format!(
                "单元几何长度与拓扑不匹配: {} != {}",
                self.cells.len(),
                self.topology.n_cells
            ));
        }

        if self.faces.len() != self.topology.n_faces as usize {
            return Err(format!(
                "面几何长度与拓扑不匹配: {} != {}",
                self.faces.len(),
                self.topology.n_faces
            ));
        }

        Ok(())
    }
}

// ============= POD类型用于GPU缓冲区 =============

/// 紧凑的单元数据（用于GPU uniform/storage buffer）
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuCellPod {
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub area: f32,
    pub z_bed: f32,
}

/// 紧凑的面数据
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFacePod {
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub normal_x: f32,
    pub normal_y: f32,
    pub length: f32,
    pub owner: u32,
    pub neighbor: u32,
    pub _padding: u32, // 对齐到32字节
}

/// 紧凑的状态数据（用于计算核心）
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuStatePod {
    pub h: f32,      // 水深
    pub hu: f32,     // x动量
    pub hv: f32,     // y动量
    pub z_bed: f32,  // 底床高程
}

/// 紧凑的通量数据
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuFluxPod {
    pub flux_h: f32,
    pub flux_hu: f32,
    pub flux_hv: f32,
    pub _padding: f32,
}

// ============= 转换trait =============

/// 从CPU网格转换到GPU网格的trait
pub trait ToGpuMesh {
    /// 转换为GPU友好的网格格式
    fn to_gpu_mesh(&self) -> GpuMeshData;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_cell_pod_size() {
        assert_eq!(std::mem::size_of::<GpuCellPod>(), 16);
    }

    #[test]
    fn test_gpu_face_pod_size() {
        assert_eq!(std::mem::size_of::<GpuFacePod>(), 32);
    }

    #[test]
    fn test_gpu_state_pod_size() {
        assert_eq!(std::mem::size_of::<GpuStatePod>(), 16);
    }

    #[test]
    fn test_gpu_flux_pod_size() {
        assert_eq!(std::mem::size_of::<GpuFluxPod>(), 16);
    }

    #[test]
    fn test_topology_csr_access() {
        let mut topo = GpuMeshTopology::with_capacity(3, 4, 4);
        
        // 设置offset
        topo.cell_faces_offset = vec![0, 2, 4, 6];
        topo.cell_faces_indices = vec![0, 1, 1, 2, 2, 3];
        topo.n_cells = 3;
        topo.n_faces = 4;

        // 测试访问
        assert_eq!(topo.cell_faces(0), &[0, 1]);
        assert_eq!(topo.cell_faces(1), &[1, 2]);
        assert_eq!(topo.cell_faces(2), &[2, 3]);
    }

    #[test]
    fn test_topology_validation() {
        let mut topo = GpuMeshTopology::empty();
        topo.n_cells = 2;
        topo.n_faces = 3;
        topo.cell_faces_offset = vec![0, 2, 4];
        topo.cell_neighbors_offset = vec![0, 1, 2];
        topo.face_owner = vec![0, 0, 1];
        topo.face_neighbor = vec![1, GPU_INVALID_CELL, GPU_INVALID_CELL];
        topo.cell_faces_indices = vec![0, 1, 1, 2];

        assert!(topo.validate().is_ok());
    }

    #[test]
    fn test_memory_estimate() {
        let mut data = GpuMeshData::empty();
        
        // 添加一些数据
        for i in 0..100 {
            data.cells.push(i as f32, i as f32, 1.0, 0.0);
        }
        
        let mem = data.gpu_memory_estimate();
        assert!(mem > 0);
    }
}
