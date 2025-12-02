// marihydro\crates\mh_mesh\src/frozen.rs

//! 冻结网格
//!
//! 从半边网格导出的只读 SoA 布局，优化计算性能。
//!
//! # 设计要点
//!
//! 1. **SoA布局**: 类似旧 UnstructuredMesh 的数组布局
//! 2. **只读**: 冻结后不可修改
//! 3. **空间索引**: 内置 R-tree 用于空间查询
//! 4. **零拷贝序列化**: 支持 mmap 加载

use mh_geo::{Point2D, Point3D};
use serde::{Deserialize, Serialize};

/// 冻结网格
///
/// 从 HalfEdgeMesh 导出的只读计算用网格
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenMesh {
    // ===== 节点数据 =====
    /// 节点数量
    pub n_nodes: usize,
    /// 节点坐标 (3D)
    pub node_coords: Vec<Point3D>,

    // ===== 单元数据 =====
    /// 单元数量
    pub n_cells: usize,
    /// 单元中心坐标（2D，用于空间查询）
    pub cell_center: Vec<Point2D>,
    /// 单元面积
    pub cell_area: Vec<f64>,
    /// 单元底床高程
    pub cell_z_bed: Vec<f64>,
    /// 单元节点索引 (压缩格式: offsets + indices)
    pub cell_node_offsets: Vec<usize>,
    /// 单元节点索引列表
    pub cell_node_indices: Vec<u32>,
    /// 单元面索引 (压缩格式)
    pub cell_face_offsets: Vec<usize>,
    /// 单元面索引列表
    pub cell_face_indices: Vec<u32>,
    /// 单元邻居索引 (压缩格式)
    pub cell_neighbor_offsets: Vec<usize>,
    /// 单元邻居索引列表 (u32::MAX 表示无邻居)
    pub cell_neighbor_indices: Vec<u32>,

    // ===== 面数据 =====
    /// 面总数
    pub n_faces: usize,
    /// 内部面数量
    pub n_interior_faces: usize,
    /// 面中心坐标（2D）
    pub face_center: Vec<Point2D>,
    /// 面法向量 (3D，用于坡度计算)
    pub face_normal: Vec<Point3D>,
    /// 面长度
    pub face_length: Vec<f64>,
    /// 面左侧高程
    pub face_z_left: Vec<f64>,
    /// 面右侧高程
    pub face_z_right: Vec<f64>,
    /// 面 owner 单元索引
    pub face_owner: Vec<u32>,
    /// 面 neighbor 单元索引 (u32::MAX 表示边界)
    pub face_neighbor: Vec<u32>,
    /// 面到 owner 中心的向量（2D）
    pub face_delta_owner: Vec<Point2D>,
    /// 面到 neighbor 中心的向量（2D）
    pub face_delta_neighbor: Vec<Point2D>,
    /// owner 到 neighbor 的距离
    pub face_dist_o2n: Vec<f64>,

    // ===== 边界数据 =====
    /// 边界面索引列表
    pub boundary_face_indices: Vec<u32>,
    /// 边界名称
    pub boundary_names: Vec<String>,
    /// 面的边界ID (None表示内部面)
    pub face_boundary_id: Vec<Option<u32>>,

    // ===== 统计 =====
    /// 最小单元尺寸
    pub min_cell_size: f64,
    /// 最大单元尺寸
    pub max_cell_size: f64,
}

impl Default for FrozenMesh {
    fn default() -> Self {
        Self::empty()
    }
}

impl FrozenMesh {
    /// 创建空的冻结网格
    pub fn empty() -> Self {
        Self {
            n_nodes: 0,
            node_coords: Vec::new(),
            n_cells: 0,
            cell_center: Vec::new(),
            cell_area: Vec::new(),
            cell_z_bed: Vec::new(),
            cell_node_offsets: vec![0],
            cell_node_indices: Vec::new(),
            cell_face_offsets: vec![0],
            cell_face_indices: Vec::new(),
            cell_neighbor_offsets: vec![0],
            cell_neighbor_indices: Vec::new(),
            n_faces: 0,
            n_interior_faces: 0,
            face_center: Vec::new(),
            face_normal: Vec::new(),
            face_length: Vec::new(),
            face_z_left: Vec::new(),
            face_z_right: Vec::new(),
            face_owner: Vec::new(),
            face_neighbor: Vec::new(),
            face_delta_owner: Vec::new(),
            face_delta_neighbor: Vec::new(),
            face_dist_o2n: Vec::new(),
            boundary_face_indices: Vec::new(),
            boundary_names: Vec::new(),
            face_boundary_id: Vec::new(),
            min_cell_size: f64::MAX,
            max_cell_size: 0.0,
        }
    }

    // =========================================================================
    // 基本统计
    // =========================================================================

    /// 节点数量
    #[inline]
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// 内部面数量
    #[inline]
    pub fn n_interior_faces(&self) -> usize {
        self.n_interior_faces
    }

    /// 边界面数量
    #[inline]
    pub fn n_boundary_faces(&self) -> usize {
        self.n_faces - self.n_interior_faces
    }

    // =========================================================================
    // 单元访问
    // =========================================================================

    /// 获取单元中心
    #[inline]
    pub fn cell_center(&self, cell: usize) -> Point2D {
        self.cell_center[cell]
    }

    /// 获取单元面积
    #[inline]
    pub fn cell_area(&self, cell: usize) -> f64 {
        self.cell_area[cell]
    }

    /// 获取单元底床高程
    #[inline]
    pub fn cell_z_bed(&self, cell: usize) -> f64 {
        self.cell_z_bed[cell]
    }

    /// 获取单元的节点索引
    #[inline]
    pub fn cell_nodes(&self, cell: usize) -> &[u32] {
        let start = self.cell_node_offsets[cell];
        let end = self.cell_node_offsets[cell + 1];
        &self.cell_node_indices[start..end]
    }

    /// 获取单元的面索引
    #[inline]
    pub fn cell_faces(&self, cell: usize) -> &[u32] {
        let start = self.cell_face_offsets[cell];
        let end = self.cell_face_offsets[cell + 1];
        &self.cell_face_indices[start..end]
    }

    /// 获取单元的邻居索引
    #[inline]
    pub fn cell_neighbors(&self, cell: usize) -> &[u32] {
        let start = self.cell_neighbor_offsets[cell];
        let end = self.cell_neighbor_offsets[cell + 1];
        &self.cell_neighbor_indices[start..end]
    }

    // =========================================================================
    // 面访问
    // =========================================================================

    /// 获取面中心
    #[inline]
    pub fn face_center(&self, face: usize) -> Point2D {
        self.face_center[face]
    }

    /// 获取面法向量（3D）
    #[inline]
    pub fn face_normal(&self, face: usize) -> Point3D {
        self.face_normal[face]
    }

    /// 获取面长度
    #[inline]
    pub fn face_length(&self, face: usize) -> f64 {
        self.face_length[face]
    }

    /// 获取面 owner
    #[inline]
    pub fn face_owner(&self, face: usize) -> u32 {
        self.face_owner[face]
    }

    /// 获取面 neighbor
    #[inline]
    pub fn face_neighbor(&self, face: usize) -> Option<u32> {
        let n = self.face_neighbor[face];
        if n == u32::MAX {
            None
        } else {
            Some(n)
        }
    }

    /// 判断是否为边界面
    #[inline]
    pub fn is_boundary_face(&self, face: usize) -> bool {
        face >= self.n_interior_faces
    }

    // =========================================================================
    // 节点访问
    // =========================================================================

    /// 获取节点3D坐标
    #[inline]
    pub fn node_coords(&self, node: usize) -> Point3D {
        self.node_coords[node]
    }

    /// 获取节点2D坐标 (x, y)
    #[inline]
    pub fn node_xy(&self, node: usize) -> Point2D {
        self.node_coords[node].xy()
    }

    /// 获取节点高程 (z)
    #[inline]
    pub fn node_z(&self, node: usize) -> f64 {
        self.node_coords[node].z
    }

    // =========================================================================
    // 范围迭代
    // =========================================================================

    /// 内部面索引范围
    #[inline]
    pub fn interior_faces(&self) -> std::ops::Range<usize> {
        0..self.n_interior_faces
    }

    /// 边界面索引范围
    #[inline]
    pub fn boundary_faces(&self) -> std::ops::Range<usize> {
        self.n_interior_faces..self.n_faces
    }

    /// 单元索引范围
    #[inline]
    pub fn cells(&self) -> std::ops::Range<usize> {
        0..self.n_cells
    }

    /// 节点索引范围
    #[inline]
    pub fn nodes(&self) -> std::ops::Range<usize> {
        0..self.n_nodes
    }

    // =========================================================================
    // 统计信息
    // =========================================================================

    /// 计算统计信息
    pub fn statistics(&self) -> MeshStatistics {
        let mut min_area = f64::MAX;
        let mut max_area = f64::MIN;
        let mut total_area = 0.0;

        for &area in &self.cell_area {
            min_area = min_area.min(area);
            max_area = max_area.max(area);
            total_area += area;
        }

        let mut min_length = f64::MAX;
        let mut max_length = f64::MIN;

        for &len in &self.face_length {
            min_length = min_length.min(len);
            max_length = max_length.max(len);
        }

        MeshStatistics {
            n_cells: self.n_cells,
            n_faces: self.n_faces,
            n_interior_faces: self.n_interior_faces,
            n_boundary_faces: self.n_faces - self.n_interior_faces,
            n_nodes: self.n_nodes,
            total_area,
            min_cell_area: min_area,
            max_cell_area: max_area,
            min_edge_length: min_length,
            max_edge_length: max_length,
        }
    }

    /// 验证网格完整性
    pub fn validate(&self) -> Result<(), String> {
        // 检查数组长度
        if self.cell_center.len() != self.n_cells {
            return Err(format!(
                "cell_center length {} != n_cells {}",
                self.cell_center.len(),
                self.n_cells
            ));
        }

        if self.node_coords.len() != self.n_nodes {
            return Err(format!(
                "node_coords length {} != n_nodes {}",
                self.node_coords.len(),
                self.n_nodes
            ));
        }

        if self.face_center.len() != self.n_faces {
            return Err(format!(
                "face_center length {} != n_faces {}",
                self.face_center.len(),
                self.n_faces
            ));
        }

        // 检查偏移数组
        if self.cell_node_offsets.len() != self.n_cells + 1 {
            return Err("cell_node_offsets length mismatch".to_string());
        }

        // 检查 owner/neighbor
        for (i, &owner) in self.face_owner.iter().enumerate() {
            if owner as usize >= self.n_cells {
                return Err(format!("face {} owner {} out of range", i, owner));
            }
        }

        for (i, &neighbor) in self.face_neighbor.iter().enumerate() {
            if neighbor != u32::MAX && neighbor as usize >= self.n_cells {
                return Err(format!("face {} neighbor {} out of range", i, neighbor));
            }
        }

        Ok(())
    }
}

/// 网格统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStatistics {
    pub n_cells: usize,
    pub n_faces: usize,
    pub n_interior_faces: usize,
    pub n_boundary_faces: usize,
    pub n_nodes: usize,
    pub total_area: f64,
    pub min_cell_area: f64,
    pub max_cell_area: f64,
    pub min_edge_length: f64,
    pub max_edge_length: f64,
}

impl std::fmt::Display for MeshStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== 网格统计 ===")?;
        writeln!(f, "单元数: {}", self.n_cells)?;
        writeln!(
            f,
            "面数: {} (内部: {}, 边界: {})",
            self.n_faces, self.n_interior_faces, self.n_boundary_faces
        )?;
        writeln!(f, "节点数: {}", self.n_nodes)?;
        writeln!(f, "总面积: {:.2} m²", self.total_area)?;
        writeln!(
            f,
            "单元面积: [{:.2}, {:.2}] m²",
            self.min_cell_area, self.max_cell_area
        )?;
        writeln!(
            f,
            "边长: [{:.2}, {:.2}] m",
            self.min_edge_length, self.max_edge_length
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_frozen_mesh() {
        let mesh = FrozenMesh::empty();
        assert_eq!(mesh.n_cells(), 0);
        assert_eq!(mesh.n_faces(), 0);
        assert_eq!(mesh.n_nodes(), 0);
    }

    #[test]
    fn test_validate_empty() {
        let mesh = FrozenMesh::empty();
        assert!(mesh.validate().is_ok());
    }
}
