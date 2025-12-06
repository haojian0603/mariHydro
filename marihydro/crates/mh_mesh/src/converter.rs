// marihydro\crates\mh_mesh\src/converter.rs

//! 网格格式转换器
//!
//! 支持不同网格表示之间的转换，以及为 GPU 传输准备简化数据。
//!
//! # 功能特性
//!
//! - `SimpleMeshData`: 简化网格数据结构，适用于 GPU 传输
//! - `MeshStatisticsExt`: 扩展的网格统计信息
//! - 格式转换工具函数
//!
//! # 示例
//!
//! ```ignore
//! use mh_mesh::converter::SimpleMeshData;
//!
//! let frozen_mesh = half_edge_mesh.freeze();
//! let simple_data = SimpleMeshData::from_frozen(&frozen_mesh);
//!
//! println!("GPU 数据大小: {} bytes", simple_data.memory_usage());
//! ```

use crate::frozen::FrozenMesh;
use serde::{Deserialize, Serialize};

/// 简化网格数据（用于 GPU 传输）
///
/// 将网格数据转换为连续的数组格式，便于 GPU 缓冲区传输。
/// 使用 f32 以减少内存占用和传输带宽。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleMeshData {
    /// 节点数
    pub n_nodes: usize,
    /// 单元数
    pub n_cells: usize,
    /// 面数
    pub n_faces: usize,
    /// 内部面数
    pub n_interior_faces: usize,

    /// 节点坐标 (x, y, z) 交错存储
    /// 长度: n_nodes * 3
    pub node_coords: Vec<f32>,

    /// 单元中心坐标 (x, y) 交错存储
    /// 长度: n_cells * 2
    pub cell_centers: Vec<f32>,

    /// 单元面积
    /// 长度: n_cells
    pub cell_areas: Vec<f32>,

    /// 单元节点索引（压缩格式）
    pub cell_nodes: Vec<u32>,

    /// 单元节点偏移
    /// 长度: n_cells + 1
    pub cell_offsets: Vec<u32>,

    /// 面 owner 单元索引
    /// 长度: n_faces
    pub face_owner: Vec<u32>,

    /// 面 neighbor 单元索引 (u32::MAX = 边界)
    /// 长度: n_faces
    pub face_neighbor: Vec<u32>,

    /// 面法向量 (nx, ny) 交错存储
    /// 长度: n_faces * 2
    pub face_normals: Vec<f32>,

    /// 面长度
    /// 长度: n_faces
    pub face_lengths: Vec<f32>,

    /// 面中心坐标 (x, y) 交错存储
    /// 长度: n_faces * 2
    pub face_centers: Vec<f32>,

    /// 边界面索引列表
    pub boundary_faces: Vec<u32>,
}

impl SimpleMeshData {
    /// 从 FrozenMesh 转换
    ///
    /// 将 FrozenMesh 中的数据转换为适合 GPU 使用的格式。
    pub fn from_frozen(mesh: &FrozenMesh) -> Self {
        // 节点坐标
        let node_coords: Vec<f32> = mesh
            .node_coords
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32, p.z as f32])
            .collect();

        // 单元中心
        let cell_centers: Vec<f32> = mesh
            .cell_center
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32])
            .collect();

        // 单元面积
        let cell_areas: Vec<f32> = mesh.cell_area.iter().map(|&x| x as f32).collect();

        // 单元节点索引
        let cell_nodes: Vec<u32> = mesh.cell_node_indices.clone();
        let cell_offsets: Vec<u32> = mesh.cell_node_offsets.iter().map(|&x| x as u32).collect();

        // 面数据
        let face_owner = mesh.face_owner.clone();
        let face_neighbor = mesh.face_neighbor.clone();

        // 面法向量（只取 x, y 分量）
        let face_normals: Vec<f32> = mesh
            .face_normal
            .iter()
            .flat_map(|n| [n.x as f32, n.y as f32])
            .collect();

        // 面长度
        let face_lengths: Vec<f32> = mesh.face_length.iter().map(|&x| x as f32).collect();

        // 面中心
        let face_centers: Vec<f32> = mesh
            .face_center
            .iter()
            .flat_map(|p| [p.x as f32, p.y as f32])
            .collect();

        // 边界面索引
        let boundary_faces = mesh.boundary_face_indices.clone();

        Self {
            n_nodes: mesh.n_nodes,
            n_cells: mesh.n_cells,
            n_faces: mesh.n_faces,
            n_interior_faces: mesh.n_interior_faces,
            node_coords,
            cell_centers,
            cell_areas,
            cell_nodes,
            cell_offsets,
            face_owner,
            face_neighbor,
            face_normals,
            face_lengths,
            face_centers,
            boundary_faces,
        }
    }

    /// 计算内存占用（字节）
    pub fn memory_usage(&self) -> usize {
        self.node_coords.len() * 4
            + self.cell_centers.len() * 4
            + self.cell_areas.len() * 4
            + self.cell_nodes.len() * 4
            + self.cell_offsets.len() * 4
            + self.face_owner.len() * 4
            + self.face_neighbor.len() * 4
            + self.face_normals.len() * 4
            + self.face_lengths.len() * 4
            + self.face_centers.len() * 4
            + self.boundary_faces.len() * 4
    }

    /// 验证数据完整性
    pub fn validate(&self) -> Result<(), String> {
        // 检查节点坐标长度
        if self.node_coords.len() != self.n_nodes * 3 {
            return Err(format!(
                "node_coords 长度 {} != n_nodes * 3 ({})",
                self.node_coords.len(),
                self.n_nodes * 3
            ));
        }

        // 检查单元中心长度
        if self.cell_centers.len() != self.n_cells * 2 {
            return Err(format!(
                "cell_centers 长度 {} != n_cells * 2 ({})",
                self.cell_centers.len(),
                self.n_cells * 2
            ));
        }

        // 检查单元面积长度
        if self.cell_areas.len() != self.n_cells {
            return Err(format!(
                "cell_areas 长度 {} != n_cells ({})",
                self.cell_areas.len(),
                self.n_cells
            ));
        }

        // 检查偏移数组长度
        if self.cell_offsets.len() != self.n_cells + 1 {
            return Err(format!(
                "cell_offsets 长度 {} != n_cells + 1 ({})",
                self.cell_offsets.len(),
                self.n_cells + 1
            ));
        }

        // 检查面数据长度
        if self.face_owner.len() != self.n_faces {
            return Err(format!(
                "face_owner 长度 {} != n_faces ({})",
                self.face_owner.len(),
                self.n_faces
            ));
        }

        if self.face_neighbor.len() != self.n_faces {
            return Err(format!(
                "face_neighbor 长度 {} != n_faces ({})",
                self.face_neighbor.len(),
                self.n_faces
            ));
        }

        if self.face_normals.len() != self.n_faces * 2 {
            return Err(format!(
                "face_normals 长度 {} != n_faces * 2 ({})",
                self.face_normals.len(),
                self.n_faces * 2
            ));
        }

        if self.face_lengths.len() != self.n_faces {
            return Err(format!(
                "face_lengths 长度 {} != n_faces ({})",
                self.face_lengths.len(),
                self.n_faces
            ));
        }

        Ok(())
    }

    /// 获取节点坐标
    pub fn get_node_coords(&self, node: usize) -> (f32, f32, f32) {
        let base = node * 3;
        (
            self.node_coords[base],
            self.node_coords[base + 1],
            self.node_coords[base + 2],
        )
    }

    /// 获取单元中心
    pub fn get_cell_center(&self, cell: usize) -> (f32, f32) {
        let base = cell * 2;
        (self.cell_centers[base], self.cell_centers[base + 1])
    }

    /// 获取面法向量
    pub fn get_face_normal(&self, face: usize) -> (f32, f32) {
        let base = face * 2;
        (self.face_normals[base], self.face_normals[base + 1])
    }

    /// 获取面中心
    pub fn get_face_center(&self, face: usize) -> (f32, f32) {
        let base = face * 2;
        (self.face_centers[base], self.face_centers[base + 1])
    }

    /// 获取单元节点索引
    pub fn get_cell_nodes(&self, cell: usize) -> &[u32] {
        let start = self.cell_offsets[cell] as usize;
        let end = self.cell_offsets[cell + 1] as usize;
        &self.cell_nodes[start..end]
    }
}

/// 扩展的网格统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStatisticsExt {
    // ===== 基础统计 =====
    /// 单元数
    pub n_cells: usize,
    /// 面数
    pub n_faces: usize,
    /// 节点数
    pub n_nodes: usize,
    /// 边界面数
    pub n_boundary_faces: usize,

    // ===== 单元统计 =====
    /// 最小单元面积
    pub min_cell_area: f64,
    /// 最大单元面积
    pub max_cell_area: f64,
    /// 平均单元面积
    pub avg_cell_area: f64,
    /// 单元面积标准差
    pub std_cell_area: f64,

    // ===== 边长统计 =====
    /// 最小边长
    pub min_face_length: f64,
    /// 最大边长
    pub max_face_length: f64,
    /// 平均边长
    pub avg_face_length: f64,

    // ===== 网格质量 =====
    /// 最小长宽比
    pub aspect_ratio_min: f64,
    /// 最大长宽比
    pub aspect_ratio_max: f64,
    /// 平均长宽比
    pub aspect_ratio_avg: f64,

    // ===== 边界统计 =====
    /// 边界名称列表
    pub boundary_names: Vec<String>,
    /// 各边界的面数
    pub boundary_face_counts: Vec<usize>,
}

impl MeshStatisticsExt {
    /// 从 FrozenMesh 计算扩展统计信息
    pub fn from_frozen(mesh: &FrozenMesh) -> Self {
        let areas = &mesh.cell_area;
        let lengths = &mesh.face_length;

        // 单元面积统计
        let min_area = areas.iter().cloned().fold(f64::MAX, f64::min);
        let max_area = areas.iter().cloned().fold(f64::MIN, f64::max);
        let sum_area: f64 = areas.iter().sum();
        let avg_area = sum_area / areas.len().max(1) as f64;

        // 计算标准差
        let variance: f64 = areas.iter().map(|&a| (a - avg_area).powi(2)).sum::<f64>()
            / areas.len().max(1) as f64;
        let std_area = variance.sqrt();

        // 边长统计
        let min_length = lengths.iter().cloned().fold(f64::MAX, f64::min);
        let max_length = lengths.iter().cloned().fold(f64::MIN, f64::max);
        let avg_length = lengths.iter().sum::<f64>() / lengths.len().max(1) as f64;

        // 长宽比（使用面积和边长估算）
        let aspect_ratio_min = if max_area > 0.0 {
            (min_area / max_area).sqrt()
        } else {
            1.0
        };
        let aspect_ratio_max = if min_area > 0.0 {
            (max_area / min_area).sqrt()
        } else {
            1.0
        };
        let aspect_ratio_avg = 0.5 * (aspect_ratio_min + aspect_ratio_max);

        // 边界统计
        let boundary_names = mesh.boundary_names.clone();
        let mut boundary_face_counts = vec![0usize; boundary_names.len()];

        for &face_idx in &mesh.boundary_face_indices {
            if let Some(Some(bid)) = mesh.face_boundary_id.get(face_idx as usize) {
                if (*bid as usize) < boundary_face_counts.len() {
                    boundary_face_counts[*bid as usize] += 1;
                }
            }
        }

        Self {
            n_cells: mesh.n_cells,
            n_faces: mesh.n_faces,
            n_nodes: mesh.n_nodes,
            n_boundary_faces: mesh.boundary_face_indices.len(),
            min_cell_area: min_area,
            max_cell_area: max_area,
            avg_cell_area: avg_area,
            std_cell_area: std_area,
            min_face_length: min_length,
            max_face_length: max_length,
            avg_face_length: avg_length,
            aspect_ratio_min,
            aspect_ratio_max,
            aspect_ratio_avg,
            boundary_names,
            boundary_face_counts,
        }
    }

    /// 格式化输出
    pub fn summary(&self) -> String {
        format!(
            "=== 网格统计 ===\n\
             单元: {}, 面: {}, 节点: {}\n\
             边界面: {} ({} 个边界)\n\
             单元面积: [{:.4}, {:.4}] m², 平均: {:.4} m², σ: {:.4}\n\
             边长: [{:.4}, {:.4}] m, 平均: {:.4} m\n\
             长宽比: [{:.2}, {:.2}], 平均: {:.2}",
            self.n_cells,
            self.n_faces,
            self.n_nodes,
            self.n_boundary_faces,
            self.boundary_names.len(),
            self.min_cell_area,
            self.max_cell_area,
            self.avg_cell_area,
            self.std_cell_area,
            self.min_face_length,
            self.max_face_length,
            self.avg_face_length,
            self.aspect_ratio_min,
            self.aspect_ratio_max,
            self.aspect_ratio_avg
        )
    }
}

impl std::fmt::Display for MeshStatisticsExt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mesh() -> FrozenMesh {
        let mut mesh = FrozenMesh::empty_with_cells(2);

        // 添加节点
        mesh.n_nodes = 4;
        mesh.node_coords = vec![
            mh_geo::Point3D::new(0.0, 0.0, 0.0),
            mh_geo::Point3D::new(1.0, 0.0, 0.0),
            mh_geo::Point3D::new(0.0, 1.0, 0.0),
            mh_geo::Point3D::new(1.0, 1.0, 0.0),
        ];

        // 单元中心
        mesh.cell_center = vec![
            mh_geo::Point2D::new(0.33, 0.33),
            mh_geo::Point2D::new(0.66, 0.66),
        ];

        // 单元面积
        mesh.cell_area = vec![0.5, 0.5];

        // 面数据
        mesh.n_faces = 5;
        mesh.n_interior_faces = 1;
        mesh.face_owner = vec![0, 0, 0, 1, 1];
        mesh.face_neighbor = vec![1, u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        mesh.face_normal = vec![
            mh_geo::Point3D::new(1.0, 0.0, 0.0),
            mh_geo::Point3D::new(0.0, -1.0, 0.0),
            mh_geo::Point3D::new(-1.0, 1.0, 0.0),
            mh_geo::Point3D::new(0.0, 1.0, 0.0),
            mh_geo::Point3D::new(1.0, 0.0, 0.0),
        ];
        mesh.face_length = vec![1.0, 1.0, 1.414, 1.0, 1.0];
        mesh.face_center = vec![
            mh_geo::Point2D::new(0.5, 0.5),
            mh_geo::Point2D::new(0.5, 0.0),
            mh_geo::Point2D::new(0.0, 0.5),
            mh_geo::Point2D::new(0.5, 1.0),
            mh_geo::Point2D::new(1.0, 0.5),
        ];

        // 单元节点
        mesh.cell_node_offsets = vec![0, 3, 6];
        mesh.cell_node_indices = vec![0, 1, 2, 1, 3, 2];

        mesh
    }

    #[test]
    fn test_simple_mesh_data_creation() {
        let mesh = create_test_mesh();
        let simple = SimpleMeshData::from_frozen(&mesh);

        assert_eq!(simple.n_nodes, 4);
        assert_eq!(simple.n_cells, 2);
        assert_eq!(simple.n_faces, 5);
    }

    #[test]
    fn test_simple_mesh_data_validate() {
        let mesh = create_test_mesh();
        let simple = SimpleMeshData::from_frozen(&mesh);

        assert!(simple.validate().is_ok());
    }

    #[test]
    fn test_simple_mesh_data_memory_usage() {
        let mesh = create_test_mesh();
        let simple = SimpleMeshData::from_frozen(&mesh);

        let memory = simple.memory_usage();
        assert!(memory > 0);
    }

    #[test]
    fn test_simple_mesh_data_getters() {
        let mesh = create_test_mesh();
        let simple = SimpleMeshData::from_frozen(&mesh);

        let (x, y, z) = simple.get_node_coords(0);
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
        assert_eq!(z, 0.0);

        let (cx, cy) = simple.get_cell_center(0);
        assert!((cx - 0.33).abs() < 0.01);
        assert!((cy - 0.33).abs() < 0.01);
    }

    #[test]
    fn test_mesh_statistics_ext() {
        let mesh = create_test_mesh();
        let stats = MeshStatisticsExt::from_frozen(&mesh);

        assert_eq!(stats.n_cells, 2);
        assert_eq!(stats.n_faces, 5);
        assert_eq!(stats.n_nodes, 4);
        assert!((stats.avg_cell_area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_mesh_statistics_display() {
        let mesh = create_test_mesh();
        let stats = MeshStatisticsExt::from_frozen(&mesh);

        let summary = stats.summary();
        assert!(summary.contains("单元: 2"));
        assert!(summary.contains("面: 5"));
    }
}
