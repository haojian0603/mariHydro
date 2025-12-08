// crates/mh_physics/src/numerics/discretization/topology.rs

//! 网格拓扑模块
//!
//! 提供有限体积法所需的网格拓扑信息，包括：
//! - 单元-面连接关系
//! - 面的几何信息（法向、长度、距离）
//! - 邻居单元查找
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::discretization::topology::CellFaceTopology;
//!
//! let topo = CellFaceTopology::from_mesh(&mesh);
//!
//! // 遍历单元的所有面
//! for face_info in topo.cell_faces(cell_idx) {
//!     let neighbor = face_info.neighbor;
//!     let normal = face_info.normal;
//!     // ...
//! }
//! ```

use crate::adapter::PhysicsMesh;
use glam::DVec2;
use mh_foundation::Scalar;

/// 邻居信息
#[derive(Debug, Clone, Copy)]
pub struct NeighborInfo {
    /// 邻居单元索引（None 表示边界）
    pub cell_idx: Option<usize>,
    /// 面索引
    pub face_idx: usize,
    /// 指向邻居的法向（归一化）
    pub normal: DVec2,
    /// 面长度 [m]
    pub length: Scalar,
    /// 单元中心到面的距离 [m]
    pub dist_to_face: Scalar,
    /// 单元中心到邻居中心的距离 [m]（边界时为 2 * dist_to_face）
    pub dist_to_neighbor: Scalar,
}

/// 面信息
#[derive(Debug, Clone, Copy)]
pub struct FaceInfo {
    /// 面索引
    pub face_idx: usize,
    /// Owner 单元索引
    pub owner: usize,
    /// Neighbor 单元索引（None 表示边界）
    pub neighbor: Option<usize>,
    /// Owner -> Neighbor 方向的法向（归一化）
    pub normal: DVec2,
    /// 面长度 [m]
    pub length: Scalar,
    /// Owner 中心到 Neighbor 中心的距离 [m]
    pub dist_o2n: Scalar,
    /// Owner 中心到面的距离 [m]
    pub dist_o2f: Scalar,
    /// Neighbor 中心到面的距离 [m]（边界时为 dist_o2f）
    pub dist_n2f: Scalar,
    /// 是否为边界面
    pub is_boundary: bool,
}

/// 单元-面拓扑
///
/// 存储网格的拓扑结构，支持高效的单元-面遍历
pub struct CellFaceTopology {
    /// 单元数量
    n_cells: usize,
    /// 面数量
    n_faces: usize,
    /// 边界面数量
    n_boundary_faces: usize,
    /// 每个单元的面索引列表起始位置
    cell_face_ptr: Vec<usize>,
    /// 所有单元的面索引列表（扁平存储）
    cell_face_idx: Vec<usize>,
    /// 面信息列表
    face_info: Vec<FaceInfo>,
    /// 内部面索引列表
    interior_faces: Vec<usize>,
    /// 边界面索引列表
    boundary_faces: Vec<usize>,
}

impl CellFaceTopology {
    /// 从物理网格构建拓扑
    pub fn from_mesh(mesh: &PhysicsMesh) -> Self {
        let n_cells = mesh.n_cells();
        let n_faces = mesh.n_faces();

        // 统计每个单元的面数
        let mut cell_face_count = vec![0usize; n_cells];
        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(face_idx);
            cell_face_count[owner] += 1;
            if let Some(neigh) = mesh.face_neighbor(face_idx) {
                cell_face_count[neigh] += 1;
            }
        }

        // 构建 cell_face_ptr
        let mut cell_face_ptr = Vec::with_capacity(n_cells + 1);
        cell_face_ptr.push(0);
        for &count in &cell_face_count {
            let last = *cell_face_ptr.last().unwrap();
            cell_face_ptr.push(last + count);
        }

        // 构建 cell_face_idx
        let total_entries = *cell_face_ptr.last().unwrap();
        let mut cell_face_idx = vec![0usize; total_entries];
        let mut current_pos = cell_face_ptr.clone();
        current_pos.pop(); // 移除最后一个

        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(face_idx);
            cell_face_idx[current_pos[owner]] = face_idx;
            current_pos[owner] += 1;

            if let Some(neigh) = mesh.face_neighbor(face_idx) {
                cell_face_idx[current_pos[neigh]] = face_idx;
                current_pos[neigh] += 1;
            }
        }

        // 构建面信息
        let mut face_info = Vec::with_capacity(n_faces);
        let mut interior_faces = Vec::new();
        let mut boundary_faces = Vec::new();

        for face_idx in 0..n_faces {
            let owner = mesh.face_owner(face_idx);
            let neighbor = mesh.face_neighbor(face_idx);
            let normal = mesh.face_normal(face_idx);
            let length = mesh.face_length(face_idx);
            let dist_o2n = mesh.face_dist_o2n(face_idx);

            let is_boundary = neighbor.is_none();
            let (dist_o2f, dist_n2f) = if is_boundary {
                (dist_o2n / 2.0, dist_o2n / 2.0)
            } else {
                // 近似：假设面在两个单元中心之间等分
                (dist_o2n / 2.0, dist_o2n / 2.0)
            };

            face_info.push(FaceInfo {
                face_idx,
                owner,
                neighbor,
                normal,
                length,
                dist_o2n,
                dist_o2f,
                dist_n2f,
                is_boundary,
            });

            if is_boundary {
                boundary_faces.push(face_idx);
            } else {
                interior_faces.push(face_idx);
            }
        }

        Self {
            n_cells,
            n_faces,
            n_boundary_faces: boundary_faces.len(),
            cell_face_ptr,
            cell_face_idx,
            face_info,
            interior_faces,
            boundary_faces,
        }
    }

    /// 获取单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 获取面数量
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    /// 获取边界面数量
    #[inline]
    pub fn n_boundary_faces(&self) -> usize {
        self.n_boundary_faces
    }

    /// 获取内部面数量
    #[inline]
    pub fn n_interior_faces(&self) -> usize {
        self.n_faces - self.n_boundary_faces
    }

    /// 获取单元的面索引列表
    #[inline]
    pub fn cell_faces(&self, cell_idx: usize) -> &[usize] {
        let start = self.cell_face_ptr[cell_idx];
        let end = self.cell_face_ptr[cell_idx + 1];
        &self.cell_face_idx[start..end]
    }

    /// 获取单元的面数量
    #[inline]
    pub fn cell_n_faces(&self, cell_idx: usize) -> usize {
        self.cell_face_ptr[cell_idx + 1] - self.cell_face_ptr[cell_idx]
    }

    /// 获取面信息
    #[inline]
    pub fn face(&self, face_idx: usize) -> &FaceInfo {
        &self.face_info[face_idx]
    }

    /// 获取所有面信息
    #[inline]
    pub fn faces(&self) -> &[FaceInfo] {
        &self.face_info
    }

    /// 获取内部面索引列表
    #[inline]
    pub fn interior_faces(&self) -> &[usize] {
        &self.interior_faces
    }

    /// 获取边界面索引列表
    #[inline]
    pub fn boundary_faces(&self) -> &[usize] {
        &self.boundary_faces
    }

    /// 遍历单元的邻居信息
    pub fn cell_neighbors(&self, cell_idx: usize) -> impl Iterator<Item = NeighborInfo> + '_ {
        self.cell_faces(cell_idx).iter().map(move |&face_idx| {
            let info = &self.face_info[face_idx];

            // 确定邻居和法向方向
            let (neighbor, normal, dist_to_face) = if info.owner == cell_idx {
                (info.neighbor, info.normal, info.dist_o2f)
            } else {
                (Some(info.owner), -info.normal, info.dist_n2f)
            };

            NeighborInfo {
                cell_idx: neighbor,
                face_idx,
                normal,
                length: info.length,
                dist_to_face,
                dist_to_neighbor: info.dist_o2n,
            }
        })
    }

    /// 获取单元的邻居单元索引列表
    pub fn cell_neighbor_indices(&self, cell_idx: usize) -> Vec<usize> {
        self.cell_neighbors(cell_idx)
            .filter_map(|n| n.cell_idx)
            .collect()
    }

    /// 获取单元的边界面数量
    pub fn cell_n_boundary_faces(&self, cell_idx: usize) -> usize {
        self.cell_neighbors(cell_idx)
            .filter(|n| n.cell_idx.is_none())
            .count()
    }

    /// 检查单元是否有边界面
    pub fn cell_is_boundary(&self, cell_idx: usize) -> bool {
        self.cell_neighbors(cell_idx)
            .any(|n| n.cell_idx.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 注意：完整测试需要 PhysicsMesh，这里只测试数据结构

    #[test]
    fn test_face_info() {
        let info = FaceInfo {
            face_idx: 0,
            owner: 0,
            neighbor: Some(1),
            normal: DVec2::new(1.0, 0.0),
            length: 10.0,
            dist_o2n: 5.0,
            dist_o2f: 2.5,
            dist_n2f: 2.5,
            is_boundary: false,
        };

        assert!(!info.is_boundary);
        assert_eq!(info.neighbor, Some(1));
    }

    #[test]
    fn test_neighbor_info() {
        let info = NeighborInfo {
            cell_idx: Some(5),
            face_idx: 10,
            normal: DVec2::new(0.0, 1.0),
            length: 8.0,
            dist_to_face: 3.0,
            dist_to_neighbor: 6.0,
        };

        assert_eq!(info.cell_idx, Some(5));
        assert!((info.length - 8.0).abs() < 1e-14);
    }
}
