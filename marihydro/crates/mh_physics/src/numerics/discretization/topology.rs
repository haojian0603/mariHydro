// crates/mh_physics/src/numerics/discretization/topology.rs

//! 网格拓扑模块
//!
//! 提供有限体积法所需的网格拓扑信息，包括：
//! - 单元-面连接关系
//! - 面的几何信息（法向、长度、距离）
//! - 邻居单元查找
//!
//! # 设计原则
//!
//! 1. **无 DVec2**: 所有法向使用 `(f64, f64)` 元组
//! 2. **元组几何接口**: 使用 `cell_center_tuple`、`face_normal_2d_tuple` 等方法
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
//!     let (nx, ny) = face_info.normal;
//!     // ...
//! }
//! ```

use crate::adapter::PhysicsMesh;

/// 邻居信息
#[derive(Debug, Clone, Copy)]
pub struct NeighborInfo {
    /// 邻居单元索引（None 表示边界）
    pub cell_idx: Option<usize>,
    /// 面索引
    pub face_idx: usize,
    /// 指向邻居的法向（归一化）- 元组版
    pub normal: (f64, f64),
    /// 面长度 [m]
    pub length: f64,
    /// 单元中心到面的距离 [m]
    pub dist_to_face: f64,
    /// 单元中心到邻居中心的距离 [m]（边界时为 2 * dist_to_face）
    pub dist_to_neighbor: f64,
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
    /// Owner -> Neighbor 方向的法向（归一化）- 元组版
    pub normal: (f64, f64),
    /// 面长度 [m]
    pub length: f64,
    /// Owner 中心到 Neighbor 中心的距离 [m]
    pub dist_o2n: f64,
    /// Owner 中心到面的距离 [m]
    pub dist_o2f: f64,
    /// Neighbor 中心到面的距离 [m]（边界时为 dist_o2f）
    pub dist_n2f: f64,
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
            let owner = mesh.face_owner(mh_runtime::FaceIndex(face_idx));
            cell_face_count[owner.0] += 1;
            if let Some(neigh) = mesh.face_neighbor(mh_runtime::FaceIndex(face_idx)) {
                cell_face_count[neigh.0] += 1;
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
            let owner = mesh.face_owner(mh_runtime::FaceIndex(face_idx));
            cell_face_idx[current_pos[owner.0]] = face_idx;
            current_pos[owner.0] += 1;

            if let Some(neigh) = mesh.face_neighbor(mh_runtime::FaceIndex(face_idx)) {
                cell_face_idx[current_pos[neigh.0]] = face_idx;
                current_pos[neigh.0] += 1;
            }
        }

        // 构建面信息
        let mut face_info = Vec::with_capacity(n_faces);
        let mut interior_faces = Vec::new();
        let mut boundary_faces = Vec::new();

        for face_idx in 0..n_faces {
            let fi = mh_runtime::FaceIndex(face_idx);
            let owner = mesh.face_owner(fi);
            let neighbor = mesh.face_neighbor(fi);
            let normal = mesh.face_normal_2d_tuple(face_idx);
            let length = mesh.face_length(fi);
            let dist_o2n = mesh.face_dist_o2n(fi);

            let is_boundary = neighbor.is_none();
            let (dist_o2f, dist_n2f) = if is_boundary {
                (dist_o2n / 2.0, dist_o2n / 2.0)
            } else {
                // 近似：假设面在两个单元中心之间等分
                (dist_o2n / 2.0, dist_o2n / 2.0)
            };

            face_info.push(FaceInfo {
                face_idx,
                owner: owner.into(),
                neighbor: neighbor.map(|n| n.into()),
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

    // =========================================================================
    // 基本访问器
    // =========================================================================

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

    // =========================================================================
    // 面信息访问
    // =========================================================================

    /// 获取面信息
    #[inline]
    pub fn face(&self, face_idx: usize) -> &FaceInfo {
        &self.face_info[face_idx]
    }

    /// 获取所有内部面索引
    #[inline]
    pub fn interior_faces(&self) -> &[usize] {
        &self.interior_faces
    }

    /// 获取所有边界面索引
    #[inline]
    pub fn boundary_faces(&self) -> &[usize] {
        &self.boundary_faces
    }

    /// 迭代所有面信息
    #[inline]
    pub fn faces(&self) -> impl Iterator<Item = &FaceInfo> {
        self.face_info.iter()
    }

    // =========================================================================
    // 单元-面关系
    // =========================================================================

    /// 获取单元的所有面索引
    #[inline]
    pub fn cell_face_indices(&self, cell_idx: usize) -> &[usize] {
        let start = self.cell_face_ptr[cell_idx];
        let end = self.cell_face_ptr[cell_idx + 1];
        &self.cell_face_idx[start..end]
    }

    /// 获取单元的所有面信息
    #[inline]
    pub fn cell_faces(&self, cell_idx: usize) -> impl Iterator<Item = &FaceInfo> {
        self.cell_face_indices(cell_idx)
            .iter()
            .map(move |&fi| &self.face_info[fi])
    }

    /// 获取单元的邻居信息
    pub fn cell_neighbors(&self, cell_idx: usize) -> Vec<NeighborInfo> {
        self.cell_face_indices(cell_idx)
            .iter()
            .map(|&face_idx| {
                let face = &self.face_info[face_idx];
                let (cell, dist_to_face, normal) = if face.owner == cell_idx {
                    // 当前单元是 owner，邻居在 neighbor
                    (face.neighbor, face.dist_o2f, face.normal)
                } else {
                    // 当前单元是 neighbor，邻居是 owner
                    (Some(face.owner), face.dist_n2f, (-face.normal.0, -face.normal.1))
                };

                NeighborInfo {
                    cell_idx: cell,
                    face_idx,
                    normal,
                    length: face.length,
                    dist_to_face,
                    dist_to_neighbor: face.dist_o2n,
                }
            })
            .collect()
    }

    // =========================================================================
    // 验证
    // =========================================================================

    /// 验证拓扑完整性
    pub fn validate(&self) -> Result<(), TopologyError> {
        // 检查面长度
        for face in &self.face_info {
            if face.length < 1e-14 {
                return Err(TopologyError::DegenerateFace {
                    face_idx: face.face_idx,
                    length: face.length,
                });
            }
        }

        // 检查自连接
        for face in &self.face_info {
            if let Some(neigh) = face.neighbor {
                if neigh == face.owner {
                    return Err(TopologyError::SelfConnectedFace {
                        face_idx: face.face_idx,
                        cell: face.owner,
                    });
                }
            }
        }

        // 检查距离
        for face in &self.face_info {
            let face_idx = face.face_idx;
            if face.dist_o2n < 1e-14 {
                return Err(TopologyError::InvalidDistance {
                    face_idx,
                    distance: face.dist_o2n,
                });
            }
        }

        Ok(())
    }

    /// 计算精确的非正交距离
    ///
    /// 对于非正交网格，考虑面法向与连接线的夹角。
    pub fn compute_orthogonal_distance(&self, face_idx: usize, mesh: &PhysicsMesh) -> f64 {
        let face = &self.face_info[face_idx];
        
        if let Some(neigh) = face.neighbor {
            let owner_center = mesh.cell_center_tuple(face.owner);
            let neigh_center = mesh.cell_center_tuple(neigh);
            let delta_x = neigh_center.0 - owner_center.0;
            let delta_y = neigh_center.1 - owner_center.1;
            
            // 投影到面法向
            let (nx, ny) = face.normal;
            let proj = delta_x * nx + delta_y * ny;
            proj.abs()
        } else {
            face.dist_o2n
        }
    }

    /// 计算面的非正交性因子
    ///
    /// 返回 0.0（完全正交）到 1.0（完全非正交）。
    pub fn compute_non_orthogonality(&self, face_idx: usize, mesh: &PhysicsMesh) -> f64 {
        let face = &self.face_info[face_idx];
        
        if let Some(neigh) = face.neighbor {
            let owner_center = mesh.cell_center_tuple(face.owner);
            let neigh_center = mesh.cell_center_tuple(neigh);
            let delta_x = neigh_center.0 - owner_center.0;
            let delta_y = neigh_center.1 - owner_center.1;
            let delta_len = (delta_x * delta_x + delta_y * delta_y).sqrt();
            
            if delta_len < 1e-14 {
                return 0.0;
            }
            
            // 计算连接线与法向的夹角余弦
            let (nx, ny) = face.normal;
            let cos_theta = (delta_x * nx + delta_y * ny).abs() / delta_len;
            
            // 非正交因子 = 1 - |cos(theta)|
            (1.0 - cos_theta).max(0.0)
        } else {
            0.0
        }
    }
}

/// 拓扑验证错误
#[derive(Debug, Clone)]
pub enum TopologyError {
    /// 退化面（长度过小）
    DegenerateFace { face_idx: usize, length: f64 },
    /// 自连接面（owner == neighbor）
    SelfConnectedFace { face_idx: usize, cell: usize },
    /// 无效距离
    InvalidDistance { face_idx: usize, distance: f64 },
    /// 负面积单元
    NegativeArea { cell_idx: usize, area: f64 },
}

impl std::fmt::Display for TopologyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopologyError::DegenerateFace { face_idx, length } => {
                write!(f, "Degenerate face {} with length {}", face_idx, length)
            }
            TopologyError::SelfConnectedFace { face_idx, cell } => {
                write!(f, "Face {} connects cell {} to itself", face_idx, cell)
            }
            TopologyError::InvalidDistance { face_idx, distance } => {
                write!(f, "Invalid distance {} for face {}", distance, face_idx)
            }
            TopologyError::NegativeArea { cell_idx, area } => {
                write!(f, "Negative area {} for cell {}", area, cell_idx)
            }
        }
    }
}

impl std::error::Error for TopologyError {}

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
            normal: (1.0, 0.0),
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
            normal: (0.0, 1.0),
            length: 8.0,
            dist_to_face: 3.0,
            dist_to_neighbor: 6.0,
        };

        assert_eq!(info.cell_idx, Some(5));
        assert!((info.length - 8.0).abs() < 1e-14);
    }
}
