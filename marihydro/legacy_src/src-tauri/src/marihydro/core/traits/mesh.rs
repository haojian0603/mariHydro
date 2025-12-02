// src-tauri/src/marihydro/core/traits/mesh.rs

//! 网格抽象接口
//!
//! 定义非结构化网格的访问接口，支持多种单元类型（三角形、四边形、多边形）。

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::{BoundaryIndex, CellIndex, FaceIndex, NodeIndex};
use glam::DVec2;

/// 网格访问接口（只读）
///
/// 提供对网格几何和拓扑信息的统一访问。
/// 实现此 trait 的类型应保证线程安全（Send + Sync）。
pub trait MeshAccess: Send + Sync {
    // ===== 基本计数 =====

    /// 单元总数
    fn n_cells(&self) -> usize;

    /// 面总数（内部面 + 边界面）
    fn n_faces(&self) -> usize;

    /// 内部面数量
    fn n_internal_faces(&self) -> usize;

    /// 边界面数量
    fn n_boundary_faces(&self) -> usize;

    /// 节点总数
    fn n_nodes(&self) -> usize;

    // ===== 几何查询 =====

    /// 单元质心
    fn cell_centroid(&self, cell: CellIndex) -> DVec2;

    /// 单元面积
    fn cell_area(&self, cell: CellIndex) -> f64;

    /// 面中点
    fn face_centroid(&self, face: FaceIndex) -> DVec2;

    /// 面长度
    fn face_length(&self, face: FaceIndex) -> f64;

    /// 面外法向量（从 owner 指向 neighbor，已归一化）
    fn face_normal(&self, face: FaceIndex) -> DVec2;

    /// 节点坐标
    fn node_position(&self, node: NodeIndex) -> DVec2;

    /// 单元底高程（用于水位计算）
    fn cell_bed_elevation(&self, cell: CellIndex) -> f64;

    // ===== 拓扑查询 =====

    /// 面的拥有者单元（总是有效的）
    fn face_owner(&self, face: FaceIndex) -> CellIndex;

    /// 面的邻居单元（边界面返回 CellIndex::INVALID）
    fn face_neighbor(&self, face: FaceIndex) -> CellIndex;

    /// 面是否为边界面
    fn is_boundary_face(&self, face: FaceIndex) -> bool {
        !self.face_neighbor(face).is_valid()
    }

    /// 面是否为内部面
    fn is_internal_face(&self, face: FaceIndex) -> bool {
        self.face_neighbor(face).is_valid()
    }

    /// 单元的相邻面列表
    fn cell_faces(&self, cell: CellIndex) -> &[FaceIndex];

    /// 单元的相邻单元列表（不含边界）
    fn cell_neighbors(&self, cell: CellIndex) -> &[CellIndex];

    /// 单元的顶点列表
    fn cell_nodes(&self, cell: CellIndex) -> &[NodeIndex];

    // ===== 边界信息 =====

    /// 边界面的边界标识
    fn boundary_id(&self, face: FaceIndex) -> Option<BoundaryIndex>;

    /// 边界名称
    fn boundary_name(&self, boundary: BoundaryIndex) -> Option<&str>;

    // ===== 批量访问（用于并行计算）=====

    /// 所有单元质心
    fn all_cell_centroids(&self) -> &[DVec2];

    /// 所有单元面积
    fn all_cell_areas(&self) -> &[f64];
}

/// 网格拓扑计算接口
pub trait MeshTopology: MeshAccess {
    /// 计算两单元中心的距离
    fn cell_distance(&self, cell1: CellIndex, cell2: CellIndex) -> f64 {
        let c1 = self.cell_centroid(cell1);
        let c2 = self.cell_centroid(cell2);
        (c2 - c1).length()
    }

    /// 计算单元中心到面中心的距离
    fn cell_to_face_distance(&self, cell: CellIndex, face: FaceIndex) -> f64 {
        let cc = self.cell_centroid(cell);
        let fc = self.face_centroid(face);
        (fc - cc).length()
    }

    /// 计算面的几何权重（用于梯度插值）
    /// 返回 owner 侧的权重，neighbor 侧权重 = 1 - weight
    fn face_weight(&self, face: FaceIndex) -> f64 {
        let owner = self.face_owner(face);
        let neighbor = self.face_neighbor(face);

        if !neighbor.is_valid() {
            return 1.0;
        }

        let d_owner = self.cell_to_face_distance(owner, face);
        let d_neighbor = self.cell_to_face_distance(neighbor, face);
        let total = d_owner + d_neighbor;

        if total < 1e-14 {
            0.5
        } else {
            d_neighbor / total
        }
    }

    /// 计算特征长度（用于CFL计算）
    fn characteristic_length(&self, cell: CellIndex) -> f64 {
        let area = self.cell_area(cell);
        let perimeter: f64 = self
            .cell_faces(cell)
            .iter()
            .map(|&f| self.face_length(f))
            .sum();

        if perimeter < 1e-14 {
            area.sqrt()
        } else {
            2.0 * area / perimeter
        }
    }

    /// 最小单元尺度（全局）
    fn min_cell_size(&self) -> f64;

    /// 最大单元尺度（全局）
    fn max_cell_size(&self) -> f64;
}

/// 单元几何信息
#[derive(Debug, Clone, Copy)]
pub struct CellGeometry {
    pub centroid: DVec2,
    pub area: f64,
    pub characteristic_length: f64,
}

/// 面几何信息
#[derive(Debug, Clone, Copy)]
pub struct FaceGeometry {
    pub centroid: DVec2,
    pub length: f64,
    pub normal: DVec2,
    pub owner: CellIndex,
    pub neighbor: CellIndex,
}

#[cfg(test)]
mod tests {
    use super::*;

    // 测试用的 mock mesh 需要 domain 层实现
    // 这里仅测试 trait 定义的编译正确性

    #[test]
    fn test_cell_index() {
        let idx = CellIndex(10);
        assert!(idx.is_valid());

        let invalid = CellIndex::INVALID;
        assert!(!invalid.is_valid());
    }
}
