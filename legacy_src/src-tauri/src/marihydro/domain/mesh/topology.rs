// src-tauri/src/marihydro/domain/mesh/topology.rs

//! 网格拓扑计算扩展

use super::unstructured::UnstructuredMesh;
use crate::marihydro::core::types::CellIndex;
use glam::DVec2;

/// 网格拓扑扩展方法
pub trait MeshTopologyExt {
    /// 计算单元特征长度
    fn cell_characteristic_length(&self, cell_idx: usize) -> f64;

    /// 计算面的几何权重
    fn face_geometric_weight(&self, face_idx: usize) -> f64;

    /// 计算两单元中心距离
    fn cell_distance(&self, cell1: usize, cell2: usize) -> f64;

    /// 计算单元中心到面中心的向量
    fn cell_to_face_vector(&self, cell_idx: usize, face_idx: usize) -> DVec2;

    /// 获取单元的平均边长
    fn cell_average_edge_length(&self, cell_idx: usize) -> f64;
}

impl MeshTopologyExt for UnstructuredMesh {
    fn cell_characteristic_length(&self, cell_idx: usize) -> f64 {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        let cell = CellIndex(cell_idx);
        let area = self.cell_area(cell);
        let faces = self.cell_faces(cell);

        let perimeter: f64 = faces.iter().map(|&f| self.face_length(f)).sum();

        if perimeter < 1e-14 {
            area.sqrt()
        } else {
            2.0 * area / perimeter
        }
    }

    fn face_geometric_weight(&self, face_idx: usize) -> f64 {
        use crate::marihydro::core::traits::mesh::MeshAccess;
        use crate::marihydro::core::types::FaceIndex;

        let face = FaceIndex(face_idx);
        let owner = self.face_owner(face);
        let neighbor = self.face_neighbor(face);

        if !neighbor.is_valid() {
            return 1.0;
        }

        let fc = self.face_centroid(face);
        let oc = self.cell_centroid(owner);
        let nc = self.cell_centroid(neighbor);

        let d_owner = (fc - oc).length();
        let d_neighbor = (fc - nc).length();
        let total = d_owner + d_neighbor;

        if total < 1e-14 {
            0.5
        } else {
            d_neighbor / total
        }
    }

    fn cell_distance(&self, cell1: usize, cell2: usize) -> f64 {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        let c1 = self.cell_centroid(CellIndex(cell1));
        let c2 = self.cell_centroid(CellIndex(cell2));
        (c2 - c1).length()
    }

    fn cell_to_face_vector(&self, cell_idx: usize, face_idx: usize) -> DVec2 {
        use crate::marihydro::core::traits::mesh::MeshAccess;
        use crate::marihydro::core::types::FaceIndex;

        let cc = self.cell_centroid(CellIndex(cell_idx));
        let fc = self.face_centroid(FaceIndex(face_idx));
        fc - cc
    }

    fn cell_average_edge_length(&self, cell_idx: usize) -> f64 {
        use crate::marihydro::core::traits::mesh::MeshAccess;

        let cell = CellIndex(cell_idx);
        let faces = self.cell_faces(cell);

        if faces.is_empty() {
            return 0.0;
        }

        let total: f64 = faces.iter().map(|&f| self.face_length(f)).sum();

        total / faces.len() as f64
    }
}

#[cfg(test)]
mod tests {
    // 拓扑测试需要完整网格，在集成测试中进行
}
