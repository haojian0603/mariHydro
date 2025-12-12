// marihydro\crates\mh_mesh\src/quality.rs

//! 网格质量评估
//!
//! 提供单元质量指标和网格质量统计。

use crate::halfedge::HalfEdgeMesh;
use mh_foundation::index::FaceIndex;

/// 计算三角形的长宽比
/// 
/// 理想的等边三角形返回 1.0，更细长的返回更大的值
pub fn aspect_ratio<V, F>(mesh: &HalfEdgeMesh<V, F>, face: FaceIndex) -> Option<f64> {
    let positions = mesh.face_positions(face);
    if positions.len() != 3 {
        return None;
    }

    // 计算三条边的长度
    let mut lengths = [0.0; 3];
    for i in 0..3 {
        let j = (i + 1) % 3;
        let dx = positions[j].x - positions[i].x;
        let dy = positions[j].y - positions[i].y;
        lengths[i] = (dx * dx + dy * dy).sqrt();
    }

    let max_len = lengths.iter().cloned().fold(0.0f64, f64::max);
    let min_len = lengths.iter().cloned().fold(f64::MAX, f64::min);

    if min_len < 1e-14 {
        None
    } else {
        Some(max_len / min_len)
    }
}

/// 计算三角形的最小角 (度)
pub fn min_angle<V, F>(mesh: &HalfEdgeMesh<V, F>, face: FaceIndex) -> Option<f64> {
    let positions = mesh.face_positions(face);
    if positions.len() != 3 {
        return None;
    }

    // 计算三个内角
    let mut min_angle = f64::MAX;

    for i in 0..3 {
        let a = (i + 2) % 3;
        let b = i;
        let c = (i + 1) % 3;

        let ab = (positions[a].x - positions[b].x, positions[a].y - positions[b].y);
        let cb = (positions[c].x - positions[b].x, positions[c].y - positions[b].y);

        let dot = ab.0 * cb.0 + ab.1 * cb.1;
        let ab_len = (ab.0 * ab.0 + ab.1 * ab.1).sqrt();
        let cb_len = (cb.0 * cb.0 + cb.1 * cb.1).sqrt();

        if ab_len > 1e-14 && cb_len > 1e-14 {
            let cos_angle = (dot / (ab_len * cb_len)).clamp(-1.0, 1.0);
            let angle = cos_angle.acos().to_degrees();
            min_angle = min_angle.min(angle);
        }
    }

    if min_angle < f64::MAX {
        Some(min_angle)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aspect_ratio_equilateral() {
        let mut mesh: HalfEdgeMesh<(), ()> = HalfEdgeMesh::new();

        // 等边三角形
        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 0.866, 0.0);

        let face = mesh.add_triangle(v0, v1, v2).unwrap();

        let ar = aspect_ratio(&mesh, face).unwrap();
        assert!((ar - 1.0).abs() < 0.01);
    }
}
