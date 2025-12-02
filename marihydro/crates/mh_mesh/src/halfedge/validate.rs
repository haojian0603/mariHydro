// marihydro\crates\mh_mesh\src/halfedge/validate.rs

//! 半边网格验证
//!
//! 提供拓扑一致性检查功能。

use super::mesh::HalfEdgeMesh;
use mh_foundation::index::{FaceIndex, HalfEdgeIndex, VertexIndex};

/// 验证错误
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// 半边的 next 无效
    InvalidNext { halfedge: HalfEdgeIndex },
    /// 半边的 prev 无效
    InvalidPrev { halfedge: HalfEdgeIndex },
    /// 半边的 twin 不对称
    TwinMismatch { halfedge: HalfEdgeIndex },
    /// 半边链不闭合
    OpenLoop { face: FaceIndex },
    /// 顶点的出发边无效
    InvalidVertexHalfEdge { vertex: VertexIndex },
    /// 面的半边无效
    InvalidFaceHalfEdge { face: FaceIndex },
    /// 半边的 origin 无效
    InvalidOrigin { halfedge: HalfEdgeIndex },
    /// 半边的 face 指向无效面
    InvalidFaceRef { halfedge: HalfEdgeIndex },
}

/// 验证报告
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// 错误列表
    pub errors: Vec<ValidationError>,
    /// 警告数量
    pub warnings: usize,
}

impl ValidationReport {
    /// 验证是否通过
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// 添加错误
    pub fn add_error(&mut self, error: ValidationError) {
        self.errors.push(error);
    }

    /// 添加警告
    pub fn add_warning(&mut self) {
        self.warnings += 1;
    }
}

impl<V, E, F> HalfEdgeMesh<V, E, F> {
    /// 验证网格拓扑完整性
    pub fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();

        // 验证所有半边
        for he_idx in self.halfedge_indices() {
            self.validate_halfedge(he_idx, &mut report);
        }

        // 验证所有顶点
        for v_idx in self.vertex_indices() {
            self.validate_vertex(v_idx, &mut report);
        }

        // 验证所有面
        for f_idx in self.face_indices() {
            self.validate_face(f_idx, &mut report);
        }

        report
    }

    /// 验证单个半边
    fn validate_halfedge(&self, he: HalfEdgeIndex, report: &mut ValidationReport) {
        let Some(he_data) = self.halfedge(he) else {
            return;
        };

        // 检查 origin
        if he_data.origin.is_valid() && !self.contains_vertex(he_data.origin) {
            report.add_error(ValidationError::InvalidOrigin { halfedge: he });
        }

        // 检查 next
        if he_data.next.is_valid() {
            if !self.contains_halfedge(he_data.next) {
                report.add_error(ValidationError::InvalidNext { halfedge: he });
            } else if let Some(next_data) = self.halfedge(he_data.next) {
                if next_data.prev != he {
                    report.add_error(ValidationError::InvalidPrev { halfedge: he_data.next });
                }
            }
        }

        // 检查 twin 对称性
        if he_data.twin.is_valid() {
            if let Some(twin_data) = self.halfedge(he_data.twin) {
                if twin_data.twin != he {
                    report.add_error(ValidationError::TwinMismatch { halfedge: he });
                }
            }
        }

        // 检查 face 引用
        if he_data.face.is_valid() && !self.contains_face(he_data.face) {
            report.add_error(ValidationError::InvalidFaceRef { halfedge: he });
        }
    }

    /// 验证单个顶点
    fn validate_vertex(&self, v: VertexIndex, report: &mut ValidationReport) {
        let Some(v_data) = self.vertex(v) else {
            return;
        };

        if v_data.halfedge.is_valid() {
            if !self.contains_halfedge(v_data.halfedge) {
                report.add_error(ValidationError::InvalidVertexHalfEdge { vertex: v });
            } else if let Some(he_data) = self.halfedge(v_data.halfedge) {
                if he_data.origin != v {
                    report.add_warning(); // 出发边的 origin 不是自己
                }
            }
        }
    }

    /// 验证单个面
    fn validate_face(&self, f: FaceIndex, report: &mut ValidationReport) {
        let Some(f_data) = self.face(f) else {
            return;
        };

        if f_data.halfedge.is_invalid() {
            report.add_error(ValidationError::InvalidFaceHalfEdge { face: f });
            return;
        }

        if !self.contains_halfedge(f_data.halfedge) {
            report.add_error(ValidationError::InvalidFaceHalfEdge { face: f });
            return;
        }

        // 检查半边链是否闭合
        let start = f_data.halfedge;
        let mut current = start;
        let mut count = 0;
        let max_count = 1000; // 防止无限循环

        loop {
            let Some(he_data) = self.halfedge(current) else {
                report.add_error(ValidationError::OpenLoop { face: f });
                break;
            };

            if he_data.face != f {
                report.add_warning(); // 半边的 face 不是当前面
            }

            current = he_data.next;
            count += 1;

            if current == start {
                break;
            }

            if count > max_count {
                report.add_error(ValidationError::OpenLoop { face: f });
                break;
            }
        }
    }

    /// 快速验证（仅检查关键不变量）
    pub fn quick_validate(&self) -> bool {
        // 检查每个面的半边链是否闭合
        for f_idx in self.face_indices() {
            if let Some(f_data) = self.face(f_idx) {
                if f_data.halfedge.is_invalid() {
                    return false;
                }

                let start = f_data.halfedge;
                let mut current = start;
                let mut count = 0;

                loop {
                    let Some(he_data) = self.halfedge(current) else {
                        return false;
                    };

                    current = he_data.next;
                    count += 1;

                    if current == start {
                        break;
                    }

                    if count > 100 {
                        return false;
                    }
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::mesh::{Face, HalfEdge, Vertex};
    use super::*;

    #[test]
    fn test_validate_empty_mesh() {
        let mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();
        let report = mesh.validate();
        assert!(report.is_valid());
    }

    #[test]
    fn test_validate_triangle() {
        let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        mesh.add_triangle(v0, v1, v2);

        let report = mesh.validate();
        assert!(report.is_valid());
    }

    #[test]
    fn test_quick_validate() {
        let mut mesh: HalfEdgeMesh<(), (), ()> = HalfEdgeMesh::new();

        let v0 = mesh.add_vertex_xyz(0.0, 0.0, 0.0);
        let v1 = mesh.add_vertex_xyz(1.0, 0.0, 0.0);
        let v2 = mesh.add_vertex_xyz(0.5, 1.0, 0.0);

        mesh.add_triangle(v0, v1, v2);

        assert!(mesh.quick_validate());
    }
}
