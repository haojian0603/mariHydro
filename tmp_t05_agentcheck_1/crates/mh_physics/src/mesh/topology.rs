//! 网格拓扑抽象
//!
//! 提供结构化和非结构化网格的统一接口。

use crate::core::Backend;
use num_traits::Float;
use mh_runtime::RuntimeScalar as Scalar;

/// 网格验证错误
#[derive(Debug, thiserror::Error)]
pub enum MeshValidationError {
    #[error("网格数量无效: n_cells={n_cells}, n_faces={n_faces}, n_interior_faces={n_interior_faces}")]
    InvalidCounts {
        n_cells: usize,
        n_faces: usize,
        n_interior_faces: usize,
    },
    #[error("网格间距无效: axis={axis}, value={value}")]
    InvalidSpacing { axis: &'static str, value: f64 },
    #[error("单元面积无效: cell={cell}, area={area}")]
    InvalidCellArea { cell: usize, area: f64 },
    #[error("面长度无效: face={face}, length={length}")]
    InvalidFaceLength { face: usize, length: f64 },
    #[error("face owner 越界: face={face}, owner={owner}, n_cells={n_cells}")]
    FaceOwnerOutOfRange { face: usize, owner: usize, n_cells: usize },
    #[error("face neighbor 越界: face={face}, neighbor={neighbor}, n_cells={n_cells}")]
    FaceNeighborOutOfRange { face: usize, neighbor: usize, n_cells: usize },
    #[error("cell face 索引越界: cell={cell}, face={face}, n_faces={n_faces}")]
    CellFaceOutOfRange { cell: usize, face: usize, n_faces: usize },
    #[error("边界面列表不一致: face={face}")]
    BoundaryFaceMismatch { face: usize },
    #[error("内部面列表不一致: face={face}")]
    InteriorFaceMismatch { face: usize },
}

/// 网格类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    /// 非结构化网格
    Unstructured,
    /// 结构化网格
    Structured { nx: usize, ny: usize },
}

/// 面信息
#[derive(Debug, Clone, Copy)]
pub struct FaceInfo<S: Scalar> {
    /// 面法向量 (指向 neighbor)
    pub normal: [S; 2],
    /// 面长度
    pub length: S,
    /// 面中心坐标
    pub center: [S; 2],
    /// 所属单元 (owner)
    pub owner: usize,
    /// 相邻单元 (neighbor)，边界面为 None
    pub neighbor: Option<usize>,
}

/// 网格拓扑 trait
pub trait MeshTopology<B: Backend>: Send + Sync {
    // ========== 基本信息 ==========
    
    /// 单元数量
    fn n_cells(&self) -> usize;
    
    /// 面数量
    fn n_faces(&self) -> usize;
    
    /// 内部面数量
    fn n_interior_faces(&self) -> usize;
    
    /// 边界面数量
    fn n_boundary_faces(&self) -> usize {
        self.n_faces() - self.n_interior_faces()
    }
    
    /// 节点数量
    fn n_nodes(&self) -> usize;
    
    // ========== 几何数据 ==========
    
    /// 单元中心坐标
    fn cell_center(&self, cell: usize) -> [B::Scalar; 2];
    
    /// 单元面积
    fn cell_area(&self, cell: usize) -> B::Scalar;
    
    /// 面法向量
    fn face_normal(&self, face: usize) -> [B::Scalar; 2];
    
    /// 面长度
    fn face_length(&self, face: usize) -> B::Scalar;
    
    /// 面中心坐标
    fn face_center(&self, face: usize) -> [B::Scalar; 2];
    
    // ========== 拓扑数据 ==========
    
    /// 面的 owner 单元
    fn face_owner(&self, face: usize) -> usize;
    
    /// 面的 neighbor 单元（边界面返回 None）
    fn face_neighbor(&self, face: usize) -> Option<usize>;
    
    /// 单元的所有面索引
    fn cell_faces(&self, cell: usize) -> &[usize];
    
    /// 单元的相邻单元索引
    fn cell_neighbors(&self, cell: usize) -> Vec<usize>;
    
    // ========== 边界信息 ==========
    
    /// 是否为边界面
    fn is_boundary_face(&self, face: usize) -> bool {
        self.face_neighbor(face).is_none()
    }
    
    /// 边界面索引列表
    fn boundary_faces(&self) -> &[usize];
    
    /// 内部面索引列表
    fn interior_faces(&self) -> &[usize];
    
    // ========== 网格类型 ==========
    
    /// 网格类型
    fn mesh_kind(&self) -> MeshKind;
    
    // ========== 批量访问（GPU 优化入口）==========
    
    /// 获取所有单元面积（设备缓冲区）
    fn cell_areas_buffer(&self) -> &B::Buffer<B::Scalar>;
    
    /// 获取所有面长度（设备缓冲区）
    fn face_lengths_buffer(&self) -> &B::Buffer<B::Scalar>;

    // ========== 一致性校验 ==========
    fn validate(&self) -> Result<(), MeshValidationError> {
        let n_cells = self.n_cells();
        let n_faces = self.n_faces();
        let n_interior_faces = self.n_interior_faces();

        if n_cells == 0 || n_faces == 0 || n_interior_faces > n_faces {
            return Err(MeshValidationError::InvalidCounts {
                n_cells,
                n_faces,
                n_interior_faces,
            });
        }

        for cell in 0..n_cells {
            let area = self.cell_area(cell);
            if !area.is_finite() || area <= B::Scalar::ZERO {
                return Err(MeshValidationError::InvalidCellArea {
                    cell,
                    area: area.to_f64_lossy(),
                });
            }

            for &face in self.cell_faces(cell) {
                if face >= n_faces {
                    return Err(MeshValidationError::CellFaceOutOfRange {
                        cell,
                        face,
                        n_faces,
                    });
                }
            }
        }

        for face in 0..n_faces {
            let length = self.face_length(face);
            if !length.is_finite() || length <= B::Scalar::ZERO {
                return Err(MeshValidationError::InvalidFaceLength {
                    face,
                    length: length.to_f64_lossy(),
                });
            }

            let owner = self.face_owner(face);
            if owner >= n_cells {
                return Err(MeshValidationError::FaceOwnerOutOfRange {
                    face,
                    owner,
                    n_cells,
                });
            }

            if let Some(neighbor) = self.face_neighbor(face) {
                if neighbor >= n_cells {
                    return Err(MeshValidationError::FaceNeighborOutOfRange {
                        face,
                        neighbor,
                        n_cells,
                    });
                }
            }
        }

        for &face in self.boundary_faces() {
            if !self.is_boundary_face(face) {
                return Err(MeshValidationError::BoundaryFaceMismatch { face });
            }
        }

        for &face in self.interior_faces() {
            if self.is_boundary_face(face) {
                return Err(MeshValidationError::InteriorFaceMismatch { face });
            }
        }

        Ok(())
    }
}

/// 网格几何计算辅助结构
///
/// 提供 Backend 感知的几何运算方法，所有计算使用泛型标量类型。
pub struct MeshGeometry;

impl MeshGeometry {
    /// 计算两点间距离
    ///
    /// # 参数
    /// - `p1`: 起点坐标 `[x1, y1]`
    /// - `p2`: 终点坐标 `[x2, y2]`
    ///
    /// # 返回
    /// 两点间的欧几里得距离
    #[inline]
    pub fn distance<S: Scalar>(p1: [S; 2], p2: [S; 2]) -> S {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        (dx * dx + dy * dy).sqrt()
    }
    
    /// 计算单位法向量（严格版本）
    ///
    /// 返回指向线段左侧的单位法向量 `[-dy/L, dx/L]`。
    ///
    /// # Panics
    /// 当线段长度接近零时 panic，避免静默错误传播。
    /// 工业级代码不允许返回零向量导致后续计算错误。
    ///
    /// # 示例
    /// ```ignore
    /// let n = MeshGeometry::unit_normal([0.0, 0.0], [1.0, 0.0]);
    /// assert!((n[0] - 0.0).abs() < 1e-10);
    /// assert!((n[1] - 1.0).abs() < 1e-10);
    /// ```
    #[inline]
    pub fn unit_normal<S: Scalar>(p1: [S; 2], p2: [S; 2]) -> [S; 2] {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let length = (dx * dx + dy * dy).sqrt();
        
        // 工业级要求：禁止返回零向量
        if length <= S::MIN_POSITIVE {
            panic!("MeshGeometry::unit_normal: 零长度线段，无法计算法向量");
        }
        
        [-dy / length, dx / length]
    }
    
    /// 安全版本的单位法向量计算
    ///
    /// # 返回
    /// - `Some([nx, ny])` 如果计算成功
    /// - `None` 如果线段长度接近零
    #[inline]
    pub fn try_unit_normal<S: Scalar>(p1: [S; 2], p2: [S; 2]) -> Option<[S; 2]> {
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];
        let length = (dx * dx + dy * dy).sqrt();
        
        if length <= S::MIN_POSITIVE {
            None
        } else {
            Some([-dy / length, dx / length])
        }
    }
    
    /// 计算三角形面积（使用叉积公式）
    ///
    /// # 参数
    /// - `p1`, `p2`, `p3`: 三角形的三个顶点
    ///
    /// # 返回
    /// 三角形面积（始终为正值）
    #[inline]
    pub fn triangle_area<S: Scalar>(p1: [S; 2], p2: [S; 2], p3: [S; 2]) -> S {
        let half = S::from_config(0.5).unwrap_or(S::ONE);
        let cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
        cross.abs() * half
    }
    
    /// 计算多边形面积（Shoelace 公式）
    ///
    /// # 参数
    /// - `vertices`: 多边形顶点数组（按顺序排列）
    ///
    /// # 返回
    /// 多边形面积（始终为正值）
    pub fn polygon_area<S: Scalar>(vertices: &[[S; 2]]) -> S {
        if vertices.len() < 3 {
            return S::ZERO;
        }
        
        let mut sum = S::ZERO;
        let n = vertices.len();
        for i in 0..n {
            let j = (i + 1) % n;
            sum = sum + vertices[i][0] * vertices[j][1];
            sum = sum - vertices[j][0] * vertices[i][1];
        }
        
        let half = S::from_config(0.5).unwrap_or(S::ONE);
        sum.abs() * half
    }
    
    /// 计算多边形质心
    ///
    /// # 参数
    /// - `vertices`: 多边形顶点数组
    ///
    /// # 返回
    /// 质心坐标 `[cx, cy]`，如果多边形无效则返回 `None`
    pub fn polygon_centroid<S: Scalar>(vertices: &[[S; 2]]) -> Option<[S; 2]> {
        if vertices.len() < 3 {
            return None;
        }
        
        let mut cx = S::ZERO;
        let mut cy = S::ZERO;
        let mut area_sum = S::ZERO;
        
        let n = vertices.len();
        for i in 0..n {
            let j = (i + 1) % n;
            let cross = vertices[i][0] * vertices[j][1] - vertices[j][0] * vertices[i][1];
            cx = cx + (vertices[i][0] + vertices[j][0]) * cross;
            cy = cy + (vertices[i][1] + vertices[j][1]) * cross;
            area_sum = area_sum + cross;
        }
        
        let area = area_sum.abs();
        if area <= S::MIN_POSITIVE {
            return None;
        }
        
        let factor = S::ONE / (S::from_config(3.0).unwrap_or(S::ONE) * area_sum);
        Some([cx * factor, cy * factor])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance() {
        let p1 = [0.0_f64, 0.0];
        let p2 = [3.0, 4.0];
        assert!((MeshGeometry::distance(p1, p2) - 5.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_distance_same_point() {
        let p = [1.0_f64, 2.0];
        assert!(MeshGeometry::distance(p, p) < 1e-15);
    }
    
    #[test]
    fn test_unit_normal_horizontal() {
        let p1 = [0.0_f64, 0.0];
        let p2 = [1.0, 0.0];
        let n = MeshGeometry::unit_normal(p1, p2);
        // 水平线段的法向量应该指向上方 [0, 1]
        assert!((n[0] - 0.0).abs() < 1e-10);
        assert!((n[1] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_unit_normal_vertical() {
        let p1 = [0.0_f64, 0.0];
        let p2 = [0.0, 1.0];
        let n = MeshGeometry::unit_normal(p1, p2);
        // 垂直线段的法向量应该指向左侧 [-1, 0]
        assert!((n[0] - (-1.0)).abs() < 1e-10);
        assert!((n[1] - 0.0).abs() < 1e-10);
    }
    
    #[test]
    #[should_panic(expected = "零长度线段")]
    fn test_unit_normal_zero_length() {
        let p = [1.0_f64, 2.0];
        let _ = MeshGeometry::unit_normal(p, p);
    }
    
    #[test]
    fn test_try_unit_normal_zero_length() {
        let p = [1.0_f64, 2.0];
        assert!(MeshGeometry::try_unit_normal(p, p).is_none());
    }
    
    #[test]
    fn test_triangle_area() {
        let p1 = [0.0_f64, 0.0];
        let p2 = [1.0, 0.0];
        let p3 = [0.0, 1.0];
        let area = MeshGeometry::triangle_area(p1, p2, p3);
        assert!((area - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_polygon_area_square() {
        let vertices = [
            [0.0_f64, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let area = MeshGeometry::polygon_area(&vertices);
        assert!((area - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_polygon_centroid() {
        let vertices = [
            [0.0_f64, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ];
        let centroid = MeshGeometry::polygon_centroid(&vertices).unwrap();
        assert!((centroid[0] - 1.0).abs() < 1e-10);
        assert!((centroid[1] - 1.0).abs() < 1e-10);
    }
}
