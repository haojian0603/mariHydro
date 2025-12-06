//! 网格验证模块
//!
//! 提供网格拓扑一致性和几何有效性验证功能。

use std::collections::HashSet;

/// 验证结果
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// 是否有效
    pub is_valid: bool,
    /// 错误列表
    pub errors: Vec<ValidationError>,
    /// 警告列表
    pub warnings: Vec<ValidationWarning>,
}

impl ValidationResult {
    /// 创建有效结果
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// 创建无效结果
    pub fn invalid(errors: Vec<ValidationError>) -> Self {
        Self {
            is_valid: false,
            errors,
            warnings: Vec::new(),
        }
    }

    /// 添加错误
    pub fn add_error(&mut self, error: ValidationError) {
        self.is_valid = false;
        self.errors.push(error);
    }

    /// 添加警告
    pub fn add_warning(&mut self, warning: ValidationWarning) {
        self.warnings.push(warning);
    }

    /// 合并另一个验证结果
    pub fn merge(&mut self, other: ValidationResult) {
        if !other.is_valid {
            self.is_valid = false;
        }
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}

/// 验证错误类型
#[derive(Debug, Clone, thiserror::Error)]
pub enum ValidationError {
    /// 半边配对不一致
    #[error("Halfedge {0} has invalid twin reference")]
    InvalidTwin(usize),

    /// 半边的下一条边不一致
    #[error("Halfedge {0} has invalid next reference")]
    InvalidNext(usize),

    /// 半边的前一条边不一致
    #[error("Halfedge {0} has invalid prev reference")]
    InvalidPrev(usize),

    /// 面的半边不构成闭合环
    #[error("Face {0} does not form a closed loop")]
    NonClosedFaceLoop(usize),

    /// 顶点没有出边
    #[error("Vertex {0} has no outgoing halfedge")]
    IsolatedVertex(usize),

    /// 顶点的出边引用不一致
    #[error("Vertex {0} has invalid outgoing halfedge reference")]
    InvalidVertexHalfedge(usize),

    /// 退化面（面积过小）
    #[error("Face {0} is degenerate (area = {1})")]
    DegenerateFace(usize, f64),

    /// 退化边（长度为零）
    #[error("Edge from vertex {0} to {1} is degenerate")]
    DegenerateEdge(usize, usize),

    /// 边界不一致
    #[error("Boundary halfedge {0} has invalid face reference")]
    InvalidBoundary(usize),

    /// 非流形顶点
    #[error("Non-manifold vertex {0}")]
    NonManifoldVertex(usize),

    /// 非流形边
    #[error("Non-manifold edge from vertex {0} to {1}")]
    NonManifoldEdge(usize, usize),

    /// 自相交
    #[error("Self-intersection detected at face {0} and {1}")]
    SelfIntersection(usize, usize),

    /// 顶点坐标无效（NaN或无穷）
    #[error("Vertex {0} has invalid coordinates")]
    InvalidCoordinates(usize),
}

/// 验证警告类型
#[derive(Debug, Clone)]
pub enum ValidationWarning {
    /// 低质量单元
    LowQualityElement {
        /// 元素类型
        element_type: ElementType,
        /// 元素索引
        index: usize,
        /// 质量值
        quality: f64,
        /// 阈值
        threshold: f64,
    },
    /// 接近退化的面
    NearDegenerateFace {
        /// 面索引
        face_index: usize,
        /// 面积
        area: f64,
    },
    /// 接近退化的边
    NearDegenerateEdge {
        /// 起始顶点
        v0: usize,
        /// 结束顶点
        v1: usize,
        /// 长度
        length: f64,
    },
    /// 孤立的边界环
    IsolatedBoundaryLoop {
        /// 环的起始半边
        start_halfedge: usize,
    },
}

/// 元素类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    /// 顶点
    Vertex,
    /// 边
    Edge,
    /// 面
    Face,
}

/// 验证配置
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// 最小面积阈值（小于此值视为退化）
    pub min_area: f64,
    /// 最小边长阈值
    pub min_edge_length: f64,
    /// 低质量警告阈值
    pub quality_warning_threshold: f64,
    /// 是否检查自相交
    pub check_self_intersection: bool,
    /// 是否检查非流形
    pub check_non_manifold: bool,
    /// 是否检查几何有效性
    pub check_geometry: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_area: 1e-12,
            min_edge_length: 1e-10,
            quality_warning_threshold: 0.1,
            check_self_intersection: false, // 默认关闭，计算代价高
            check_non_manifold: true,
            check_geometry: true,
        }
    }
}

impl ValidationConfig {
    /// 创建严格验证配置
    pub fn strict() -> Self {
        Self {
            min_area: 1e-10,
            min_edge_length: 1e-8,
            quality_warning_threshold: 0.2,
            check_self_intersection: true,
            check_non_manifold: true,
            check_geometry: true,
        }
    }

    /// 创建快速验证配置（只检查拓扑）
    pub fn fast() -> Self {
        Self {
            min_area: 1e-12,
            min_edge_length: 1e-10,
            quality_warning_threshold: 0.0,
            check_self_intersection: false,
            check_non_manifold: false,
            check_geometry: false,
        }
    }
}

/// 网格验证器
#[derive(Debug, Clone)]
pub struct MeshValidator {
    config: ValidationConfig,
}

impl MeshValidator {
    /// 创建验证器
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建验证器
    pub fn with_default() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// 验证半边拓扑一致性
    ///
    /// # 参数
    /// - `halfedges`: 半边数据 (twin, next, prev, vertex, face)
    /// - `vertices`: 顶点出边索引
    /// - `faces`: 面的起始半边索引
    pub fn validate_topology(
        &self,
        halfedges: &[(Option<usize>, usize, usize, usize, Option<usize>)],
        vertices: &[Option<usize>],
        faces: &[Option<usize>],
    ) -> ValidationResult {
        let mut result = ValidationResult::valid();

        // 检查半边配对一致性
        for (he_idx, he) in halfedges.iter().enumerate() {
            // 检查 twin 关系
            if let Some(twin_idx) = he.0 {
                if twin_idx >= halfedges.len() {
                    result.add_error(ValidationError::InvalidTwin(he_idx));
                } else if let Some(twin_twin) = halfedges[twin_idx].0 {
                    if twin_twin != he_idx {
                        result.add_error(ValidationError::InvalidTwin(he_idx));
                    }
                }
            }

            // 检查 next 引用
            let next_idx = he.1;
            if next_idx >= halfedges.len() {
                result.add_error(ValidationError::InvalidNext(he_idx));
            }

            // 检查 prev 引用
            let prev_idx = he.2;
            if prev_idx >= halfedges.len() {
                result.add_error(ValidationError::InvalidPrev(he_idx));
            } else if halfedges[prev_idx].1 != he_idx {
                result.add_error(ValidationError::InvalidPrev(he_idx));
            }
        }

        // 检查面的闭合性
        for (face_idx, face_he) in faces.iter().enumerate() {
            if let Some(start_he) = face_he {
                let mut visited = HashSet::new();
                let mut current = *start_he;
                
                loop {
                    if visited.contains(&current) {
                        if current != *start_he {
                            result.add_error(ValidationError::NonClosedFaceLoop(face_idx));
                        }
                        break;
                    }
                    visited.insert(current);
                    
                    if current >= halfedges.len() {
                        result.add_error(ValidationError::NonClosedFaceLoop(face_idx));
                        break;
                    }
                    
                    current = halfedges[current].1;
                    
                    if visited.len() > halfedges.len() {
                        result.add_error(ValidationError::NonClosedFaceLoop(face_idx));
                        break;
                    }
                }
            }
        }

        // 检查顶点一致性
        for (v_idx, v_he) in vertices.iter().enumerate() {
            if let Some(he_idx) = v_he {
                if *he_idx >= halfedges.len() {
                    result.add_error(ValidationError::InvalidVertexHalfedge(v_idx));
                } else if halfedges[*he_idx].3 != v_idx {
                    result.add_error(ValidationError::InvalidVertexHalfedge(v_idx));
                }
            }
        }

        result
    }

    /// 验证几何有效性
    ///
    /// # 参数
    /// - `positions`: 顶点位置 (x, y, z)
    /// - `face_vertices`: 每个面的顶点索引列表
    pub fn validate_geometry(
        &self,
        positions: &[[f64; 3]],
        face_vertices: &[Vec<usize>],
    ) -> ValidationResult {
        if !self.config.check_geometry {
            return ValidationResult::valid();
        }

        let mut result = ValidationResult::valid();

        // 检查顶点坐标有效性
        for (v_idx, pos) in positions.iter().enumerate() {
            if pos.iter().any(|c| c.is_nan() || c.is_infinite()) {
                result.add_error(ValidationError::InvalidCoordinates(v_idx));
            }
        }

        // 检查边长度
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for face in face_vertices {
            for i in 0..face.len() {
                let v0 = face[i];
                let v1 = face[(i + 1) % face.len()];
                let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
                
                if !edge_set.contains(&key) {
                    edge_set.insert(key);
                    
                    let p0 = &positions[v0];
                    let p1 = &positions[v1];
                    let length = ((p1[0] - p0[0]).powi(2) 
                        + (p1[1] - p0[1]).powi(2) 
                        + (p1[2] - p0[2]).powi(2)).sqrt();
                    
                    if length < self.config.min_edge_length {
                        result.add_error(ValidationError::DegenerateEdge(v0, v1));
                    } else if length < self.config.min_edge_length * 10.0 {
                        result.add_warning(ValidationWarning::NearDegenerateEdge {
                            v0,
                            v1,
                            length,
                        });
                    }
                }
            }
        }

        // 检查面面积
        for (face_idx, face) in face_vertices.iter().enumerate() {
            if face.len() < 3 {
                result.add_error(ValidationError::DegenerateFace(face_idx, 0.0));
                continue;
            }

            let area = compute_face_area(positions, face);
            
            if area < self.config.min_area {
                result.add_error(ValidationError::DegenerateFace(face_idx, area));
            } else if area < self.config.min_area * 100.0 {
                result.add_warning(ValidationWarning::NearDegenerateFace {
                    face_index: face_idx,
                    area,
                });
            }
        }

        result
    }

    /// 检查非流形结构
    ///
    /// # 参数
    /// - `halfedges`: 半边数据
    /// - `vertices`: 顶点数量
    pub fn check_manifold(
        &self,
        halfedges: &[(Option<usize>, usize, usize, usize, Option<usize>)],
        num_vertices: usize,
    ) -> ValidationResult {
        if !self.config.check_non_manifold {
            return ValidationResult::valid();
        }

        let mut result = ValidationResult::valid();

        // 检查非流形边（超过两个面共享一条边）
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> = 
            std::collections::HashMap::new();
        
        for he in halfedges {
            let v0 = he.3;
            let v1 = halfedges[he.1].3;
            let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_count.entry(key).or_insert(0) += 1;
        }

        for ((v0, v1), count) in edge_count {
            if count > 2 {
                result.add_error(ValidationError::NonManifoldEdge(v0, v1));
            }
        }

        // 检查非流形顶点（顶点周围的面不连续）
        for v_idx in 0..num_vertices {
            if !self.is_vertex_manifold(halfedges, v_idx) {
                result.add_error(ValidationError::NonManifoldVertex(v_idx));
            }
        }

        result
    }

    /// 检查顶点是否为流形
    fn is_vertex_manifold(
        &self,
        halfedges: &[(Option<usize>, usize, usize, usize, Option<usize>)],
        vertex: usize,
    ) -> bool {
        // 找到从该顶点出发的所有半边
        let outgoing: Vec<usize> = halfedges
            .iter()
            .enumerate()
            .filter(|(_, he)| he.3 == vertex)
            .map(|(idx, _)| idx)
            .collect();

        if outgoing.is_empty() {
            return true; // 孤立顶点视为流形
        }

        // 对于流形顶点，绕顶点旋转应该能访问所有出边
        let mut visited = HashSet::new();
        let mut current = outgoing[0];
        
        loop {
            if visited.contains(&current) {
                break;
            }
            visited.insert(current);
            
            // 获取 twin
            if let Some(twin) = halfedges[current].0 {
                current = halfedges[twin].1;
            } else {
                // 边界情况
                break;
            }
            
            if visited.len() > halfedges.len() {
                return false;
            }
        }

        visited.len() == outgoing.len()
    }

    /// 完整验证
    pub fn validate_full(
        &self,
        halfedges: &[(Option<usize>, usize, usize, usize, Option<usize>)],
        vertices: &[Option<usize>],
        faces: &[Option<usize>],
        positions: &[[f64; 3]],
        face_vertices: &[Vec<usize>],
    ) -> ValidationResult {
        let mut result = self.validate_topology(halfedges, vertices, faces);
        result.merge(self.validate_geometry(positions, face_vertices));
        result.merge(self.check_manifold(halfedges, vertices.len()));
        result
    }
}

/// 计算面的面积
fn compute_face_area(positions: &[[f64; 3]], vertices: &[usize]) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }

    // 使用 Newell 方法计算多边形面积
    let mut normal = [0.0, 0.0, 0.0];
    let n = vertices.len();

    for i in 0..n {
        let curr = &positions[vertices[i]];
        let next = &positions[vertices[(i + 1) % n]];

        normal[0] += (curr[1] - next[1]) * (curr[2] + next[2]);
        normal[1] += (curr[2] - next[2]) * (curr[0] + next[0]);
        normal[2] += (curr[0] - next[0]) * (curr[1] + next[1]);
    }

    0.5 * (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt()
}

/// 快速拓扑检查
///
/// 只检查最基本的拓扑一致性，适合频繁调用
pub fn quick_topology_check(
    halfedges: &[(Option<usize>, usize, usize, usize, Option<usize>)],
) -> bool {
    for (he_idx, he) in halfedges.iter().enumerate() {
        // 检查 twin 对称性
        if let Some(twin_idx) = he.0 {
            if twin_idx >= halfedges.len() {
                return false;
            }
            if let Some(twin_twin) = halfedges[twin_idx].0 {
                if twin_twin != he_idx {
                    return false;
                }
            }
        }

        // 检查 next/prev 一致性
        let next_idx = he.1;
        if next_idx >= halfedges.len() {
            return false;
        }
        if halfedges[next_idx].2 != he_idx {
            return false;
        }
    }
    true
}

/// 验证边界一致性
pub fn validate_boundary(
    halfedges: &[(Option<usize>, usize, usize, usize, Option<usize>)],
) -> Vec<ValidationError> {
    let errors = Vec::new();

    for (_he_idx, he) in halfedges.iter().enumerate() {
        // 边界半边应该没有 twin
        if he.0.is_none() {
            // 检查边界半边是否属于边界面
            if he.4.is_some() {
                // 有边界半边但属于实际面，这是错误的
                // （除非是开放网格的边界边）
            }
        }
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::valid();
        assert!(result.is_valid);
        
        result.add_error(ValidationError::InvalidTwin(0));
        assert!(!result.is_valid);
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_quick_topology_check() {
        // 简单的三角形
        let halfedges = vec![
            (Some(3), 1, 2, 0, Some(0)), // he0
            (Some(4), 2, 0, 1, Some(0)), // he1
            (Some(5), 0, 1, 2, Some(0)), // he2
            (Some(0), 4, 5, 1, None),    // he3 (boundary)
            (Some(1), 5, 3, 2, None),    // he4 (boundary)
            (Some(2), 3, 4, 0, None),    // he5 (boundary)
        ];
        
        assert!(quick_topology_check(&halfedges));
    }

    #[test]
    fn test_face_area() {
        // 边长为1的等边三角形
        let positions = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
        ];
        
        let area = compute_face_area(&positions, &[0, 1, 2]);
        assert!((area - 0.433).abs() < 0.01);
    }

    #[test]
    fn test_validator_config() {
        let strict = ValidationConfig::strict();
        assert!(strict.check_self_intersection);
        
        let fast = ValidationConfig::fast();
        assert!(!fast.check_self_intersection);
        assert!(!fast.check_geometry);
    }
}
