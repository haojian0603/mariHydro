// marihydro\crates\mh_mesh\src/algorithms/triangulate.rs

//! 多边形三角化算法
//!
//! 支持:
//! - 耳切法 (Ear Clipping) - 简单多边形
//! - Delaunay三角化 (可选)

/// 三角化配置
#[derive(Debug, Clone)]
pub struct TriangulateConfig {
    /// 最小角度约束 (度)
    pub min_angle: f64,
    /// 最大面积约束
    pub max_area: Option<f64>,
    /// 是否进行质量优化
    pub quality_refinement: bool,
}

impl Default for TriangulateConfig {
    fn default() -> Self {
        Self {
            min_angle: 20.0,
            max_area: None,
            quality_refinement: false,
        }
    }
}

/// 三角化器
pub struct Triangulator {
    config: TriangulateConfig,
}

impl Triangulator {
    /// 创建三角化器
    pub fn new(config: TriangulateConfig) -> Self {
        Self { config }
    }

    /// 使用默认配置创建
    pub fn default_config() -> Self {
        Self::new(TriangulateConfig::default())
    }

    /// 三角化简单多边形 (耳切法)
    ///
    /// # 参数
    /// - `vertices`: 多边形顶点坐标 (逆时针顺序)
    ///
    /// # 返回
    /// 三角形顶点索引列表
    pub fn triangulate_polygon(&self, vertices: &[[f64; 2]]) -> Vec<[usize; 3]> {
        let n = vertices.len();
        
        if n < 3 {
            return Vec::new();
        }
        
        if n == 3 {
            return vec![[0, 1, 2]];
        }
        
        self.ear_clipping(vertices)
    }

    /// 耳切法实现
    fn ear_clipping(&self, vertices: &[[f64; 2]]) -> Vec<[usize; 3]> {
        let n = vertices.len();
        let mut triangles = Vec::with_capacity(n - 2);
        let mut remaining: Vec<usize> = (0..n).collect();
        
        // 确保多边形是逆时针方向
        let area = self.signed_area(vertices);
        if area < 0.0 {
            remaining.reverse();
        }
        
        let mut safety_counter = 0;
        let max_iterations = n * n; // 防止无限循环
        
        while remaining.len() > 3 && safety_counter < max_iterations {
            safety_counter += 1;
            let mut found_ear = false;
            
            let len = remaining.len();
            for i in 0..len {
                let prev_idx = if i == 0 { len - 1 } else { i - 1 };
                let next_idx = (i + 1) % len;
                
                let prev = remaining[prev_idx];
                let curr = remaining[i];
                let next = remaining[next_idx];
                
                if self.is_ear(vertices, &remaining, prev, curr, next) {
                    triangles.push([prev, curr, next]);
                    remaining.remove(i);
                    found_ear = true;
                    break;
                }
            }
            
            if !found_ear {
                // 退化情况，跳过
                break;
            }
        }
        
        // 处理剩余的三角形
        if remaining.len() == 3 {
            triangles.push([remaining[0], remaining[1], remaining[2]]);
        }
        
        triangles
    }

    /// 计算有符号面积
    fn signed_area(&self, vertices: &[[f64; 2]]) -> f64 {
        let n = vertices.len();
        let mut area = 0.0;
        
        for i in 0..n {
            let j = (i + 1) % n;
            area += vertices[i][0] * vertices[j][1];
            area -= vertices[j][0] * vertices[i][1];
        }
        
        area / 2.0
    }

    /// 检查是否为耳朵
    fn is_ear(
        &self,
        vertices: &[[f64; 2]],
        remaining: &[usize],
        prev: usize,
        curr: usize,
        next: usize,
    ) -> bool {
        let p0 = vertices[prev];
        let p1 = vertices[curr];
        let p2 = vertices[next];
        
        // 检查是否凸角
        if !self.is_convex(p0, p1, p2) {
            return false;
        }
        
        // 检查是否有其他点在三角形内
        for &i in remaining {
            if i == prev || i == curr || i == next {
                continue;
            }
            
            if self.point_in_triangle(vertices[i], p0, p1, p2) {
                return false;
            }
        }
        
        true
    }

    /// 检查凸角 (逆时针方向的凸角叉积为正)
    fn is_convex(&self, p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> bool {
        let cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        cross > 0.0
    }

    /// 点在三角形内 (使用重心坐标法)
    fn point_in_triangle(
        &self,
        p: [f64; 2],
        a: [f64; 2],
        b: [f64; 2],
        c: [f64; 2],
    ) -> bool {
        let sign = |p1: [f64; 2], p2: [f64; 2], p3: [f64; 2]| {
            (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        };
        
        let d1 = sign(p, a, b);
        let d2 = sign(p, b, c);
        let d3 = sign(p, c, a);
        
        let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
        let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
        
        // 如果在边上，不算在内部
        !(has_neg && has_pos)
    }

    /// 计算三角形面积
    pub fn triangle_area(&self, a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs() / 2.0
    }

    /// 计算三角形最小角 (度)
    pub fn triangle_min_angle(&self, a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
        let ab = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let bc = ((c[0] - b[0]).powi(2) + (c[1] - b[1]).powi(2)).sqrt();
        let ca = ((a[0] - c[0]).powi(2) + (a[1] - c[1]).powi(2)).sqrt();
        
        // 使用余弦定理计算角度
        let cos_a = (ab * ab + ca * ca - bc * bc) / (2.0 * ab * ca);
        let cos_b = (ab * ab + bc * bc - ca * ca) / (2.0 * ab * bc);
        let cos_c = (bc * bc + ca * ca - ab * ab) / (2.0 * bc * ca);
        
        let angle_a = cos_a.clamp(-1.0, 1.0).acos().to_degrees();
        let angle_b = cos_b.clamp(-1.0, 1.0).acos().to_degrees();
        let angle_c = cos_c.clamp(-1.0, 1.0).acos().to_degrees();
        
        angle_a.min(angle_b).min(angle_c)
    }
}

/// 三角化错误
#[derive(Debug, Clone, thiserror::Error)]
pub enum TriangulateError {
    #[error("Degenerate polygon: less than 3 vertices")]
    DegeneratePolygon,
    
    #[error("Self-intersecting polygon")]
    SelfIntersecting,
    
    #[error("Failed to triangulate: {0}")]
    Failed(String),
}

/// 便捷函数：三角化简单多边形
pub fn triangulate_simple(vertices: &[[f64; 2]]) -> Vec<[usize; 3]> {
    Triangulator::default_config().triangulate_polygon(vertices)
}

/// 计算三角剖分的质量指标
#[derive(Debug, Clone)]
pub struct TriangulationQuality {
    /// 三角形数量
    pub num_triangles: usize,
    /// 最小角度 (度)
    pub min_angle: f64,
    /// 最大角度 (度)
    pub max_angle: f64,
    /// 平均角度 (度)
    pub avg_angle: f64,
    /// 最小面积
    pub min_area: f64,
    /// 最大面积
    pub max_area: f64,
    /// 面积比率 (max/min)
    pub area_ratio: f64,
}

impl TriangulationQuality {
    /// 计算三角剖分质量
    pub fn compute(vertices: &[[f64; 2]], triangles: &[[usize; 3]]) -> Self {
        let triangulator = Triangulator::default_config();
        
        let mut min_angle = f64::MAX;
        let mut max_angle = f64::MIN;
        let mut sum_angle = 0.0;
        let mut min_area = f64::MAX;
        let mut max_area = f64::MIN;
        
        for tri in triangles {
            let a = vertices[tri[0]];
            let b = vertices[tri[1]];
            let c = vertices[tri[2]];
            
            let angle = triangulator.triangle_min_angle(a, b, c);
            let area = triangulator.triangle_area(a, b, c);
            
            min_angle = min_angle.min(angle);
            max_angle = max_angle.max(180.0 - angle * 2.0); // 估计最大角
            sum_angle += angle;
            min_area = min_area.min(area);
            max_area = max_area.max(area);
        }
        
        let num_triangles = triangles.len();
        
        Self {
            num_triangles,
            min_angle,
            max_angle,
            avg_angle: sum_angle / num_triangles as f64,
            min_area,
            max_area,
            area_ratio: if min_area > 0.0 { max_area / min_area } else { f64::INFINITY },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle() {
        let vertices = [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let triangles = triangulate_simple(&vertices);
        
        assert_eq!(triangles.len(), 1);
        assert_eq!(triangles[0], [0, 1, 2]);
    }

    #[test]
    fn test_square() {
        let vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let triangles = triangulate_simple(&vertices);
        
        assert_eq!(triangles.len(), 2);
    }

    #[test]
    fn test_pentagon() {
        // 正五边形
        let n = 5;
        let vertices: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64 - std::f64::consts::PI / 2.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        
        let triangles = triangulate_simple(&vertices);
        
        assert_eq!(triangles.len(), 3); // n - 2 = 3
    }

    #[test]
    fn test_convex_check() {
        let triangulator = Triangulator::default_config();
        
        // 逆时针三角形的顶点
        assert!(triangulator.is_convex([0.0, 0.0], [1.0, 0.0], [0.5, 1.0]));
        
        // 顺时针三角形的顶点
        assert!(!triangulator.is_convex([0.0, 0.0], [0.5, 1.0], [1.0, 0.0]));
    }

    #[test]
    fn test_point_in_triangle() {
        let triangulator = Triangulator::default_config();
        let a = [0.0, 0.0];
        let b = [2.0, 0.0];
        let c = [1.0, 2.0];
        
        // 内部点
        assert!(triangulator.point_in_triangle([1.0, 0.5], a, b, c));
        
        // 外部点
        assert!(!triangulator.point_in_triangle([0.0, 1.0], a, b, c));
    }

    #[test]
    fn test_triangulation_quality() {
        let vertices = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let triangles = triangulate_simple(&vertices);
        
        let quality = TriangulationQuality::compute(&vertices, &triangles);
        
        assert_eq!(quality.num_triangles, 2);
        assert!(quality.min_angle > 0.0);
        assert!(quality.area_ratio >= 1.0);
    }
}
