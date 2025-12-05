// marihydro\crates\mh_mesh\src/algorithms/boundary.rs

//! 边界检测和处理
//!
//! 提供边界循环检测、边界提取等功能。

use std::collections::{HashMap, HashSet};

/// 边界循环
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// 边界顶点索引 (有序)
    pub vertices: Vec<usize>,
    /// 边界边索引 (有序)
    pub edges: Vec<(usize, usize)>,
    /// 是否为外边界
    pub is_outer: bool,
    /// 周长
    pub perimeter: f64,
    /// 包围盒
    pub bbox: BoundingBox2D,
}

impl BoundaryLoop {
    /// 创建边界循环
    pub fn new(vertices: Vec<usize>, is_outer: bool) -> Self {
        let edges: Vec<_> = (0..vertices.len())
            .map(|i| (vertices[i], vertices[(i + 1) % vertices.len()]))
            .collect();

        Self {
            vertices,
            edges,
            is_outer,
            perimeter: 0.0,
            bbox: BoundingBox2D::default(),
        }
    }

    /// 计算周长
    pub fn compute_perimeter(&mut self, coords: &[[f64; 2]]) {
        self.perimeter = 0.0;
        for i in 0..self.vertices.len() {
            let j = (i + 1) % self.vertices.len();
            let a = coords[self.vertices[i]];
            let b = coords[self.vertices[j]];
            self.perimeter += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        }
    }

    /// 计算包围盒
    pub fn compute_bbox(&mut self, coords: &[[f64; 2]]) {
        if self.vertices.is_empty() {
            return;
        }

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for &v in &self.vertices {
            let p = coords[v];
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_y = min_y.min(p[1]);
            max_y = max_y.max(p[1]);
        }

        self.bbox = BoundingBox2D {
            min: [min_x, min_y],
            max: [max_x, max_y],
        };
    }

    /// 顶点数
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// 2D 包围盒
#[derive(Debug, Clone, Default)]
pub struct BoundingBox2D {
    /// 最小坐标
    pub min: [f64; 2],
    /// 最大坐标
    pub max: [f64; 2],
}

impl BoundingBox2D {
    /// 宽度
    pub fn width(&self) -> f64 {
        self.max[0] - self.min[0]
    }

    /// 高度
    pub fn height(&self) -> f64 {
        self.max[1] - self.min[1]
    }

    /// 面积
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// 中心
    pub fn center(&self) -> [f64; 2] {
        [
            (self.min[0] + self.max[0]) / 2.0,
            (self.min[1] + self.max[1]) / 2.0,
        ]
    }

    /// 是否包含点
    pub fn contains(&self, point: [f64; 2]) -> bool {
        point[0] >= self.min[0]
            && point[0] <= self.max[0]
            && point[1] >= self.min[1]
            && point[1] <= self.max[1]
    }
}

/// 边界提取器
pub struct BoundaryExtractor;

impl BoundaryExtractor {
    /// 从三角形网格提取边界
    ///
    /// # 参数
    /// - `num_vertices`: 顶点数量
    /// - `triangles`: 三角形顶点索引
    /// - `coords`: 顶点坐标 (可选，用于计算几何属性)
    ///
    /// # 返回
    /// 边界循环列表
    pub fn extract_from_triangles(
        triangles: &[[usize; 3]],
        coords: Option<&[[f64; 2]]>,
    ) -> Vec<BoundaryLoop> {
        // 收集边界边（只出现一次的边）
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

        for tri in triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        // 找出边界边
        let boundary_edges: Vec<(usize, usize)> = edge_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(edge, _)| edge)
            .collect();

        if boundary_edges.is_empty() {
            return Vec::new();
        }

        // 构建边界循环
        Self::build_loops_from_edges(&boundary_edges, coords)
    }

    /// 从边界边构建循环
    fn build_loops_from_edges(
        edges: &[(usize, usize)],
        coords: Option<&[[f64; 2]]>,
    ) -> Vec<BoundaryLoop> {
        if edges.is_empty() {
            return Vec::new();
        }

        // 构建邻接表
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in edges {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }

        let mut visited_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut loops = Vec::new();

        // 遍历所有边，构建循环
        for &(start_a, start_b) in edges {
            let edge = if start_a < start_b {
                (start_a, start_b)
            } else {
                (start_b, start_a)
            };

            if visited_edges.contains(&edge) {
                continue;
            }

            // 开始一个新循环
            let mut loop_vertices = vec![start_a];
            let mut current = start_b;
            let mut prev = start_a;

            visited_edges.insert(edge);

            while current != start_a {
                loop_vertices.push(current);

                // 找下一个顶点
                let neighbors = adjacency.get(&current).unwrap();
                let next = neighbors.iter().find(|&&n| n != prev);

                if let Some(&next_v) = next {
                    let edge = if current < next_v {
                        (current, next_v)
                    } else {
                        (next_v, current)
                    };
                    visited_edges.insert(edge);
                    prev = current;
                    current = next_v;
                } else {
                    // 非流形边界，停止
                    break;
                }
            }

            let mut boundary_loop = BoundaryLoop::new(loop_vertices, false);

            // 计算几何属性
            if let Some(c) = coords {
                boundary_loop.compute_perimeter(c);
                boundary_loop.compute_bbox(c);
            }

            loops.push(boundary_loop);
        }

        // 确定外边界（最大的循环）
        if !loops.is_empty() {
            if let Some(coords) = coords {
                // 使用面积确定外边界
                let outer_idx = loops
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        let area_a = Self::signed_area(&a.vertices, coords);
                        let area_b = Self::signed_area(&b.vertices, coords);
                        area_a.abs().partial_cmp(&area_b.abs()).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                loops[outer_idx].is_outer = true;
            } else {
                // 使用顶点数估计
                let outer_idx = loops
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, l)| l.len())
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                loops[outer_idx].is_outer = true;
            }
        }

        loops
    }

    /// 计算有符号面积
    fn signed_area(vertices: &[usize], coords: &[[f64; 2]]) -> f64 {
        let n = vertices.len();
        if n < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let a = coords[vertices[i]];
            let b = coords[vertices[j]];
            area += a[0] * b[1] - b[0] * a[1];
        }

        area / 2.0
    }

    /// 提取边界顶点集合
    pub fn boundary_vertices(triangles: &[[usize; 3]]) -> HashSet<usize> {
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

        for tri in triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        let mut boundary = HashSet::new();
        for ((a, b), count) in edge_count {
            if count == 1 {
                boundary.insert(a);
                boundary.insert(b);
            }
        }

        boundary
    }

    /// 提取边界边
    pub fn boundary_edges(triangles: &[[usize; 3]]) -> Vec<(usize, usize)> {
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();

        for tri in triangles {
            for i in 0..3 {
                let a = tri[i];
                let b = tri[(i + 1) % 3];
                let edge = if a < b { (a, b) } else { (b, a) };
                *edge_count.entry(edge).or_insert(0) += 1;
            }
        }

        edge_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(edge, _)| edge)
            .collect()
    }
}

/// 边界分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// 外边界
    Outer,
    /// 内边界（孔）
    Inner,
    /// 未分类
    Unknown,
}

/// 边界信息
#[derive(Debug, Clone)]
pub struct BoundaryInfo {
    /// 边界类型
    pub boundary_type: BoundaryType,
    /// 边界循环
    pub loop_ref: usize,
    /// 边界标签
    pub label: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_triangle() {
        let triangles = vec![[0, 1, 2]];

        let loops = BoundaryExtractor::extract_from_triangles(&triangles, None);

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 3);
        assert!(loops[0].is_outer);
    }

    #[test]
    fn test_two_triangles() {
        // 两个共享一边的三角形
        let triangles = vec![[0, 1, 2], [1, 3, 2]];

        let loops = BoundaryExtractor::extract_from_triangles(&triangles, None);

        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 4);
    }

    #[test]
    fn test_with_hole() {
        // 有孔的网格 (外部4个三角形，中间是孔)
        // 简化：只测试边界提取的正确性
        let triangles = vec![
            [0, 1, 4],
            [1, 2, 5],
            [2, 3, 6],
            [3, 0, 7],
            [0, 4, 7],
            [1, 5, 4],
            [2, 6, 5],
            [3, 7, 6],
        ];

        let boundary_vertices = BoundaryExtractor::boundary_vertices(&triangles);
        let boundary_edges = BoundaryExtractor::boundary_edges(&triangles);

        // 应该有边界顶点
        assert!(!boundary_vertices.is_empty());
        assert!(!boundary_edges.is_empty());
    }

    #[test]
    fn test_boundary_with_coords() {
        let triangles = vec![[0, 1, 2], [0, 2, 3]];
        let coords = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];

        let loops = BoundaryExtractor::extract_from_triangles(&triangles, Some(&coords));

        assert_eq!(loops.len(), 1);
        assert!(loops[0].perimeter > 0.0);
        assert!(loops[0].bbox.width() > 0.0);
        assert!(loops[0].bbox.height() > 0.0);
    }

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox2D {
            min: [0.0, 0.0],
            max: [10.0, 5.0],
        };

        assert_eq!(bbox.width(), 10.0);
        assert_eq!(bbox.height(), 5.0);
        assert_eq!(bbox.area(), 50.0);
        assert_eq!(bbox.center(), [5.0, 2.5]);
        assert!(bbox.contains([5.0, 2.5]));
        assert!(!bbox.contains([15.0, 2.5]));
    }
}
