// marihydro\crates\mh_terrain\src/interpolation/natural_neighbor.rs

//! Natural Neighbor (Sibson) 插值
//!
//! 基于 Voronoi 图的局部插值方法。
//!
//! # 算法原理
//!
//! Natural Neighbor 插值使用 Voronoi 图的面积作为权重：
//! 1. 构建原始采样点的 Voronoi 图
//! 2. 将插值点加入，计算新的 Voronoi 单元
//! 3. 权重 = 新单元"偷取"的旧单元面积 / 新单元总面积
//!
//! # 特点
//!
//! - 优点：C1 连续，在采样点处精确通过，局部性好
//! - 缺点：计算复杂度高，需要构建 Voronoi 图
//!
//! # 实现说明
//!
//! 本模块提供简化实现，使用加权距离近似真正的 Natural Neighbor。
//! 完整实现需要 Delaunay 三角化和 Voronoi 图构建。
//!
//! # 示例
//!
//! ```ignore
//! use mh_terrain::interpolation::natural_neighbor::NaturalNeighborInterpolator;
//! use mh_geo::Point2D;
//!
//! let points = vec![...];
//! let values = vec![...];
//!
//! let nn = NaturalNeighborInterpolator::new(points, values)
//!     .with_search_radius(100.0);
//!
//! let z = nn.interpolate(50.0, 50.0);
//! ```

use mh_geo::Point2D;
use serde::{Deserialize, Serialize};

/// Natural Neighbor 插值配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaturalNeighborConfig {
    /// 搜索半径
    pub search_radius: f64,
    /// 最小邻居数
    pub min_neighbors: usize,
    /// 最大邻居数
    pub max_neighbors: usize,
    /// 距离容差（小于此值视为在采样点上）
    pub distance_tolerance: f64,
    /// 是否使用光滑权重函数
    pub use_smooth_weights: bool,
}

impl Default for NaturalNeighborConfig {
    fn default() -> Self {
        Self {
            search_radius: f64::MAX,
            min_neighbors: 3,
            max_neighbors: 20,
            distance_tolerance: 1e-10,
            use_smooth_weights: true,
        }
    }
}

impl NaturalNeighborConfig {
    /// 创建新配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置搜索半径
    pub fn with_search_radius(mut self, radius: f64) -> Self {
        self.search_radius = radius;
        self
    }

    /// 设置邻居数范围
    pub fn with_neighbor_range(mut self, min: usize, max: usize) -> Self {
        self.min_neighbors = min;
        self.max_neighbors = max;
        self
    }
}

/// Natural Neighbor 插值器
///
/// 使用基于距离的权重近似 Natural Neighbor 插值。
/// 
/// 注意：这是简化实现，真正的 Natural Neighbor 需要 Voronoi 图计算。
#[derive(Debug, Clone)]
pub struct NaturalNeighborInterpolator {
    /// 采样点坐标
    points: Vec<Point2D>,
    /// 采样点值
    values: Vec<f64>,
    /// 配置
    config: NaturalNeighborConfig,
}

impl NaturalNeighborInterpolator {
    /// 创建新的 Natural Neighbor 插值器
    ///
    /// # 参数
    /// - `points`: 采样点坐标列表
    /// - `values`: 采样点值列表
    pub fn new(points: Vec<Point2D>, values: Vec<f64>) -> Self {
        assert_eq!(
            points.len(),
            values.len(),
            "采样点数量必须等于值数量"
        );

        Self {
            points,
            values,
            config: NaturalNeighborConfig::default(),
        }
    }

    /// 使用指定配置创建
    pub fn with_config(
        points: Vec<Point2D>,
        values: Vec<f64>,
        config: NaturalNeighborConfig,
    ) -> Self {
        assert_eq!(points.len(), values.len());
        Self {
            points,
            values,
            config,
        }
    }

    /// 设置搜索半径
    pub fn with_search_radius(mut self, radius: f64) -> Self {
        self.config.search_radius = radius;
        self
    }

    /// 设置邻居数范围
    pub fn with_neighbor_range(mut self, min: usize, max: usize) -> Self {
        self.config.min_neighbors = min;
        self.config.max_neighbors = max;
        self
    }

    /// 获取采样点数量
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// 获取配置
    pub fn config(&self) -> &NaturalNeighborConfig {
        &self.config
    }

    /// 在指定点插值
    ///
    /// 使用改进的 Natural Neighbor 近似算法。
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        // 计算到所有点的距离
        let mut neighbors: Vec<(usize, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dx = p.x - x;
                let dy = p.y - y;
                (i, (dx * dx + dy * dy).sqrt())
            })
            .filter(|&(_, d)| d <= self.config.search_radius)
            .collect();

        // 检查是否在采样点上
        for &(idx, dist) in &neighbors {
            if dist < self.config.distance_tolerance {
                return Some(self.values[idx]);
            }
        }

        // 按距离排序
        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // 检查邻居数量
        if neighbors.len() < self.config.min_neighbors {
            // 如果邻居不足，回退到最近邻
            return self.nearest_neighbor(x, y);
        }

        // 限制邻居数量
        neighbors.truncate(self.config.max_neighbors);

        // 计算 Natural Neighbor 风格的权重
        // 使用 Sibson 风格的权重近似
        let weights = self.compute_weights(&neighbors);

        // 加权插值
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;

        for (i, &(idx, _)) in neighbors.iter().enumerate() {
            let w = weights[i];
            weight_sum += w;
            value_sum += w * self.values[idx];
        }

        if weight_sum > 1e-10 {
            Some(value_sum / weight_sum)
        } else {
            self.nearest_neighbor(x, y)
        }
    }

    /// 计算 Natural Neighbor 风格的权重
    ///
    /// 使用改进的权重函数近似真正的 Natural Neighbor 权重。
    fn compute_weights(&self, neighbors: &[(usize, f64)]) -> Vec<f64> {
        if neighbors.is_empty() {
            return vec![];
        }

        let n = neighbors.len();
        let mut weights = vec![0.0; n];

        if n == 1 {
            weights[0] = 1.0;
            return weights;
        }

        // 计算参考距离（最近邻居的距离）
        let d_min = neighbors[0].1.max(1e-10);

        if self.config.use_smooth_weights {
            // 使用光滑权重函数
            // 类似于 Sibson 坐标的近似
            for (i, &(_, dist)) in neighbors.iter().enumerate() {
                let r = dist / d_min;
                // 使用紧支撑函数
                if r < 3.0 {
                    // Wendland C2 核函数
                    let t = 1.0 - r / 3.0;
                    weights[i] = t.powi(4) * (4.0 * r / 3.0 + 1.0);
                } else {
                    weights[i] = 0.0;
                }
            }
        } else {
            // 使用简单的逆距离权重
            for (i, &(_, dist)) in neighbors.iter().enumerate() {
                weights[i] = 1.0 / (dist * dist + 1e-10);
            }
        }

        // 归一化权重
        let sum: f64 = weights.iter().sum();
        if sum > 1e-10 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        weights
    }

    /// 最近邻插值（回退方法）
    fn nearest_neighbor(&self, x: f64, y: f64) -> Option<f64> {
        let mut min_dist = f64::MAX;
        let mut nearest_value = None;

        for (i, p) in self.points.iter().enumerate() {
            let dist = ((p.x - x).powi(2) + (p.y - y).powi(2)).sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest_value = Some(self.values[i]);
            }
        }

        nearest_value
    }

    /// 批量插值
    pub fn interpolate_batch(&self, points: &[(f64, f64)]) -> Vec<Option<f64>> {
        points
            .iter()
            .map(|&(x, y)| self.interpolate(x, y))
            .collect()
    }

    /// 获取指定点的邻居信息
    ///
    /// # 返回
    /// (邻居索引列表, 权重列表)
    pub fn get_neighbors(&self, x: f64, y: f64) -> (Vec<usize>, Vec<f64>) {
        if self.points.is_empty() {
            return (vec![], vec![]);
        }

        let mut neighbors: Vec<(usize, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dx = p.x - x;
                let dy = p.y - y;
                (i, (dx * dx + dy * dy).sqrt())
            })
            .filter(|&(_, d)| d <= self.config.search_radius)
            .collect();

        neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.truncate(self.config.max_neighbors);

        let weights = self.compute_weights(&neighbors);
        let indices: Vec<usize> = neighbors.iter().map(|&(idx, _)| idx).collect();

        (indices, weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_neighbor_at_sample_point() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
        let values = vec![5.0, 10.0];

        let nn = NaturalNeighborInterpolator::new(points, values);

        // 在采样点上应该返回精确值
        assert!((nn.interpolate(0.0, 0.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((nn.interpolate(1.0, 0.0).unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_natural_neighbor_interpolation() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(2.0, 0.0)];
        let values = vec![0.0, 10.0];

        let nn = NaturalNeighborInterpolator::new(points, values);

        // 中点应该接近 5.0
        let result = nn.interpolate(1.0, 0.0).unwrap();
        assert!((result - 5.0).abs() < 2.0);
    }

    #[test]
    fn test_natural_neighbor_2d() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        // z = x + y
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let nn = NaturalNeighborInterpolator::new(points, values);

        // 中心点应该接近 1.0 (= 0.5 + 0.5)
        let result = nn.interpolate(0.5, 0.5).unwrap();
        assert!((result - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_natural_neighbor_search_radius() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(100.0, 0.0), // 远点
        ];
        let values = vec![0.0, 100.0];

        let nn = NaturalNeighborInterpolator::new(points, values).with_search_radius(10.0);

        // 半径内只有一个点
        let result = nn.interpolate(1.0, 0.0).unwrap();
        assert!((result - 0.0).abs() < 0.1);
    }

    #[test]
    fn test_natural_neighbor_get_neighbors() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let values = vec![0.0, 1.0, 2.0];

        let nn = NaturalNeighborInterpolator::new(points, values);
        let (indices, weights) = nn.get_neighbors(0.5, 0.5);

        assert_eq!(indices.len(), 3);
        assert_eq!(weights.len(), 3);

        // 权重和应该接近 1
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_natural_neighbor_empty() {
        let nn = NaturalNeighborInterpolator::new(vec![], vec![]);
        assert!(nn.interpolate(0.0, 0.0).is_none());
    }

    #[test]
    fn test_natural_neighbor_batch() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];
        let values = vec![0.0, 1.0, 2.0];

        let nn = NaturalNeighborInterpolator::new(points, values);
        let test_points = vec![(0.5, 0.5), (0.0, 0.0), (10.0, 10.0)];
        let results = nn.interpolate_batch(&test_points);

        assert_eq!(results.len(), 3);
        assert!(results[0].is_some());
        assert!(results[1].is_some());
    }
}

