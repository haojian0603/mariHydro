// marihydro\crates\mh_terrain\src/interpolation/idw.rs

//! IDW (Inverse Distance Weighting) 插值
//!
//! 反距离加权插值方法，是最常用的空间插值方法之一。
//!
//! # 算法原理
//!
//! IDW 基于"距离越近，相关性越强"的假设，使用距离的倒数作为权重：
//!
//! $$
//! z(x) = \frac{\sum_{i=1}^{n} w_i \cdot z_i}{\sum_{i=1}^{n} w_i}
//! $$
//!
//! 其中权重 $w_i = \frac{1}{d_i^p}$，$p$ 为距离指数（通常为 2）。
//!
//! # 特点
//!
//! - 优点：计算简单，直观易懂
//! - 缺点：存在"牛眼效应"，在采样点处梯度不连续
//!
//! # 示例
//!
//! ```ignore
//! use mh_terrain::interpolation::idw::IdwInterpolator;
//! use mh_geo::Point2D;
//!
//! let points = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(1.0, 0.0),
//!     Point2D::new(0.0, 1.0),
//! ];
//! let values = vec![0.0, 1.0, 2.0];
//!
//! let idw = IdwInterpolator::new(points, values)
//!     .with_power(2.0)
//!     .with_search_radius(100.0);
//!
//! let z = idw.interpolate(0.5, 0.5);
//! ```

use mh_geo::Point2D;
use serde::{Deserialize, Serialize};

/// IDW 插值配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdwConfig {
    /// 距离指数 (p)，通常为 1-3，默认 2
    pub power: f64,
    /// 搜索半径限制，None 表示无限制
    pub search_radius: Option<f64>,
    /// 最大邻居数限制，None 表示无限制
    pub max_neighbors: Option<usize>,
    /// 最小邻居数，如果邻居不足则返回 None
    pub min_neighbors: usize,
    /// 距离容差（小于此值视为在采样点上）
    pub distance_tolerance: f64,
}

impl Default for IdwConfig {
    fn default() -> Self {
        Self {
            power: 2.0,
            search_radius: None,
            max_neighbors: None,
            min_neighbors: 1,
            distance_tolerance: 1e-10,
        }
    }
}

impl IdwConfig {
    /// 创建新配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置距离指数
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// 设置搜索半径
    pub fn with_search_radius(mut self, radius: f64) -> Self {
        self.search_radius = Some(radius);
        self
    }

    /// 设置最大邻居数
    pub fn with_max_neighbors(mut self, n: usize) -> Self {
        self.max_neighbors = Some(n);
        self
    }

    /// 设置最小邻居数
    pub fn with_min_neighbors(mut self, n: usize) -> Self {
        self.min_neighbors = n;
        self
    }
}

/// IDW 插值器
///
/// 使用反距离加权进行空间插值
#[derive(Debug, Clone)]
pub struct IdwInterpolator {
    /// 采样点坐标
    points: Vec<Point2D>,
    /// 采样点值
    values: Vec<f64>,
    /// 配置
    config: IdwConfig,
}

impl IdwInterpolator {
    /// 创建新的 IDW 插值器
    ///
    /// # 参数
    /// - `points`: 采样点坐标列表
    /// - `values`: 采样点值列表（长度必须与 points 相同）
    ///
    /// # Panics
    /// 如果 points 和 values 长度不同则 panic
    pub fn new(points: Vec<Point2D>, values: Vec<f64>) -> Self {
        assert_eq!(
            points.len(),
            values.len(),
            "采样点数量 ({}) 必须等于值数量 ({})",
            points.len(),
            values.len()
        );

        Self {
            points,
            values,
            config: IdwConfig::default(),
        }
    }

    /// 使用指定配置创建
    pub fn with_config(points: Vec<Point2D>, values: Vec<f64>, config: IdwConfig) -> Self {
        assert_eq!(points.len(), values.len());
        Self {
            points,
            values,
            config,
        }
    }

    /// 设置距离指数
    pub fn with_power(mut self, power: f64) -> Self {
        self.config.power = power;
        self
    }

    /// 设置搜索半径
    pub fn with_search_radius(mut self, radius: f64) -> Self {
        self.config.search_radius = Some(radius);
        self
    }

    /// 设置最大邻居数
    pub fn with_max_neighbors(mut self, n: usize) -> Self {
        self.config.max_neighbors = Some(n);
        self
    }

    /// 获取采样点数量
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// 获取配置
    pub fn config(&self) -> &IdwConfig {
        &self.config
    }

    /// 在指定点插值
    ///
    /// # 参数
    /// - `x`: 插值点 x 坐标
    /// - `y`: 插值点 y 坐标
    ///
    /// # 返回
    /// 如果有足够的邻居点，返回 Some(插值结果)；否则返回 None
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        if self.points.is_empty() {
            return None;
        }

        // 计算到所有点的距离
        let mut distances: Vec<(usize, f64)> = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let dx = p.x - x;
                let dy = p.y - y;
                (i, (dx * dx + dy * dy).sqrt())
            })
            .collect();

        // 检查是否恰好在采样点上
        for &(idx, dist) in &distances {
            if dist < self.config.distance_tolerance {
                return Some(self.values[idx]);
            }
        }

        // 应用搜索半径过滤
        if let Some(radius) = self.config.search_radius {
            distances.retain(|&(_, d)| d <= radius);
        }

        // 检查最小邻居数
        if distances.len() < self.config.min_neighbors {
            return None;
        }

        // 按距离排序
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // 应用最大邻居数限制
        if let Some(max_n) = self.config.max_neighbors {
            distances.truncate(max_n);
        }

        // 计算权重和加权值
        let p = self.config.power;
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;

        for (idx, dist) in distances {
            let weight = 1.0 / dist.powf(p);
            weight_sum += weight;
            value_sum += weight * self.values[idx];
        }

        if weight_sum > 0.0 {
            Some(value_sum / weight_sum)
        } else {
            None
        }
    }

    /// 批量插值
    pub fn interpolate_batch(&self, points: &[(f64, f64)]) -> Vec<Option<f64>> {
        points
            .iter()
            .map(|&(x, y)| self.interpolate(x, y))
            .collect()
    }

    /// 在网格点上插值
    ///
    /// # 参数
    /// - `x_min`, `y_min`: 网格左下角坐标
    /// - `x_max`, `y_max`: 网格右上角坐标
    /// - `nx`, `ny`: x 和 y 方向的网格点数
    ///
    /// # 返回
    /// 二维数组，行优先存储 (ny 行, nx 列)
    pub fn interpolate_grid(
        &self,
        x_min: f64,
        y_min: f64,
        x_max: f64,
        y_max: f64,
        nx: usize,
        ny: usize,
    ) -> Vec<Vec<Option<f64>>> {
        let dx = if nx > 1 {
            (x_max - x_min) / (nx - 1) as f64
        } else {
            0.0
        };
        let dy = if ny > 1 {
            (y_max - y_min) / (ny - 1) as f64
        } else {
            0.0
        };

        (0..ny)
            .map(|j| {
                let y = y_min + j as f64 * dy;
                (0..nx)
                    .map(|i| {
                        let x = x_min + i as f64 * dx;
                        self.interpolate(x, y)
                    })
                    .collect()
            })
            .collect()
    }

    /// 计算交叉验证误差（留一法）
    ///
    /// 对每个采样点，使用其他所有点进行插值并计算误差。
    ///
    /// # 返回
    /// (均方根误差, 平均绝对误差, 最大绝对误差)
    pub fn cross_validation(&self) -> (f64, f64, f64) {
        if self.points.len() < 2 {
            return (0.0, 0.0, 0.0);
        }

        let mut sum_sq_error: f64 = 0.0;
        let mut sum_abs_error: f64 = 0.0;
        let mut max_abs_error: f64 = 0.0;
        let mut count: usize = 0;

        for i in 0..self.points.len() {
            // 创建不包含第 i 个点的临时插值器
            let mut temp_points = self.points.clone();
            let mut temp_values = self.values.clone();
            let test_point = temp_points.remove(i);
            let true_value = temp_values.remove(i);

            let temp_interp = IdwInterpolator::with_config(temp_points, temp_values, self.config.clone());

            if let Some(predicted) = temp_interp.interpolate(test_point.x, test_point.y) {
                let error = (predicted - true_value).abs();
                sum_sq_error += error * error;
                sum_abs_error += error;
                max_abs_error = max_abs_error.max(error);
                count += 1;
            }
        }

        if count > 0 {
            let rmse = (sum_sq_error / count as f64).sqrt();
            let mae = sum_abs_error / count as f64;
            (rmse, mae, max_abs_error)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idw_basic() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let idw = IdwInterpolator::new(points, values);

        // 中心点应该接近平均值
        let result = idw.interpolate(0.5, 0.5).unwrap();
        assert!((result - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_idw_at_sample_point() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(1.0, 0.0)];
        let values = vec![5.0, 10.0];

        let idw = IdwInterpolator::new(points, values);

        // 在采样点上应该返回精确值
        assert!((idw.interpolate(0.0, 0.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((idw.interpolate(1.0, 0.0).unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_idw_with_power() {
        let points = vec![Point2D::new(0.0, 0.0), Point2D::new(2.0, 0.0)];
        let values = vec![0.0, 10.0];

        // 高指数使权重更集中在近点
        let idw_p1 = IdwInterpolator::new(points.clone(), values.clone()).with_power(1.0);
        let idw_p4 = IdwInterpolator::new(points.clone(), values.clone()).with_power(4.0);

        let v1 = idw_p1.interpolate(0.5, 0.0).unwrap();
        let v4 = idw_p4.interpolate(0.5, 0.0).unwrap();

        // 高指数时，更靠近左边的点 (0,0)，所以值更小
        assert!(v4 < v1);
    }

    #[test]
    fn test_idw_search_radius() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0), // 远点
        ];
        let values = vec![0.0, 100.0];

        let idw = IdwInterpolator::new(points, values)
            .with_search_radius(5.0)
            .with_power(2.0);

        // 半径内只有一个点
        let result = idw.interpolate(1.0, 0.0).unwrap();
        assert!((result - 0.0).abs() < 0.01); // 应该接近 0，因为远点被排除
    }

    #[test]
    fn test_idw_max_neighbors() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
        ];
        let values = vec![0.0, 1.0, 2.0, 3.0];

        let idw = IdwInterpolator::new(points, values).with_max_neighbors(2);

        // 只使用最近的 2 个点
        let result = idw.interpolate(0.5, 0.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_idw_grid_interpolation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let idw = IdwInterpolator::new(points, values);
        let grid = idw.interpolate_grid(0.0, 0.0, 1.0, 1.0, 3, 3);

        assert_eq!(grid.len(), 3);
        assert_eq!(grid[0].len(), 3);

        // 角点应该等于采样值
        assert!((grid[0][0].unwrap() - 0.0).abs() < 1e-6);
        assert!((grid[0][2].unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_idw_cross_validation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        // 完美线性场: z = x + y
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let idw = IdwInterpolator::new(points, values);
        let (rmse, mae, max_error) = idw.cross_validation();

        // IDW 对线性场应该有较小的误差
        assert!(rmse < 1.0);
        assert!(mae < 1.0);
        assert!(max_error < 1.0);
    }

    #[test]
    fn test_idw_empty() {
        let idw = IdwInterpolator::new(vec![], vec![]);
        assert!(idw.interpolate(0.0, 0.0).is_none());
    }
}

