// marihydro\crates\mh_mesh\src/spatial_index.rs

//! 网格空间索引
//!
//! 基于 R-Tree 的空间索引，用于快速查找点所在单元。
//! 使用 rstar crate 实现高效的空间查询。
//!
//! # 功能特性
//!
//! - 快速点定位：O(log n) 时间复杂度查找点所在单元
//! - 范围查询：查找指定矩形区域内的所有单元
//! - 最近邻查询：查找距离指定点最近的单元
//! - 序列化支持：可保存和加载索引数据
//!
//! # 示例
//!
//! ```ignore
//! use mh_mesh::spatial_index::MeshSpatialIndex;
//! use mh_geo::Point2D;
//!
//! // 从网格构建空间索引
//! let index = MeshSpatialIndex::build(n_cells, |i| mesh.get_cell_vertices(i));
//!
//! // 查找点所在单元
//! if let Some(cell_idx) = index.locate_point(100.0, 200.0) {
//!     println!("点在单元 {} 内", cell_idx);
//! }
//!
//! // 保存索引
//! let data = index.to_serializable();
//! // ... 保存到文件
//!
//! // 加载索引
//! let loaded = MeshSpatialIndex::from_serializable(data);
//! ```

use mh_geo::Point2D;
use rstar::{PointDistance, RTree, RTreeObject, AABB};
use serde::{Deserialize, Serialize};

/// 单元包围盒
///
/// 存储单元的轴对齐包围盒（AABB），用于 R-Tree 索引。
/// 支持序列化，可用于保存和加载空间索引。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellEnvelope {
    /// 单元索引
    pub cell_index: usize,
    /// 最小 x 坐标
    pub min_x: f64,
    /// 最小 y 坐标
    pub min_y: f64,
    /// 最大 x 坐标
    pub max_x: f64,
    /// 最大 y 坐标
    pub max_y: f64,
}

impl CellEnvelope {
    /// 创建单元包围盒
    ///
    /// # 参数
    /// - `cell_index`: 单元索引
    /// - `vertices`: 单元顶点坐标列表
    ///
    /// # Panics
    /// 如果顶点列表为空则 panic
    pub fn new(cell_index: usize, vertices: &[Point2D]) -> Self {
        debug_assert!(!vertices.is_empty(), "单元顶点列表不能为空");
        
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        for v in vertices {
            min_x = min_x.min(v.x);
            min_y = min_y.min(v.y);
            max_x = max_x.max(v.x);
            max_y = max_y.max(v.y);
        }

        Self {
            cell_index,
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// 从边界坐标直接创建
    pub fn from_bounds(cell_index: usize, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            cell_index,
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    /// 检查点是否在包围盒内
    ///
    /// 使用闭区间判断，边界上的点也算在内
    #[inline]
    pub fn contains_point(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }

    /// 包围盒宽度
    #[inline]
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// 包围盒高度
    #[inline]
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// 包围盒面积
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// 包围盒中心点
    #[inline]
    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
        )
    }
}

impl RTreeObject for CellEnvelope {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.min_x, self.min_y], [self.max_x, self.max_y])
    }
}

impl PointDistance for CellEnvelope {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        // 计算点到 AABB 的最短距离的平方
        let dx = if point[0] < self.min_x {
            self.min_x - point[0]
        } else if point[0] > self.max_x {
            point[0] - self.max_x
        } else {
            0.0
        };

        let dy = if point[1] < self.min_y {
            self.min_y - point[1]
        } else if point[1] > self.max_y {
            point[1] - self.max_y
        } else {
            0.0
        };

        dx * dx + dy * dy
    }

    fn contains_point(&self, point: &[f64; 2]) -> bool {
        point[0] >= self.min_x
            && point[0] <= self.max_x
            && point[1] >= self.min_y
            && point[1] <= self.max_y
    }
}

// ============================================================
// 可序列化数据结构
// ============================================================

/// 可序列化的空间索引数据
///
/// 用于将空间索引保存到文件或传输。由于 R-Tree 本身不支持序列化，
/// 此结构存储构建索引所需的所有数据，加载时重建 R-Tree。
///
/// # 序列化格式
///
/// 推荐使用 bincode 进行高效的二进制序列化：
///
/// ```ignore
/// use bincode;
///
/// // 保存
/// let data = index.to_serializable();
/// let bytes = bincode::serialize(&data).unwrap();
/// std::fs::write("index.bin", bytes).unwrap();
///
/// // 加载
/// let bytes = std::fs::read("index.bin").unwrap();
/// let data: SpatialIndexData = bincode::deserialize(&bytes).unwrap();
/// let index = MeshSpatialIndex::from_serializable(data);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialIndexData {
    /// 版本号（用于向后兼容）
    pub version: u32,
    /// 单元包围盒列表
    pub envelopes: Vec<CellEnvelope>,
    /// 单元顶点数据
    pub cell_vertices: Vec<Vec<Point2D>>,
    /// 单元数量
    pub n_cells: usize,
    /// 全局边界（用于快速检查）
    pub bounds: Option<SpatialBounds>,
}

/// 空间边界
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpatialBounds {
    /// 最小 X 坐标
    pub min_x: f64,
    /// 最小 Y 坐标
    pub min_y: f64,
    /// 最大 X 坐标
    pub max_x: f64,
    /// 最大 Y 坐标
    pub max_y: f64,
}

impl SpatialBounds {
    /// 计算边界面积
    #[inline]
    pub fn area(&self) -> f64 {
        (self.max_x - self.min_x) * (self.max_y - self.min_y)
    }

    /// 检查点是否在边界内
    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.min_x && x <= self.max_x && y >= self.min_y && y <= self.max_y
    }
}

/// 空间索引数据版本号
const SPATIAL_INDEX_VERSION: u32 = 1;

/// 网格空间索引
///
/// 基于 R-Tree 的网格空间索引，提供高效的点定位和范围查询功能。
pub struct MeshSpatialIndex {
    /// R-Tree 索引
    tree: RTree<CellEnvelope>,
    /// 单元顶点（用于精确点位测试）
    cell_vertices: Vec<Vec<Point2D>>,
    /// 单元数量
    n_cells: usize,
}

impl MeshSpatialIndex {
    /// 从网格数据构建空间索引
    ///
    /// # 参数
    /// - `n_cells`: 单元总数
    /// - `get_cell_vertices`: 获取单元顶点的闭包，输入单元索引，返回顶点列表
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let index = MeshSpatialIndex::build(mesh.n_cells(), |i| {
    ///     mesh.get_cell_vertices(i)
    /// });
    /// ```
    pub fn build<F>(n_cells: usize, get_cell_vertices: F) -> Self
    where
        F: Fn(usize) -> Vec<Point2D>,
    {
        let mut envelopes = Vec::with_capacity(n_cells);
        let mut cell_vertices = Vec::with_capacity(n_cells);

        for i in 0..n_cells {
            let vertices = get_cell_vertices(i);
            if !vertices.is_empty() {
                envelopes.push(CellEnvelope::new(i, &vertices));
            }
            cell_vertices.push(vertices);
        }

        Self {
            tree: RTree::bulk_load(envelopes),
            cell_vertices,
            n_cells,
        }
    }

    /// 从已有的包围盒和顶点数据构建索引
    pub fn from_data(envelopes: Vec<CellEnvelope>, cell_vertices: Vec<Vec<Point2D>>) -> Self {
        let n_cells = cell_vertices.len();
        Self {
            tree: RTree::bulk_load(envelopes),
            cell_vertices,
            n_cells,
        }
    }

    // ============================================================
    // 序列化/反序列化方法
    // ============================================================

    /// 转换为可序列化数据结构
    ///
    /// 将空间索引转换为可序列化的格式，便于保存到文件。
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let data = index.to_serializable();
    /// let json = serde_json::to_string(&data).unwrap();
    /// ```
    pub fn to_serializable(&self) -> SpatialIndexData {
        // 收集所有包围盒
        let envelopes: Vec<CellEnvelope> = self.tree.iter().cloned().collect();
        
        // 计算全局边界
        let bounds = if !envelopes.is_empty() {
            let mut min_x = f64::MAX;
            let mut min_y = f64::MAX;
            let mut max_x = f64::MIN;
            let mut max_y = f64::MIN;
            
            for env in &envelopes {
                min_x = min_x.min(env.min_x);
                min_y = min_y.min(env.min_y);
                max_x = max_x.max(env.max_x);
                max_y = max_y.max(env.max_y);
            }
            
            Some(SpatialBounds { min_x, min_y, max_x, max_y })
        } else {
            None
        };

        SpatialIndexData {
            version: SPATIAL_INDEX_VERSION,
            envelopes,
            cell_vertices: self.cell_vertices.clone(),
            n_cells: self.n_cells,
            bounds,
        }
    }

    /// 从可序列化数据恢复空间索引
    ///
    /// # 参数
    /// - `data`: 序列化数据结构
    ///
    /// # 示例
    ///
    /// ```ignore
    /// let json = std::fs::read_to_string("index.json").unwrap();
    /// let data: SpatialIndexData = serde_json::from_str(&json).unwrap();
    /// let index = MeshSpatialIndex::from_serializable(data);
    /// ```
    pub fn from_serializable(data: SpatialIndexData) -> Self {
        Self {
            tree: RTree::bulk_load(data.envelopes),
            cell_vertices: data.cell_vertices,
            n_cells: data.n_cells,
        }
    }

    /// 获取空间边界
    ///
    /// 返回索引中所有单元的总边界框。
    pub fn bounds(&self) -> Option<SpatialBounds> {
        let mut iter = self.tree.iter();
        let first = iter.next()?;
        
        let mut min_x = first.min_x;
        let mut min_y = first.min_y;
        let mut max_x = first.max_x;
        let mut max_y = first.max_y;
        
        for env in iter {
            min_x = min_x.min(env.min_x);
            min_y = min_y.min(env.min_y);
            max_x = max_x.max(env.max_x);
            max_y = max_y.max(env.max_y);
        }
        
        Some(SpatialBounds { min_x, min_y, max_x, max_y })
    }

    /// 估算序列化后的数据大小（字节）
    ///
    /// 用于预分配缓冲区或显示进度信息。
    pub fn estimated_serialized_size(&self) -> usize {
        // 估算：每个包围盒约 48 字节 (5 * f64 + usize)
        // 每个顶点约 16 字节 (2 * f64)
        let envelope_size = self.n_cells * 48;
        let vertex_size: usize = self.cell_vertices.iter()
            .map(|v| v.len() * 16 + 8)  // +8 for Vec metadata
            .sum();
        envelope_size + vertex_size + 100  // +100 for header overhead
    }

    /// 获取单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 查找包含指定点的单元
    ///
    /// 首先使用 R-Tree 快速筛选候选单元，然后进行精确的点位测试。
    ///
    /// # 参数
    /// - `x`: 点的 x 坐标
    /// - `y`: 点的 y 坐标
    ///
    /// # 返回
    /// 如果点在某个单元内，返回 Some(单元索引)；否则返回 None
    pub fn locate_point(&self, x: f64, y: f64) -> Option<usize> {
        // 首先用 R-Tree 快速筛选候选单元
        let candidates = self.tree.locate_all_at_point(&[x, y]);

        // 然后对候选单元进行精确的点位测试
        for envelope in candidates {
            let cell_idx = envelope.cell_index;
            if self.point_in_polygon(x, y, &self.cell_vertices[cell_idx]) {
                return Some(cell_idx);
            }
        }

        None
    }

    /// 查找距离指定点最近的单元
    ///
    /// 基于单元包围盒中心点的距离查找最近单元。
    /// 注意：返回的是包围盒最近的单元，不一定是几何中心最近的。
    ///
    /// # 参数
    /// - `x`: 点的 x 坐标
    /// - `y`: 点的 y 坐标
    ///
    /// # 返回
    /// 最近单元的索引，如果索引为空则返回 None
    pub fn locate_nearest(&self, x: f64, y: f64) -> Option<usize> {
        self.tree
            .nearest_neighbor(&[x, y])
            .map(|env| env.cell_index)
    }

    /// 查找 K 个最近的单元
    ///
    /// # 参数
    /// - `x`: 点的 x 坐标
    /// - `y`: 点的 y 坐标
    /// - `k`: 返回的单元数量
    ///
    /// # 返回
    /// 最近的 k 个单元索引列表
    pub fn locate_k_nearest(&self, x: f64, y: f64, k: usize) -> Vec<usize> {
        self.tree
            .nearest_neighbor_iter(&[x, y])
            .take(k)
            .map(|env| env.cell_index)
            .collect()
    }

    /// 查找与指定矩形范围相交的所有单元
    ///
    /// # 参数
    /// - `min_x`: 矩形左边界
    /// - `min_y`: 矩形下边界
    /// - `max_x`: 矩形右边界
    /// - `max_y`: 矩形上边界
    ///
    /// # 返回
    /// 与矩形相交的单元索引列表
    pub fn locate_in_rect(&self, min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<usize> {
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);
        self.tree
            .locate_in_envelope(&envelope)
            .map(|env| env.cell_index)
            .collect()
    }

    /// 查找与指定矩形完全包含的单元
    pub fn locate_contained_in_rect(
        &self,
        min_x: f64,
        min_y: f64,
        max_x: f64,
        max_y: f64,
    ) -> Vec<usize> {
        let envelope = AABB::from_corners([min_x, min_y], [max_x, max_y]);
        self.tree
            .locate_in_envelope(&envelope)
            .filter(|env| {
                env.min_x >= min_x && env.max_x <= max_x && env.min_y >= min_y && env.max_y <= max_y
            })
            .map(|env| env.cell_index)
            .collect()
    }

    /// 查找指定圆形范围内的单元
    ///
    /// # 参数
    /// - `center_x`: 圆心 x 坐标
    /// - `center_y`: 圆心 y 坐标
    /// - `radius`: 搜索半径
    ///
    /// # 返回
    /// 与圆形相交的单元索引列表
    pub fn locate_in_circle(&self, center_x: f64, center_y: f64, radius: f64) -> Vec<usize> {
        // 首先用包围矩形快速筛选
        let candidates = self.locate_in_rect(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        );

        // 然后精确过滤
        let r2 = radius * radius;
        candidates
            .into_iter()
            .filter(|&idx| {
                // 检查单元是否与圆相交（简化：检查包围盒中心是否在圆内）
                if let Some(env) = self.tree.iter().find(|e| e.cell_index == idx) {
                    let center = env.center();
                    let dx = center.x - center_x;
                    let dy = center.y - center_y;
                    dx * dx + dy * dy <= r2
                } else {
                    false
                }
            })
            .collect()
    }

    /// 射线法判断点是否在多边形内
    ///
    /// 使用经典的射线交叉法（Ray Casting Algorithm）判断点是否在多边形内部。
    ///
    /// # 算法说明
    /// 从待测点向右发射水平射线，统计与多边形边的交点数：
    /// - 奇数个交点：点在多边形内
    /// - 偶数个交点：点在多边形外
    fn point_in_polygon(&self, x: f64, y: f64, vertices: &[Point2D]) -> bool {
        let n = vertices.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;
        let mut j = n - 1;

        for i in 0..n {
            let vi = &vertices[i];
            let vj = &vertices[j];

            // 检查射线与边的交点
            if ((vi.y > y) != (vj.y > y))
                && (x < (vj.x - vi.x) * (y - vi.y) / (vj.y - vi.y) + vi.x)
            {
                inside = !inside;
            }

            j = i;
        }

        inside
    }

    /// 检查点是否在任意单元内
    #[inline]
    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.locate_point(x, y).is_some()
    }

    /// 批量定位点
    ///
    /// 对多个点进行定位，返回每个点所在的单元索引
    pub fn locate_points_batch(&self, points: &[(f64, f64)]) -> Vec<Option<usize>> {
        points
            .iter()
            .map(|&(x, y)| self.locate_point(x, y))
            .collect()
    }

    /// 并行批量定位点
    #[cfg(feature = "parallel")]
    pub fn locate_points_parallel(&self, points: &[(f64, f64)]) -> Vec<Option<usize>> {
        use rayon::prelude::*;
        points
            .par_iter()
            .map(|&(x, y)| self.locate_point(x, y))
            .collect()
    }

    /// 获取单元顶点
    pub fn cell_vertices(&self, cell: usize) -> &[Point2D] {
        &self.cell_vertices[cell]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_index() -> MeshSpatialIndex {
        // 创建 2x2 的四个正方形单元网格
        // (0,2)---(1,2)---(2,2)
        //   |  2   |   3   |
        // (0,1)---(1,1)---(2,1)
        //   |  0   |   1   |
        // (0,0)---(1,0)---(2,0)
        MeshSpatialIndex::build(4, |i| match i {
            0 => vec![
                Point2D::new(0.0, 0.0),
                Point2D::new(1.0, 0.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(0.0, 1.0),
            ],
            1 => vec![
                Point2D::new(1.0, 0.0),
                Point2D::new(2.0, 0.0),
                Point2D::new(2.0, 1.0),
                Point2D::new(1.0, 1.0),
            ],
            2 => vec![
                Point2D::new(0.0, 1.0),
                Point2D::new(1.0, 1.0),
                Point2D::new(1.0, 2.0),
                Point2D::new(0.0, 2.0),
            ],
            3 => vec![
                Point2D::new(1.0, 1.0),
                Point2D::new(2.0, 1.0),
                Point2D::new(2.0, 2.0),
                Point2D::new(1.0, 2.0),
            ],
            _ => vec![],
        })
    }

    #[test]
    fn test_cell_envelope() {
        let vertices = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];
        let env = CellEnvelope::new(0, &vertices);

        assert_eq!(env.cell_index, 0);
        assert_eq!(env.min_x, 0.0);
        assert_eq!(env.max_x, 1.0);
        assert_eq!(env.min_y, 0.0);
        assert_eq!(env.max_y, 1.0);

        assert!(env.contains_point(0.5, 0.5));
        assert!(!env.contains_point(2.0, 0.5));
    }

    #[test]
    fn test_cell_envelope_dimensions() {
        let env = CellEnvelope::from_bounds(0, 0.0, 0.0, 2.0, 3.0);

        assert_eq!(env.width(), 2.0);
        assert_eq!(env.height(), 3.0);
        assert_eq!(env.area(), 6.0);

        let center = env.center();
        assert_eq!(center.x, 1.0);
        assert_eq!(center.y, 1.5);
    }

    #[test]
    fn test_locate_point() {
        let index = create_test_index();

        assert_eq!(index.locate_point(0.5, 0.5), Some(0));
        assert_eq!(index.locate_point(1.5, 0.5), Some(1));
        assert_eq!(index.locate_point(0.5, 1.5), Some(2));
        assert_eq!(index.locate_point(1.5, 1.5), Some(3));
        assert_eq!(index.locate_point(3.0, 3.0), None);
    }

    #[test]
    fn test_locate_nearest() {
        let index = create_test_index();

        // 在单元内的点
        assert!(index.locate_nearest(0.5, 0.5).is_some());

        // 在网格外的点应该返回最近的单元
        assert!(index.locate_nearest(10.0, 10.0).is_some());
    }

    #[test]
    fn test_locate_k_nearest() {
        let index = create_test_index();

        let nearest = index.locate_k_nearest(1.0, 1.0, 4);
        assert_eq!(nearest.len(), 4);
    }

    #[test]
    fn test_locate_in_rect() {
        let index = create_test_index();

        // 覆盖所有单元的矩形
        let cells = index.locate_in_rect(0.0, 0.0, 2.0, 2.0);
        assert_eq!(cells.len(), 4);

        // 只覆盖左下角单元的矩形
        let cells = index.locate_in_rect(0.0, 0.0, 0.5, 0.5);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0], 0);
    }

    #[test]
    fn test_contains() {
        let index = create_test_index();

        assert!(index.contains(0.5, 0.5));
        assert!(index.contains(1.5, 1.5));
        assert!(!index.contains(3.0, 3.0));
    }

    #[test]
    fn test_batch_locate() {
        let index = create_test_index();

        let points = vec![(0.5, 0.5), (1.5, 0.5), (3.0, 3.0)];
        let results = index.locate_points_batch(&points);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(0));
        assert_eq!(results[1], Some(1));
        assert_eq!(results[2], None);
    }

    #[test]
    fn test_point_in_polygon() {
        let index = create_test_index();

        // 三角形
        let triangle = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(1.0, 2.0),
        ];

        assert!(index.point_in_polygon(1.0, 0.5, &triangle));
        assert!(!index.point_in_polygon(0.0, 2.0, &triangle));
    }

    #[test]
    fn test_locate_in_circle() {
        let index = create_test_index();

        // 以 (1, 1) 为圆心，半径 0.6 的圆应该覆盖所有 4 个单元中心
        let cells = index.locate_in_circle(1.0, 1.0, 0.6);
        assert!(cells.len() >= 1); // 至少应该有一些单元
    }
}
