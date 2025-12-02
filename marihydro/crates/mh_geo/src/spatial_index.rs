// marihydro\crates\mh_geo\src/spatial_index.rs
//! 空间索引实现
//!
//! 基于 R-tree 的空间索引，用于高效的空间查询
//!
//! # 示例
//!
//! ```
//! use mh_geo::spatial_index::{SpatialIndex, BoundingBox};
//! use mh_geo::geometry::Point2D;
//!
//! let mut index: SpatialIndex<u32> = SpatialIndex::new();
//!
//! // 插入点
//! index.insert(Point2D::new(10.0, 20.0), 1);
//! index.insert(Point2D::new(15.0, 25.0), 2);
//!
//! // 查询范围内的点
//! let bbox = BoundingBox::new(5.0, 15.0, 12.0, 22.0);
//! let results = index.query_range(&bbox);
//! ```

use crate::geometry::Point2D;
use rstar::{RTree, RTreeObject, AABB};

/// 边界框
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox {
    /// 最小 x
    pub min_x: f64,
    /// 最小 y
    pub min_y: f64,
    /// 最大 x
    pub max_x: f64,
    /// 最大 y
    pub max_y: f64,
}

impl BoundingBox {
    /// 创建新的边界框
    #[must_use]
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Self {
            min_x: min_x.min(max_x),
            min_y: min_y.min(max_y),
            max_x: min_x.max(max_x),
            max_y: min_y.max(max_y),
        }
    }

    /// 从两个角点创建
    #[must_use]
    pub fn from_corners(p1: Point2D, p2: Point2D) -> Self {
        Self::new(p1.x, p1.y, p2.x, p2.y)
    }

    /// 检查点是否在边界框内
    #[must_use]
    pub fn contains_point(&self, point: &Point2D) -> bool {
        point.x >= self.min_x
            && point.x <= self.max_x
            && point.y >= self.min_y
            && point.y <= self.max_y
    }

    /// 检查两个边界框是否相交
    #[must_use]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
    }

    /// 合并两个边界框
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// 计算宽度
    #[must_use]
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// 计算高度
    #[must_use]
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// 计算面积
    #[must_use]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// 计算中心点
    #[must_use]
    pub fn center(&self) -> Point2D {
        Point2D::new(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    /// 扩展边界框
    #[must_use]
    pub fn expand(&self, amount: f64) -> Self {
        Self {
            min_x: self.min_x - amount,
            min_y: self.min_y - amount,
            max_x: self.max_x + amount,
            max_y: self.max_y + amount,
        }
    }
}

// ============================================================================
// R-tree 包装
// ============================================================================

/// 空间索引条目
#[derive(Debug, Clone)]
struct SpatialEntry<T> {
    point: Point2D,
    data: T,
}

impl<T> RTreeObject for SpatialEntry<T> {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point([self.point.x, self.point.y])
    }
}

impl<T> rstar::PointDistance for SpatialEntry<T> {
    fn distance_2(&self, point: &[f64; 2]) -> f64 {
        let dx = self.point.x - point[0];
        let dy = self.point.y - point[1];
        dx * dx + dy * dy
    }
}

/// 空间索引
///
/// 基于 R-tree 的空间索引，用于高效的空间查询
pub struct SpatialIndex<T> {
    tree: RTree<SpatialEntry<T>>,
}

impl<T: Clone> Default for SpatialIndex<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone> SpatialIndex<T> {
    /// 创建空的空间索引
    #[must_use]
    pub fn new() -> Self {
        Self { tree: RTree::new() }
    }

    /// 从点集批量构建
    #[must_use]
    pub fn bulk_load(points: Vec<(Point2D, T)>) -> Self {
        let entries: Vec<SpatialEntry<T>> = points
            .into_iter()
            .map(|(point, data)| SpatialEntry { point, data })
            .collect();
        Self {
            tree: RTree::bulk_load(entries),
        }
    }

    /// 插入点
    pub fn insert(&mut self, point: Point2D, data: T) {
        self.tree.insert(SpatialEntry { point, data });
    }

    /// 查询范围内的点
    #[must_use]
    pub fn query_range(&self, bbox: &BoundingBox) -> Vec<(&Point2D, &T)> {
        let envelope = AABB::from_corners(
            [bbox.min_x, bbox.min_y],
            [bbox.max_x, bbox.max_y],
        );
        self.tree
            .locate_in_envelope(&envelope)
            .map(|entry| (&entry.point, &entry.data))
            .collect()
    }

    /// 查询最近的 k 个点
    #[must_use]
    pub fn query_nearest(&self, point: &Point2D, k: usize) -> Vec<(&Point2D, &T)> {
        self.tree
            .nearest_neighbor_iter(&[point.x, point.y])
            .take(k)
            .map(|entry| (&entry.point, &entry.data))
            .collect()
    }

    /// 查询指定距离内的点
    #[must_use]
    pub fn query_within_distance(&self, point: &Point2D, distance: f64) -> Vec<(&Point2D, &T)> {
        let dist_squared = distance * distance;
        self.tree
            .nearest_neighbor_iter(&[point.x, point.y])
            .take_while(|entry| {
                let dx = entry.point.x - point.x;
                let dy = entry.point.y - point.y;
                dx * dx + dy * dy <= dist_squared
            })
            .map(|entry| (&entry.point, &entry.data))
            .collect()
    }

    /// 返回索引中的点数量
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.size()
    }

    /// 检查索引是否为空
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.size() == 0
    }

    /// 迭代所有点
    pub fn iter(&self) -> impl Iterator<Item = (&Point2D, &T)> {
        self.tree.iter().map(|entry| (&entry.point, &entry.data))
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point2d() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        assert!(bbox.contains_point(&Point2D::new(5.0, 5.0)));
        assert!(!bbox.contains_point(&Point2D::new(15.0, 5.0)));

        assert!((bbox.width() - 10.0).abs() < 1e-10);
        assert!((bbox.height() - 10.0).abs() < 1e-10);
        assert!((bbox.area() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_intersects() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        let bbox3 = BoundingBox::new(20.0, 20.0, 30.0, 30.0);

        assert!(bbox1.intersects(&bbox2));
        assert!(!bbox1.intersects(&bbox3));
    }

    #[test]
    fn test_spatial_index_insert_query() {
        let mut index: SpatialIndex<u32> = SpatialIndex::new();

        index.insert(Point2D::new(10.0, 10.0), 1);
        index.insert(Point2D::new(20.0, 20.0), 2);
        index.insert(Point2D::new(30.0, 30.0), 3);

        assert_eq!(index.len(), 3);

        let bbox = BoundingBox::new(5.0, 5.0, 15.0, 15.0);
        let results = index.query_range(&bbox);
        assert_eq!(results.len(), 1);
        assert_eq!(*results[0].1, 1);
    }

    #[test]
    fn test_spatial_index_nearest() {
        let mut index: SpatialIndex<u32> = SpatialIndex::new();

        index.insert(Point2D::new(0.0, 0.0), 1);
        index.insert(Point2D::new(10.0, 10.0), 2);
        index.insert(Point2D::new(20.0, 20.0), 3);

        let results = index.query_nearest(&Point2D::new(5.0, 5.0), 2);
        assert_eq!(results.len(), 2);
        // 最近的应该是 (0,0) 和 (10,10)
    }

    #[test]
    fn test_spatial_index_within_distance() {
        let mut index: SpatialIndex<u32> = SpatialIndex::new();

        index.insert(Point2D::new(0.0, 0.0), 1);
        index.insert(Point2D::new(5.0, 0.0), 2);
        index.insert(Point2D::new(100.0, 0.0), 3);

        let results = index.query_within_distance(&Point2D::new(0.0, 0.0), 10.0);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_spatial_index_bulk_load() {
        let points = vec![
            (Point2D::new(0.0, 0.0), 1u32),
            (Point2D::new(10.0, 10.0), 2),
            (Point2D::new(20.0, 20.0), 3),
        ];

        let index = SpatialIndex::bulk_load(points);
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn test_bounding_box_merge() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::new(5.0, 5.0, 20.0, 20.0);
        let merged = bbox1.merge(&bbox2);

        assert!((merged.min_x - 0.0).abs() < 1e-10);
        assert!((merged.min_y - 0.0).abs() < 1e-10);
        assert!((merged.max_x - 20.0).abs() < 1e-10);
        assert!((merged.max_y - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_point_from() {
        let p1: Point2D = (1.0, 2.0).into();
        assert!((p1.x - 1.0).abs() < 1e-10);
        assert!((p1.y - 2.0).abs() < 1e-10);

        let p2: Point2D = [3.0, 4.0].into();
        assert!((p2.x - 3.0).abs() < 1e-10);
        assert!((p2.y - 4.0).abs() < 1e-10);
    }
}
