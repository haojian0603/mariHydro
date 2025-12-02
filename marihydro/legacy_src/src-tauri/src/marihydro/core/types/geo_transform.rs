// src-tauri/src/marihydro/core/types/geo_transform.rs

//! 地理变换类型定义
//!
//! 提供坐标变换相关的数据类型，不依赖具体的投影库实现。

use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::fmt;

/// 仿射变换参数（6参数）
///
/// 用于栅格数据的坐标变换：
/// ```text
/// x_geo = a + b * col + c * row
/// y_geo = d + e * col + f * row
/// ```
///
/// 通常 c = e = 0（正交栅格）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GeoTransform {
    /// 左上角 x 坐标
    pub origin_x: f64,
    /// 列方向像素宽度（通常为正）
    pub pixel_width: f64,
    /// 行旋转（通常为 0）
    pub row_rotation: f64,
    /// 左上角 y 坐标
    pub origin_y: f64,
    /// 列旋转（通常为 0）
    pub col_rotation: f64,
    /// 行方向像素高度（通常为负，因为 y 向下增加）
    pub pixel_height: f64,
}

impl Default for GeoTransform {
    fn default() -> Self {
        Self {
            origin_x: 0.0,
            pixel_width: 1.0,
            row_rotation: 0.0,
            origin_y: 0.0,
            col_rotation: 0.0,
            pixel_height: -1.0, // y 向下递减
        }
    }
}

impl GeoTransform {
    /// 创建新的仿射变换
    pub fn new(
        origin_x: f64,
        pixel_width: f64,
        row_rotation: f64,
        origin_y: f64,
        col_rotation: f64,
        pixel_height: f64,
    ) -> Self {
        Self {
            origin_x,
            pixel_width,
            row_rotation,
            origin_y,
            col_rotation,
            pixel_height,
        }
    }

    /// 从边界框和尺寸创建
    ///
    /// # 参数
    ///
    /// - `min_x`, `max_x`, `min_y`, `max_y`: 边界框
    /// - `cols`, `rows`: 栅格尺寸
    pub fn from_bounds(
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        cols: usize,
        rows: usize,
    ) -> Self {
        let pixel_width = (max_x - min_x) / cols as f64;
        let pixel_height = -(max_y - min_y) / rows as f64; // 负数

        Self {
            origin_x: min_x,
            pixel_width,
            row_rotation: 0.0,
            origin_y: max_y, // 左上角
            col_rotation: 0.0,
            pixel_height,
        }
    }

    /// 从 GDAL 格式的数组创建
    ///
    /// GDAL 顺序: [origin_x, pixel_width, row_rotation, origin_y, col_rotation, pixel_height]
    pub fn from_gdal_array(arr: [f64; 6]) -> Self {
        Self {
            origin_x: arr[0],
            pixel_width: arr[1],
            row_rotation: arr[2],
            origin_y: arr[3],
            col_rotation: arr[4],
            pixel_height: arr[5],
        }
    }

    /// 转换为 GDAL 格式的数组
    pub fn to_gdal_array(&self) -> [f64; 6] {
        [
            self.origin_x,
            self.pixel_width,
            self.row_rotation,
            self.origin_y,
            self.col_rotation,
            self.pixel_height,
        ]
    }

    /// 像素坐标到地理坐标
    ///
    /// # 参数
    ///
    /// - `col`: 列索引（可以是小数）
    /// - `row`: 行索引（可以是小数）
    #[inline]
    pub fn pixel_to_geo(&self, col: f64, row: f64) -> DVec2 {
        let x = self.origin_x + col * self.pixel_width + row * self.row_rotation;
        let y = self.origin_y + col * self.col_rotation + row * self.pixel_height;
        DVec2::new(x, y)
    }

    /// 像素中心到地理坐标
    #[inline]
    pub fn pixel_center_to_geo(&self, col: usize, row: usize) -> DVec2 {
        self.pixel_to_geo(col as f64 + 0.5, row as f64 + 0.5)
    }

    /// 地理坐标到像素坐标（反向变换）
    ///
    /// 如果变换矩阵是奇异的（行列式为零），返回 None
    #[inline]
    pub fn geo_to_pixel(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        // 计算 2x2 矩阵的行列式
        let det = self.pixel_width * self.pixel_height - self.row_rotation * self.col_rotation;

        if det.abs() < 1e-14 {
            return None;
        }

        let dx = x - self.origin_x;
        let dy = y - self.origin_y;

        // 使用克拉默法则求解
        let col = (self.pixel_height * dx - self.row_rotation * dy) / det;
        let row = (-self.col_rotation * dx + self.pixel_width * dy) / det;

        Some((col, row))
    }

    /// 获取像素中心对应的行列索引
    pub fn geo_to_pixel_index(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let (col, row) = self.geo_to_pixel(x, y)?;

        if col < 0.0 || row < 0.0 {
            return None;
        }

        Some((col.floor() as usize, row.floor() as usize))
    }

    /// 检查变换是否为正交（无旋转）
    #[inline]
    pub fn is_orthogonal(&self) -> bool {
        self.row_rotation.abs() < 1e-10 && self.col_rotation.abs() < 1e-10
    }

    /// 检查是否为北上（pixel_height < 0）
    #[inline]
    pub fn is_north_up(&self) -> bool {
        self.pixel_height < 0.0
    }

    /// 获取像素宽度（绝对值）
    #[inline]
    pub fn abs_pixel_width(&self) -> f64 {
        self.pixel_width.abs()
    }

    /// 获取像素高度（绝对值）
    #[inline]
    pub fn abs_pixel_height(&self) -> f64 {
        self.pixel_height.abs()
    }

    /// 获取像素面积
    #[inline]
    pub fn pixel_area(&self) -> f64 {
        (self.pixel_width * self.pixel_height).abs()
    }

    /// 获取边界框
    ///
    /// # 返回
    ///
    /// (min_x, max_x, min_y, max_y)
    pub fn bounds(&self, cols: usize, rows: usize) -> (f64, f64, f64, f64) {
        let corners = [
            self.pixel_to_geo(0.0, 0.0),
            self.pixel_to_geo(cols as f64, 0.0),
            self.pixel_to_geo(0.0, rows as f64),
            self.pixel_to_geo(cols as f64, rows as f64),
        ];

        let min_x = corners.iter().map(|c| c.x).fold(f64::INFINITY, f64::min);
        let max_x = corners
            .iter()
            .map(|c| c.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let min_y = corners.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
        let max_y = corners
            .iter()
            .map(|c| c.y)
            .fold(f64::NEG_INFINITY, f64::max);

        (min_x, max_x, min_y, max_y)
    }

    /// 缩放变换（用于重采样）
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            origin_x: self.origin_x,
            pixel_width: self.pixel_width * factor,
            row_rotation: self.row_rotation * factor,
            origin_y: self.origin_y,
            col_rotation: self.col_rotation * factor,
            pixel_height: self.pixel_height * factor,
        }
    }

    /// 平移变换
    pub fn translate(&self, dx: f64, dy: f64) -> Self {
        Self {
            origin_x: self.origin_x + dx,
            origin_y: self.origin_y + dy,
            ..*self
        }
    }
}

impl fmt::Display for GeoTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GeoTransform(origin=({:.2}, {:.2}), pixel=({:.4}, {:.4}))",
            self.origin_x, self.origin_y, self.pixel_width, self.pixel_height
        )
    }
}

/// 边界框
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl BoundingBox {
    /// 创建新的边界框
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Self {
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
        }
    }

    /// 从点集计算边界框
    pub fn from_points(points: &[DVec2]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;

        for p in points {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }

        Some(Self {
            min_x,
            max_x,
            min_y,
            max_y,
        })
    }

    /// 宽度
    #[inline]
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// 高度
    #[inline]
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// 面积
    #[inline]
    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// 中心点
    #[inline]
    pub fn center(&self) -> DVec2 {
        DVec2::new(
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }

    /// 检查点是否在边界框内
    #[inline]
    pub fn contains(&self, point: DVec2) -> bool {
        point.x >= self.min_x
            && point.x <= self.max_x
            && point.y >= self.min_y
            && point.y <= self.max_y
    }

    /// 检查两个边界框是否相交
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        !(self.max_x < other.min_x
            || self.min_x > other.max_x
            || self.max_y < other.min_y
            || self.min_y > other.max_y)
    }

    /// 计算两个边界框的交集
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if !self.intersects(other) {
            return None;
        }

        Some(Self {
            min_x: self.min_x.max(other.min_x),
            max_x: self.max_x.min(other.max_x),
            min_y: self.min_y.max(other.min_y),
            max_y: self.max_y.min(other.max_y),
        })
    }

    /// 计算两个边界框的并集
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min_x: self.min_x.min(other.min_x),
            max_x: self.max_x.max(other.max_x),
            min_y: self.min_y.min(other.min_y),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// 扩展边界框
    pub fn expand(&self, margin: f64) -> Self {
        Self {
            min_x: self.min_x - margin,
            max_x: self.max_x + margin,
            min_y: self.min_y - margin,
            max_y: self.max_y + margin,
        }
    }

    /// 转换为四个角点
    pub fn corners(&self) -> [DVec2; 4] {
        [
            DVec2::new(self.min_x, self.min_y),
            DVec2::new(self.max_x, self.min_y),
            DVec2::new(self.max_x, self.max_y),
            DVec2::new(self.min_x, self.max_y),
        ]
    }
}

impl fmt::Display for BoundingBox {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BBox([{:.2}, {:.2}] x [{:.2}, {:.2}])",
            self.min_x, self.max_x, self.min_y, self.max_y
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geo_transform_pixel_to_geo() {
        let gt = GeoTransform::new(100.0, 10.0, 0.0, 200.0, 0.0, -10.0);

        let p = gt.pixel_to_geo(0.0, 0.0);
        assert!((p.x - 100.0).abs() < 1e-10);
        assert!((p.y - 200.0).abs() < 1e-10);

        let p = gt.pixel_to_geo(1.0, 1.0);
        assert!((p.x - 110.0).abs() < 1e-10);
        assert!((p.y - 190.0).abs() < 1e-10);
    }

    #[test]
    fn test_geo_transform_roundtrip() {
        let gt = GeoTransform::new(100.0, 10.0, 0.0, 200.0, 0.0, -10.0);

        let (col, row) = (5.5, 3.5);
        let geo = gt.pixel_to_geo(col, row);
        let (col2, row2) = gt.geo_to_pixel(geo.x, geo.y).unwrap();

        assert!((col - col2).abs() < 1e-10);
        assert!((row - row2).abs() < 1e-10);
    }

    #[test]
    fn test_geo_transform_from_bounds() {
        let gt = GeoTransform::from_bounds(0.0, 100.0, 0.0, 50.0, 10, 5);

        // 左上角
        let p = gt.pixel_to_geo(0.0, 0.0);
        assert!((p.x - 0.0).abs() < 1e-10);
        assert!((p.y - 50.0).abs() < 1e-10);

        // 右下角
        let p = gt.pixel_to_geo(10.0, 5.0);
        assert!((p.x - 100.0).abs() < 1e-10);
        assert!((p.y - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_geo_transform_is_orthogonal() {
        let orthogonal = GeoTransform::new(0.0, 10.0, 0.0, 0.0, 0.0, -10.0);
        assert!(orthogonal.is_orthogonal());

        let rotated = GeoTransform::new(0.0, 10.0, 1.0, 0.0, 1.0, -10.0);
        assert!(!rotated.is_orthogonal());
    }

    #[test]
    fn test_bounding_box() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 5.0);

        assert!((bbox.width() - 10.0).abs() < 1e-10);
        assert!((bbox.height() - 5.0).abs() < 1e-10);
        assert!((bbox.area() - 50.0).abs() < 1e-10);

        let center = bbox.center();
        assert!((center.x - 5.0).abs() < 1e-10);
        assert!((center.y - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox::new(0.0, 10.0, 0.0, 10.0);

        assert!(bbox.contains(DVec2::new(5.0, 5.0)));
        assert!(bbox.contains(DVec2::new(0.0, 0.0)));
        assert!(!bbox.contains(DVec2::new(-1.0, 5.0)));
        assert!(!bbox.contains(DVec2::new(11.0, 5.0)));
    }

    #[test]
    fn test_bounding_box_intersection() {
        let a = BoundingBox::new(0.0, 10.0, 0.0, 10.0);
        let b = BoundingBox::new(5.0, 15.0, 5.0, 15.0);

        let inter = a.intersection(&b).unwrap();
        assert!((inter.min_x - 5.0).abs() < 1e-10);
        assert!((inter.max_x - 10.0).abs() < 1e-10);
        assert!((inter.min_y - 5.0).abs() < 1e-10);
        assert!((inter.max_y - 10.0).abs() < 1e-10);

        let c = BoundingBox::new(20.0, 30.0, 0.0, 10.0);
        assert!(a.intersection(&c).is_none());
    }

    #[test]
    fn test_bounding_box_from_points() {
        let points = vec![
            DVec2::new(1.0, 2.0),
            DVec2::new(5.0, 8.0),
            DVec2::new(3.0, 1.0),
        ];

        let bbox = BoundingBox::from_points(&points).unwrap();
        assert!((bbox.min_x - 1.0).abs() < 1e-10);
        assert!((bbox.max_x - 5.0).abs() < 1e-10);
        assert!((bbox.min_y - 1.0).abs() < 1e-10);
        assert!((bbox.max_y - 8.0).abs() < 1e-10);
    }
}
