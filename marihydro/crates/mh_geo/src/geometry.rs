// marihydro\crates\mh_geo\src/geometry.rs

//! 几何类型定义
//!
//! 提供项目统一的几何类型，包括2D和3D点。

use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul, Neg, Sub};

// ============================================================================
// Point3D - 3D点（项目统一几何类型）
// ============================================================================

/// 3D点 - 项目统一几何类型
///
/// 用于存储顶点位置、法向量等3D几何数据。
///
/// # 示例
///
/// ```
/// use mh_geo::geometry::Point3D;
///
/// let p1 = Point3D::new(1.0, 2.0, 3.0);
/// let p2 = Point3D::new(4.0, 5.0, 6.0);
///
/// // 向量运算
/// let sum = p1 + p2;
/// let diff = p2 - p1;
/// let dot = p1.dot(&p2);
/// let cross = p1.cross(&p2);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point3D {
    /// X坐标
    pub x: f64,
    /// Y坐标
    pub y: f64,
    /// Z坐标
    pub z: f64,
}

impl Point3D {
    /// 零点常量
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// 单位X向量
    pub const UNIT_X: Self = Self {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };

    /// 单位Y向量
    pub const UNIT_Y: Self = Self {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };

    /// 单位Z向量
    pub const UNIT_Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// 创建新的3D点
    #[inline]
    #[must_use]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// 从2D点创建，指定Z坐标
    #[inline]
    #[must_use]
    pub const fn from_xy_z(xy: Point2D, z: f64) -> Self {
        Self {
            x: xy.x,
            y: xy.y,
            z,
        }
    }

    /// 投影到XY平面（忽略Z坐标）
    #[inline]
    #[must_use]
    pub const fn xy(&self) -> Point2D {
        Point2D {
            x: self.x,
            y: self.y,
        }
    }

    /// 设置Z坐标，返回新点
    #[inline]
    #[must_use]
    pub const fn with_z(self, z: f64) -> Self {
        Self {
            x: self.x,
            y: self.y,
            z,
        }
    }

    /// 点积（内积）
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// 叉积（外积）
    #[inline]
    #[must_use]
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// 向量长度（模）
    #[inline]
    #[must_use]
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// 向量长度的平方
    #[inline]
    #[must_use]
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// 归一化向量
    ///
    /// 如果向量长度接近零，返回 None
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len < 1e-14 {
            None
        } else {
            Some(Self {
                x: self.x / len,
                y: self.y / len,
                z: self.z / len,
            })
        }
    }

    /// 强制归一化向量
    ///
    /// 如果向量长度接近零，返回零向量
    #[inline]
    #[must_use]
    pub fn normalize_or_zero(&self) -> Self {
        self.normalize().unwrap_or(Self::ZERO)
    }

    /// 计算到另一个点的距离
    #[inline]
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        (*self - *other).length()
    }

    /// 计算到另一个点的距离的平方
    #[inline]
    #[must_use]
    pub fn distance_squared(&self, other: &Self) -> f64 {
        (*self - *other).length_squared()
    }

    /// 线性插值
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
            z: self.z + (other.z - self.z) * t,
        }
    }

    /// 标量乘法
    #[inline]
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
            z: self.z * factor,
        }
    }

    /// 判断是否为有限数（非NaN、非Inf）
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// 判断是否为零向量
    #[inline]
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.z == 0.0
    }

    /// 分量最小值
    #[inline]
    #[must_use]
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// 分量最大值
    #[inline]
    #[must_use]
    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }
}

// ============================================================================
// Point3D 运算符实现
// ============================================================================

impl Add for Point3D {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl Sub for Point3D {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Neg for Point3D {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<f64> for Point3D {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self {
        self.scale(scalar)
    }
}

impl Mul<Point3D> for f64 {
    type Output = Point3D;

    #[inline]
    fn mul(self, point: Point3D) -> Point3D {
        point.scale(self)
    }
}

// ============================================================================
// Point3D 转换实现
// ============================================================================

impl From<[f64; 3]> for Point3D {
    fn from([x, y, z]: [f64; 3]) -> Self {
        Self::new(x, y, z)
    }
}

impl From<Point3D> for [f64; 3] {
    fn from(p: Point3D) -> Self {
        [p.x, p.y, p.z]
    }
}

impl From<(f64, f64, f64)> for Point3D {
    fn from((x, y, z): (f64, f64, f64)) -> Self {
        Self::new(x, y, z)
    }
}

impl From<Point3D> for (f64, f64, f64) {
    fn from(p: Point3D) -> Self {
        (p.x, p.y, p.z)
    }
}

// ============================================================================
// Point2D - 2D点（仅用于明确的平面计算）
// ============================================================================

/// 2D点 - 仅用于明确的平面计算
///
/// 用于空间索引、平面投影等2D场景。
///
/// # 示例
///
/// ```
/// use mh_geo::geometry::Point2D;
///
/// let p = Point2D::new(1.0, 2.0);
/// let dist = p.distance_to(&Point2D::new(4.0, 6.0));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    /// X坐标
    pub x: f64,
    /// Y坐标
    pub y: f64,
}

impl Point2D {
    /// 零点常量
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// 单位X向量
    pub const UNIT_X: Self = Self { x: 1.0, y: 0.0 };

    /// 单位Y向量
    pub const UNIT_Y: Self = Self { x: 0.0, y: 1.0 };

    /// 创建新的2D点
    #[inline]
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// 扩展为3D点，指定Z坐标
    #[inline]
    #[must_use]
    pub const fn with_z(self, z: f64) -> Point3D {
        Point3D::new(self.x, self.y, z)
    }

    /// 计算到另一个点的距离
    #[inline]
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// 计算到另一个点的距离的平方
    #[inline]
    #[must_use]
    pub fn distance_squared_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// 点积
    #[inline]
    #[must_use]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 叉积（返回标量，即Z分量）
    #[inline]
    #[must_use]
    pub fn cross(&self, other: &Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// 向量长度
    #[inline]
    #[must_use]
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// 向量长度的平方
    #[inline]
    #[must_use]
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// 归一化向量
    #[inline]
    #[must_use]
    pub fn normalize(&self) -> Option<Self> {
        let len = self.length();
        if len < 1e-14 {
            None
        } else {
            Some(Self {
                x: self.x / len,
                y: self.y / len,
            })
        }
    }

    /// 强制归一化向量
    #[inline]
    #[must_use]
    pub fn normalize_or_zero(&self) -> Self {
        self.normalize().unwrap_or(Self::ZERO)
    }

    /// 线性插值
    #[inline]
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: self.x + (other.x - self.x) * t,
            y: self.y + (other.y - self.y) * t,
        }
    }

    /// 标量乘法
    #[inline]
    #[must_use]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            x: self.x * factor,
            y: self.y * factor,
        }
    }

    /// 判断是否为有限数
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// 旋转90度（逆时针）
    #[inline]
    #[must_use]
    pub fn perpendicular(&self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }
}

// ============================================================================
// Point2D 运算符实现
// ============================================================================

impl Add for Point2D {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Point2D {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Neg for Point2D {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl Mul<f64> for Point2D {
    type Output = Self;

    #[inline]
    fn mul(self, scalar: f64) -> Self {
        self.scale(scalar)
    }
}

impl Mul<Point2D> for f64 {
    type Output = Point2D;

    #[inline]
    fn mul(self, point: Point2D) -> Point2D {
        point.scale(self)
    }
}

// ============================================================================
// Point2D 转换实现
// ============================================================================

impl From<[f64; 2]> for Point2D {
    fn from([x, y]: [f64; 2]) -> Self {
        Self::new(x, y)
    }
}

impl From<Point2D> for [f64; 2] {
    fn from(p: Point2D) -> Self {
        [p.x, p.y]
    }
}

impl From<(f64, f64)> for Point2D {
    fn from((x, y): (f64, f64)) -> Self {
        Self::new(x, y)
    }
}

impl From<Point2D> for (f64, f64) {
    fn from(p: Point2D) -> Self {
        (p.x, p.y)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point3d_new() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
        assert_eq!(p.z, 3.0);
    }

    #[test]
    fn test_point3d_dot() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(4.0, 5.0, 6.0);
        assert!((p1.dot(&p2) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_cross() {
        let i = Point3D::UNIT_X;
        let j = Point3D::UNIT_Y;
        let k = i.cross(&j);
        assert!((k.x - 0.0).abs() < 1e-10);
        assert!((k.y - 0.0).abs() < 1e-10);
        assert!((k.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_length() {
        let p = Point3D::new(3.0, 4.0, 0.0);
        assert!((p.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_normalize() {
        let p = Point3D::new(3.0, 4.0, 0.0);
        let n = p.normalize().unwrap();
        assert!((n.length() - 1.0).abs() < 1e-10);
        assert!((n.x - 0.6).abs() < 1e-10);
        assert!((n.y - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_normalize_zero() {
        let p = Point3D::ZERO;
        assert!(p.normalize().is_none());
        assert!(p.normalize_or_zero().is_zero());
    }

    #[test]
    fn test_point3d_distance() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(3.0, 4.0, 0.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_xy_projection() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let xy = p.xy();
        assert_eq!(xy.x, 1.0);
        assert_eq!(xy.y, 2.0);
    }

    #[test]
    fn test_point3d_arithmetic() {
        let p1 = Point3D::new(1.0, 2.0, 3.0);
        let p2 = Point3D::new(4.0, 5.0, 6.0);

        let sum = p1 + p2;
        assert_eq!(sum, Point3D::new(5.0, 7.0, 9.0));

        let diff = p2 - p1;
        assert_eq!(diff, Point3D::new(3.0, 3.0, 3.0));

        let neg = -p1;
        assert_eq!(neg, Point3D::new(-1.0, -2.0, -3.0));

        let scaled = p1 * 2.0;
        assert_eq!(scaled, Point3D::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_point3d_conversions() {
        let p = Point3D::new(1.0, 2.0, 3.0);
        let arr: [f64; 3] = p.into();
        assert_eq!(arr, [1.0, 2.0, 3.0]);

        let p2 = Point3D::from([4.0, 5.0, 6.0]);
        assert_eq!(p2, Point3D::new(4.0, 5.0, 6.0));

        let tup: (f64, f64, f64) = p.into();
        assert_eq!(tup, (1.0, 2.0, 3.0));
    }

    #[test]
    fn test_point2d_new() {
        let p = Point2D::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }

    #[test]
    fn test_point2d_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_point2d_with_z() {
        let p2d = Point2D::new(1.0, 2.0);
        let p3d = p2d.with_z(3.0);
        assert_eq!(p3d, Point3D::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_point2d_perpendicular() {
        let p = Point2D::new(1.0, 0.0);
        let perp = p.perpendicular();
        assert!((perp.x - 0.0).abs() < 1e-10);
        assert!((perp.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_lerp() {
        let p1 = Point3D::new(0.0, 0.0, 0.0);
        let p2 = Point3D::new(10.0, 10.0, 10.0);

        let mid = p1.lerp(&p2, 0.5);
        assert_eq!(mid, Point3D::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_point3d_min_max() {
        let p1 = Point3D::new(1.0, 5.0, 3.0);
        let p2 = Point3D::new(4.0, 2.0, 6.0);

        let min_p = p1.min(&p2);
        assert_eq!(min_p, Point3D::new(1.0, 2.0, 3.0));

        let max_p = p1.max(&p2);
        assert_eq!(max_p, Point3D::new(4.0, 5.0, 6.0));
    }
}
