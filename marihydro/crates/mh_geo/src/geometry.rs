//! 几何类型定义
//!
//! 提供项目统一的几何类型，包括2D和3D点，以及地理距离计算。
//!
//! # 距离计算
//!
//! - `distance_to`: 欧几里得距离（适用于投影坐标）
//! - `geodesic_distance_to`: Haversine 公式（适用于经纬度）
//! - `vincenty_distance_to`: Vincenty 公式（高精度椭球面距离）

use crate::ellipsoid::Ellipsoid;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

// ============================================================================
// 地球物理常量
// ============================================================================

/// 地球平均半径 (米) - 用于 Haversine 公式
pub const EARTH_MEAN_RADIUS: f64 = 6_371_008.8;

/// 角度转弧度
#[inline]
fn deg_to_rad(deg: f64) -> f64 {
    deg * PI / 180.0
}

/// 弧度转角度
#[inline]
fn rad_to_deg(rad: f64) -> f64 {
    rad * 180.0 / PI
}

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

    /// 计算到另一个点的欧几里得距离
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

/// 2D点 - 用于平面几何和地理坐标
///
/// # 距离计算方法
///
/// - `distance_to`: 欧几里得距离，适用于投影坐标（米）
/// - `geodesic_distance_to`: Haversine 公式，适用于经纬度（度）
/// - `vincenty_distance_to`: Vincenty 公式，高精度椭球面距离
///
/// # 示例
///
/// ```
/// use mh_geo::geometry::Point2D;
///
/// // 投影坐标距离
/// let p1 = Point2D::new(500000.0, 4000000.0);
/// let p2 = Point2D::new(500100.0, 4000100.0);
/// let dist = p1.distance_to(&p2); // 约 141.4 米
///
/// // 经纬度距离（Haversine）
/// let beijing = Point2D::new(116.4, 39.9);
/// let shanghai = Point2D::new(121.5, 31.2);
/// let dist_km = beijing.geodesic_distance_to(&shanghai) / 1000.0; // 约 1068 km
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point2D {
    /// X坐标（或经度）
    pub x: f64,
    /// Y坐标（或纬度）
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

    /// 从经纬度创建（lon, lat）
    #[inline]
    #[must_use]
    pub const fn from_lonlat(lon: f64, lat: f64) -> Self {
        Self { x: lon, y: lat }
    }

    /// 获取经度（假设 x 为经度）
    #[inline]
    #[must_use]
    pub const fn lon(&self) -> f64 {
        self.x
    }

    /// 获取纬度（假设 y 为纬度）
    #[inline]
    #[must_use]
    pub const fn lat(&self) -> f64 {
        self.y
    }

    /// 扩展为3D点，指定Z坐标
    #[inline]
    #[must_use]
    pub const fn with_z(self, z: f64) -> Point3D {
        Point3D::new(self.x, self.y, z)
    }

    // ========================================================================
    // 欧几里得距离（投影坐标用）
    // ========================================================================

    /// 计算到另一个点的欧几里得距离
    ///
    /// 适用于投影坐标（如 UTM、高斯-克吕格），单位与坐标单位一致。
    /// **不要用于经纬度坐标！**
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

    // ========================================================================
    // 大地测量距离（经纬度用）
    // ========================================================================

    /// Haversine 公式计算大圆距离
    ///
    /// 将地球视为正球体，精度约 0.5%。适用于快速估算。
    ///
    /// # Arguments
    /// - `other`: 另一个点（经纬度，度）
    ///
    /// # Returns
    /// 大圆距离（米）
    ///
    /// # 注意
    /// 假设 self.x = 经度, self.y = 纬度
    #[must_use]
    pub fn geodesic_distance_to(&self, other: &Self) -> f64 {
        self.haversine_distance(other, EARTH_MEAN_RADIUS)
    }

    /// Haversine 公式（可自定义球体半径）
    #[must_use]
    pub fn haversine_distance(&self, other: &Self, radius: f64) -> f64 {
        let lat1 = deg_to_rad(self.y);
        let lat2 = deg_to_rad(other.y);
        let dlat = lat2 - lat1;
        let dlon = deg_to_rad(other.x - self.x);

        let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);

        let c = 2.0 * a.sqrt().asin();

        radius * c
    }

    /// Vincenty 公式计算椭球面距离
    ///
    /// 使用 WGS84 椭球体参数，精度可达毫米级。
    ///
    /// # Arguments
    /// - `other`: 另一个点（经纬度，度）
    ///
    /// # Returns
    /// 测地线距离（米），如果迭代不收敛返回 None
    #[must_use]
    pub fn vincenty_distance_to(&self, other: &Self) -> Option<f64> {
        self.vincenty_distance(other, &Ellipsoid::WGS84)
    }

    /// Vincenty 公式（可自定义椭球体）
    #[must_use]
    pub fn vincenty_distance(&self, other: &Self, ellipsoid: &Ellipsoid) -> Option<f64> {
        let a = ellipsoid.a;
        let f = ellipsoid.f;
        let b = ellipsoid.b();

        let phi1 = deg_to_rad(self.y);
        let phi2 = deg_to_rad(other.y);
        let l = deg_to_rad(other.x - self.x);

        // Reduced latitudes
        let u1 = ((1.0 - f) * phi1.tan()).atan();
        let u2 = ((1.0 - f) * phi2.tan()).atan();

        let sin_u1 = u1.sin();
        let cos_u1 = u1.cos();
        let sin_u2 = u2.sin();
        let cos_u2 = u2.cos();

        // 迭代求解 λ
        let mut lambda = l;
        let mut lambda_prev;
        let mut cos_sq_alpha = 0.0;
        let mut sin_sigma = 0.0;
        let mut cos_sigma = 0.0;
        let mut cos_2sigma_m = 0.0;
        let mut sigma = 0.0;

        const MAX_ITER: usize = 100;
        const TOLERANCE: f64 = 1e-12;

        for _ in 0..MAX_ITER {
            let sin_lambda = lambda.sin();
            let cos_lambda = lambda.cos();

            sin_sigma = ((cos_u2 * sin_lambda).powi(2)
                + (cos_u1 * sin_u2 - sin_u1 * cos_u2 * cos_lambda).powi(2))
            .sqrt();

            if sin_sigma < 1e-12 {
                // 两点重合
                return Some(0.0);
            }

            cos_sigma = sin_u1 * sin_u2 + cos_u1 * cos_u2 * cos_lambda;
            sigma = sin_sigma.atan2(cos_sigma);

            let sin_alpha = cos_u1 * cos_u2 * sin_lambda / sin_sigma;
            cos_sq_alpha = 1.0 - sin_alpha.powi(2);

            cos_2sigma_m = if cos_sq_alpha.abs() < 1e-12 {
                0.0
            } else {
                cos_sigma - 2.0 * sin_u1 * sin_u2 / cos_sq_alpha
            };

            let c = f / 16.0 * cos_sq_alpha * (4.0 + f * (4.0 - 3.0 * cos_sq_alpha));

            lambda_prev = lambda;
            lambda = l
                + (1.0 - c)
                    * f
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos_2sigma_m + c * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))));

            if (lambda - lambda_prev).abs() < TOLERANCE {
                break;
            }
        }

        // 计算距离
        let u_sq = cos_sq_alpha * (a * a - b * b) / (b * b);
        let aa = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
        let bb = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));

        let delta_sigma = bb
            * sin_sigma
            * (cos_2sigma_m
                + bb / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos_2sigma_m.powi(2))
                        - bb / 6.0
                            * cos_2sigma_m
                            * (-3.0 + 4.0 * sin_sigma.powi(2))
                            * (-3.0 + 4.0 * cos_2sigma_m.powi(2))));

        let s = b * aa * (sigma - delta_sigma);

        Some(s)
    }

    /// 计算初始方位角（从 self 到 other）
    ///
    /// 返回从正北顺时针的角度（度）
    #[must_use]
    pub fn initial_bearing_to(&self, other: &Self) -> f64 {
        let lat1 = deg_to_rad(self.y);
        let lat2 = deg_to_rad(other.y);
        let dlon = deg_to_rad(other.x - self.x);

        let x = lat2.cos() * dlon.sin();
        let y = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();

        let bearing = x.atan2(y);
        (rad_to_deg(bearing) + 360.0) % 360.0
    }

    /// 根据距离和方位角计算目标点
    ///
    /// # Arguments
    /// - `distance`: 距离（米）
    /// - `bearing`: 方位角（度，从正北顺时针）
    ///
    /// # Returns
    /// 目标点（经纬度，度）
    #[must_use]
    pub fn destination_point(&self, distance: f64, bearing: f64) -> Self {
        let lat1 = deg_to_rad(self.y);
        let lon1 = deg_to_rad(self.x);
        let bearing_rad = deg_to_rad(bearing);
        let delta = distance / EARTH_MEAN_RADIUS;

        let lat2 = (lat1.sin() * delta.cos() + lat1.cos() * delta.sin() * bearing_rad.cos()).asin();

        let lon2 = lon1
            + (bearing_rad.sin() * delta.sin() * lat1.cos())
                .atan2(delta.cos() - lat1.sin() * lat2.sin());

        Self::new(rad_to_deg(lon2), rad_to_deg(lat2))
    }

    /// 计算中点（球面几何）
    #[must_use]
    pub fn midpoint_geodesic(&self, other: &Self) -> Self {
        let lat1 = deg_to_rad(self.y);
        let lon1 = deg_to_rad(self.x);
        let lat2 = deg_to_rad(other.y);
        let dlon = deg_to_rad(other.x - self.x);

        let bx = lat2.cos() * dlon.cos();
        let by = lat2.cos() * dlon.sin();

        let lat3 = (lat1.sin() + lat2.sin()).atan2(((lat1.cos() + bx).powi(2) + by.powi(2)).sqrt());
        let lon3 = lon1 + by.atan2(lat1.cos() + bx);

        Self::new(rad_to_deg(lon3), rad_to_deg(lat3))
    }

    // ========================================================================
    // 向量运算
    // ========================================================================

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

    /// 旋转指定角度（弧度，逆时针）
    #[inline]
    #[must_use]
    pub fn rotate(&self, angle: f64) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
        }
    }

    /// 分量最小值
    #[inline]
    #[must_use]
    pub fn min(&self, other: &Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// 分量最大值
    #[inline]
    #[must_use]
    pub fn max(&self, other: &Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
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

    // Point3D 测试（保持原有测试）...
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
        assert!((k.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_point3d_length() {
        let p = Point3D::new(3.0, 4.0, 0.0);
        assert!((p.length() - 5.0).abs() < 1e-10);
    }

    // Point2D 欧几里得距离测试
    #[test]
    fn test_point2d_euclidean_distance() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(3.0, 4.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 1e-10);
    }

    // Haversine 测试
    #[test]
    fn test_haversine_distance() {
        // 北京到上海
        let beijing = Point2D::from_lonlat(116.4, 39.9);
        let shanghai = Point2D::from_lonlat(121.5, 31.2);

        let dist = beijing.geodesic_distance_to(&shanghai);
        let dist_km = dist / 1000.0;

        // 实际距离约 1068 km
        assert!(
            (dist_km - 1068.0).abs() < 20.0,
            "Beijing-Shanghai: {dist_km} km"
        );
    }

    #[test]
    fn test_haversine_same_point() {
        let p = Point2D::from_lonlat(116.4, 39.9);
        let dist = p.geodesic_distance_to(&p);
        assert!(dist.abs() < 1e-10);
    }

    #[test]
    fn test_haversine_antipodal() {
        // 对跖点（地球两端）
        let p1 = Point2D::from_lonlat(0.0, 0.0);
        let p2 = Point2D::from_lonlat(180.0, 0.0);

        let dist = p1.geodesic_distance_to(&p2);
        let half_circumference = PI * EARTH_MEAN_RADIUS;

        assert!(
            (dist - half_circumference).abs() < 1000.0,
            "Antipodal distance: {dist}"
        );
    }

    // Vincenty 测试
    #[test]
    fn test_vincenty_distance() {
        let beijing = Point2D::from_lonlat(116.4, 39.9);
        let shanghai = Point2D::from_lonlat(121.5, 31.2);

        let dist = beijing.vincenty_distance_to(&shanghai);
        assert!(dist.is_some());

        let dist_km = dist.unwrap() / 1000.0;
        // Vincenty 精度更高
        assert!(
            (dist_km - 1068.0).abs() < 10.0,
            "Vincenty Beijing-Shanghai: {dist_km} km"
        );
    }

    #[test]
    fn test_vincenty_same_point() {
        let p = Point2D::from_lonlat(116.4, 39.9);
        let dist = p.vincenty_distance_to(&p);
        assert!(dist.is_some());
        assert!(dist.unwrap() < 1e-6);
    }

    // 方位角测试
    #[test]
    fn test_initial_bearing() {
        // 正北
        let p1 = Point2D::from_lonlat(0.0, 0.0);
        let p2 = Point2D::from_lonlat(0.0, 10.0);
        let bearing = p1.initial_bearing_to(&p2);
        assert!((bearing - 0.0).abs() < 0.1, "North bearing: {bearing}");

        // 正东
        let p3 = Point2D::from_lonlat(10.0, 0.0);
        let bearing_east = p1.initial_bearing_to(&p3);
        assert!(
            (bearing_east - 90.0).abs() < 0.1,
            "East bearing: {bearing_east}"
        );
    }

    // 目标点测试
    #[test]
    fn test_destination_point() {
        let start = Point2D::from_lonlat(0.0, 0.0);
        let distance = 111_000.0; // 约 1 度纬度
        let bearing = 0.0; // 正北

        let dest = start.destination_point(distance, bearing);

        // 应该向北移动约 1 度
        assert!((dest.x - 0.0).abs() < 0.1, "lon: {}", dest.x);
        assert!((dest.y - 1.0).abs() < 0.1, "lat: {}", dest.y);
    }

    // 中点测试
    #[test]
    fn test_midpoint_geodesic() {
        let p1 = Point2D::from_lonlat(0.0, 0.0);
        let p2 = Point2D::from_lonlat(10.0, 0.0);

        let mid = p1.midpoint_geodesic(&p2);

        assert!((mid.x - 5.0).abs() < 0.1, "mid lon: {}", mid.x);
        assert!(mid.y.abs() < 0.1, "mid lat: {}", mid.y);
    }

    #[test]
    fn test_point2d_rotate() {
        let p = Point2D::new(1.0, 0.0);
        let rotated = p.rotate(PI / 2.0); // 90度逆时针

        assert!(rotated.x.abs() < 1e-10);
        assert!((rotated.y - 1.0).abs() < 1e-10);
    }
}