// marihydro\crates\mh_geo\src\transform.rs
//! 坐标转换器
//!
//! 基于纯 Rust 实现的投影转换，不依赖外部 C 库
//!
//! # 示例
//!
//! ```
//! use mh_geo::transform::{GeoTransformer, AffineTransform};
//!
//! // 创建仿射变换
//! let affine = AffineTransform::identity();
//! let (x, y) = affine.apply(10.0, 20.0);
//!
//! // 创建坐标转换器
//! let transformer = GeoTransformer::from_epsg(4326, 32650).unwrap();
//! let (utm_x, utm_y) = transformer.transform_point(116.0, 40.0).unwrap();
//! ```

use crate::crs::Crs;
use crate::projection::FastProjection;
use mh_foundation::error::MhResult;

// ============================================================================
// 仿射变换矩阵
// ============================================================================

/// 仿射变换矩阵
///
/// 用于像素坐标到地理坐标的转换
///
/// 变换公式：
/// - x' = a*x + b*y + c
/// - y' = d*x + e*y + f
#[derive(Debug, Clone, Copy)]
pub struct AffineTransform {
    /// x 方向缩放系数
    pub a: f64,
    /// x 方向倾斜系数
    pub b: f64,
    /// x 平移量
    pub c: f64,
    /// y 方向倾斜系数
    pub d: f64,
    /// y 方向缩放系数
    pub e: f64,
    /// y 平移量
    pub f: f64,
}

impl Default for AffineTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl AffineTransform {
    /// 恒等变换
    #[must_use]
    pub fn identity() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        }
    }

    /// 创建平移变换
    #[must_use]
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            c: tx,
            d: 0.0,
            e: 1.0,
            f: ty,
        }
    }

    /// 创建缩放变换
    #[must_use]
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            a: sx,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: sy,
            f: 0.0,
        }
    }

    /// 创建旋转变换（弧度，逆时针）
    #[must_use]
    pub fn rotation(angle: f64) -> Self {
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        Self {
            a: cos_a,
            b: -sin_a,
            c: 0.0,
            d: sin_a,
            e: cos_a,
            f: 0.0,
        }
    }

    /// 从 GDAL `GeoTransform` 数组创建
    ///
    /// GDAL 格式: [c, a, b, f, d, e]
    #[must_use]
    pub fn from_gdal_geotransform(gt: [f64; 6]) -> Self {
        Self {
            c: gt[0],
            a: gt[1],
            b: gt[2],
            f: gt[3],
            d: gt[4],
            e: gt[5],
        }
    }

    /// 转换为 GDAL `GeoTransform` 格式
    #[must_use]
    pub fn to_gdal_geotransform(&self) -> [f64; 6] {
        [self.c, self.a, self.b, self.f, self.d, self.e]
    }

    /// 应用正向变换
    #[inline]
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.c,
            self.d * x + self.e * y + self.f,
        )
    }

    /// 计算逆变换
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.a * self.e - self.b * self.d;
        if det.abs() < 1e-15 {
            return None;
        }
        let inv_det = 1.0 / det;
        Some(Self {
            a: self.e * inv_det,
            b: -self.b * inv_det,
            c: (self.b * self.f - self.c * self.e) * inv_det,
            d: -self.d * inv_det,
            e: self.a * inv_det,
            f: (self.c * self.d - self.a * self.f) * inv_det,
        })
    }

    /// 应用逆变换
    #[must_use]
    pub fn apply_inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.inverse().map(|inv| inv.apply(x, y))
    }

    /// 变换多个点
    #[must_use]
    pub fn apply_batch(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        points.iter().map(|&(x, y)| self.apply(x, y)).collect()
    }

    /// 组合两个变换：self * other
    ///
    /// 结果变换先应用 other，再应用 self
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            a: self.a * other.a + self.b * other.d,
            b: self.a * other.b + self.b * other.e,
            c: self.a * other.c + self.b * other.f + self.c,
            d: self.d * other.a + self.e * other.d,
            e: self.d * other.b + self.e * other.e,
            f: self.d * other.c + self.e * other.f + self.f,
        }
    }

    /// 获取变换的行列式
    #[must_use]
    pub fn determinant(&self) -> f64 {
        self.a * self.e - self.b * self.d
    }

    /// 是否为恒等变换
    #[must_use]
    pub fn is_identity(&self) -> bool {
        (self.a - 1.0).abs() < 1e-10
            && self.b.abs() < 1e-10
            && self.c.abs() < 1e-10
            && self.d.abs() < 1e-10
            && (self.e - 1.0).abs() < 1e-10
            && self.f.abs() < 1e-10
    }
}

// ============================================================================
// 地理坐标转换器
// ============================================================================

/// 地理坐标转换器
pub struct GeoTransformer {
    /// 源 CRS
    source_crs: Crs,
    /// 目标 CRS
    target_crs: Crs,
    /// 快速投影（源）
    source_proj: FastProjection,
    /// 快速投影（目标）
    target_proj: FastProjection,
    /// 是否为恒等变换
    is_identity: bool,
}

impl GeoTransformer {
    /// 创建新的坐标转换器
    ///
    /// # Errors
    /// 如果 CRS 不支持则返回错误
    pub fn new(source: &Crs, target: &Crs) -> MhResult<Self> {
        let is_identity = source.definition == target.definition;
        let source_proj = source.to_fast_projection();
        let target_proj = target.to_fast_projection();

        Ok(Self {
            source_crs: source.clone(),
            target_crs: target.clone(),
            source_proj,
            target_proj,
            is_identity,
        })
    }

    /// 从 EPSG 代码创建转换器
    ///
    /// # Errors
    /// 如果 EPSG 代码无效则返回错误
    pub fn from_epsg(source_epsg: u32, target_epsg: u32) -> MhResult<Self> {
        let source = Crs::from_epsg(source_epsg)?;
        let target = Crs::from_epsg(target_epsg)?;
        Self::new(&source, &target)
    }

    /// 创建恒等变换
    #[must_use]
    pub fn identity() -> Self {
        let crs = Crs::wgs84();
        Self {
            source_crs: crs.clone(),
            target_crs: crs,
            source_proj: FastProjection::Geographic(crate::ellipsoid::Ellipsoid::WGS84),
            target_proj: FastProjection::Geographic(crate::ellipsoid::Ellipsoid::WGS84),
            is_identity: true,
        }
    }

    /// 正向变换单点
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    #[inline]
    pub fn transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        if self.is_identity {
            return Ok((x, y));
        }

        // 源坐标 -> 地理坐标
        let (lon, lat) = self.source_proj.inverse(x, y)?;
        // 地理坐标 -> 目标坐标
        self.target_proj.forward(lon, lat)
    }

    /// 逆向变换单点
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    #[inline]
    pub fn inverse_transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        if self.is_identity {
            return Ok((x, y));
        }

        let (lon, lat) = self.target_proj.inverse(x, y)?;
        self.source_proj.forward(lon, lat)
    }

    /// 批量正向变换
    ///
    /// # Errors
    /// 如果任意坐标超出有效范围则返回错误
    pub fn transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if self.is_identity {
            return Ok(points.to_vec());
        }

        points
            .iter()
            .map(|&(x, y)| self.transform_point(x, y))
            .collect()
    }

    /// 批量逆向变换
    ///
    /// # Errors
    /// 如果任意坐标超出有效范围则返回错误
    pub fn inverse_transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if self.is_identity {
            return Ok(points.to_vec());
        }

        points
            .iter()
            .map(|&(x, y)| self.inverse_transform_point(x, y))
            .collect()
    }

    /// 就地变换坐标数组
    ///
    /// # Errors
    /// 如果任意坐标超出有效范围则返回错误
    pub fn transform_inplace(&self, x: &mut [f64], y: &mut [f64]) -> MhResult<()> {
        if self.is_identity {
            return Ok(());
        }

        let n = x.len().min(y.len());
        for i in 0..n {
            let (nx, ny) = self.transform_point(x[i], y[i])?;
            x[i] = nx;
            y[i] = ny;
        }
        Ok(())
    }

    /// 计算投影收敛角（用于矢量旋转）
    ///
    /// 返回从真北到网格北的顺时针角度（弧度）
    #[must_use]
    pub fn compute_convergence_angle(&self, x: f64, y: f64) -> f64 {
        if self.is_identity || self.target_crs.is_geographic() {
            return 0.0;
        }

        // 使用有限差分计算收敛角
        let delta = 0.0001; // 约11米（在赤道）

        // 获取点在目标 CRS 中的坐标
        let (px, py) = match self.transform_point(x, y) {
            Ok(p) => p,
            Err(_) => return 0.0,
        };

        // 北向偏移
        let (px_n, py_n) = match self.transform_point(x, y + delta) {
            Ok(p) => p,
            Err(_) => return 0.0,
        };

        // 计算网格北方向
        let dx = px_n - px;
        let dy = py_n - py;

        // 收敛角 = arctan(dx/dy)
        dx.atan2(dy)
    }

    /// 旋转矢量以补偿投影收敛角
    #[must_use]
    pub fn rotate_vector(&self, u: f64, v: f64, x: f64, y: f64) -> (f64, f64) {
        let angle = self.compute_convergence_angle(x, y);
        if angle.abs() < 1e-10 {
            return (u, v);
        }
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        (u * cos_a - v * sin_a, u * sin_a + v * cos_a)
    }

    /// 批量旋转矢量
    pub fn rotate_vectors(&self, u: &mut [f64], v: &mut [f64], x: &[f64], y: &[f64]) {
        if self.is_identity {
            return;
        }
        let n = u.len().min(v.len()).min(x.len()).min(y.len());
        for i in 0..n {
            let (nu, nv) = self.rotate_vector(u[i], v[i], x[i], y[i]);
            u[i] = nu;
            v[i] = nv;
        }
    }

    /// 获取源 CRS
    #[must_use]
    pub fn source_crs(&self) -> &Crs {
        &self.source_crs
    }

    /// 获取目标 CRS
    #[must_use]
    pub fn target_crs(&self) -> &Crs {
        &self.target_crs
    }

    /// 是否为恒等变换
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.is_identity
    }
}

// ============================================================================
// 快捷转换函数
// ============================================================================

/// 快捷转换函数
pub mod conversions {
    use super::MhResult;
    use crate::projection;

    /// WGS84 经纬度转 UTM
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    pub fn wgs84_to_utm(lon: f64, lat: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
        projection::wgs84_to_utm(lon, lat, zone, north)
    }

    /// UTM 转 WGS84 经纬度
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    pub fn utm_to_wgs84(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
        projection::utm_to_wgs84(x, y, zone, north)
    }

    /// 自动检测 UTM 区域并转换
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    pub fn wgs84_to_auto_utm(lon: f64, lat: f64) -> MhResult<(f64, f64, u8, bool)> {
        projection::wgs84_to_auto_utm(lon, lat)
    }

    /// WGS84 转 Web Mercator
    ///
    /// # Errors
    /// 如果坐标超出有效范围则返回错误
    pub fn wgs84_to_web_mercator(lon: f64, lat: f64) -> MhResult<(f64, f64)> {
        projection::wgs84_to_web_mercator(lon, lat)
    }

    /// Web Mercator 转 WGS84
    ///
    /// # Errors
    /// 返回可能的错误
    pub fn web_mercator_to_wgs84(x: f64, y: f64) -> MhResult<(f64, f64)> {
        projection::web_mercator_to_wgs84(x, y)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let transformer = GeoTransformer::identity();
        let (x, y) = transformer
            .transform_point(116.0, 40.0)
            .expect("transform failed");
        assert!((x - 116.0).abs() < 1e-10);
        assert!((y - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_transform() {
        let affine = AffineTransform {
            a: 2.0,
            b: 0.0,
            c: 10.0,
            d: 0.0,
            e: 3.0,
            f: 20.0,
        };
        let (x, y) = affine.apply(5.0, 5.0);
        assert!((x - 20.0).abs() < 1e-10); // 2*5 + 10
        assert!((y - 35.0).abs() < 1e-10); // 3*5 + 20

        let inv = affine.inverse().expect("inverse failed");
        let (ox, oy) = inv.apply(x, y);
        assert!((ox - 5.0).abs() < 1e-10);
        assert!((oy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_compose() {
        let scale = AffineTransform::scale(2.0, 2.0);
        let translate = AffineTransform::translation(10.0, 20.0);

        // 先缩放再平移
        let combined = translate.compose(&scale);
        let (x, y) = combined.apply(5.0, 5.0);
        assert!((x - 20.0).abs() < 1e-10); // 2*5 + 10
        assert!((y - 30.0).abs() < 1e-10); // 2*5 + 20
    }

    #[test]
    fn test_affine_rotation() {
        let rot90 = AffineTransform::rotation(std::f64::consts::FRAC_PI_2);
        let (x, y) = rot90.apply(1.0, 0.0);
        assert!(x.abs() < 1e-10);
        assert!((y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wgs84_to_utm() {
        // 北京 116°E, 40°N -> UTM 50N
        let result = conversions::wgs84_to_utm(116.0, 40.0, 50, true);
        assert!(result.is_ok());
        let (x, y) = result.expect("utm conversion failed");
        assert!(x > 400_000.0 && x < 600_000.0, "x out of range: {x}");
        assert!(y > 4_000_000.0 && y < 5_000_000.0, "y out of range: {y}");
    }

    #[test]
    fn test_geo_transformer() {
        let transformer =
            GeoTransformer::from_epsg(4326, 32650).expect("create transformer failed");
        let (x, y) = transformer
            .transform_point(116.0, 40.0)
            .expect("transform failed");

        assert!(x > 400_000.0 && x < 600_000.0, "x out of range: {x}");
        assert!(y > 4_000_000.0 && y < 5_000_000.0, "y out of range: {y}");

        // 测试逆变换
        let (lon, lat) = transformer
            .inverse_transform_point(x, y)
            .expect("inverse failed");
        assert!((lon - 116.0).abs() < 1e-9, "lon mismatch: {lon}");
        assert!((lat - 40.0).abs() < 1e-9, "lat mismatch: {lat}");
    }

    #[test]
    fn test_web_mercator_conversion() {
        let (x, y) =
            conversions::wgs84_to_web_mercator(116.0, 40.0).expect("web mercator failed");

        assert!(x > 12_900_000.0 && x < 12_950_000.0, "x out of range: {x}");
        assert!(y > 4_800_000.0 && y < 4_900_000.0, "y out of range: {y}");

        let (lon, lat) = conversions::web_mercator_to_wgs84(x, y).expect("inverse failed");
        assert!((lon - 116.0).abs() < 1e-9);
        assert!((lat - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_affine_gdal_format() {
        let gt = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0];
        let affine = AffineTransform::from_gdal_geotransform(gt);

        let (x, y) = affine.apply(10.0, 20.0);
        assert!((x - 110.0).abs() < 1e-10);
        assert!((y - 180.0).abs() < 1e-10);

        let gt2 = affine.to_gdal_geotransform();
        assert_eq!(gt, gt2);
    }

    #[test]
    fn test_batch_transform() {
        let transformer =
            GeoTransformer::from_epsg(4326, 32650).expect("create transformer failed");

        let points = vec![(116.0, 40.0), (117.0, 41.0), (118.0, 42.0)];

        let transformed = transformer.transform_points(&points).expect("batch failed");
        assert_eq!(transformed.len(), 3);

        for (x, y) in &transformed {
            assert!(*x > 100_000.0 && *x < 900_000.0);
            assert!(*y > 4_000_000.0 && *y < 5_000_000.0);
        }
    }

    #[test]
    fn test_affine_is_identity() {
        assert!(AffineTransform::identity().is_identity());
        assert!(!AffineTransform::translation(1.0, 0.0).is_identity());
    }
}