// src-tauri/src/marihydro/geo/transform.rs
//! 坐标转换器
//! 基于 PROJ 库实现高精度地理坐标转换

use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::geo::crs::Crs;
use glam::DVec2;
use proj::Proj;
use rayon::prelude::*;

/// 并行处理阈值
const PARALLEL_THRESHOLD: usize = 1000;

/// 仿射变换矩阵
/// 用于像素坐标到地理坐标的转换
#[derive(Debug, Clone, Copy)]
pub struct AffineTransform {
    /// x' = a*x + b*y + c
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// y' = d*x + e*y + f
    pub d: f64,
    pub e: f64,
    pub f: f64,
}

impl AffineTransform {
    /// 恒等变换
    pub fn identity() -> Self {
        Self {
            a: 1.0, b: 0.0, c: 0.0,
            d: 0.0, e: 1.0, f: 0.0,
        }
    }

    /// 从 GDAL GeoTransform 数组创建
    /// GDAL 格式: [c, a, b, f, d, e]
    pub fn from_gdal_geotransform(gt: [f64; 6]) -> Self {
        Self {
            c: gt[0], a: gt[1], b: gt[2],
            f: gt[3], d: gt[4], e: gt[5],
        }
    }

    /// 转换为 GDAL GeoTransform 格式
    pub fn to_gdal_geotransform(&self) -> [f64; 6] {
        [self.c, self.a, self.b, self.f, self.d, self.e]
    }

    /// 应用正向变换
    #[inline]
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (
            self.a * x + self.b * y + self.c,
            self.d * x + self.e * y + self.f,
        )
    }

    /// 计算逆变换
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
    pub fn apply_inverse(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        self.inverse().map(|inv| inv.apply(x, y))
    }

    /// 变换多个点
    pub fn apply_batch(&self, points: &[(f64, f64)]) -> Vec<(f64, f64)> {
        points.iter().map(|&(x, y)| self.apply(x, y)).collect()
    }
}

/// 地理坐标转换器
pub struct GeoTransformer {
    /// 源 CRS
    source_crs: Crs,
    /// 目标 CRS
    target_crs: Crs,
    /// PROJ 转换对象
    proj: Option<Proj>,
    /// 逆向 PROJ 转换对象
    proj_inverse: Option<Proj>,
    /// 是否为恒等变换
    is_identity: bool,
}

impl GeoTransformer {
    /// 创建新的坐标转换器
    pub fn new(source: &Crs, target: &Crs) -> MhResult<Self> {
        let is_identity = source.definition == target.definition;

        if is_identity {
            return Ok(Self {
                source_crs: source.clone(),
                target_crs: target.clone(),
                proj: None,
                proj_inverse: None,
                is_identity: true,
            });
        }

        // 创建 PROJ 转换对象
        let proj = Proj::new_known_crs(&source.definition, &target.definition, None)
            .map_err(|e| MhError::Config(format!("Failed to create PROJ transform: {}", e)))?;

        // 创建逆向转换
        let proj_inverse = Proj::new_known_crs(&target.definition, &source.definition, None)
            .map_err(|e| MhError::Config(format!("Failed to create inverse PROJ transform: {}", e)))?;

        Ok(Self {
            source_crs: source.clone(),
            target_crs: target.clone(),
            proj: Some(proj),
            proj_inverse: Some(proj_inverse),
            is_identity: false,
        })
    }

    /// 从 EPSG 代码创建转换器
    pub fn from_epsg(source_epsg: u32, target_epsg: u32) -> MhResult<Self> {
        let source = Crs::from_epsg(source_epsg)?;
        let target = Crs::from_epsg(target_epsg)?;
        Self::new(&source, &target)
    }

    /// 创建恒等变换
    pub fn identity() -> Self {
        let crs = Crs::wgs84();
        Self {
            source_crs: crs.clone(),
            target_crs: crs,
            proj: None,
            proj_inverse: None,
            is_identity: true,
        }
    }

    /// 正向变换单点
    #[inline]
    pub fn transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        if self.is_identity {
            return Ok((x, y));
        }

        let proj = self.proj.as_ref()
            .ok_or_else(|| MhError::Config("PROJ not initialized".into()))?;

        proj.convert((x, y))
            .map_err(|e| MhError::Config(format!("PROJ transform failed: {}", e)))
    }

    /// 逆向变换单点
    #[inline]
    pub fn inverse_transform_point(&self, x: f64, y: f64) -> MhResult<(f64, f64)> {
        if self.is_identity {
            return Ok((x, y));
        }

        let proj = self.proj_inverse.as_ref()
            .ok_or_else(|| MhError::Config("Inverse PROJ not initialized".into()))?;

        proj.convert((x, y))
            .map_err(|e| MhError::Config(format!("Inverse PROJ transform failed: {}", e)))
    }

    /// 批量正向变换（自动并行）
    pub fn transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if self.is_identity {
            return Ok(points.to_vec());
        }

        if points.len() < PARALLEL_THRESHOLD {
            points.iter().map(|&(x, y)| self.transform_point(x, y)).collect()
        } else {
            // 并行处理时需要线程安全的 PROJ 实例
            // 每个线程创建独立的 PROJ 对象
            let src_def = self.source_crs.definition.clone();
            let tgt_def = self.target_crs.definition.clone();
            
            points.par_iter()
                .map(|&(x, y)| {
                    // 线程本地 PROJ
                    thread_local! {
                        static PROJ_CACHE: std::cell::RefCell<Option<Proj>> = const { std::cell::RefCell::new(None) };
                    }
                    
                    // 简化：直接创建新的 Proj（实际可用 thread_local 缓存）
                    let proj = Proj::new_known_crs(&src_def, &tgt_def, None)
                        .map_err(|e| MhError::Config(format!("PROJ error: {}", e)))?;
                    proj.convert((x, y))
                        .map_err(|e| MhError::Config(format!("Transform error: {}", e)))
                })
                .collect()
        }
    }

    /// 批量逆向变换
    pub fn inverse_transform_points(&self, points: &[(f64, f64)]) -> MhResult<Vec<(f64, f64)>> {
        if self.is_identity {
            return Ok(points.to_vec());
        }

        points.iter().map(|&(x, y)| self.inverse_transform_point(x, y)).collect()
    }

    /// 变换 DVec2 数组
    pub fn transform_dvec2(&self, points: &[DVec2]) -> MhResult<Vec<DVec2>> {
        if self.is_identity {
            return Ok(points.to_vec());
        }
        points
            .iter()
            .map(|p| self.transform_point(p.x, p.y).map(|(x, y)| DVec2::new(x, y)))
            .collect()
    }

    /// 就地变换坐标数组
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
    /// 返回从真北到网格北的顺时针角度（弧度）
    pub fn compute_convergence_angle(&self, x: f64, y: f64) -> f64 {
        if self.is_identity || !self.target_crs.is_projected() {
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
    pub fn rotate_vector(&self, u: f64, v: f64, x: f64, y: f64) -> (f64, f64) {
        let angle = self.compute_convergence_angle(x, y);
        if angle.abs() < 1e-10 {
            return (u, v);
        }
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        (
            u * cos_a - v * sin_a,
            u * sin_a + v * cos_a,
        )
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
    pub fn source_crs(&self) -> &Crs {
        &self.source_crs
    }

    /// 获取目标 CRS
    pub fn target_crs(&self) -> &Crs {
        &self.target_crs
    }

    /// 是否为恒等变换
    pub fn is_identity(&self) -> bool {
        self.is_identity
    }
}

/// 快捷转换函数
pub mod conversions {
    use super::*;

    /// WGS84 经纬度转 UTM
    pub fn wgs84_to_utm(lon: f64, lat: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
        let target_epsg = if north { 32600 + zone as u32 } else { 32700 + zone as u32 };
        let transformer = GeoTransformer::from_epsg(4326, target_epsg)?;
        transformer.transform_point(lon, lat)
    }

    /// UTM 转 WGS84 经纬度
    pub fn utm_to_wgs84(x: f64, y: f64, zone: u8, north: bool) -> MhResult<(f64, f64)> {
        let source_epsg = if north { 32600 + zone as u32 } else { 32700 + zone as u32 };
        let transformer = GeoTransformer::from_epsg(source_epsg, 4326)?;
        transformer.transform_point(x, y)
    }

    /// 自动检测 UTM 区域并转换
    pub fn wgs84_to_auto_utm(lon: f64, lat: f64) -> MhResult<(f64, f64, u8, bool)> {
        let zone = ((lon + 180.0) / 6.0).floor() as u8 + 1;
        let north = lat >= 0.0;
        let (x, y) = wgs84_to_utm(lon, lat, zone, north)?;
        Ok((x, y, zone, north))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_transform() {
        let transformer = GeoTransformer::identity();
        let (x, y) = transformer.transform_point(116.0, 40.0).expect("transform failed");
        assert!((x - 116.0).abs() < 1e-10);
        assert!((y - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_transform() {
        let affine = AffineTransform {
            a: 2.0, b: 0.0, c: 10.0,
            d: 0.0, e: 3.0, f: 20.0,
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
    fn test_wgs84_to_utm() {
        // 北京 116°E, 40°N -> UTM 50N
        let result = conversions::wgs84_to_utm(116.0, 40.0, 50, true);
        assert!(result.is_ok());
        let (x, y) = result.expect("utm conversion failed");
        // UTM 坐标应该在合理范围内
        assert!(x > 400000.0 && x < 600000.0);
        assert!(y > 4000000.0 && y < 5000000.0);
    }
}

