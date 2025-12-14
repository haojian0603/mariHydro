// marihydro\crates\mh_geo\src\ellipsoid.rs
//! 椭球体定义
//!
//! 提供地球椭球体参数，支持 WGS84、CGCS2000、GRS80 等标准椭球体。
//!
//! # 示例
//!
//! ```
//! use mh_geo::ellipsoid::Ellipsoid;
//!
//! let wgs84 = Ellipsoid::WGS84;
//! println!("长半轴: {} m", wgs84.a);
//! println!("第一偏心率平方: {}", wgs84.e2());
//! ```

use serde::{Deserialize, Serialize};

/// 地球椭球体
///
/// 定义椭球体的几何参数，并提供派生参数的计算方法。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ellipsoid {
    /// 长半轴 (m)
    pub a: f64,
    /// 扁率 (flattening)
    pub f: f64,
}

impl Ellipsoid {
    // ========================================================================
    // 预定义椭球体
    // ========================================================================

    /// WGS84 椭球体 (GPS 标准)
    ///
    /// - EPSG: 7030
    /// - 长半轴: 6378137.0 m
    /// - 扁率: 1/298.257223563
    pub const WGS84: Self = Self {
        a: 6_378_137.0,
        f: 1.0 / 298.257_223_563,
    };

    /// CGCS2000 椭球体 (中国大地坐标系)
    ///
    /// - EPSG: 1024
    /// - 长半轴: 6378137.0 m
    /// - 扁率: 1/298.257222101
    ///
    /// 注意：与 WGS84 极为相似，扁率微有差异
    pub const CGCS2000: Self = Self {
        a: 6_378_137.0,
        f: 1.0 / 298.257_222_101,
    };

    /// GRS80 椭球体
    ///
    /// - EPSG: 7019
    /// - 等同于 CGCS2000
    pub const GRS80: Self = Self::CGCS2000;

    /// 克拉索夫斯基椭球体 (北京54坐标系)
    ///
    /// - 长半轴: 6378245.0 m
    /// - 扁率: 1/298.3
    pub const KRASSOVSKY: Self = Self {
        a: 6_378_245.0,
        f: 1.0 / 298.3,
    };

    /// 国际椭球体 1924 (西安80坐标系采用的 IAG75)
    pub const INTERNATIONAL_1924: Self = Self {
        a: 6_378_388.0,
        f: 1.0 / 297.0,
    };

    // ========================================================================
    // 构造方法
    // ========================================================================

    /// 从长半轴和扁率创建椭球体
    #[must_use]
    pub const fn new(a: f64, f: f64) -> Self {
        Self { a, f }
    }

    /// 从长半轴和短半轴创建椭球体
    #[must_use]
    pub fn from_semi_axes(a: f64, b: f64) -> Self {
        let f = (a - b) / a;
        Self { a, f }
    }

    /// 从 EPSG 椭球体代码获取
    #[must_use]
    pub fn from_epsg(code: u32) -> Option<Self> {
        match code {
            7030 => Some(Self::WGS84),
            7019 => Some(Self::GRS80),
            1024 => Some(Self::CGCS2000),
            7024 => Some(Self::KRASSOVSKY),
            7022 => Some(Self::INTERNATIONAL_1924),
            _ => None,
        }
    }

    // ========================================================================
    // 派生参数（几何常量）
    // ========================================================================

    /// 短半轴 b = a(1-f)
    #[inline]
    #[must_use]
    pub fn b(&self) -> f64 {
        self.a * (1.0 - self.f)
    }

    /// 第一偏心率的平方 e² = 2f - f²
    #[inline]
    #[must_use]
    pub fn e2(&self) -> f64 {
        self.f * (2.0 - self.f)
    }

    /// 第一偏心率 e = √e²
    #[inline]
    #[must_use]
    pub fn e(&self) -> f64 {
        self.e2().sqrt()
    }

    /// 第二偏心率的平方 e'² = e²/(1-e²)
    #[inline]
    #[must_use]
    pub fn ep2(&self) -> f64 {
        let e2 = self.e2();
        e2 / (1.0 - e2)
    }

    /// 第二偏心率 e' = √e'²
    #[inline]
    #[must_use]
    pub fn ep(&self) -> f64 {
        self.ep2().sqrt()
    }

    /// 第三扁率 n = (a-b)/(a+b) = f/(2-f)
    ///
    /// 这是 Karney 算法的关键参数
    #[inline]
    #[must_use]
    pub fn n(&self) -> f64 {
        self.f / (2.0 - self.f)
    }

    /// 子午圈曲率半径（在纬度 φ 处）
    ///
    /// M = a(1-e²) / (1-e²sin²φ)^(3/2)
    #[inline]
    #[must_use]
    pub fn meridional_radius(&self, lat_rad: f64) -> f64 {
        let sin_lat = lat_rad.sin();
        let e2 = self.e2();
        self.a * (1.0 - e2) / (1.0 - e2 * sin_lat * sin_lat).powf(1.5)
    }

    /// 卯酉圈曲率半径（在纬度 φ 处）
    ///
    /// N = a / √(1-e²sin²φ)
    #[inline]
    #[must_use]
    pub fn prime_vertical_radius(&self, lat_rad: f64) -> f64 {
        let sin_lat = lat_rad.sin();
        let e2 = self.e2();
        self.a / (1.0 - e2 * sin_lat * sin_lat).sqrt()
    }

    /// 平均曲率半径（几何平均）
    ///
    /// R = √(M·N)
    #[inline]
    #[must_use]
    pub fn mean_radius(&self, lat_rad: f64) -> f64 {
        (self.meridional_radius(lat_rad) * self.prime_vertical_radius(lat_rad)).sqrt()
    }

    /// 自转椭球体体积
    #[inline]
    #[must_use]
    pub fn volume(&self) -> f64 {
        (4.0 / 3.0) * std::f64::consts::PI * self.a * self.a * self.b()
    }

    /// 椭球体表面积（近似公式）
    #[must_use]
    pub fn surface_area(&self) -> f64 {
        let a = self.a;
        let e = self.e();
        
        if e < 1e-10 {
            // 近似球体
            4.0 * std::f64::consts::PI * a * a
        } else {
            // 扁椭球体
            2.0 * std::f64::consts::PI * a * a 
                * (1.0 + ((1.0 - e * e) / e) * ((1.0 + e) / (1.0 - e)).ln() / 2.0)
        }
    }

    // ========================================================================
    // Karney 算法所需的预计算系数
    // ========================================================================

    /// 计算 Krüger α 系数（正向投影用）
    ///
    /// 返回 6 阶系数数组
    #[must_use]
    pub fn krueger_alpha(&self) -> [f64; 6] {
        let n = self.n();
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n3 * n;
        let n5 = n4 * n;
        let n6 = n5 * n;

        [
            // α₁
            n / 2.0 - (2.0 / 3.0) * n2 + (5.0 / 16.0) * n3 
                + (41.0 / 180.0) * n4 - (127.0 / 288.0) * n5 + (7891.0 / 37800.0) * n6,
            // α₂
            (13.0 / 48.0) * n2 - (3.0 / 5.0) * n3 + (557.0 / 1440.0) * n4 
                + (281.0 / 630.0) * n5 - (1983433.0 / 1935360.0) * n6,
            // α₃
            (61.0 / 240.0) * n3 - (103.0 / 140.0) * n4 + (15061.0 / 26880.0) * n5
                + (167603.0 / 181440.0) * n6,
            // α₄
            (49561.0 / 161280.0) * n4 - (179.0 / 168.0) * n5 + (6601661.0 / 7257600.0) * n6,
            // α₅
            (34729.0 / 80640.0) * n5 - (3418889.0 / 1995840.0) * n6,
            // α₆
            (212378941.0 / 319334400.0) * n6,
        ]
    }

    /// 计算 Krüger β 系数（逆向投影用）
    ///
    /// 返回 6 阶系数数组
    #[must_use]
    pub fn krueger_beta(&self) -> [f64; 6] {
        let n = self.n();
        let n2 = n * n;
        let n3 = n2 * n;
        let n4 = n3 * n;
        let n5 = n4 * n;
        let n6 = n5 * n;

        [
            // β₁
            n / 2.0 - (2.0 / 3.0) * n2 + (37.0 / 96.0) * n3 
                - (1.0 / 360.0) * n4 - (81.0 / 512.0) * n5 + (96199.0 / 604800.0) * n6,
            // β₂
            (1.0 / 48.0) * n2 + (1.0 / 15.0) * n3 - (437.0 / 1440.0) * n4 
                + (46.0 / 105.0) * n5 - (1118711.0 / 3870720.0) * n6,
            // β₃
            (17.0 / 480.0) * n3 - (37.0 / 840.0) * n4 - (209.0 / 4480.0) * n5
                + (5569.0 / 90720.0) * n6,
            // β₄
            (4397.0 / 161280.0) * n4 - (11.0 / 504.0) * n5 - (830251.0 / 7257600.0) * n6,
            // β₅
            (4583.0 / 161280.0) * n5 - (108847.0 / 3991680.0) * n6,
            // β₆
            (20648693.0 / 638668800.0) * n6,
        ]
    }

    /// 计算缩放常数 A
    ///
    /// A = a/(1+n) * (1 + n²/4 + n⁴/64 + n⁶/256 + ...)
    #[must_use]
    pub fn krueger_a(&self) -> f64 {
        let n = self.n();
        let n2 = n * n;
        let n4 = n2 * n2;
        let n6 = n4 * n2;
        let n8 = n4 * n4;

        (self.a / (1.0 + n)) 
            * (1.0 + n2 / 4.0 + n4 / 64.0 + n6 / 256.0 + (25.0 / 16384.0) * n8)
    }
}

impl Default for Ellipsoid {
    fn default() -> Self {
        Self::WGS84
    }
}

impl std::fmt::Display for Ellipsoid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Ellipsoid(a={}, f=1/{:.6})", self.a, 1.0 / self.f)
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgs84_parameters() {
        let e = Ellipsoid::WGS84;
        
        // 验证长半轴
        assert!((e.a - 6_378_137.0).abs() < 1e-6);
        
        // 验证短半轴 (标准值约 6356752.314245)
        assert!((e.b() - 6_356_752.314_245).abs() < 0.001);
        
        // 验证第一偏心率平方 (约 0.00669437999014)
        assert!((e.e2() - 0.006_694_379_990_14).abs() < 1e-12);
        
        // 验证第三扁率
        let n_expected = e.f / (2.0 - e.f);
        assert!((e.n() - n_expected).abs() < 1e-15);
    }

    #[test]
    fn test_cgcs2000_vs_wgs84() {
        let wgs84 = Ellipsoid::WGS84;
        let cgcs = Ellipsoid::CGCS2000;
        
        // 长半轴相同
        assert_eq!(wgs84.a, cgcs.a);
        
        // 扁率略有不同
        assert!((wgs84.f - cgcs.f).abs() > 1e-12);
        assert!((wgs84.f - cgcs.f).abs() < 1e-9);
    }

    #[test]
    fn test_curvature_radius() {
        let e = Ellipsoid::WGS84;
        
        // 赤道处
        let m_equator = e.meridional_radius(0.0);
        let n_equator = e.prime_vertical_radius(0.0);
        
        // N > M 在赤道
        assert!(n_equator > m_equator);
        
        // N(0) = a
        assert!((n_equator - e.a).abs() < 1e-6);
    }

    #[test]
    fn test_krueger_coefficients() {
        let e = Ellipsoid::WGS84;
        
        let alpha = e.krueger_alpha();
        let beta = e.krueger_beta();
        
        // 系数应该是小量
        for i in 0..6 {
            assert!(alpha[i].abs() < 1.0);
            assert!(beta[i].abs() < 1.0);
        }
        
        // 高阶系数应该更小
        assert!(alpha[5].abs() < alpha[0].abs());
        assert!(beta[5].abs() < beta[0].abs());
    }

    #[test]
    fn test_from_epsg() {
        assert_eq!(Ellipsoid::from_epsg(7030), Some(Ellipsoid::WGS84));
        assert_eq!(Ellipsoid::from_epsg(7019), Some(Ellipsoid::GRS80));
        assert_eq!(Ellipsoid::from_epsg(9999), None);
    }

    #[test]
    fn test_from_semi_axes() {
        let e = Ellipsoid::from_semi_axes(6_378_137.0, 6_356_752.314_245);
        assert!((e.f - Ellipsoid::WGS84.f).abs() < 1e-9);
    }
}