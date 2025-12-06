// marihydro\crates\mh_terrain\src/interpolation/kriging.rs

//! Kriging 插值
//!
//! 克里金地统计插值方法，提供最优线性无偏估计（BLUE）。
//!
//! # 算法原理
//!
//! Kriging 基于区域化变量理论，假设空间变异可以用变异函数描述。
//! 通过求解 Kriging 系统获得最优权重：
//!
//! $$
//! \begin{bmatrix} \gamma_{11} & \cdots & \gamma_{1n} & 1 \\
//!                 \vdots & \ddots & \vdots & \vdots \\
//!                 \gamma_{n1} & \cdots & \gamma_{nn} & 1 \\
//!                 1 & \cdots & 1 & 0 \end{bmatrix}
//! \begin{bmatrix} w_1 \\ \vdots \\ w_n \\ \mu \end{bmatrix} =
//! \begin{bmatrix} \gamma_{10} \\ \vdots \\ \gamma_{n0} \\ 1 \end{bmatrix}
//! $$
//!
//! # 变异函数模型
//!
//! 支持三种常用模型：
//! - 球状模型 (Spherical)
//! - 指数模型 (Exponential)
//! - 高斯模型 (Gaussian)
//!
//! # 示例
//!
//! ```ignore
//! use mh_terrain::interpolation::kriging::{KrigingInterpolator, VariogramModel};
//! use mh_geo::Point2D;
//!
//! let points = vec![...];
//! let values = vec![...];
//!
//! let variogram = VariogramModel::spherical(0.0, 1.0, 100.0);
//! let kriging = KrigingInterpolator::new(points, values, variogram);
//!
//! let (value, variance) = kriging.interpolate_with_variance(50.0, 50.0);
//! ```

use mh_geo::Point2D;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

/// 变异函数模型
///
/// 描述空间相关性随距离变化的模型。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VariogramModel {
    /// 球状模型
    ///
    /// 最常用的模型，具有明确的变程。
    /// γ(h) = C₀ + C * (1.5h/a - 0.5(h/a)³) 当 h < a
    /// γ(h) = C₀ + C 当 h >= a
    Spherical {
        /// 块金值 (nugget)，即 h=0 处的变异
        nugget: f64,
        /// 基台值 (sill)，不含块金
        sill: f64,
        /// 变程 (range)，相关性消失的距离
        range: f64,
    },

    /// 指数模型
    ///
    /// 渐近趋近于基台，没有明确的变程。
    /// γ(h) = C₀ + C * (1 - exp(-3h/a))
    Exponential {
        nugget: f64,
        sill: f64,
        range: f64,
    },

    /// 高斯模型
    ///
    /// 光滑的变异函数，在原点处导数为零。
    /// γ(h) = C₀ + C * (1 - exp(-3(h/a)²))
    Gaussian {
        nugget: f64,
        sill: f64,
        range: f64,
    },

    /// 线性模型
    ///
    /// 最简单的模型，变异随距离线性增加。
    /// γ(h) = C₀ + slope * h
    Linear {
        nugget: f64,
        slope: f64,
    },

    /// 纯块金效应模型
    ///
    /// 表示完全随机变异，无空间相关性。
    PureNugget { nugget: f64 },
}

impl VariogramModel {
    /// 创建球状模型
    pub fn spherical(nugget: f64, sill: f64, range: f64) -> Self {
        Self::Spherical {
            nugget,
            sill,
            range,
        }
    }

    /// 创建指数模型
    pub fn exponential(nugget: f64, sill: f64, range: f64) -> Self {
        Self::Exponential {
            nugget,
            sill,
            range,
        }
    }

    /// 创建高斯模型
    pub fn gaussian(nugget: f64, sill: f64, range: f64) -> Self {
        Self::Gaussian {
            nugget,
            sill,
            range,
        }
    }

    /// 创建默认球状模型
    pub fn default_spherical() -> Self {
        Self::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 100.0,
        }
    }

    /// 计算半变异函数值 γ(h)
    ///
    /// # 参数
    /// - `h`: 滞后距离
    ///
    /// # 返回
    /// 半变异函数值
    pub fn gamma(&self, h: f64) -> f64 {
        match *self {
            Self::Spherical {
                nugget,
                sill,
                range,
            } => {
                if h < 1e-10 {
                    0.0
                } else if h >= range {
                    nugget + sill
                } else {
                    let ratio = h / range;
                    nugget + sill * (1.5 * ratio - 0.5 * ratio.powi(3))
                }
            }
            Self::Exponential {
                nugget,
                sill,
                range,
            } => {
                if h < 1e-10 {
                    0.0
                } else {
                    nugget + sill * (1.0 - (-3.0 * h / range).exp())
                }
            }
            Self::Gaussian {
                nugget,
                sill,
                range,
            } => {
                if h < 1e-10 {
                    0.0
                } else {
                    nugget + sill * (1.0 - (-3.0 * (h / range).powi(2)).exp())
                }
            }
            Self::Linear { nugget, slope } => {
                if h < 1e-10 {
                    0.0
                } else {
                    nugget + slope * h
                }
            }
            Self::PureNugget { nugget } => {
                if h < 1e-10 {
                    0.0
                } else {
                    nugget
                }
            }
        }
    }

    /// 计算协方差 C(h)
    ///
    /// C(h) = C₀ + C - γ(h) = sill - γ(h) + nugget
    pub fn covariance(&self, h: f64) -> f64 {
        match *self {
            Self::Spherical { nugget, sill, .. }
            | Self::Exponential { nugget, sill, .. }
            | Self::Gaussian { nugget, sill, .. } => nugget + sill - self.gamma(h),
            Self::Linear { nugget, .. } => nugget - self.gamma(h),
            Self::PureNugget { nugget } => {
                if h < 1e-10 {
                    nugget
                } else {
                    0.0
                }
            }
        }
    }

    /// 获取基台值（总方差）
    pub fn sill_total(&self) -> f64 {
        match *self {
            Self::Spherical { nugget, sill, .. }
            | Self::Exponential { nugget, sill, .. }
            | Self::Gaussian { nugget, sill, .. } => nugget + sill,
            Self::Linear { nugget, .. } => nugget,
            Self::PureNugget { nugget } => nugget,
        }
    }
}

impl Default for VariogramModel {
    fn default() -> Self {
        Self::default_spherical()
    }
}

/// Kriging 插值器
///
/// 提供普通 Kriging (Ordinary Kriging) 插值功能。
pub struct KrigingInterpolator {
    /// 采样点坐标
    points: Vec<Point2D>,
    /// 采样点值
    values: Vec<f64>,
    /// 变异函数模型
    variogram: VariogramModel,
    /// 预计算的 Kriging 矩阵逆
    k_inv: Option<DMatrix<f64>>,
}

impl KrigingInterpolator {
    /// 创建新的 Kriging 插值器
    ///
    /// # 参数
    /// - `points`: 采样点坐标列表
    /// - `values`: 采样点值列表
    /// - `variogram`: 变异函数模型
    pub fn new(points: Vec<Point2D>, values: Vec<f64>, variogram: VariogramModel) -> Self {
        assert_eq!(
            points.len(),
            values.len(),
            "采样点数量必须等于值数量"
        );

        let mut interpolator = Self {
            points,
            values,
            variogram,
            k_inv: None,
        };

        interpolator.compute_kriging_matrix();
        interpolator
    }

    /// 更新变异函数模型
    pub fn set_variogram(&mut self, variogram: VariogramModel) {
        self.variogram = variogram;
        self.compute_kriging_matrix();
    }

    /// 获取变异函数模型
    pub fn variogram(&self) -> &VariogramModel {
        &self.variogram
    }

    /// 获取采样点数量
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// 计算 Kriging 矩阵并求逆
    fn compute_kriging_matrix(&mut self) {
        let n = self.points.len();
        if n == 0 {
            self.k_inv = None;
            return;
        }

        // 构建扩展的 Kriging 矩阵 (n+1) x (n+1)
        // 包含变异函数矩阵和 Lagrange 乘子
        let mut k = DMatrix::zeros(n + 1, n + 1);

        // 填充变异函数矩阵
        for i in 0..n {
            for j in 0..n {
                let h = self.distance(i, j);
                k[(i, j)] = self.variogram.gamma(h);
            }
            // Lagrange 乘子行/列
            k[(i, n)] = 1.0;
            k[(n, i)] = 1.0;
        }
        k[(n, n)] = 0.0;

        // 计算逆矩阵
        self.k_inv = k.try_inverse();

        if self.k_inv.is_none() {
            log::warn!("Kriging 矩阵奇异，无法求逆");
        }
    }

    /// 计算两个采样点之间的距离
    #[inline]
    fn distance(&self, i: usize, j: usize) -> f64 {
        let p1 = &self.points[i];
        let p2 = &self.points[j];
        ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt()
    }

    /// 在指定点插值
    ///
    /// # 返回
    /// 如果 Kriging 矩阵可逆，返回 Some(插值结果)；否则返回 None
    pub fn interpolate(&self, x: f64, y: f64) -> Option<f64> {
        let k_inv = self.k_inv.as_ref()?;
        let n = self.points.len();

        if n == 0 {
            return None;
        }

        // 构建右端向量 k₀
        let mut k0 = DVector::zeros(n + 1);
        for i in 0..n {
            let h = ((self.points[i].x - x).powi(2) + (self.points[i].y - y).powi(2)).sqrt();
            k0[i] = self.variogram.gamma(h);
        }
        k0[n] = 1.0;

        // 计算权重 w = K⁻¹ * k₀
        let weights = k_inv * k0;

        // 加权求和
        let mut result = 0.0;
        for i in 0..n {
            result += weights[i] * self.values[i];
        }

        Some(result)
    }

    /// 插值并返回 Kriging 方差
    ///
    /// # 返回
    /// (插值值, Kriging 方差)
    ///
    /// Kriging 方差可用于评估插值不确定性。
    pub fn interpolate_with_variance(&self, x: f64, y: f64) -> Option<(f64, f64)> {
        let k_inv = self.k_inv.as_ref()?;
        let n = self.points.len();

        if n == 0 {
            return None;
        }

        // 构建右端向量 k₀
        let mut k0 = DVector::zeros(n + 1);
        for i in 0..n {
            let h = ((self.points[i].x - x).powi(2) + (self.points[i].y - y).powi(2)).sqrt();
            k0[i] = self.variogram.gamma(h);
        }
        k0[n] = 1.0;

        // 计算权重
        let weights = k_inv * &k0;

        // 加权求和得到插值
        let mut result = 0.0;
        for i in 0..n {
            result += weights[i] * self.values[i];
        }

        // 计算 Kriging 方差
        // σ²_k = Σ wᵢ γ(xᵢ, x₀) + μ
        let variance = k0.dot(&weights).max(0.0);

        Some((result, variance))
    }

    /// 计算指定点的 Kriging 标准差
    pub fn standard_deviation(&self, x: f64, y: f64) -> Option<f64> {
        self.interpolate_with_variance(x, y)
            .map(|(_, var)| var.sqrt())
    }

    /// 批量插值
    pub fn interpolate_batch(&self, points: &[(f64, f64)]) -> Vec<Option<f64>> {
        points
            .iter()
            .map(|&(x, y)| self.interpolate(x, y))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_variogram() {
        let model = VariogramModel::Spherical {
            nugget: 0.1,
            sill: 0.9,
            range: 100.0,
        };

        // h=0 时应为 0
        assert!((model.gamma(0.0) - 0.0).abs() < 1e-10);

        // h=range 时应达到基台
        assert!((model.gamma(100.0) - 1.0).abs() < 1e-10);

        // h>range 时保持基台
        assert!((model.gamma(200.0) - 1.0).abs() < 1e-10);

        // 中间值应该在 0 和 1 之间
        let g50 = model.gamma(50.0);
        assert!(g50 > 0.0 && g50 < 1.0);
    }

    #[test]
    fn test_exponential_variogram() {
        let model = VariogramModel::Exponential {
            nugget: 0.0,
            sill: 1.0,
            range: 100.0,
        };

        assert!((model.gamma(0.0) - 0.0).abs() < 1e-10);

        // 指数模型渐近趋近基台
        let g_large = model.gamma(1000.0);
        assert!((g_large - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_gaussian_variogram() {
        let model = VariogramModel::Gaussian {
            nugget: 0.0,
            sill: 1.0,
            range: 100.0,
        };

        assert!((model.gamma(0.0) - 0.0).abs() < 1e-10);

        // 高斯模型也渐近趋近基台
        let g_large = model.gamma(500.0);
        assert!((g_large - 1.0).abs() < 0.05);
    }

    #[test]
    fn test_kriging_interpolation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];
        let values = vec![0.0, 1.0, 1.0, 2.0];

        let kriging = KrigingInterpolator::new(
            points,
            values,
            VariogramModel::default_spherical(),
        );

        // 中心点应该接近 1.0
        let result = kriging.interpolate(0.5, 0.5);
        assert!(result.is_some());
        let value = result.unwrap();
        assert!((value - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_kriging_at_sample_point() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
        ];
        let values = vec![5.0, 10.0];

        let kriging = KrigingInterpolator::new(
            points,
            values,
            VariogramModel::spherical(0.0, 1.0, 10.0),
        );

        // 在采样点附近应该接近采样值
        let result = kriging.interpolate(0.01, 0.0);
        assert!(result.is_some());
        assert!((result.unwrap() - 5.0).abs() < 0.5);
    }

    #[test]
    fn test_kriging_variance() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(10.0, 0.0),
        ];
        let values = vec![0.0, 10.0];

        let kriging = KrigingInterpolator::new(
            points,
            values,
            VariogramModel::spherical(0.0, 1.0, 20.0),
        );

        // 采样点附近方差应该较小
        let (_, var_near) = kriging.interpolate_with_variance(0.1, 0.0).unwrap();
        
        // 远离采样点方差应该较大
        let (_, var_far) = kriging.interpolate_with_variance(5.0, 5.0).unwrap();

        assert!(var_near < var_far);
    }

    #[test]
    fn test_kriging_empty() {
        let kriging = KrigingInterpolator::new(vec![], vec![], VariogramModel::default_spherical());
        assert!(kriging.interpolate(0.0, 0.0).is_none());
    }
}

