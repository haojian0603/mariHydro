//! marihydro\crates\mh_physics\src\vertical\sigma.rs
//! σ坐标系统
//!
//! 实现地形跟随的 σ 坐标用于垂向分层：
//! - σ = 0 在水面
//! - σ = -1 在底床
//!
//! # 分布类型
//!
//! - 均匀分布：等间距层
//! - 对数分布：底部加密（边界层解析）
//! - 双曲正切：表层+底层加密

use serde::{Deserialize, Serialize};

/// σ层分布类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[derive(Default)]
pub enum SigmaDistribution {
    /// 均匀分布
    #[default]
    Uniform,
    /// 对数分布（底部加密）
    Logarithmic {
        /// 加密因子（>1加密底部）
        factor: f64,
    },
    /// 双曲正切分布（表层+底层加密）
    DoubleTanh {
        /// 表层加密参数
        surface_param: f64,
        /// 底层加密参数
        bottom_param: f64,
    },
}


/// σ坐标定义
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SigmaCoordinate {
    /// 层数
    n_layers: usize,
    /// 层界面的 σ 值（长度 = n_layers + 1）
    /// sigma_w[0] = 0 (水面), sigma_w[n_layers] = -1 (底床)
    sigma_w: Vec<f64>,
    /// 层中心的 σ 值（长度 = n_layers）
    sigma_c: Vec<f64>,
    /// 分布类型
    distribution: SigmaDistribution,
}

impl SigmaCoordinate {
    /// 创建均匀分布的 σ 层
    pub fn uniform(n_layers: usize) -> Self {
        Self::new(n_layers, SigmaDistribution::Uniform)
    }

    /// 创建对数分布的 σ 层（底部加密）
    pub fn logarithmic(n_layers: usize, factor: f64) -> Self {
        Self::new(n_layers, SigmaDistribution::Logarithmic { factor })
    }

    /// 创建双曲正切分布（表层+底层加密）
    pub fn double_tanh(n_layers: usize, surface_param: f64, bottom_param: f64) -> Self {
        Self::new(n_layers, SigmaDistribution::DoubleTanh { surface_param, bottom_param })
    }

    /// 根据分布类型创建 σ 坐标
    pub fn new(n_layers: usize, distribution: SigmaDistribution) -> Self {
        assert!(n_layers >= 1, "至少需要1层");

        let mut sigma_w = vec![0.0; n_layers + 1];
        let mut sigma_c = vec![0.0; n_layers];

        match distribution {
            SigmaDistribution::Uniform => {
                // 均匀分布
                for i in 0..=n_layers {
                    sigma_w[i] = -(i as f64) / (n_layers as f64);
                }
            }
            SigmaDistribution::Logarithmic { factor } => {
                // 对数分布（底部加密）
                let f = factor.max(1.0);
                for i in 0..=n_layers {
                    let xi = (i as f64) / (n_layers as f64);
                    // σ = -(e^(f*ξ) - 1) / (e^f - 1)
                    sigma_w[i] = -((f * xi).exp() - 1.0) / (f.exp() - 1.0);
                }
            }
            SigmaDistribution::DoubleTanh { surface_param, bottom_param } => {
                // 双曲正切分布
                let ts = surface_param;
                let tb = bottom_param;
                for i in 0..=n_layers {
                    let xi = (i as f64) / (n_layers as f64);
                    // 组合表层和底层加密
                    let s = ((xi - 1.0) * ts).tanh() / ts.tanh();
                    let b = (xi * tb).tanh() / tb.tanh();
                    sigma_w[i] = -0.5 * (s + b);
                }
            }
        }

        // 计算层中心
        for k in 0..n_layers {
            sigma_c[k] = 0.5 * (sigma_w[k] + sigma_w[k + 1]);
        }

        Self {
            n_layers,
            sigma_w,
            sigma_c,
            distribution,
        }
    }

    /// 层数
    #[inline]
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }

    /// 层界面的 σ 值
    #[inline]
    pub fn sigma_at_interface(&self, k: usize) -> f64 {
        self.sigma_w[k]
    }

    /// 层中心的 σ 值
    #[inline]
    pub fn sigma_at_center(&self, k: usize) -> f64 {
        self.sigma_c[k]
    }

    /// 层厚度（无量纲）
    #[inline]
    pub fn layer_thickness_sigma(&self, k: usize) -> f64 {
        (self.sigma_w[k] - self.sigma_w[k + 1]).abs()
    }

    /// 层厚度（有量纲）
    #[inline]
    pub fn layer_thickness(&self, k: usize, water_depth: f64) -> f64 {
        self.layer_thickness_sigma(k) * water_depth
    }

    /// 从 σ 值计算实际深度
    #[inline]
    pub fn depth_from_sigma(&self, sigma: f64, water_depth: f64, eta: f64) -> f64 {
        eta + sigma * water_depth
    }

    /// 从实际深度计算 σ 值
    #[inline]
    pub fn sigma_from_depth(&self, z: f64, water_depth: f64, eta: f64) -> f64 {
        if water_depth < 1e-10 {
            return 0.0;
        }
        (z - eta) / water_depth
    }

    /// 层界面的 σ 值切片
    pub fn sigma_interfaces(&self) -> &[f64] {
        &self.sigma_w
    }

    /// 层中心的 σ 值切片
    pub fn sigma_centers(&self) -> &[f64] {
        &self.sigma_c
    }

    /// 分布类型
    pub fn distribution(&self) -> SigmaDistribution {
        self.distribution
    }
}

impl Default for SigmaCoordinate {
    fn default() -> Self {
        Self::uniform(10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_sigma() {
        let sigma = SigmaCoordinate::uniform(5);
        
        assert_eq!(sigma.n_layers(), 5);
        assert!((sigma.sigma_at_interface(0) - 0.0).abs() < 1e-10);
        assert!((sigma.sigma_at_interface(5) - (-1.0)).abs() < 1e-10);
        
        // 均匀层厚
        for k in 0..5 {
            assert!((sigma.layer_thickness_sigma(k) - 0.2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_logarithmic_sigma() {
        let sigma = SigmaCoordinate::logarithmic(10, 2.0);
        
        // 底部层应该更薄
        let top_thickness = sigma.layer_thickness_sigma(0);
        let bottom_thickness = sigma.layer_thickness_sigma(9);
        assert!(bottom_thickness > top_thickness);
    }

    #[test]
    fn test_depth_conversion() {
        let sigma = SigmaCoordinate::uniform(10);
        let h = 10.0;  // 水深
        let eta = 5.0; // 水位
        
        // 水面
        let z_surface = sigma.depth_from_sigma(0.0, h, eta);
        assert!((z_surface - 5.0).abs() < 1e-10);
        
        // 底床
        let z_bottom = sigma.depth_from_sigma(-1.0, h, eta);
        assert!((z_bottom - (-5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_layer_thickness() {
        let sigma = SigmaCoordinate::uniform(10);
        let h = 5.0;
        
        // 每层厚度应该是 0.5m
        for k in 0..10 {
            assert!((sigma.layer_thickness(k, h) - 0.5).abs() < 1e-10);
        }
    }
}
