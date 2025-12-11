//! 垂向混合模型
//!
//! 实现垂向涡粘性/扩散系数的计算：
//! - 常数混合
//! - Pacanowski-Philander (1981)
//! - k-ε 模型驱动（需外部提供 k, ε）

use super::sigma::SigmaCoordinate;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// 垂向混合模型类型
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VerticalMixingModel {
    /// 常数混合系数
    Constant {
        /// 动量涡粘性 [m²/s]
        nu_v: f64,
        /// 标量扩散系数 [m²/s]
        kappa_v: f64,
    },
    /// Pacanowski-Philander (1981) 模型
    /// 
    /// ν_v = ν_0 + ν_max / (1 + α × Ri)^n
    PacanowskiPhilander {
        /// 背景粘性 [m²/s]
        nu_0: f64,
        /// 最大粘性增量 [m²/s]
        nu_max: f64,
        /// Richardson 数系数 α
        alpha: f64,
        /// 指数 n
        n: f64,
    },
    /// 由 k-ε 模型驱动
    /// 
    /// ν_v = c_μ × k² / ε
    KEpsilonDriven,
}

impl Default for VerticalMixingModel {
    fn default() -> Self {
        Self::Constant {
            nu_v: 1e-4,
            kappa_v: 1e-5,
        }
    }
}

/// 垂向混合计算器
pub struct VerticalMixing {
    /// σ坐标
    sigma: SigmaCoordinate,
    /// 单元数量
    n_cells: usize,
    /// 混合模型
    model: VerticalMixingModel,
    /// 动量涡粘性场 [m²/s]（n_cells × n_layers）
    nu_v: Vec<AlignedVec<f64>>,
    /// 标量扩散系数场 [m²/s]（n_cells × n_layers）
    kappa_v: Vec<AlignedVec<f64>>,
    /// Richardson 数（可选）
    ri: Option<Vec<AlignedVec<f64>>>,
}

impl VerticalMixing {
    /// 创建新的垂向混合计算器
    pub fn new(n_cells: usize, sigma: SigmaCoordinate, model: VerticalMixingModel) -> Self {
        let n_layers = sigma.n_layers();
        let nu_v = (0..n_layers).map(|_| AlignedVec::zeros(n_cells)).collect();
        let kappa_v = (0..n_layers).map(|_| AlignedVec::zeros(n_cells)).collect();

        let mut mixing = Self {
            sigma,
            n_cells,
            model,
            nu_v,
            kappa_v,
            ri: None,
        };

        // 初始化默认值
        mixing.initialize();
        mixing
    }

    /// 初始化混合系数
    fn initialize(&mut self) {
        match self.model {
            VerticalMixingModel::Constant { nu_v, kappa_v } => {
                for k in 0..self.sigma.n_layers() {
                    self.nu_v[k].fill(nu_v);
                    self.kappa_v[k].fill(kappa_v);
                }
            }
            VerticalMixingModel::PacanowskiPhilander { nu_0, .. } => {
                // 初始化为背景值
                for k in 0..self.sigma.n_layers() {
                    self.nu_v[k].fill(nu_0);
                    self.kappa_v[k].fill(nu_0 * 0.1); // 初始扩散系数
                }
                // 初始化 Ri 存储
                self.ri = Some(
                    (0..self.sigma.n_layers())
                        .map(|_| AlignedVec::zeros(self.n_cells))
                        .collect(),
                );
            }
            VerticalMixingModel::KEpsilonDriven => {
                // k-ε 驱动模式：等待外部更新
            }
        }
    }

    /// 更新混合系数（Pacanowski-Philander 模型）
    ///
    /// # 参数
    /// - `drho_dz`: 密度垂向梯度 [kg/m⁴]
    /// - `du_dz`: 速度垂向梯度 [1/s]
    pub fn update_pp(
        &mut self,
        drho_dz: &[&[f64]],
        du_dz: &[&[f64]],
        rho_0: f64,
        g: f64,
    ) {
        if let VerticalMixingModel::PacanowskiPhilander { nu_0, nu_max, alpha, n } = self.model {
            let n_layers = self.sigma.n_layers();

            for k in 0..n_layers.min(drho_dz.len()).min(du_dz.len()) {
                for cell in 0..self.n_cells.min(drho_dz[k].len()).min(du_dz[k].len()) {
                    // 计算 Richardson 数
                    // Ri = -(g/ρ₀) × (∂ρ/∂z) / (∂u/∂z)²
                    let shear_sq = du_dz[k][cell].powi(2).max(1e-10);
                    let buoyancy = -(g / rho_0) * drho_dz[k][cell];
                    let ri = buoyancy / shear_sq;

                    // 存储 Ri
                    if let Some(ref mut ri_field) = self.ri {
                        ri_field[k][cell] = ri;
                    }

                    // PP 公式
                    let denominator = (1.0 + alpha * ri.max(0.0)).powf(n);
                    let nu = nu_0 + nu_max / denominator;

                    self.nu_v[k][cell] = nu;
                    // 标量扩散使用 Prandtl 数 Pr ≈ 0.75
                    self.kappa_v[k][cell] = nu / 0.75;
                }
            }
        }
    }

    /// 从 k-ε 模型更新混合系数
    ///
    /// ν_v = c_μ × k² / ε
    pub fn update_from_k_epsilon(
        &mut self,
        k_field: &[&[f64]],
        epsilon_field: &[&[f64]],
        c_mu: f64,
    ) {
        let n_layers = self.sigma.n_layers();

        for layer in 0..n_layers.min(k_field.len()).min(epsilon_field.len()) {
            for cell in 0..self.n_cells.min(k_field[layer].len()) {
                let k = k_field[layer][cell].max(1e-12);
                let eps = epsilon_field[layer][cell].max(1e-12);

                let nu = c_mu * k * k / eps;
                self.nu_v[layer][cell] = nu.clamp(1e-7, 1.0);
                self.kappa_v[layer][cell] = nu / 0.9; // Pr_t ≈ 0.9
            }
        }
    }

    /// 获取特定层的涡粘性
    pub fn nu_v_at_layer(&self, k: usize) -> &[f64] {
        &self.nu_v[k]
    }

    /// 获取特定层的扩散系数
    pub fn kappa_v_at_layer(&self, k: usize) -> &[f64] {
        &self.kappa_v[k]
    }

    /// 获取特定单元、层的涡粘性
    pub fn nu_v(&self, cell: usize, k: usize) -> f64 {
        self.nu_v[k][cell]
    }

    /// 获取特定单元、层的扩散系数
    pub fn kappa_v(&self, cell: usize, k: usize) -> f64 {
        self.kappa_v[k][cell]
    }

    /// 层数
    pub fn n_layers(&self) -> usize {
        self.sigma.n_layers()
    }

    /// 单元数
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_mixing() {
        let sigma = SigmaCoordinate::uniform(5);
        let mixing = VerticalMixing::new(
            10,
            sigma,
            VerticalMixingModel::Constant {
                nu_v: 1e-3,
                kappa_v: 1e-4,
            },
        );

        for k in 0..5 {
            for cell in 0..10 {
                assert!((mixing.nu_v(cell, k) - 1e-3).abs() < 1e-10);
                assert!((mixing.kappa_v(cell, k) - 1e-4).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pp_model_creation() {
        let sigma = SigmaCoordinate::uniform(5);
        let mixing = VerticalMixing::new(
            10,
            sigma,
            VerticalMixingModel::PacanowskiPhilander {
                nu_0: 1e-4,
                nu_max: 1e-2,
                alpha: 5.0,
                n: 2.0,
            },
        );

        // 应该有 Ri 存储
        assert!(mixing.ri.is_some());
    }
}
