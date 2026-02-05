//! 垂向混合模型
//!
//! 实现垂向涡粘性/扩散系数的计算：
//! - 常数混合
//! - Pacanowski-Philander (1981)
//! - k-ε 模型驱动（需外部提供 k, ε）

use super::sigma::SigmaCoordinate;
use mh_runtime::{Backend, DeviceBuffer, RuntimeScalar};
use num_traits::Float;
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
pub struct VerticalMixing<B: Backend> {
    /// σ坐标
    sigma: SigmaCoordinate,
    /// 单元数量
    n_cells: usize,
    /// 混合模型
    model: VerticalMixingModel,
    /// 动量涡粘性场 [m²/s]（n_cells × n_layers）
    nu_v: Vec<B::Buffer<B::Scalar>>,
    /// 标量扩散系数场 [m²/s]（n_cells × n_layers）
    kappa_v: Vec<B::Buffer<B::Scalar>>,
    /// Richardson 数（可选）
    ri: Option<Vec<B::Buffer<B::Scalar>>>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> VerticalMixing<B> {
    /// 创建新的垂向混合计算器
    pub fn new_with_backend(
        backend: B,
        n_cells: usize,
        sigma: SigmaCoordinate,
        model: VerticalMixingModel,
    ) -> Self {
        let n_layers = sigma.n_layers();
        let mut nu_v: Vec<B::Buffer<B::Scalar>> =
            (0..n_layers).map(|_| backend.alloc(n_cells)).collect();
        let mut kappa_v: Vec<B::Buffer<B::Scalar>> =
            (0..n_layers).map(|_| backend.alloc(n_cells)).collect();
        for layer in &mut nu_v {
            layer.fill(<B::Scalar as RuntimeScalar>::ZERO);
        }
        for layer in &mut kappa_v {
            layer.fill(<B::Scalar as RuntimeScalar>::ZERO);
        }

        let mut mixing = Self {
            sigma,
            n_cells,
            model,
            nu_v,
            kappa_v,
            ri: None,
            backend,
        };

        // 初始化默认值
        mixing.initialize();
        mixing
    }

    /// 初始化混合系数
    fn initialize(&mut self) {
        match self.model {
            VerticalMixingModel::Constant { nu_v, kappa_v } => {
                let nu_v = self.backend.scalar_from_f64(nu_v);
                let kappa_v = self.backend.scalar_from_f64(kappa_v);
                for k in 0..self.sigma.n_layers() {
                    self.nu_v[k].fill(nu_v);
                    self.kappa_v[k].fill(kappa_v);
                }
            }
            VerticalMixingModel::PacanowskiPhilander { nu_0, .. } => {
                let nu_0 = self.backend.scalar_from_f64(nu_0);
                let kappa_0 = self.backend.scalar_from_f64(0.1) * nu_0;
                // 初始化为背景值
                for k in 0..self.sigma.n_layers() {
                    self.nu_v[k].fill(nu_0);
                    self.kappa_v[k].fill(kappa_0); // 初始扩散系数
                }
                // 初始化 Ri 存储
                self.ri = Some(
                    (0..self.sigma.n_layers())
                        .map(|_| self.backend.alloc(self.n_cells))
                        .collect(),
                );
                if let Some(ref mut ri_field) = self.ri {
                    for layer in ri_field.iter_mut() {
                        layer.fill(<B::Scalar as RuntimeScalar>::ZERO);
                    }
                }
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
        drho_dz: &[&B::Buffer<B::Scalar>],
        du_dz: &[&B::Buffer<B::Scalar>],
        rho_0: f64,
        g: f64,
    ) {
        if let VerticalMixingModel::PacanowskiPhilander { nu_0, nu_max, alpha, n } = self.model {
            let rho_0 = self.backend.scalar_from_f64(rho_0);
            let g = self.backend.scalar_from_f64(g);
            let nu_0 = self.backend.scalar_from_f64(nu_0);
            let nu_max = self.backend.scalar_from_f64(nu_max);
            let alpha = self.backend.scalar_from_f64(alpha);
            let n = self.backend.scalar_from_f64(n);
            let zero = <B::Scalar as RuntimeScalar>::ZERO;
            let one = <B::Scalar as RuntimeScalar>::ONE;
            let min_shear = <B::Scalar as RuntimeScalar>::from_config(1e-10).unwrap_or(zero);
            let pr = self.backend.scalar_from_f64(0.75);
            let kappa_0 = self.backend.scalar_from_f64(0.1) * nu_0;
            let n_layers = self.sigma.n_layers();

            for k in 0..n_layers.min(drho_dz.len()).min(du_dz.len()) {
                for cell in 0..self.n_cells.min(drho_dz[k].len()).min(du_dz[k].len()) {
                    // 计算 Richardson 数
                    // Ri = -(g/ρ₀) × (∂ρ/∂z) / (∂u/∂z)²
                    let du = du_dz[k][cell];
                    let drho = drho_dz[k][cell];
                    if !du.is_finite() || !drho.is_finite() {
                        self.nu_v[k][cell] = nu_0;
                        self.kappa_v[k][cell] = kappa_0;
                        continue;
                    }
                    let shear_sq = (du * du).max(min_shear);
                    let buoyancy = -(g / rho_0) * drho;
                    let mut ri = buoyancy / shear_sq;
                    if !ri.is_finite() {
                        ri = zero;
                    }

                    // 存储 Ri
                    if let Some(ref mut ri_field) = self.ri {
                        ri_field[k][cell] = ri;
                    }

                    // PP 公式
                    let denominator = (one + alpha * ri.max(zero)).powf(n);
                    let nu = nu_0 + nu_max / denominator;

                    self.nu_v[k][cell] = nu;
                    // 标量扩散使用 Prandtl 数 Pr ≈ 0.75
                    self.kappa_v[k][cell] = nu / pr;
                }
            }
        }
    }

    /// 从 k-ε 模型更新混合系数
    ///
    /// ν_v = c_μ × k² / ε
    pub fn update_from_k_epsilon(
        &mut self,
        k_field: &[&B::Buffer<B::Scalar>],
        epsilon_field: &[&B::Buffer<B::Scalar>],
        c_mu: f64,
    ) {
        let c_mu = self.backend.scalar_from_f64(c_mu);
        let min_val = <B::Scalar as RuntimeScalar>::from_config(1e-12)
            .unwrap_or(<B::Scalar as RuntimeScalar>::ZERO);
        let min_nu = self.backend.scalar_from_f64(1e-7);
        let max_nu = self.backend.scalar_from_f64(1.0);
        let pr_t = self.backend.scalar_from_f64(0.9);
        let n_layers = self.sigma.n_layers();

        for layer in 0..n_layers.min(k_field.len()).min(epsilon_field.len()) {
            for cell in 0..self.n_cells.min(k_field[layer].len()) {
                let k_raw = k_field[layer][cell];
                let eps_raw = epsilon_field[layer][cell];
                if !k_raw.is_finite() || !eps_raw.is_finite() {
                    continue;
                }
                let k = k_raw.max(min_val);
                let eps = eps_raw.max(min_val);

                let nu = c_mu * k * k / eps;
                self.nu_v[layer][cell] = nu.max(min_nu).min(max_nu);
                self.kappa_v[layer][cell] = nu / pr_t; // Pr_t ≈ 0.9
            }
        }
    }

    /// 获取特定层的 Richardson 数（仅 PP 模型可用）
    pub fn ri_at_layer(&self, k: usize) -> Option<&B::Buffer<B::Scalar>> {
        self.ri.as_ref().and_then(|ri| ri.get(k))
    }

    /// 是否启用 Richardson 数存储
    pub fn has_ri(&self) -> bool {
        self.ri.is_some()
    }

    /// 获取特定层的涡粘性
    pub fn nu_v_at_layer(&self, k: usize) -> &B::Buffer<B::Scalar> {
        &self.nu_v[k]
    }

    /// 获取特定层的扩散系数
    pub fn kappa_v_at_layer(&self, k: usize) -> &B::Buffer<B::Scalar> {
        &self.kappa_v[k]
    }

    /// 获取特定单元、层的涡粘性
    pub fn nu_v(&self, cell: usize, k: usize) -> B::Scalar {
        self.nu_v[k][cell]
    }

    /// 获取特定单元、层的扩散系数
    pub fn kappa_v(&self, cell: usize, k: usize) -> B::Scalar {
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
    use mh_runtime::CpuBackend;

    #[test]
    fn test_constant_mixing() {
        let sigma = SigmaCoordinate::uniform(5);
        let backend = CpuBackend::<f64>::new();
        let mixing = VerticalMixing::new_with_backend(
            backend,
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
        let backend = CpuBackend::<f64>::new();
        let mixing = VerticalMixing::new_with_backend(
            backend,
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
