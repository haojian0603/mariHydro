// src-tauri/src/marihydro/physics/sources/implicit.rs

//! 隐式源项处理模块
//!
//! 提供隐式时间积分方法，用于处理刚性源项（如摩擦、扩散）。
//!
//! # 物理背景
//!
//! 某些源项（如底摩擦）在小水深时可能变得非常刚性，
//! 导致显式方法需要极小的时间步长。隐式处理可以避免这个问题。
//!
//! # 关联问题
//!
//! - P5-007: 隐式摩擦处理
//! - P5-017: 隐式扩散
//! - P5-018: 源项稳定性

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::state::StateAccess;
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

/// 隐式处理方法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImplicitMethod {
    /// 显式 Euler（不推荐用于刚性问题）
    Explicit,
    /// 隐式 Euler（一阶，无条件稳定）
    #[default]
    ImplicitEuler,
    /// Crank-Nicolson（二阶，A-稳定）
    CrankNicolson,
    /// 解析衰减（用于线性阻尼）
    AnalyticDecay,
}

/// 隐式源项配置
#[derive(Debug, Clone, Copy)]
pub struct ImplicitConfig {
    /// 处理方法
    pub method: ImplicitMethod,
    /// 最大迭代次数（用于非线性隐式）
    pub max_iterations: usize,
    /// 收敛容差
    pub tolerance: f64,
    /// 欠松弛因子
    pub relaxation: f64,
}

impl Default for ImplicitConfig {
    fn default() -> Self {
        Self {
            method: ImplicitMethod::default(),
            max_iterations: 5,
            tolerance: 1e-8,
            relaxation: 1.0,
        }
    }
}

/// 阻尼系数计算器
///
/// 将各种摩擦公式统一为阻尼系数 γ
pub trait DampingCoefficient {
    /// 计算阻尼系数 γ（单位：1/s）
    ///
    /// 动量方程中的阻尼项形式为 -(γ/h) * (hu, hv)
    fn compute_gamma(&self, h: f64, speed: f64, cell_idx: usize) -> f64;

    /// 名称
    fn name(&self) -> &'static str;
}

/// Manning 阻尼系数
#[derive(Debug, Clone)]
pub struct ManningDamping {
    g: f64,
    n: f64,      // 均匀 Manning 系数
    gn2: f64,    // 预计算 g*n²
}

impl ManningDamping {
    pub fn new(g: f64, n: f64) -> Self {
        Self {
            g,
            n,
            gn2: g * n * n,
        }
    }
}

impl DampingCoefficient for ManningDamping {
    fn compute_gamma(&self, h: f64, speed: f64, _cell_idx: usize) -> f64 {
        if h < 1e-6 {
            return 0.0;
        }
        // γ = g * n² * |u| / h^(4/3)
        self.gn2 * speed / h.powf(4.0 / 3.0)
    }

    fn name(&self) -> &'static str {
        "Manning"
    }
}

/// Chezy 阻尼系数
#[derive(Debug, Clone)]
pub struct ChezyDamping {
    g: f64,
    c: f64,      // Chezy 系数
    g_c2: f64,   // 预计算 g/C²
}

impl ChezyDamping {
    pub fn new(g: f64, c: f64) -> Self {
        Self {
            g,
            c,
            g_c2: g / (c * c),
        }
    }
}

impl DampingCoefficient for ChezyDamping {
    fn compute_gamma(&self, h: f64, speed: f64, _cell_idx: usize) -> f64 {
        if h < 1e-6 {
            return 0.0;
        }
        // γ = g * |u| / (C² * h)
        self.g_c2 * speed / h
    }

    fn name(&self) -> &'static str {
        "Chezy"
    }
}

/// 隐式动量衰减求解器
///
/// 求解 d(hu)/dt = -γ * hu
#[derive(Debug)]
pub struct ImplicitMomentumDecay {
    config: ImplicitConfig,
}

impl Default for ImplicitMomentumDecay {
    fn default() -> Self {
        Self {
            config: ImplicitConfig::default(),
        }
    }
}

impl ImplicitMomentumDecay {
    /// 创建求解器
    pub fn new(config: ImplicitConfig) -> Self {
        Self { config }
    }

    /// 应用隐式衰减到单个单元
    ///
    /// # 参数
    /// - `hu`, `hv`: 动量分量
    /// - `gamma`: 阻尼系数
    /// - `dt`: 时间步长
    ///
    /// # 返回
    /// 更新后的 (hu, hv)
    pub fn apply(&self, hu: f64, hv: f64, gamma: f64, dt: f64) -> (f64, f64) {
        if gamma.abs() < 1e-20 {
            return (hu, hv);
        }

        match self.config.method {
            ImplicitMethod::Explicit => {
                // du/dt = -γ*u  =>  u_new = u - dt*γ*u
                let factor = 1.0 - dt * gamma;
                (hu * factor, hv * factor)
            }

            ImplicitMethod::ImplicitEuler => {
                // (u_new - u)/dt = -γ*u_new  =>  u_new = u / (1 + dt*γ)
                let factor = 1.0 / (1.0 + dt * gamma);
                (hu * factor, hv * factor)
            }

            ImplicitMethod::CrankNicolson => {
                // (u_new - u)/dt = -γ*(u + u_new)/2
                // u_new = u * (1 - dt*γ/2) / (1 + dt*γ/2)
                let factor = (1.0 - 0.5 * dt * gamma) / (1.0 + 0.5 * dt * gamma);
                (hu * factor, hv * factor)
            }

            ImplicitMethod::AnalyticDecay => {
                // du/dt = -γ*u  =>  u(t) = u₀ * exp(-γ*t)
                let factor = (-gamma * dt).exp();
                (hu * factor, hv * factor)
            }
        }
    }

    /// 应用隐式衰减到向量
    pub fn apply_vec(&self, momentum: DVec2, gamma: f64, dt: f64) -> DVec2 {
        let (hu, hv) = self.apply(momentum.x, momentum.y, gamma, dt);
        DVec2::new(hu, hv)
    }

    /// 批量应用隐式衰减
    pub fn apply_batch<D: DampingCoefficient>(
        &self,
        h: &[f64],
        hu: &mut [f64],
        hv: &mut [f64],
        damping: &D,
        dt: f64,
        h_dry: f64,
    ) {
        for i in 0..h.len() {
            if h[i] < h_dry {
                // 干区：动量归零
                hu[i] = 0.0;
                hv[i] = 0.0;
                continue;
            }

            let h_safe = h[i].max(1e-6);
            let speed = ((hu[i] / h_safe).powi(2) + (hv[i] / h_safe).powi(2)).sqrt();
            let gamma = damping.compute_gamma(h[i], speed, i);

            let (new_hu, new_hv) = self.apply(hu[i], hv[i], gamma, dt);
            hu[i] = new_hu;
            hv[i] = new_hv;
        }
    }
}

/// 隐式扩散求解器（标量场）
///
/// 求解 dφ/dt = ν∇²φ
#[derive(Debug)]
pub struct ImplicitDiffusion {
    config: ImplicitConfig,
    /// 扩散系数
    pub diffusivity: f64,
}

impl ImplicitDiffusion {
    /// 创建求解器
    pub fn new(diffusivity: f64) -> Self {
        Self {
            config: ImplicitConfig::default(),
            diffusivity,
        }
    }

    /// 设置配置
    pub fn with_config(mut self, config: ImplicitConfig) -> Self {
        self.config = config;
        self
    }

    /// 应用隐式扩散（单步 Jacobi 迭代）
    ///
    /// 对于结构化网格，可以用这个简单实现。
    /// 非结构化网格需要更复杂的稀疏矩阵求解。
    pub fn apply_jacobi<M: MeshAccess>(
        &self,
        field: &mut [f64],
        mesh: &M,
        dt: f64,
    ) -> MhResult<()> {
        use crate::marihydro::core::types::CellIndex;

        let n = mesh.n_cells();
        let mut new_field = vec![0.0; n];

        for _ in 0..self.config.max_iterations {
            for i in 0..n {
                let cell = CellIndex(i);
                let area = mesh.cell_area(cell);
                let mut sum = 0.0;
                let mut coeff = 0.0;

                for &nb in mesh.cell_neighbors(cell) {
                    if nb.is_valid() {
                        // 简化：假设均匀网格
                        let dx = (mesh.cell_centroid(cell) - mesh.cell_centroid(nb)).length();
                        let flux_coeff = self.diffusivity / dx;
                        sum += flux_coeff * field[nb.0];
                        coeff += flux_coeff;
                    }
                }

                let alpha = dt * coeff / area;
                new_field[i] = (field[i] + dt * sum / area) / (1.0 + alpha);
            }

            // 更新
            field.copy_from_slice(&new_field);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_euler() {
        let solver = ImplicitMomentumDecay::new(ImplicitConfig {
            method: ImplicitMethod::ImplicitEuler,
            ..Default::default()
        });

        // γ = 1, dt = 1  =>  factor = 0.5
        let (hu, hv) = solver.apply(1.0, 0.5, 1.0, 1.0);
        assert!((hu - 0.5).abs() < 1e-10);
        assert!((hv - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_analytic_decay() {
        let solver = ImplicitMomentumDecay::new(ImplicitConfig {
            method: ImplicitMethod::AnalyticDecay,
            ..Default::default()
        });

        // γ = 1, dt = 1  =>  factor = e^(-1) ≈ 0.368
        let (hu, _) = solver.apply(1.0, 0.0, 1.0, 1.0);
        assert!((hu - (-1.0f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_crank_nicolson() {
        let solver = ImplicitMomentumDecay::new(ImplicitConfig {
            method: ImplicitMethod::CrankNicolson,
            ..Default::default()
        });

        // γ = 2, dt = 1  =>  factor = (1-1)/(1+1) = 0
        let (hu, _) = solver.apply(1.0, 0.0, 2.0, 1.0);
        assert!(hu.abs() < 1e-10);
    }

    #[test]
    fn test_manning_damping() {
        let damping = ManningDamping::new(9.81, 0.03);

        let h = 1.0;
        let speed = 1.0;
        let gamma = damping.compute_gamma(h, speed, 0);

        // γ = g * n² * |u| / h^(4/3) = 9.81 * 0.0009 * 1 / 1 = 0.008829
        assert!((gamma - 9.81 * 0.0009).abs() < 1e-6);
    }
}
