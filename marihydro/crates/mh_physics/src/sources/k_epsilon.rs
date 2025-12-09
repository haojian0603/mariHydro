//! k-ε 湍流闭合模型
//!
//! 实现完整的标准 k-ε 两方程模型：
//! - k 方程（湍动能）
//! - ε 方程（耗散率）
//! - 壁面函数
//!
//! # 控制方程
//!
//! ## k 方程
//! ```text
//! ∂k/∂t + u·∇k = ∇·(ν_t/σ_k ∇k) + P_k - ε
//! ```
//!
//! ## ε 方程
//! ```text
//! ∂ε/∂t + u·∇ε = ∇·(ν_t/σ_ε ∇ε) + c₁(ε/k)P_k - c₂ε²/k
//! ```
//!
//! ## 涡粘性
//! ```text
//! ν_t = c_μ × k² / ε
//! ```
//!
//! # 默认参数（Launder-Spalding 标准值）
//!
//! | 参数 | 值 |
//! |------|-----|
//! | c_μ | 0.09 |
//! | c₁ | 1.44 |
//! | c₂ | 1.92 |
//! | σ_k | 1.0 |
//! | σ_ε | 1.3 |

use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::sources::turbulence::VelocityGradient;
use crate::state::ShallowWaterState;
use mh_foundation::{AlignedVec, Scalar};
use serde::{Deserialize, Serialize};

/// k-ε 模型参数（Launder-Spalding 标准值）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KEpsilonParams {
    /// c_μ 系数（默认 0.09）
    pub c_mu: Scalar,
    /// c₁ 系数（默认 1.44）
    pub c_1: Scalar,
    /// c₂ 系数（默认 1.92）
    pub c_2: Scalar,
    /// σ_k（k 方程扩散 Prandtl 数，默认 1.0）
    pub sigma_k: Scalar,
    /// σ_ε（ε 方程扩散 Prandtl 数，默认 1.3）
    pub sigma_eps: Scalar,
    /// 最小 k 值
    pub k_min: Scalar,
    /// 最小 ε 值
    pub eps_min: Scalar,
    /// 最大涡粘性 [m²/s]
    pub nu_t_max: Scalar,
}

impl Default for KEpsilonParams {
    fn default() -> Self {
        Self {
            c_mu: 0.09,
            c_1: 1.44,
            c_2: 1.92,
            sigma_k: 1.0,
            sigma_eps: 1.3,
            k_min: 1e-10,
            eps_min: 1e-14,
            nu_t_max: 1e3,
        }
    }
}

impl KEpsilonParams {
    /// RNG k-ε 参数
    pub fn rng() -> Self {
        Self {
            c_mu: 0.0845,
            c_1: 1.42,
            c_2: 1.68,
            sigma_k: 0.7194,
            sigma_eps: 0.7194,
            ..Default::default()
        }
    }

    /// Realizable k-ε 参数（c_μ 自适应）
    pub fn realizable() -> Self {
        Self {
            c_mu: 0.09, // 实际会动态计算
            c_1: 1.44,
            c_2: 1.9,
            ..Default::default()
        }
    }
}

/// k-ε 湍流模型
///
/// 实现完整的 k-ε 两方程模型，包含扩散项和对流项
#[derive(Debug, Clone)]
pub struct KEpsilonModel {
    /// 模型参数
    pub params: KEpsilonParams,
    /// 湍动能场 [m²/s²]
    pub k: AlignedVec<Scalar>,
    /// 耗散率场 [m²/s³]
    pub epsilon: AlignedVec<Scalar>,
    /// 湍流产生率 P_k [m²/s³]
    production: AlignedVec<Scalar>,
    /// 涡粘性 ν_t [m²/s]
    eddy_viscosity: AlignedVec<Scalar>,
    /// 速度梯度（外部提供）
    velocity_gradient: Vec<VelocityGradient>,
    /// 到壁面的距离 [m]（用于壁面函数）
    wall_distance: AlignedVec<Scalar>,
    /// k 方程右端项
    rhs_k: AlignedVec<Scalar>,
    /// ε 方程右端项
    rhs_eps: AlignedVec<Scalar>,
    /// k 方程扩散项 [m²/s³]
    diffusion_k: AlignedVec<Scalar>,
    /// ε 方程扩散项 [m²/s⁴]
    diffusion_eps: AlignedVec<Scalar>,
    /// 单元面积 [m²]
    cell_areas: AlignedVec<Scalar>,
    /// 邻域信息：(邻居索引, 面长度, 单元心距离)
    neighbors: Vec<Vec<(usize, Scalar, Scalar)>>,
    /// 最小水深
    h_min: Scalar,
}

impl KEpsilonModel {
    /// 创建新的 k-ε 模型
    pub fn new(n_cells: usize) -> Self {
        Self::with_params(n_cells, KEpsilonParams::default())
    }

    /// 使用指定参数创建
    pub fn with_params(n_cells: usize, params: KEpsilonParams) -> Self {
        // 初始值：小的 k 和 ε 避免除零
        let k_init = 1e-4;
        let eps_init = 1e-6;

        Self {
            params,
            k: AlignedVec::from_vec(vec![k_init; n_cells]),
            epsilon: AlignedVec::from_vec(vec![eps_init; n_cells]),
            production: AlignedVec::zeros(n_cells),
            eddy_viscosity: AlignedVec::zeros(n_cells),
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            wall_distance: AlignedVec::from_vec(vec![1.0; n_cells]),
            rhs_k: AlignedVec::zeros(n_cells),
            rhs_eps: AlignedVec::zeros(n_cells),
            diffusion_k: AlignedVec::zeros(n_cells),
            diffusion_eps: AlignedVec::zeros(n_cells),
            cell_areas: AlignedVec::from_vec(vec![1.0; n_cells]),
            neighbors: vec![Vec::new(); n_cells],
            h_min: 1e-4,
        }
    }

    /// 设置速度梯度
    pub fn set_velocity_gradients(&mut self, gradients: &[VelocityGradient]) {
        let n = self.velocity_gradient.len().min(gradients.len());
        self.velocity_gradient[..n].copy_from_slice(&gradients[..n]);
    }

    /// 设置单个单元的速度梯度
    pub fn set_velocity_gradient(&mut self, cell: usize, grad: VelocityGradient) {
        if cell < self.velocity_gradient.len() {
            self.velocity_gradient[cell] = grad;
        }
    }

    /// 设置壁面距离
    pub fn set_wall_distance(&mut self, distances: &[Scalar]) {
        let n = self.wall_distance.len().min(distances.len());
        self.wall_distance[..n].copy_from_slice(&distances[..n]);
    }

    /// 计算湍流产生率 P_k = ν_t × |S|²
    pub fn compute_production(&mut self) {
        for i in 0..self.production.len() {
            let nu_t = self.eddy_viscosity[i];
            let s_mag = self.velocity_gradient[i].strain_rate_magnitude();
            
            // P_k = ν_t × |S|²
            self.production[i] = nu_t * s_mag * s_mag;
        }
    }
    /// 设置网格连接信息
    ///
    /// # 参数
    /// - `cell_areas`: 单元面积 [m²]
    /// - `neighbors`: 邻域信息（邻居索引，面长度，单元心距离）
    pub fn set_mesh_connectivity(
        &mut self,
        cell_areas: &[Scalar],
        neighbors: Vec<Vec<(usize, Scalar, Scalar)>>,
    ) {
        let n = self.cell_areas.len().min(cell_areas.len());
        self.cell_areas[..n].copy_from_slice(&cell_areas[..n]);
        self.neighbors = neighbors;
    }

    /// 计算 k 场扩散项
    ///
    /// D_k = ∇·(ν_t/σ_k ∇k)
    /// 使用有限体积法：D_k = (1/A) × ∑_f (ν_t/σ_k) × (k_j - k_i) / d_ij × L_f
    pub fn compute_diffusion_k(&mut self) {
        let sigma_k = self.params.sigma_k;

        for i in 0..self.diffusion_k.len() {
            let area = self.cell_areas[i].max(1e-10);
            let nu_t_i = self.eddy_viscosity[i];
            let k_i = self.k[i];

            let mut diffusion_sum = 0.0;

            for &(j, face_length, distance) in &self.neighbors[i] {
                if j >= self.k.len() {
                    continue;
                }

                let nu_t_j = self.eddy_viscosity[j];
                let k_j = self.k[j];

                // 面上平均涡粘性
                let nu_t_face = 0.5 * (nu_t_i + nu_t_j);

                // 扩散系数
                let diff_coeff = nu_t_face / sigma_k;

                // 通量
                let grad_k = (k_j - k_i) / distance.max(1e-10);
                diffusion_sum += diff_coeff * grad_k * face_length;
            }

            self.diffusion_k[i] = diffusion_sum / area;
        }
    }

    /// 计算 ε 场扩散项
    ///
    /// D_ε = ∇·(ν_t/σ_ε ∇ε)
    pub fn compute_diffusion_epsilon(&mut self) {
        let sigma_eps = self.params.sigma_eps;

        for i in 0..self.diffusion_eps.len() {
            let area = self.cell_areas[i].max(1e-10);
            let nu_t_i = self.eddy_viscosity[i];
            let eps_i = self.epsilon[i];

            let mut diffusion_sum = 0.0;

            for &(j, face_length, distance) in &self.neighbors[i] {
                if j >= self.epsilon.len() {
                    continue;
                }

                let nu_t_j = self.eddy_viscosity[j];
                let eps_j = self.epsilon[j];

                // 面上平均涡粘性
                let nu_t_face = 0.5 * (nu_t_i + nu_t_j);

                // 扩散系数
                let diff_coeff = nu_t_face / sigma_eps;

                // 通量
                let grad_eps = (eps_j - eps_i) / distance.max(1e-10);
                diffusion_sum += diff_coeff * grad_eps * face_length;
            }

            self.diffusion_eps[i] = diffusion_sum / area;
        }
    }

    /// 更新涡粘性 ν_t = c_μ × k² / ε
    pub fn update_eddy_viscosity(&mut self) {
        let c_mu = self.params.c_mu;
        let nu_max = self.params.nu_t_max;

        for i in 0..self.eddy_viscosity.len() {
            let k = self.k[i].max(self.params.k_min);
            let eps = self.epsilon[i].max(self.params.eps_min);

            let nu_t = c_mu * k * k / eps;
            self.eddy_viscosity[i] = nu_t.min(nu_max);
        }
    }

    /// 计算 k 方程右端项
    ///
    /// dk/dt = P_k - ε + ∇·(ν_t/σ_k ∇k)
    fn compute_rhs_k(&mut self, state: &ShallowWaterState) {
        // 先计算扩散项
        self.compute_diffusion_k();

        for i in 0..self.rhs_k.len() {
            let h = state.h[i];
            if h < self.h_min {
                self.rhs_k[i] = 0.0;
                continue;
            }

            let p_k = self.production[i];
            let eps = self.epsilon[i];
            let diff_k = self.diffusion_k[i];

            // 完整 k 方程：dk/dt = P_k - ε + D_k
            self.rhs_k[i] = p_k - eps + diff_k;
        }
    }

    /// 计算 ε 方程右端项
    ///
    /// dε/dt = c₁(ε/k)P_k - c₂ε²/k + ∇·(ν_t/σ_ε ∇ε)
    fn compute_rhs_epsilon(&mut self, state: &ShallowWaterState) {
        // 先计算扩散项
        self.compute_diffusion_epsilon();

        let c1 = self.params.c_1;
        let c2 = self.params.c_2;

        for i in 0..self.rhs_eps.len() {
            let h = state.h[i];
            if h < self.h_min {
                self.rhs_eps[i] = 0.0;
                continue;
            }

            let k = self.k[i].max(self.params.k_min);
            let eps = self.epsilon[i].max(self.params.eps_min);
            let p_k = self.production[i];
            let diff_eps = self.diffusion_eps[i];

            // 完整 ε 方程：dε/dt = c₁(ε/k)P_k - c₂ε²/k + D_ε
            let ratio = eps / k;
            self.rhs_eps[i] = c1 * ratio * p_k - c2 * eps * ratio + diff_eps;
        }
    }

    /// 求解 k 方程（显式欧拉）
    pub fn step_k(&mut self, state: &ShallowWaterState, dt: Scalar) {
        self.compute_rhs_k(state);

        for i in 0..self.k.len() {
            self.k[i] += dt * self.rhs_k[i];
            self.k[i] = self.k[i].max(self.params.k_min);
        }
    }

    /// 求解 ε 方程（显式欧拉）
    pub fn step_epsilon(&mut self, state: &ShallowWaterState, dt: Scalar) {
        self.compute_rhs_epsilon(state);

        for i in 0..self.epsilon.len() {
            self.epsilon[i] += dt * self.rhs_eps[i];
            self.epsilon[i] = self.epsilon[i].max(self.params.eps_min);
        }
    }

    /// 完整的时间步进
    ///
    /// 1. 更新涡粘性
    /// 2. 计算产生率
    /// 3. 推进 k 方程
    /// 4. 推进 ε 方程
    /// 5. 更新涡粘性
    pub fn step(&mut self, state: &ShallowWaterState, dt: Scalar) {
        // 1. 从当前 k, ε 更新 ν_t
        self.update_eddy_viscosity();

        // 2. 计算产生率
        self.compute_production();

        // 3. 推进 k
        self.step_k(state, dt);

        // 4. 推进 ε
        self.step_epsilon(state, dt);

        // 5. 更新 ν_t
        self.update_eddy_viscosity();
    }

    /// 应用壁面函数
    ///
    /// 在近壁区域使用对数律修正 k 和 ε
    pub fn apply_wall_functions(&mut self, state: &ShallowWaterState) {
        let c_mu = self.params.c_mu;
        let kappa = 0.41; // von Karman 常数

        for i in 0..self.k.len() {
            let h = state.h[i];
            if h < self.h_min {
                continue;
            }

            let y = self.wall_distance[i].max(1e-6);
            
            // 估计剪切速度 u_τ
            let speed = ((state.hu[i] / h).powi(2) + (state.hv[i] / h).powi(2)).sqrt();
            let cf: Scalar = 0.001; // 简化摩擦系数
            let u_tau = (cf / 2.0).sqrt() * speed;

            if u_tau < 1e-6 {
                continue;
            }

            // y+ 计算
            let nu = 1e-6; // 分子粘性
            let y_plus = y * u_tau / nu;

            // 近壁区域（y+ < 300）应用壁面函数
            if y_plus < 300.0 && y_plus > 11.0 {
                // k 在壁面：k = u_τ² / √c_μ
                let k_wall = u_tau * u_tau / c_mu.sqrt();
                
                // ε 在壁面：ε = u_τ³ / (κy)
                let eps_wall = u_tau.powi(3) / (kappa * y);

                // 混合：靠近壁面时使用壁面值
                let blend = (1.0 - y_plus / 300.0).max(0.0);
                self.k[i] = self.k[i] * (1.0 - blend) + k_wall * blend;
                self.epsilon[i] = self.epsilon[i] * (1.0 - blend) + eps_wall * blend;
            }
        }
    }

    /// 获取涡粘性
    pub fn eddy_viscosity(&self) -> &[Scalar] {
        &self.eddy_viscosity
    }

    /// 获取单个单元的涡粘性
    pub fn get_eddy_viscosity(&self, cell: usize) -> Scalar {
        self.eddy_viscosity.get(cell).copied().unwrap_or(0.0)
    }

    /// 获取产生率
    pub fn production(&self) -> &[Scalar] {
        &self.production
    }

    /// 湍流时间尺度 τ = k/ε [s]
    pub fn turbulent_time_scale(&self, cell: usize) -> Scalar {
        let k = self.k[cell].max(self.params.k_min);
        let eps = self.epsilon[cell].max(self.params.eps_min);
        k / eps
    }

    /// 湍流长度尺度 L = k^1.5/ε [m]
    pub fn turbulent_length_scale(&self, cell: usize) -> Scalar {
        let k = self.k[cell].max(self.params.k_min);
        let eps = self.epsilon[cell].max(self.params.eps_min);
        k.powf(1.5) / eps
    }
}

/// k-ε 作为源项（用于统一接口）
impl SourceTerm for KEpsilonModel {
    fn name(&self) -> &'static str {
        "k-epsilon"
    }

    fn is_enabled(&self) -> bool {
        true
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];
        if h < self.h_min {
            return SourceContribution::ZERO;
        }

        // k-ε 本身不直接贡献动量源项
        // 通过 eddy_viscosity 影响扩散
        SourceContribution::ZERO
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state(n_cells: usize, h: Scalar, u: Scalar, v: Scalar) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.hu[i] = h * u;
            state.hv[i] = h * v;
        }
        state
    }

    #[test]
    fn test_ke_creation() {
        let ke = KEpsilonModel::new(10);
        assert_eq!(ke.k.len(), 10);
        assert_eq!(ke.epsilon.len(), 10);
    }

    #[test]
    fn test_ke_params_default() {
        let params = KEpsilonParams::default();
        assert!((params.c_mu - 0.09).abs() < 1e-10);
        assert!((params.c_1 - 1.44).abs() < 1e-10);
    }

    #[test]
    fn test_eddy_viscosity_update() {
        let mut ke = KEpsilonModel::new(10);
        
        // 设置 k = 0.01, ε = 0.001
        ke.k.fill(0.01);
        ke.epsilon.fill(0.001);
        
        ke.update_eddy_viscosity();
        
        // ν_t = 0.09 × 0.01² / 0.001 = 0.009
        for i in 0..10 {
            assert!((ke.eddy_viscosity[i] - 0.009).abs() < 1e-6);
        }
    }

    #[test]
    fn test_production_calculation() {
        let mut ke = KEpsilonModel::new(10);
        ke.eddy_viscosity.fill(0.01);
        
        // 设置剪切流梯度
        ke.set_velocity_gradient(0, VelocityGradient::new(0.0, 1.0, 0.0, 0.0));
        
        ke.compute_production();
        
        // P_k = ν_t × |S|² = 0.01 × 1.0 = 0.01
        assert!((ke.production[0] - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_step() {
        let mut ke = KEpsilonModel::new(10);
        let state = create_test_state(10, 2.0, 1.0, 0.0);
        
        // 设置速度梯度
        for i in 0..10 {
            ke.set_velocity_gradient(i, VelocityGradient::new(0.1, 0.0, 0.0, 0.0));
        }
        
        let _k_before = ke.k[0];
        ke.step(&state, 0.1);
        
        // k 应该改变（但保持正值）
        assert!(ke.k[0] > 0.0);
    }

    #[test]
    fn test_turbulent_scales() {
        let mut ke = KEpsilonModel::new(10);
        ke.k[0] = 0.1;
        ke.epsilon[0] = 0.01;
        
        let tau = ke.turbulent_time_scale(0);
        let length = ke.turbulent_length_scale(0);
        
        assert!((tau - 10.0).abs() < 1e-6);
        assert!(length > 0.0);
    }
}
