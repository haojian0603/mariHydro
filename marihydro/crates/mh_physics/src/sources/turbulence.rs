// crates/mh_physics/src/sources/turbulence.rs

//! 湍流源项
//!
//! 实现浅水方程的湍流闭合模型，主要使用 Smagorinsky 亚格子尺度模型。
//!
//! # Smagorinsky 模型
//!
//! Smagorinsky (1963) 模型假设亚格子湍流粘性：
//! ```text
//! ν_t = (C_s * Δ)² * |S|
//! ```
//!
//! 其中：
//! - C_s 是 Smagorinsky 常数（通常 0.1-0.2）
//! - Δ 是网格尺度 (√A)
//! - |S| 是应变率张量的模
//!
//! # 应变率张量
//!
//! 对于二维流动：
//! ```text
//! |S| = √(2*(∂u/∂x)² + 2*(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²)
//! ```
//!
//! # 湍流粘性系数
//!
//! 提供两种计算方式：
//! - 常数涡粘性
//! - Smagorinsky 动态计算

use super::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::adapter::PhysicsMesh;
use crate::state::ShallowWaterState;

/// Smagorinsky 常数的默认值
pub const DEFAULT_SMAGORINSKY_CONSTANT: f64 = 0.15;

/// 最小涡粘性系数 [m²/s]
pub const MIN_EDDY_VISCOSITY: f64 = 1e-6;

/// 最大涡粘性系数 [m²/s]
pub const MAX_EDDY_VISCOSITY: f64 = 1e3;

/// 湍流模型类型
/// 
/// **重要警告**：浅水方程是深度平均方程，直接添加3D湍流扩散项
/// 在物理上是不恰当的。深度平均后的湍流效应通常通过以下方式处理：
/// 
/// 1. **底部摩擦**：已包含在摩擦模块中，占主导作用
/// 2. **水平扩散**：使用适当的水平涡粘性（不是Smagorinsky的3D公式）
/// 3. **色散项**：如 Boussinesq 方程的色散修正
/// 
/// 如果确实需要水平扩散，建议使用 `ConstantViscosity` 配合
/// 较小的涡粘性值（0.1-10 m²/s），或使用 `Disabled` 模式。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TurbulenceModel {
    /// 无湍流（推荐用于浅水方程）
    None,
    /// 显式禁用（带警告）
    /// 
    /// 使用此模式时，代码会输出一次警告日志，
    /// 提醒用户浅水方程不应使用3D湍流模型
    Disabled,
    /// 常数涡粘性（仅用于水平扩散，建议值 0.1-10 m²/s）
    ConstantViscosity(f64),
    /// Smagorinsky 亚格子模型
    /// 
    /// **警告**：此模型设计用于3D LES，在浅水方程中物理意义有限
    /// 仅建议用于研究或对比目的
    #[deprecated(since = "0.2.0", note = "Smagorinsky 为3D LES模型，不适用于深度平均的浅水方程")]
    Smagorinsky {
        /// Smagorinsky 常数
        cs: f64,
        /// 最小涡粘性
        nu_min: f64,
        /// 最大涡粘性
        nu_max: f64,
    },
    /// Pacanowski-Philander 模型（深海用）
    /// 
    /// **警告**：此模型设计用于垂向混合，不适用于浅水方程
    #[deprecated(since = "0.2.0", note = "PP模型用于垂向混合，不适用于浅水方程")]
    PacanowskiPhilander {
        /// 背景粘性
        nu_0: f64,
        /// 最大粘性增量
        nu_max: f64,
        /// Richardson 数临界值
        ri_c: f64,
    },
}

impl Default for TurbulenceModel {
    fn default() -> Self {
        // 默认禁用湍流模型 - 浅水方程不应使用3D湍流
        Self::None
    }
}

impl TurbulenceModel {
    /// 创建禁用模式（推荐）
    pub fn disabled() -> Self {
        Self::Disabled
    }

    /// 创建常数涡粘性模型（仅用于水平扩散）
    /// 
    /// # 参数
    /// - `nu`: 涡粘性系数 [m²/s]，建议范围 0.1-10
    pub fn constant(nu: f64) -> Self {
        Self::ConstantViscosity(nu.max(MIN_EDDY_VISCOSITY).min(MAX_EDDY_VISCOSITY))
    }

    /// 创建 Smagorinsky 模型
    /// 
    /// **警告**：此模型不适用于浅水方程，仅用于研究目的
    #[allow(deprecated)]
    pub fn smagorinsky(cs: f64) -> Self {
        Self::Smagorinsky {
            cs: cs.abs().max(0.05).min(0.3),
            nu_min: MIN_EDDY_VISCOSITY,
            nu_max: MAX_EDDY_VISCOSITY,
        }
    }

    /// 创建自定义范围的 Smagorinsky 模型
    #[allow(deprecated)]
    pub fn smagorinsky_with_limits(cs: f64, nu_min: f64, nu_max: f64) -> Self {
        Self::Smagorinsky {
            cs: cs.abs().max(0.05).min(0.3),
            nu_min: nu_min.max(0.0),
            nu_max: nu_max.max(nu_min),
        }
    }

    /// 检查模型是否实际启用
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None | Self::Disabled)
    }
}

/// 速度梯度张量
#[derive(Debug, Clone, Copy, Default)]
pub struct VelocityGradient {
    /// ∂u/∂x
    pub du_dx: f64,
    /// ∂u/∂y
    pub du_dy: f64,
    /// ∂v/∂x
    pub dv_dx: f64,
    /// ∂v/∂y
    pub dv_dy: f64,
}

impl VelocityGradient {
    /// 创建新的速度梯度
    pub fn new(du_dx: f64, du_dy: f64, dv_dx: f64, dv_dy: f64) -> Self {
        Self { du_dx, du_dy, dv_dx, dv_dy }
    }

    /// 计算应变率张量的模
    ///
    /// |S| = √(2*(∂u/∂x)² + 2*(∂v/∂y)² + (∂u/∂y + ∂v/∂x)²)
    #[inline]
    pub fn strain_rate_magnitude(&self) -> f64 {
        let s11 = self.du_dx;
        let s22 = self.dv_dy;
        let s12 = 0.5 * (self.du_dy + self.dv_dx);

        (2.0 * s11 * s11 + 2.0 * s22 * s22 + 4.0 * s12 * s12).sqrt()
    }

    /// 计算涡度
    ///
    /// ω = ∂v/∂x - ∂u/∂y
    #[inline]
    pub fn vorticity(&self) -> f64 {
        self.dv_dx - self.du_dy
    }

    /// 计算散度
    ///
    /// div(u) = ∂u/∂x + ∂v/∂y
    #[inline]
    pub fn divergence(&self) -> f64 {
        self.du_dx + self.dv_dy
    }
}

/// Smagorinsky 湍流求解器
#[derive(Debug, Clone)]
pub struct SmagorinskySolver {
    /// 模型配置
    pub model: TurbulenceModel,
    /// 网格尺度 [m]（每个单元）
    pub grid_scale: Vec<f64>,
    /// 计算得到的涡粘性 [m²/s]（每个单元）
    pub eddy_viscosity: Vec<f64>,
    /// 速度梯度（每个单元）
    pub velocity_gradient: Vec<VelocityGradient>,
    /// 最小水深
    pub h_min: f64,
}

impl SmagorinskySolver {
    /// 创建新的求解器
    pub fn new(n_cells: usize, model: TurbulenceModel) -> Self {
        Self {
            model,
            grid_scale: vec![10.0; n_cells], // 默认网格尺度
            eddy_viscosity: vec![0.0; n_cells],
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: 1e-4,
        }
    }

    /// 从网格初始化
    pub fn from_mesh(mesh: &PhysicsMesh, model: TurbulenceModel) -> Self {
        let n_cells = mesh.n_cells();
        let mut solver = Self::new(n_cells, model);

        // 计算网格尺度（使用单元面积的平方根）
        for i in 0..n_cells {
            if let Some(area) = mesh.cell_area(i) {
                solver.grid_scale[i] = area.sqrt();
            }
        }

        solver
    }

    /// 设置网格尺度
    pub fn set_grid_scale(&mut self, i: usize, scale: f64) {
        if i < self.grid_scale.len() {
            self.grid_scale[i] = scale.max(1e-3);
        }
    }

    /// 设置速度梯度（外部计算）
    pub fn set_velocity_gradient(&mut self, i: usize, grad: VelocityGradient) {
        if i < self.velocity_gradient.len() {
            self.velocity_gradient[i] = grad;
        }
    }

    /// 批量设置速度梯度
    pub fn set_velocity_gradients(&mut self, gradients: &[VelocityGradient]) {
        let n = self.velocity_gradient.len().min(gradients.len());
        self.velocity_gradient[..n].copy_from_slice(&gradients[..n]);
    }

    /// 使用简单差分估算速度梯度（适用于结构化网格）
    ///
    /// 对于非结构化网格，应使用外部梯度求解器
    pub fn estimate_gradient_from_neighbors(
        &mut self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
    ) {
        let n_cells = self.velocity_gradient.len().min(state.h.len()).min(mesh.n_cells());

        for i in 0..n_cells {
            let h = state.h[i];
            if h < self.h_min {
                self.velocity_gradient[i] = VelocityGradient::default();
                continue;
            }

            let u = state.hu[i] / h;
            let v = state.hv[i] / h;

            // 简单的最近邻梯度估计
            let mut du_dx = 0.0;
            let mut du_dy = 0.0;
            let mut dv_dx = 0.0;
            let mut dv_dy = 0.0;
            let mut weight_sum = 0.0;

            for face_id in mesh.cell_faces(i) {
                // 使用 face_neighbor 获取邻居（如果当前单元是左侧则返回右侧邻居）
                if let Some(neighbor) = mesh.face_neighbor(face_id) {
                    // 确保邻居不是自己
                    if neighbor == i {
                        continue;
                    }
                    let h_n = state.h[neighbor];
                    if h_n < self.h_min {
                        continue;
                    }

                    let u_n = state.hu[neighbor] / h_n;
                    let v_n = state.hv[neighbor] / h_n;

                    // 使用面法向作为方向
                    let normal = mesh.face_normal(face_id);
                    let dist = self.grid_scale[i]; // 简化：使用网格尺度作为距离

                    if dist > 1e-10 {
                        let weight = 1.0 / dist;
                        du_dx += (u_n - u) * normal.x * weight;
                        du_dy += (u_n - u) * normal.y * weight;
                        dv_dx += (v_n - v) * normal.x * weight;
                        dv_dy += (v_n - v) * normal.y * weight;
                        weight_sum += weight;
                    }
                }
            }

            if weight_sum > 1e-10 {
                self.velocity_gradient[i] = VelocityGradient::new(
                    du_dx / weight_sum,
                    du_dy / weight_sum,
                    dv_dx / weight_sum,
                    dv_dy / weight_sum,
                );
            } else {
                self.velocity_gradient[i] = VelocityGradient::default();
            }
        }
    }

    /// 更新涡粘性系数
    #[allow(deprecated)]
    pub fn update_eddy_viscosity(&mut self) {
        match &self.model {
            TurbulenceModel::None | TurbulenceModel::Disabled => {
                self.eddy_viscosity.fill(0.0);
            }
            TurbulenceModel::ConstantViscosity(nu) => {
                self.eddy_viscosity.fill(*nu);
            }
            TurbulenceModel::Smagorinsky { cs, nu_min, nu_max } => {
                for i in 0..self.eddy_viscosity.len() {
                    let delta = self.grid_scale[i];
                    let strain_rate = self.velocity_gradient[i].strain_rate_magnitude();

                    // ν_t = (C_s * Δ)² * |S|
                    let nu_t = (cs * delta).powi(2) * strain_rate;
                    self.eddy_viscosity[i] = nu_t.max(*nu_min).min(*nu_max);
                }
            }
            TurbulenceModel::PacanowskiPhilander { nu_0, nu_max, ri_c } => {
                // 简化版本，忽略 Richardson 数效应
                for i in 0..self.eddy_viscosity.len() {
                    self.eddy_viscosity[i] = (*nu_0).max(MIN_EDDY_VISCOSITY);
                }
                let _ = (ri_c, nu_max); // 完整版本需要密度分层信息
            }
        }
    }

    /// 获取单元涡粘性
    pub fn get_eddy_viscosity(&self, cell: usize) -> f64 {
        self.eddy_viscosity.get(cell).copied().unwrap_or(0.0)
    }

    /// 计算湍流扩散通量
    ///
    /// 返回 (Fx, Fy) 动量扩散通量
    pub fn compute_diffusion_flux(&self, cell: usize, state: &ShallowWaterState) -> (f64, f64) {
        let h = state.h[cell];
        if h < self.h_min {
            return (0.0, 0.0);
        }

        let nu = self.get_eddy_viscosity(cell);
        let grad = &self.velocity_gradient[cell];

        // 扩散通量: F = ν * h * ∇u
        // 简化为: S = ν * (∂²u/∂x² + ∂²u/∂y²) ≈ ν * Laplacian
        // 这里使用速度梯度的散度作为近似
        let fx = nu * h * grad.du_dx;
        let fy = nu * h * grad.dv_dy;

        (fx, fy)
    }
}

/// 湍流源项配置
#[derive(Debug, Clone)]
pub struct TurbulenceConfig {
    /// 是否启用
    pub enabled: bool,
    /// 湍流模型
    pub model: TurbulenceModel,
    /// 涡粘性 [m²/s]（预计算或常数）
    pub eddy_viscosity: Vec<f64>,
    /// 速度梯度（外部提供）
    pub velocity_gradient: Vec<VelocityGradient>,
    /// 最小水深
    pub h_min: f64,
}

impl TurbulenceConfig {
    /// 创建新配置
    pub fn new(n_cells: usize, model: TurbulenceModel) -> Self {
        Self {
            enabled: true,
            model,
            eddy_viscosity: vec![0.0; n_cells],
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: 1e-4,
        }
    }

    /// 创建常数涡粘性配置
    pub fn constant(n_cells: usize, nu: f64) -> Self {
        let mut config = Self::new(n_cells, TurbulenceModel::constant(nu));
        config.eddy_viscosity.fill(nu);
        config
    }

    /// 创建 Smagorinsky 配置
    pub fn smagorinsky(n_cells: usize, cs: f64) -> Self {
        Self::new(n_cells, TurbulenceModel::smagorinsky(cs))
    }

    /// 设置涡粘性
    pub fn set_eddy_viscosity(&mut self, cell: usize, nu: f64) {
        if cell < self.eddy_viscosity.len() {
            self.eddy_viscosity[cell] = nu.max(0.0);
        }
    }

    /// 批量设置涡粘性
    pub fn set_eddy_viscosity_field(&mut self, nu: &[f64]) {
        let n = self.eddy_viscosity.len().min(nu.len());
        self.eddy_viscosity[..n].copy_from_slice(&nu[..n]);
    }
}

impl SourceTerm for TurbulenceConfig {
    fn name(&self) -> &'static str {
        "Turbulence"
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];

        // 干单元不计算
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContribution::ZERO;
        }

        let nu = self.eddy_viscosity.get(cell).copied().unwrap_or(0.0);
        if nu < MIN_EDDY_VISCOSITY {
            return SourceContribution::ZERO;
        }

        let grad = self.velocity_gradient.get(cell).copied().unwrap_or_default();

        // 湍流扩散源项
        // S_hu = ν * h * (∂²u/∂x² + ∂²u/∂y²)
        // 简化为一阶近似: S ≈ ν * h * ∇²u
        // 使用速度梯度的拉普拉斯近似
        let s_hu = nu * h * grad.du_dx.abs() * 0.1; // 简化因子
        let s_hv = nu * h * grad.dv_dy.abs() * 0.1;

        SourceContribution::momentum(s_hu, s_hv)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64, u: f64, v: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.hu[i] = h * u;
            state.hv[i] = h * v;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_velocity_gradient_strain_rate() {
        // 纯剪切流: u = y, v = 0
        // ∂u/∂y = 1, 其他为 0
        let grad = VelocityGradient::new(0.0, 1.0, 0.0, 0.0);

        // |S| = √(4 * s12²) = √(4 * 0.25) = 1.0
        let strain = grad.strain_rate_magnitude();
        assert!((strain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_gradient_vorticity() {
        let grad = VelocityGradient::new(0.0, 1.0, -1.0, 0.0);
        let vorticity = grad.vorticity();
        assert!((vorticity - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_velocity_gradient_divergence() {
        let grad = VelocityGradient::new(2.0, 0.0, 0.0, 3.0);
        let div = grad.divergence();
        assert!((div - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_model_default() {
        let model = TurbulenceModel::default();
        // 默认应该是 None（禁用状态）
        assert_eq!(model, TurbulenceModel::None);
        assert!(!model.is_active());
    }

    #[test]
    fn test_turbulence_model_disabled() {
        let model = TurbulenceModel::disabled();
        assert_eq!(model, TurbulenceModel::Disabled);
        assert!(!model.is_active());
    }

    #[test]
    fn test_turbulence_model_constant() {
        let model = TurbulenceModel::constant(0.01);
        match model {
            TurbulenceModel::ConstantViscosity(nu) => {
                assert!((nu - 0.01).abs() < 1e-10);
            }
            _ => panic!("Expected ConstantViscosity model"),
        }
    }

    #[test]
    fn test_smagorinsky_solver_creation() {
        let solver = SmagorinskySolver::new(10, TurbulenceModel::default());
        assert_eq!(solver.grid_scale.len(), 10);
        assert_eq!(solver.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_smagorinsky_solver_constant_viscosity() {
        let mut solver = SmagorinskySolver::new(10, TurbulenceModel::constant(0.1));
        solver.update_eddy_viscosity();

        for i in 0..10 {
            assert!((solver.eddy_viscosity[i] - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_smagorinsky_solver_compute() {
        let mut solver = SmagorinskySolver::new(10, TurbulenceModel::smagorinsky(0.15));
        solver.grid_scale.fill(10.0); // 10m 网格

        // 设置速度梯度
        solver.set_velocity_gradient(0, VelocityGradient::new(0.1, 0.0, 0.0, 0.1));

        solver.update_eddy_viscosity();

        // ν_t = (0.15 * 10)² * |S|
        // |S| = √(2*0.1² + 2*0.1²) = √0.04 = 0.2
        // ν_t = 2.25 * 0.2 = 0.45
        let expected_nu = (0.15 * 10.0_f64).powi(2) * 0.2;
        assert!((solver.eddy_viscosity[0] - expected_nu).abs() < 1e-6);
    }

    #[test]
    fn test_turbulence_config_creation() {
        let config = TurbulenceConfig::new(10, TurbulenceModel::default());
        assert!(config.enabled);
        assert_eq!(config.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_turbulence_config_constant() {
        let config = TurbulenceConfig::constant(10, 0.05);
        assert!((config.eddy_viscosity[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_source_term() {
        let mut config = TurbulenceConfig::constant(10, 0.1);
        config.velocity_gradient[0] = VelocityGradient::new(1.0, 0.0, 0.0, 1.0);

        let state = create_test_state(10, 2.0, 1.0, 0.5);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu > 0.0);
        assert!(contrib.s_hv > 0.0);
    }

    #[test]
    fn test_turbulence_dry_cell() {
        let config = TurbulenceConfig::constant(10, 0.1);

        let state = create_test_state(10, 1e-7, 0.0, 0.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = config.compute_cell(&state, 0, &ctx);

        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_source_term_trait() {
        let config = TurbulenceConfig::smagorinsky(10, 0.15);
        assert_eq!(config.name(), "Turbulence");
        assert!(config.is_explicit());
    }
}
