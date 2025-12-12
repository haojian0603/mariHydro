// crates/mh_physics/src/sources/turbulence/smagorinsky.rs

//! Smagorinsky 亚格子尺度湍流模型
//!
//! 实现 2D 浅水方程的水平湍流闭合，主要用于水平涡粘性计算。
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
//! # 物理适用性
//!
//! **重要警告**：Smagorinsky 模型原本设计用于 3D LES。
//! 在 2D 浅水方程中，其物理意义有限，因为：
//!
//! 1. 深度平均消除了垂向湍流结构
//! 2. 底部摩擦通常是主导耗散机制
//! 3. 2D 湍流动力学与 3D 本质不同
//!
//! 推荐用法：
//! - 使用 `TurbulenceModel::None` 或 `TurbulenceModel::Disabled`
//! - 如需水平扩散，使用 `TurbulenceModel::ConstantViscosity(0.1~10.0)`

use super::traits::{TurbulenceClosure, VelocityGradient};
use crate::adapter::PhysicsMesh;
use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;
use mh_core::Scalar;

/// Smagorinsky 常数的默认值
pub const DEFAULT_SMAGORINSKY_CONSTANT: Scalar = 0.15;

/// 最小涡粘性系数 [m²/s]
pub const MIN_EDDY_VISCOSITY: Scalar = 1e-6;

/// 最大涡粘性系数 [m²/s]
pub const MAX_EDDY_VISCOSITY: Scalar = 1e3;

/// 湍流模型类型
/// 
/// **重要警告**：浅水方程是深度平均方程，直接添加 3D 湍流扩散项
/// 在物理上是不恰当的。深度平均后的湍流效应通常通过以下方式处理：
/// 
/// 1. **底部摩擦**：已包含在摩擦模块中，占主导作用
/// 2. **水平扩散**：使用适当的水平涡粘性（不是 Smagorinsky 的 3D 公式）
/// 3. **色散项**：如 Boussinesq 方程的色散修正
/// 
/// 如果确实需要水平扩散，建议使用 `ConstantViscosity` 配合
/// 较小的涡粘性值（0.1-10 m²/s），或使用 `Disabled` 模式。
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum TurbulenceModel {
    /// 无湍流（推荐用于浅水方程）
    #[default]
    None,
    /// 显式禁用（带警告）
    /// 
    /// 使用此模式时，代码会输出一次警告日志，
    /// 提醒用户浅水方程不应使用 3D 湍流模型
    Disabled,
    /// 常数涡粘性（仅用于水平扩散，建议值 0.1-10 m²/s）
    ConstantViscosity(Scalar),
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
    pub fn constant(nu: Scalar) -> Self {
        Self::ConstantViscosity(nu.max(MIN_EDDY_VISCOSITY).min(MAX_EDDY_VISCOSITY))
    }

    /// 检查模型是否实际启用
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None | Self::Disabled)
    }
}

/// Smagorinsky 湍流求解器
///
/// 2D 水平涡粘性计算器。
#[derive(Debug, Clone)]
pub struct SmagorinskySolver {
    /// 模型配置
    pub model: TurbulenceModel,
    /// 网格尺度 [m]（每个单元）
    pub grid_scale: Vec<Scalar>,
    /// 计算得到的涡粘性 [m²/s]（每个单元）
    pub eddy_viscosity: Vec<Scalar>,
    /// 速度梯度（每个单元）
    pub velocity_gradient: Vec<VelocityGradient>,
    /// 最小水深
    pub h_min: Scalar,
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
    pub fn set_grid_scale(&mut self, i: usize, scale: Scalar) {
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
                // 使用 face_neighbor 获取邻居
                if let Some(neighbor) = mesh.face_neighbor(face_id) {
                    if neighbor == i {
                        continue;
                    }
                    let h_n = state.h[neighbor];
                    if h_n < self.h_min {
                        continue;
                    }

                    let u_n = state.hu[neighbor] / h_n;
                    let v_n = state.hv[neighbor] / h_n;

                    let normal = mesh.face_normal(face_id);
                    let dist = self.grid_scale[i];

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
    pub fn update_eddy_viscosity(&mut self) {
        match &self.model {
            TurbulenceModel::None | TurbulenceModel::Disabled => {
                self.eddy_viscosity.fill(0.0);
            }
            TurbulenceModel::ConstantViscosity(nu) => {
                self.eddy_viscosity.fill(*nu);
            }
        }
    }

    /// 获取单元涡粘性
    pub fn get_eddy_viscosity(&self, cell: usize) -> Scalar {
        self.eddy_viscosity.get(cell).copied().unwrap_or(0.0)
    }

    /// 计算湍流扩散通量
    ///
    /// 返回 (Fx, Fy) 动量扩散通量
    pub fn compute_diffusion_flux(&self, cell: usize, state: &ShallowWaterState) -> (Scalar, Scalar) {
        let h = state.h[cell];
        if h < self.h_min {
            return (0.0, 0.0);
        }

        let nu = self.get_eddy_viscosity(cell);
        let grad = &self.velocity_gradient[cell];

        let fx = nu * h * grad.du_dx;
        let fy = nu * h * grad.dv_dy;

        (fx, fy)
    }
}

// 实现 TurbulenceClosure trait
impl TurbulenceClosure for SmagorinskySolver {
    fn name(&self) -> &'static str {
        "Smagorinsky"
    }
    
    fn is_3d(&self) -> bool {
        false // Smagorinsky 适用于 2D
    }
    
    fn eddy_viscosity(&self) -> &[Scalar] {
        &self.eddy_viscosity
    }
    
    fn update(&mut self, velocity_gradients: &[VelocityGradient], cell_sizes: &[Scalar]) {
        self.set_velocity_gradients(velocity_gradients);
        let n = self.grid_scale.len().min(cell_sizes.len());
        self.grid_scale[..n].copy_from_slice(&cell_sizes[..n]);
        self.update_eddy_viscosity();
    }
    
    fn is_enabled(&self) -> bool {
        self.model.is_active()
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
    pub eddy_viscosity: Vec<Scalar>,
    /// 速度梯度（外部提供）
    pub velocity_gradient: Vec<VelocityGradient>,
    /// 最小水深
    pub h_min: Scalar,
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
    pub fn constant(n_cells: usize, nu: Scalar) -> Self {
        let mut config = Self::new(n_cells, TurbulenceModel::constant(nu));
        config.eddy_viscosity.fill(nu);
        config
    }

    /// 设置涡粘性
    pub fn set_eddy_viscosity(&mut self, cell: usize, nu: Scalar) {
        if cell < self.eddy_viscosity.len() {
            self.eddy_viscosity[cell] = nu.max(0.0);
        }
    }

    /// 批量设置涡粘性
    pub fn set_eddy_viscosity_field(&mut self, nu: &[Scalar]) {
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

        // 粘性应力源项（简化形式）
        let s11 = 2.0 * grad.du_dx;
        let s22 = 2.0 * grad.dv_dy;
        let s12 = grad.du_dy + grad.dv_dx;
        
        let char_length = h.max(0.1);
        
        let s_hu = nu * h * (s11 + s12) / char_length;
        let s_hv = nu * h * (s12 + s22) / char_length;
        
        // 限制源项大小
        let max_source = nu * h * 10.0;
        let s_hu_clamped = s_hu.clamp(-max_source, max_source);
        let s_hv_clamped = s_hv.clamp(-max_source, max_source);

        SourceContribution::momentum(s_hu_clamped, s_hv_clamped)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: Scalar, u: Scalar, v: Scalar) -> ShallowWaterState {
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
    fn test_turbulence_model_default() {
        let model = TurbulenceModel::default();
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
        let config = TurbulenceConfig::constant(10, 0.15);
        assert_eq!(config.name(), "Turbulence");
        assert!(config.is_explicit());
    }
    
    #[test]
    fn test_turbulence_closure_trait() {
        let mut solver = SmagorinskySolver::new(10, TurbulenceModel::constant(0.5));
        assert_eq!(solver.name(), "Smagorinsky");
        assert!(!solver.is_3d());
        
        let grads = vec![VelocityGradient::default(); 10];
        let sizes = vec![10.0; 10];
        solver.update(&grads, &sizes);
        
        assert!((solver.eddy_viscosity[0] - 0.5).abs() < 1e-10);
    }
}
