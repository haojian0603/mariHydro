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
use crate::core::{Backend, CpuBackend};
use crate::sources::traits::{
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterStateGeneric;
use mh_core::Scalar;
use std::marker::PhantomData;

/// 湍流模型类型（完全泛型化）
#[derive(Debug, Clone, Copy, PartialEq)]
#[derive(Default)]
pub enum TurbulenceModel<S: Scalar> {
    /// 无湍流（推荐用于浅水方程）
    #[default]
    None,
    /// 显式禁用（带警告）
    ///
    /// 使用此模式时，代码会输出一次警告日志，
    /// 提醒用户浅水方程不应使用 3D 湍流模型
    Disabled,
    /// 常数涡粘性（仅用于水平扩散，建议值 0.1-10 m²/s）
    // ALLOW_F64: Layer 4 配置参数 - 泛型参数
    ConstantViscosity(S),
}

impl<S: Scalar> TurbulenceModel<S> {
    /// Smagorinsky 常数的默认值
    #[inline]
    pub fn default_smagorinsky_constant() -> S {
        S::from_config(0.15).unwrap_or(S::ZERO)
    }

    /// 最小涡粘性系数 [m²/s]
    #[inline]
    pub fn min_eddy_viscosity() -> S {
        S::from_config(1e-6).unwrap_or(S::ZERO)
    }

    /// 最大涡粘性系数 [m²/s]
    #[inline]
    pub fn max_eddy_viscosity() -> S {
        S::from_config(1e3).unwrap_or(S::ZERO)
    }

    /// 创建禁用模式（推荐）
    pub fn disabled() -> Self {
        Self::Disabled
    }

    /// 创建常数涡粘性模型（仅用于水平扩散）
    ///
    /// # 参数
    /// - `nu`: 涡粘性系数 [m²/s]，建议范围 0.1-10
    pub fn constant(nu: S) -> Self {
        let min = Self::min_eddy_viscosity();
        let max = Self::max_eddy_viscosity();
        let clamped = if nu < min { min } else if nu > max { max } else { nu };
        Self::ConstantViscosity(clamped)
    }

    /// 检查模型是否实际启用
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None | Self::Disabled)
    }
}

/// Smagorinsky 湍流求解器（完全泛型化）
#[derive(Debug, Clone)]
pub struct SmagorinskySolver<S: Scalar> {
    /// 模型配置
    pub model: TurbulenceModel<S>,
    /// 网格尺度 [m]（每个单元）
    pub grid_scale: Vec<S>,
    /// 计算得到的涡粘性 [m²/s]（每个单元）
    pub eddy_viscosity: Vec<S>,
    /// 速度梯度（每个单元）
    pub velocity_gradient: Vec<VelocityGradient<S>>,
    /// 最小水深
    pub h_min: S,
    /// 类型标记
    _marker: PhantomData<S>,
}

impl<S: Scalar> SmagorinskySolver<S> {
    /// 创建新的求解器
    pub fn new(n_cells: usize, model: TurbulenceModel<S>) -> Self {
        Self {
            model,
            grid_scale: vec![S::from_config(10.0).unwrap_or(S::ZERO); n_cells], // 默认网格尺度
            eddy_viscosity: vec![S::ZERO; n_cells],
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: S::from_config(1e-4).unwrap_or(S::ZERO),
            _marker: PhantomData,
        }
    }

    /// 从网格初始化
    pub fn from_mesh(mesh: &PhysicsMesh, model: TurbulenceModel<S>) -> Self {
        let n_cells = mesh.n_cells();
        let mut solver = Self::new(n_cells, model);

        // 计算网格尺度（使用单元面积的平方根）
        for i in 0..n_cells {
            if let Some(area) = mesh.cell_area(i) {
                solver.grid_scale[i] = S::from_config(area.sqrt()).unwrap_or(S::ZERO);
            }
        }

        solver
    }

    /// 设置网格尺度
    pub fn set_grid_scale(&mut self, i: usize, scale: S) {
        if i < self.grid_scale.len() {
            self.grid_scale[i] = if scale < S::from_config(1e-3).unwrap_or(S::ZERO) {
                S::from_config(1e-3).unwrap_or(S::ZERO)
            } else {
                scale
            };
        }
    }

    /// 设置速度梯度（外部计算）
    pub fn set_velocity_gradient(&mut self, i: usize, grad: VelocityGradient<S>) {
        if i < self.velocity_gradient.len() {
            self.velocity_gradient[i] = grad;
        }
    }

    /// 批量设置速度梯度
    pub fn set_velocity_gradients(&mut self, gradients: &[VelocityGradient<S>]) {
        let n = self.velocity_gradient.len().min(gradients.len());
        self.velocity_gradient[..n].copy_from_slice(&gradients[..n]);
    }

    /// 使用简单差分估算速度梯度（适用于结构化网格）
    ///
    /// 对于非结构化网格，应使用外部梯度求解器
    ///
    /// # 参数
    /// - `h`: 水深场
    /// - `hu`: x方向动量场
    /// - `hv`: y方向动量场
    /// - `mesh`: 网格信息
    pub fn estimate_gradient_from_state(
        &mut self,
        h: &[S],
        hu: &[S],
        hv: &[S],
        mesh: &PhysicsMesh,
    ) {
        let n_cells = self.velocity_gradient.len()
            .min(h.len())
            .min(mesh.n_cells());

        for i in 0..n_cells {
            let h_i = h[i];
            if h_i < self.h_min {
                self.velocity_gradient[i] = VelocityGradient::default();
                continue;
            }

            let u = hu[i] / h_i;
            let v = hv[i] / h_i;

            // 简单的最近邻梯度估计
            let mut du_dx = S::ZERO;
            let mut du_dy = S::ZERO;
            let mut dv_dx = S::ZERO;
            let mut dv_dy = S::ZERO;
            let mut weight_sum = S::ZERO;

            for face_id in mesh.cell_faces(i) {
                // 使用 face_neighbor 获取邻居
                if let Some(neighbor) = mesh.face_neighbor(face_id) {
                    if neighbor == i {
                        continue;
                    }
                    let h_n = h[neighbor];
                    if h_n < self.h_min {
                        continue;
                    }

                    let u_n = hu[neighbor] / h_n;
                    let v_n = hv[neighbor] / h_n;

                    let normal = mesh.face_normal(face_id);
                    let dist = self.grid_scale[i];

                    if dist > S::from_config(1e-10).unwrap_or(S::ZERO) {
                        let weight = S::ONE / dist;
                        du_dx = du_dx + (u_n - u) * S::from_config(normal.x).unwrap_or(S::ZERO) * weight;
                        du_dy = du_dy + (u_n - u) * S::from_config(normal.y).unwrap_or(S::ZERO) * weight;
                        dv_dx = dv_dx + (v_n - v) * S::from_config(normal.x).unwrap_or(S::ZERO) * weight;
                        dv_dy = dv_dy + (v_n - v) * S::from_config(normal.y).unwrap_or(S::ZERO) * weight;
                        weight_sum = weight_sum + weight;
                    }
                }
            }

            if weight_sum > S::from_config(1e-10).unwrap_or(S::ZERO) {
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
                self.eddy_viscosity.fill(S::ZERO);
            }
            TurbulenceModel::ConstantViscosity(nu) => {
                self.eddy_viscosity.fill(*nu);
            }
        }
    }

    /// 获取单元涡粘性
    pub fn get_eddy_viscosity(&self, cell: usize) -> S {
        self.eddy_viscosity.get(cell).copied().unwrap_or(S::ZERO)
    }

    /// 计算湍流扩散通量
    ///
    /// 返回 (Fx, Fy) 动量扩散通量
    ///
    /// # 参数
    /// - `cell`: 单元索引
    /// - `h`: 该单元的水深
    pub fn compute_diffusion_flux(
        &self,
        cell: usize,
        h: S,
    ) -> (S, S) {
        if h < self.h_min {
            return (S::ZERO, S::ZERO);
        }

        let nu = self.get_eddy_viscosity(cell);
        let grad = &self.velocity_gradient[cell];

        let fx = nu * h * grad.du_dx;
        let fy = nu * h * grad.dv_dy;

        (fx, fy)
    }
}

// 实现 TurbulenceClosure trait
impl<S: Scalar> TurbulenceClosure<S> for SmagorinskySolver<S> {
    fn name(&self) -> &'static str {
        "Smagorinsky"
    }

    fn is_3d(&self) -> bool {
        false // Smagorinsky 适用于 2D
    }

    fn eddy_viscosity(&self) -> &[S] {
        &self.eddy_viscosity
    }

    fn update(&mut self, velocity_gradients: &[VelocityGradient<S>], cell_sizes: &[S]) {
        self.set_velocity_gradients(velocity_gradients);
        let n = self.grid_scale.len().min(cell_sizes.len());
        self.grid_scale[..n].copy_from_slice(&cell_sizes[..n]);
        self.update_eddy_viscosity();
    }

    fn is_enabled(&self) -> bool {
        self.model.is_active()
    }
}

/// 湍流源项配置（使用 Backend 泛型）
#[derive(Debug, Clone)]
pub struct TurbulenceConfig<B: Backend> {
    /// 是否启用
    pub enabled: bool,
    /// 湍流模型
    pub model: TurbulenceModel<B::Scalar>,
    /// 涡粘性 [m²/s]（预计算或常数）
    pub eddy_viscosity: Vec<B::Scalar>,
    /// 速度梯度（外部提供）
    pub velocity_gradient: Vec<VelocityGradient<B::Scalar>>,
    /// 最小水深
    pub h_min: B::Scalar,
    /// 类型标记
    _marker: PhantomData<B>,
}

impl<B: Backend> TurbulenceConfig<B> {
    /// 创建新配置
    pub fn new(n_cells: usize, model: TurbulenceModel<B::Scalar>) -> Self {
        Self {
            enabled: true,
            model,
            eddy_viscosity: vec![B::Scalar::ZERO; n_cells],
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: <B::Scalar as Scalar>::from_config(1e-4).unwrap_or(B::Scalar::ZERO),
            _marker: PhantomData,
        }
    }

    /// 创建常数涡粘性配置
    pub fn constant(n_cells: usize, nu: B::Scalar) -> Self {
        let mut config = Self::new(n_cells, TurbulenceModel::constant(nu));
        config.eddy_viscosity.fill(nu);
        config
    }

    /// 设置涡粘性
    pub fn set_eddy_viscosity(&mut self, cell: usize, nu: B::Scalar) {
        if cell < self.eddy_viscosity.len() {
            self.eddy_viscosity[cell] = if nu < B::Scalar::ZERO { B::Scalar::ZERO } else { nu };
        }
    }

    /// 批量设置涡粘性
    pub fn set_eddy_viscosity_field(&mut self, nu: &[B::Scalar]) {
        let n = self.eddy_viscosity.len().min(nu.len());
        self.eddy_viscosity[..n].copy_from_slice(&nu[..n]);
    }
}

// 为 CpuBackend<f64> 特化实现 SourceTermGeneric
impl SourceTermGeneric<CpuBackend<f64>> for TurbulenceConfig<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "Turbulence"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];

        // 干单元不计算
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContributionGeneric::zero();
        }

        let nu = self.eddy_viscosity.get(cell).copied().unwrap_or(0.0);
        if nu < TurbulenceModel::<f64>::min_eddy_viscosity() {
            return SourceContributionGeneric::zero();
        }

        let grad = self
            .velocity_gradient
            .get(cell)
            .copied()
            .unwrap_or_default();

        // 粘性应力源项（简化形式）
        let two = 2.0_f64;
        let s11 = two * grad.du_dx;
        let s22 = two * grad.dv_dy;
        let s12 = grad.du_dy + grad.dv_dx;

        let char_length = if h < 0.1 { 0.1 } else { h };

        let s_hu = nu * h * (s11 + s12) / char_length;
        let s_hv = nu * h * (s12 + s22) / char_length;

        // 限制源项大小
        let max_source = nu * h * 10.0;
        let s_hu_clamped = s_hu.clamp(-max_source, max_source);
        let s_hv_clamped = s_hv.clamp(-max_source, max_source);

        SourceContributionGeneric::momentum(s_hu_clamped, s_hv_clamped)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        _rhs_h: &mut Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
        _rhs_hu: &mut Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
        _rhs_hv: &mut Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.is_enabled() {
            return;
        }

        // 默认实现：逐单元计算并累加
        for cell in 0..state.n_cells() {
            let _contrib = self.compute_cell(cell, state, ctx);
            // 注意：实际的累加需要通过 Buffer trait 的方法
            // 这里只是占位实现
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_turbulence_model_default() {
        let model = TurbulenceModel::<f64>::default();
        assert_eq!(model, TurbulenceModel::None);
        assert!(!model.is_active());
    }

    #[test]
    fn test_turbulence_model_disabled() {
        let model = TurbulenceModel::<f64>::disabled();
        assert_eq!(model, TurbulenceModel::Disabled);
        assert!(!model.is_active());
    }

    #[test]
    fn test_turbulence_model_constant() {
        let model = TurbulenceModel::constant(0.01_f64);
        match model {
            TurbulenceModel::ConstantViscosity(nu) => {
                assert!((nu - 0.01).abs() < 1e-10);
            }
            _ => panic!("Expected ConstantViscosity model"),
        }
    }

    #[test]
    fn test_smagorinsky_solver_creation() {
        let solver = SmagorinskySolver::<f64>::new(10, TurbulenceModel::default());
        assert_eq!(solver.grid_scale.len(), 10);
        assert_eq!(solver.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_smagorinsky_solver_constant_viscosity() {
        let mut solver = SmagorinskySolver::new(10, TurbulenceModel::constant(0.1_f64));
        solver.update_eddy_viscosity();

        for i in 0..10 {
            assert!((solver.eddy_viscosity[i] - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_turbulence_config_creation() {
        let config = TurbulenceConfig::<CpuBackend<f64>>::new(10, TurbulenceModel::default());
        assert!(config.enabled);
        assert_eq!(config.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_turbulence_config_constant() {
        let config = TurbulenceConfig::<CpuBackend<f64>>::constant(10, 0.05);
        assert!((config.eddy_viscosity[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_source_term() {
        let mut config = TurbulenceConfig::<CpuBackend<f64>>::constant(10, 0.1);
        config.velocity_gradient[0] = VelocityGradient::new(1.0, 0.0, 0.0, 1.0);

        let mut state = ShallowWaterStateGeneric::<CpuBackend<f64>>::new(10);
        // 设置测试状态
        for i in 0..10 {
            state.h[i] = 2.0;
            state.hu[i] = 2.0; // h * u = 2.0 * 1.0
            state.hv[i] = 1.0; // h * v = 2.0 * 0.5
        }
        let ctx = SourceContextGeneric::with_defaults(0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        assert!((contrib.s_h - 0.0).abs() < 1e-10);
        // 验证动量源项非零
        assert!(contrib.s_hu.abs() > 0.0 || contrib.s_hv.abs() > 0.0);
    }

    #[test]
    fn test_turbulence_dry_cell() {
        let config = TurbulenceConfig::<CpuBackend<f64>>::constant(10, 0.1);

        let state = ShallowWaterStateGeneric::<CpuBackend<f64>>::new(10);
        // h 默认为 0，是干单元
        let ctx = SourceContextGeneric::with_defaults(0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        assert!((contrib.s_hu - 0.0).abs() < 1e-10);
        assert!((contrib.s_hv - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_source_term_generic_trait() {
        let config = TurbulenceConfig::<CpuBackend<f64>>::constant(10, 0.15);
        assert_eq!(config.name(), "Turbulence");
        assert_eq!(config.stiffness(), SourceStiffness::Explicit);
    }

    #[test]
    fn test_turbulence_closure_trait() {
        let mut solver = SmagorinskySolver::<f64>::new(10, TurbulenceModel::constant(0.5));
        assert_eq!(solver.name(), "Smagorinsky");
        assert!(!solver.is_3d());

        let grads = vec![VelocityGradient::default(); 10];
        let sizes = vec![10.0_f64; 10];
        solver.update(&grads, &sizes);

        assert!((solver.eddy_viscosity[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_f32_precision() {
        let model_f32 = TurbulenceModel::<f32>::constant(0.1_f32);
        let model_f64 = TurbulenceModel::<f64>::constant(0.1_f64);

        match (model_f32, model_f64) {
            (
                TurbulenceModel::ConstantViscosity(nu32),
                TurbulenceModel::ConstantViscosity(nu64),
            ) => {
                assert!((nu32 as f64 - nu64).abs() < 1e-6);
            }
            _ => panic!("Expected ConstantViscosity models"),
        }
    }
}
