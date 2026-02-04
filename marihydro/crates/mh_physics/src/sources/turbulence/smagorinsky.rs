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
use crate::types::CellIndex;
use crate::core::{Backend, CpuBackend};
use crate::sources::traits::{
    SourceContributionGeneric, SourceContextGeneric, SourceStiffness, SourceTermGeneric,
};
use crate::state::ShallowWaterState;
use mh_runtime::{RuntimeScalar as Scalar, Vector2D};
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
    /// Smagorinsky 模型
    Smagorinsky { cs: S },
}

impl<S: Scalar> TurbulenceModel<S> {
    /// Smagorinsky 常数的默认值
    #[inline]
    pub fn default_smagorinsky_constant<B: Backend<Scalar = S>>(backend: &B) -> S {
        backend.scalar_from_f64(0.15)
    }

    /// 最小涡粘性系数 [m²/s]
    #[inline]
    pub fn min_eddy_viscosity<B: Backend<Scalar = S>>(backend: &B) -> S {
        backend.scalar_from_f64(1e-6)
    }

    /// 最大涡粘性系数 [m²/s]
    #[inline]
    pub fn max_eddy_viscosity<B: Backend<Scalar = S>>(backend: &B) -> S {
        backend.scalar_from_f64(1e3)
    }

    /// 创建禁用模式（推荐）
    pub fn disabled() -> Self {
        Self::Disabled
    }

    /// 创建常数涡粘性模型（仅用于水平扩散）
    ///
    /// # 参数
    /// - `nu`: 涡粘性系数 [m²/s]，建议范围 0.1-10
    pub fn constant<B: Backend<Scalar = S>>(backend: &B, nu: S) -> Self {
        let min = Self::min_eddy_viscosity(backend);
        let max = Self::max_eddy_viscosity(backend);
        let clamped = if nu < min { min } else if nu > max { max } else { nu };
        Self::ConstantViscosity(clamped)
    }

    /// 创建 Smagorinsky 模型
    pub fn smagorinsky<B: Backend<Scalar = S>>(backend: &B, cs: S) -> Self {
        let min = backend.scalar_from_f64(0.05);
        let max = backend.scalar_from_f64(0.3);
        let clamped = if cs < min { min } else if cs > max { max } else { cs };
        Self::Smagorinsky { cs: clamped }
    }

    /// 检查模型是否实际启用
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None | Self::Disabled)
    }
}

/// Smagorinsky 湍流求解器（完全泛型化）
#[derive(Debug, Clone)]
pub struct SmagorinskySolver<B: Backend> {
    /// 模型配置
    pub model: TurbulenceModel<B::Scalar>,
    /// 网格尺度 [m]（每个单元）
    pub grid_scale: B::Buffer<B::Scalar>,
    /// 计算得到的涡粘性 [m²/s]（每个单元）
    pub eddy_viscosity: B::Buffer<B::Scalar>,
    /// 速度梯度（每个单元）
    pub velocity_gradient: Vec<VelocityGradient<B::Scalar>>,
    /// 最小水深
    pub h_min: B::Scalar,
    /// 后端
    backend: B,
    /// 类型标记
    _marker: PhantomData<B>,
}

impl<B: Backend> SmagorinskySolver<B> {
    /// 创建新的求解器
    pub fn new(backend: B, n_cells: usize, model: TurbulenceModel<B::Scalar>) -> Self {
        let grid_scale = backend.alloc_init(n_cells, backend.scalar_from_f64(10.0));
        let eddy_viscosity = backend.alloc_init(n_cells, B::Scalar::ZERO);
        Self {
            model,
            grid_scale,
            eddy_viscosity,
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: backend.scalar_from_f64(1e-4),
            backend,
            _marker: PhantomData,
        }
    }

    /// 从网格初始化
    pub fn from_mesh(backend: B, mesh: &PhysicsMesh, model: TurbulenceModel<B::Scalar>) -> Self {
        let n_cells = mesh.cell_count();
        let mut solver = Self::new(backend, n_cells, model);

        // 计算网格尺度（使用单元面积的平方根）
        for i in 0..n_cells {
            if let Some(area) = mesh.cell_area(mh_runtime::CellIndex(i)) {
                solver.grid_scale[i] = solver.backend.scalar_from_f64(area.sqrt());
            }
        }

        solver
    }

    /// 设置网格尺度
    pub fn set_grid_scale(&mut self, i: usize, scale: B::Scalar) {
        if i < self.grid_scale.len() {
            let min_scale = self.backend.scalar_from_f64(1e-3);
            self.grid_scale[i] = if scale < min_scale {
                min_scale
            } else {
                scale
            };
        }
    }

    /// 设置速度梯度（外部计算）
    pub fn set_velocity_gradient(&mut self, i: usize, grad: VelocityGradient<B::Scalar>) {
        if i < self.velocity_gradient.len() {
            self.velocity_gradient[i] = grad;
        }
    }

    /// 批量设置速度梯度
    pub fn set_velocity_gradients(&mut self, gradients: &[VelocityGradient<B::Scalar>]) {
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
        h: &[B::Scalar],
        hu: &[B::Scalar],
        hv: &[B::Scalar],
        mesh: &PhysicsMesh,
    ) {
        let n_cells = self.velocity_gradient.len()
            .min(h.len())
            .min(mesh.cell_count());

        for i in 0..n_cells {
            let h_i = h[i];
            if h_i < self.h_min {
                self.velocity_gradient[i] = VelocityGradient::default();
                continue;
            }

            let u = hu[i] / h_i;
            let v = hv[i] / h_i;

            // 简单的最近邻梯度估计
            let mut du_dx = B::Scalar::ZERO;
            let mut du_dy = B::Scalar::ZERO;
            let mut dv_dx = B::Scalar::ZERO;
            let mut dv_dy = B::Scalar::ZERO;
            let mut weight_sum = B::Scalar::ZERO;

            for face_id in mesh.cell_faces(CellIndex::new(i)) {
                // 使用 face_neighbor 获取邻居
                if let Some(neighbor) = mesh.face_neighbor(face_id) {
                    let neigh_idx: usize = neighbor.into();
                    if neigh_idx == i {
                        continue;
                    }
                    let h_n = h[neigh_idx];
                    if h_n < self.h_min {
                        continue;
                    }

                    let u_n = hu[neigh_idx] / h_n;
                    let v_n = hv[neigh_idx] / h_n;

                    let normal = mesh
                        .face_normal_generic::<CpuBackend<f64>>(face_id)
                        .expect("face_normal out of range");
                    let nx = self.backend.scalar_from_f64(normal.x());
                    let ny = self.backend.scalar_from_f64(normal.y());
                    let dist = self.grid_scale[i];

                    if dist > self.backend.scalar_from_f64(1e-10) {
                        let weight = B::Scalar::ONE / dist;
                        du_dx = du_dx + (u_n - u) * nx * weight;
                        du_dy = du_dy + (u_n - u) * ny * weight;
                        dv_dx = dv_dx + (v_n - v) * nx * weight;
                        dv_dy = dv_dy + (v_n - v) * ny * weight;
                        weight_sum = weight_sum + weight;
                    }
                }
            }

            if weight_sum > self.backend.scalar_from_f64(1e-10) {
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
                self.eddy_viscosity.fill(B::Scalar::ZERO);
            }
            TurbulenceModel::ConstantViscosity(nu) => {
                self.eddy_viscosity.fill(*nu);
            }
            TurbulenceModel::Smagorinsky { cs } => {
                let min = TurbulenceModel::<B::Scalar>::min_eddy_viscosity(&self.backend);
                let max = TurbulenceModel::<B::Scalar>::max_eddy_viscosity(&self.backend);
                for (i, nu) in self.eddy_viscosity.iter_mut().enumerate() {
                    let delta = self.grid_scale.get(i).copied().unwrap_or(B::Scalar::ZERO);
                    let strain = self
                        .velocity_gradient
                        .get(i)
                        .copied()
                        .unwrap_or_default()
                        .strain_rate_magnitude();
                    let nu_sgs = (*cs * delta) * (*cs * delta) * strain;
                    *nu = if nu_sgs < min { min } else if nu_sgs > max { max } else { nu_sgs };
                }
            }
        }
    }

    /// 获取单元涡粘性
    pub fn get_eddy_viscosity(&self, cell: usize) -> B::Scalar {
        self.eddy_viscosity.get(cell).copied().unwrap_or(B::Scalar::ZERO)
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
        h: B::Scalar,
    ) -> (B::Scalar, B::Scalar) {
        if h < self.h_min {
            return (B::Scalar::ZERO, B::Scalar::ZERO);
        }

        let nu = self.get_eddy_viscosity(cell);
        let grad = &self.velocity_gradient[cell];

        let fx = nu * h * grad.du_dx;
        let fy = nu * h * grad.dv_dy;

        (fx, fy)
    }
}

// 实现 TurbulenceClosure trait
impl<B: Backend> TurbulenceClosure<B::Scalar> for SmagorinskySolver<B> {
    fn name(&self) -> &'static str {
        "Smagorinsky"
    }

    fn is_3d(&self) -> bool {
        false // Smagorinsky 适用于 2D
    }

    fn eddy_viscosity(&self) -> &[B::Scalar] {
        self.eddy_viscosity.as_slice()
    }

    fn update(&mut self, velocity_gradients: &[VelocityGradient<B::Scalar>], cell_sizes: &[B::Scalar]) {
        self.set_velocity_gradients(velocity_gradients);
        let n = self.grid_scale.len().min(cell_sizes.len());
        self.grid_scale.as_slice_mut()[..n].copy_from_slice(&cell_sizes[..n]);
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
    pub eddy_viscosity: B::Buffer<B::Scalar>,
    /// 网格尺度 [m]（每个单元）
    pub grid_scale: B::Buffer<B::Scalar>,
    /// 速度梯度（外部提供）
    pub velocity_gradient: Vec<VelocityGradient<B::Scalar>>,
    /// 最小水深
    pub h_min: B::Scalar,
    /// 后端
    backend: B,
    /// 类型标记
    _marker: PhantomData<B>,
}

impl<B: Backend> TurbulenceConfig<B> {
    /// 创建新配置
    pub fn new(backend: B, n_cells: usize, model: TurbulenceModel<B::Scalar>) -> Self {
        Self {
            enabled: true,
            model,
            eddy_viscosity: backend.alloc_init(n_cells, B::Scalar::ZERO),
            grid_scale: backend.alloc_init(n_cells, backend.scalar_from_f64(10.0)),
            velocity_gradient: vec![VelocityGradient::default(); n_cells],
            h_min: backend.scalar_from_f64(1e-4),
            backend,
            _marker: PhantomData,
        }
    }

    /// 创建常数涡粘性配置
    pub fn constant(backend: B, n_cells: usize, nu: B::Scalar) -> Self {
        let model = TurbulenceModel::constant(&backend, nu);
        let mut config = Self::new(backend, n_cells, model);
        config.eddy_viscosity.fill(nu);
        config
    }

    /// 设置网格尺度
    pub fn set_grid_scale(&mut self, cell: usize, scale: B::Scalar) {
        if cell < self.grid_scale.len() {
            self.grid_scale[cell] = if scale <= B::Scalar::ZERO { B::Scalar::ZERO } else { scale };
        }
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
        self.eddy_viscosity.as_slice_mut()[..n].copy_from_slice(&nu[..n]);
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
        state: &ShallowWaterState<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];

        // 干单元不计算
        if h < self.h_min || ctx.is_dry(h) {
            return SourceContributionGeneric::zero();
        }

        let nu = match self.model {
            TurbulenceModel::None | TurbulenceModel::Disabled => 0.0,
            TurbulenceModel::ConstantViscosity(nu) => nu,
            TurbulenceModel::Smagorinsky { cs } => {
                let delta = self.grid_scale.get(cell).copied().unwrap_or(0.0);
                let grad = self
                    .velocity_gradient
                    .get(cell)
                    .copied()
                    .unwrap_or_default();
                let strain = grad.strain_rate_magnitude();
                let nu_sgs = (cs * delta) * (cs * delta) * strain;
                let min = TurbulenceModel::<f64>::min_eddy_viscosity(&self.backend);
                let max = TurbulenceModel::<f64>::max_eddy_viscosity(&self.backend);
                if nu_sgs < min { min } else if nu_sgs > max { max } else { nu_sgs }
            }
        };
        if nu < TurbulenceModel::<f64>::min_eddy_viscosity(&self.backend) {
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
        state: &ShallowWaterState<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
        rhs_hu: &mut Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
        rhs_hv: &mut Vec<f64>, // ALLOW_F64: 与 CpuBackend<f64> 配合
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.is_enabled() {
            return;
        }

        // 默认实现：逐单元计算并累加
        let n = state.n_cells();
        if rhs_h.len() < n || rhs_hu.len() < n || rhs_hv.len() < n {
            return;
        }

        for cell in 0..n {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
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
        let backend = CpuBackend::<f64>::new();
        let model = TurbulenceModel::constant(&backend, 0.01_f64);
        match model {
            TurbulenceModel::ConstantViscosity(nu) => {
                assert!((nu - 0.01).abs() < 1e-10);
            }
            _ => panic!("Expected ConstantViscosity model"),
        }
    }

    #[test]
    fn test_smagorinsky_solver_creation() {
        let backend = CpuBackend::<f64>::new();
        let solver = SmagorinskySolver::new(backend, 10, TurbulenceModel::default());
        assert_eq!(solver.grid_scale.len(), 10);
        assert_eq!(solver.eddy_viscosity.len(), 10);
    }

    #[test]
    fn test_smagorinsky_solver_constant_viscosity() {
        let backend = CpuBackend::<f64>::new();
        let model = TurbulenceModel::constant(&backend, 0.1_f64);
        let mut solver = SmagorinskySolver::new(backend, 10, model);
        solver.update_eddy_viscosity();

        for i in 0..10 {
            assert!((solver.eddy_viscosity[i] - 0.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_turbulence_config_creation() {
        let backend = CpuBackend::<f64>::new();
        let config = TurbulenceConfig::new(backend, 10, TurbulenceModel::default());
        assert!(config.enabled);
        assert_eq!(config.eddy_viscosity.len(), 10);
        assert_eq!(config.grid_scale.len(), 10);
    }

    #[test]
    fn test_turbulence_config_constant() {
        let backend = CpuBackend::<f64>::new();
        let config = TurbulenceConfig::constant(backend, 10, 0.05);
        assert!((config.eddy_viscosity[0] - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_source_term() {
        let backend = CpuBackend::<f64>::new();
        let mut config = TurbulenceConfig::constant(backend, 10, 0.1);
        config.velocity_gradient[0] = VelocityGradient::new(1.0, 0.0, 0.0, 1.0);

        let mut state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(
            CpuBackend::<f64>::new(),
            10,
        );
        // 设置测试状态
        for i in 0..10 {
            state.h[i] = 2.0;
            state.hu[i] = 2.0; // h * u = 2.0 * 1.0
            state.hv[i] = 1.0; // h * v = 2.0 * 0.5
        }
        let backend_ctx = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend_ctx, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        assert!((contrib.s_h - 0.0).abs() < 1e-10);
        // 验证动量源项非零
        assert!(contrib.s_hu.abs() > 0.0 || contrib.s_hv.abs() > 0.0);
    }

    #[test]
    fn test_turbulence_dry_cell() {
        let backend = CpuBackend::<f64>::new();
        let config = TurbulenceConfig::constant(backend, 10, 0.1);

        let state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(
            CpuBackend::<f64>::new(),
            10,
        );
        // h 默认为 0，是干单元
        let backend_ctx = CpuBackend::<f64>::new();
        let ctx = SourceContextGeneric::with_defaults(&backend_ctx, 0.0, 1.0);

        let contrib = config.compute_cell(0, &state, &ctx);

        assert!((contrib.s_hu - 0.0).abs() < 1e-10);
        assert!((contrib.s_hv - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_source_term_generic_trait() {
        let backend = CpuBackend::<f64>::new();
        let config = TurbulenceConfig::constant(backend, 10, 0.15);
        assert_eq!(config.name(), "Turbulence");
        assert_eq!(config.stiffness(), SourceStiffness::Explicit);
    }

    #[test]
    fn test_turbulence_closure_trait() {
        let backend = CpuBackend::<f64>::new();
        let model = TurbulenceModel::constant(&backend, 0.5);
        let mut solver = SmagorinskySolver::new(backend, 10, model);
        assert_eq!(solver.name(), "Smagorinsky");
        assert!(!solver.is_3d());

        let grads = vec![VelocityGradient::default(); 10];
        let sizes = vec![10.0_f64; 10];
        solver.update(&grads, &sizes);

        assert!((solver.eddy_viscosity[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_turbulence_model_smagorinsky() {
        let backend = CpuBackend::<f64>::new();
        let model = TurbulenceModel::smagorinsky(&backend, 0.2_f64);
        match model {
            TurbulenceModel::Smagorinsky { cs } => assert!((cs - 0.2).abs() < 1e-10),
            _ => panic!("Expected Smagorinsky model"),
        }
    }

    #[test]
    fn test_f32_precision() {
        let backend_f32 = CpuBackend::<f32>::new();
        let backend_f64 = CpuBackend::<f64>::new();
        let model_f32 = TurbulenceModel::<f32>::constant(&backend_f32, 0.1_f32);
        let model_f64 = TurbulenceModel::<f64>::constant(&backend_f64, 0.1_f64);

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
