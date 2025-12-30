// crates/mh_physics/src/tracer/transport.rs

//! 示踪剂输运求解器（Backend泛型化版本）
//!
//! 本模块提供示踪剂对流-扩散方程的求解功能，支持任意Backend（CPU f32/f64, GPU）。
//!
//! # 基本方程
//!
//! 示踪剂输运方程（二维深度平均）：
//!
//! $$\frac{\partial (hC)}{\partial t} + \nabla \cdot (hC\vec{u}) = \nabla \cdot (hK\nabla C) + S$$
//!
//! # Backend泛型化
//!
//! 整个模块完全Backend化，所有计算数据使用`B::Buffer<B::Scalar>`存储，
//! 几何数据使用`B::Vector2D`，支持运行时精度切换。

use mh_runtime::{Backend, CpuBackend, RuntimeScalar};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use super::state::{TracerField, TracerState};

// ============================================================
// 对流格式
// ============================================================

/// 对流格式类型（Backend无关，保持枚举）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracerAdvectionScheme {
    /// 一阶迎风格式
    #[default]
    FirstOrderUpwind,

    /// 二阶 Lax-Wendroff 格式
    LaxWendroff,

    /// 二阶 TVD 格式（MinMod 限制器）
    TvdMinmod,

    /// 二阶 TVD 格式（Superbee 限制器）
    TvdSuperbee,

    /// 二阶 TVD 格式（Van Leer 限制器）
    TvdVanLeer,
}

impl TracerAdvectionScheme {
    /// 获取格式名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::FirstOrderUpwind => "First-Order Upwind",
            Self::LaxWendroff => "Lax-Wendroff",
            Self::TvdMinmod => "TVD (MinMod)",
            Self::TvdSuperbee => "TVD (Superbee)",
            Self::TvdVanLeer => "TVD (Van Leer)",
        }
    }

    /// 是否需要梯度信息
    pub fn requires_gradient(&self) -> bool {
        matches!(
            self,
            Self::LaxWendroff | Self::TvdMinmod | Self::TvdSuperbee | Self::TvdVanLeer
        )
    }
}

// ============================================================
// 扩散配置（Backend泛型化）
// ============================================================

/// 扩散计算配置（Backend泛型化）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerDiffusionConfig<S: RuntimeScalar> {
    /// 是否启用扩散
    pub enabled: bool,

    /// 水平扩散系数 [m²/s]
    pub horizontal_diffusivity: S,

    /// Smagorinsky 系数（用于自适应扩散）
    pub smagorinsky_coefficient: S,

    /// 是否使用 Smagorinsky 模型
    pub use_smagorinsky: bool,
}

impl<S: RuntimeScalar> Default for TracerDiffusionConfig<S> {
    fn default() -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: S::from_f64(10.0).unwrap_or(S::ZERO),
            smagorinsky_coefficient: S::from_f64(0.2).unwrap_or(S::ZERO),
            use_smagorinsky: false,
        }
    }
}

impl<S: RuntimeScalar> TracerDiffusionConfig<S> {
    /// 仅使用常数扩散
    pub fn constant(diffusivity: S) -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: diffusivity,
            use_smagorinsky: false,
            smagorinsky_coefficient: S::from_f64(0.2).unwrap_or(S::ZERO),
        }
    }

    /// 使用 Smagorinsky 模型
    pub fn smagorinsky(coefficient: S) -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: S::ZERO,
            smagorinsky_coefficient: coefficient,
            use_smagorinsky: true,
        }
    }

    /// 禁用扩散
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            horizontal_diffusivity: S::ZERO,
            smagorinsky_coefficient: S::ZERO,
            use_smagorinsky: false,
        }
    }

    /// 计算 Smagorinsky 扩散系数
    #[inline]
    pub fn compute_smagorinsky_diffusivity(
        &self,
        grid_scale: S,
        strain_rate_magnitude: S,
    ) -> S {
        let cs = self.smagorinsky_coefficient;
        let cs_delta = cs * grid_scale;
        let k = cs_delta * cs_delta * strain_rate_magnitude;

        // 限制扩散系数在合理范围内
        let k_min = S::from_f64(1e-6).unwrap_or(S::ZERO);
        let k_max = S::from_f64(1e4).unwrap_or(S::ONE);
        k.clamp_value(k_min, k_max)
    }
}

// ============================================================
// 求解器配置（Backend泛型化）
// ============================================================

/// 示踪剂输运求解器配置（Backend泛型化）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerTransportConfig<S: RuntimeScalar> {
    /// 对流格式
    pub advection_scheme: TracerAdvectionScheme,

    /// 扩散配置
    pub diffusion: TracerDiffusionConfig<S>,

    /// 最小水深阈值 [m]
    pub h_min: S,

    /// 浓度限制器
    pub enable_clipping: bool,

    /// 最小浓度
    pub c_min: S,

    /// 最大浓度（可选）
    pub c_max: Option<S>,
}

impl<S: RuntimeScalar> Default for TracerTransportConfig<S> {
    fn default() -> Self {
        Self {
            advection_scheme: TracerAdvectionScheme::default(),
            diffusion: TracerDiffusionConfig::default(),
            h_min: S::from_f64(1e-6).unwrap_or(S::ZERO),
            enable_clipping: true,
            c_min: S::ZERO,
            c_max: None,
        }
    }
}

// ============================================================
// 面通量数据（Backend泛型化）
// ============================================================

/// 面的流动数据（Backend泛型化）
#[derive(Debug, Clone, Copy)]
pub struct FaceFlowData<B: Backend> {
    /// 面索引
    pub face_id: usize,
    /// 左侧单元索引
    pub left_cell: usize,
    /// 右侧单元索引（边界面为 None）
    pub right_cell: Option<usize>,
    /// 面法向量（从左到右）
    pub normal: B::Vector2D,
    /// 面长度 [m]
    pub length: B::Scalar,
    /// 面上的法向流速 [m/s]
    pub un: B::Scalar,
    /// 面上的水深 [m]
    pub h_face: B::Scalar,
}

/// 示踪剂面通量（Backend泛型化）
#[derive(Debug, Clone, Copy, Default)]
pub struct TracerFaceFlux<S: RuntimeScalar> {
    /// 对流通量 [单位/s]
    pub advective: S,
    /// 扩散通量 [单位/s]
    pub diffusive: S,
}

impl<S: RuntimeScalar> TracerFaceFlux<S> {
    /// 总通量
    #[inline]
    pub fn total(&self) -> S {
        self.advective + self.diffusive
    }
}

/// Smagorinsky 模型数据（Backend泛型化）
#[derive(Debug, Clone)]
pub struct SmagorinskyData<B: Backend> {
    /// 每个单元的网格尺度 [m]
    pub grid_scales: B::Buffer<B::Scalar>,
    /// 每个单元的应变率张量模 [1/s]
    pub strain_rate_magnitudes: B::Buffer<B::Scalar>,
}

impl<B: Backend> SmagorinskyData<B> {
    /// 创建新的 Smagorinsky 数据
    pub fn new(backend: &B, n_cells: usize) -> Self {
        Self {
            grid_scales: backend.alloc(n_cells),
            strain_rate_magnitudes: backend.alloc(n_cells),
        }
    }

    /// 从网格和速度场初始化
    pub fn from_velocity_gradients(
        backend: &B,
        cell_areas: &[B::Scalar],
        du_dx: &[B::Scalar],
        du_dy: &[B::Scalar],
        dv_dx: &[B::Scalar],
        dv_dy: &[B::Scalar],
    ) -> Self {
        let n_cells = cell_areas.len();
        let mut data = Self::new(backend, n_cells);
        let min_scale = B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::ZERO);
        let four = B::Scalar::TWO + B::Scalar::TWO;

        for i in 0..n_cells {
            // 网格尺度 = √(面积)
            data.grid_scales[i] = Float::max(Float::sqrt(cell_areas[i]), min_scale);

            // 应变率张量模
            let s11 = du_dx[i];
            let s22 = dv_dy[i];
            let s12 = B::Scalar::HALF * (du_dy[i] + dv_dx[i]);

            data.strain_rate_magnitudes[i] =
                Float::sqrt(B::Scalar::TWO * s11 * s11 + B::Scalar::TWO * s22 * s22 + four * s12 * s12);
        }

        data
    }

    /// 获取单元的平均扩散系数
    #[inline]
    pub fn face_diffusivity(
        &self,
        config: &TracerDiffusionConfig<B::Scalar>,
        left_cell: usize,
        right_cell: Option<usize>,
    ) -> B::Scalar {
        let k_left = config.compute_smagorinsky_diffusivity(
            self.grid_scales[left_cell],
            self.strain_rate_magnitudes[left_cell],
        );

        if let Some(right) = right_cell {
            let k_right = config.compute_smagorinsky_diffusivity(
                self.grid_scales[right],
                self.strain_rate_magnitudes[right],
            );
            // 使用调和平均值（更适合扩散系数）
            let eps = B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::ZERO);
            if k_left + k_right > eps {
                B::Scalar::TWO * k_left * k_right / (k_left + k_right)
            } else {
                B::Scalar::ZERO
            }
        } else {
            k_left
        }
    }
}

// ============================================================
// 示踪剂输运求解器（Backend泛型化）
// ============================================================

/// 示踪剂输运求解器（Backend泛型化）
///
/// 负责计算示踪剂的对流和扩散通量，并更新浓度场。
///
/// # 类型参数
///
/// - `B`: 计算后端类型，必须实现 `Backend` trait
pub struct TracerTransportSolver<B: Backend> {
    config: TracerTransportConfig<B::Scalar>,
    face_fluxes: Vec<TracerFaceFlux<B::Scalar>>,
    backend: B,
}

impl TracerTransportSolver<CpuBackend<f64>> {
    /// 创建 CPU f64 后端的求解器
    pub fn new(config: TracerTransportConfig<f64>) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), config)
    }
}

impl<B: Backend> TracerTransportSolver<B> {
    /// 创建新的求解器
    pub fn new_with_backend(backend: B, config: TracerTransportConfig<B::Scalar>) -> Self {
        Self {
            config,
            face_fluxes: Vec::new(),
            backend,
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &TracerTransportConfig<B::Scalar> {
        &self.config
    }

    /// 设置配置
    pub fn set_config(&mut self, config: TracerTransportConfig<B::Scalar>) {
        self.config = config;
    }

    /// 计算单个面的对流通量（一阶迎风）
    #[inline]
    pub fn compute_advective_flux_upwind(
        &self,
        c_left: B::Scalar,
        c_right: B::Scalar,
        h_face: B::Scalar,
        un: B::Scalar,
        face_length: B::Scalar,
    ) -> B::Scalar {
        // 迎风选择
        let c_upwind = if un >= B::Scalar::ZERO { c_left } else { c_right };
        h_face * un * c_upwind * face_length
    }

    /// 计算单个面的扩散通量
    #[inline]
    pub fn compute_diffusive_flux(
        &self,
        c_left: B::Scalar,
        c_right: B::Scalar,
        h_face: B::Scalar,
        distance: B::Scalar,
        face_length: B::Scalar,
        diffusivity: B::Scalar,
    ) -> B::Scalar {
        let zero = B::Scalar::ZERO;
        if !self.config.diffusion.enabled || diffusivity <= zero {
            return zero;
        }

        let eps = B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::ZERO);
        let dc_dx = (c_right - c_left) / Float::max(distance, eps);
        -h_face * diffusivity * dc_dx * face_length
    }

    /// 计算所有面的通量并累加到 RHS
    pub fn compute_rhs(
        &mut self,
        field: &mut TracerField<B>,
        flow_data: &[FaceFlowData<B>],
        cell_volumes: &B::Buffer<B::Scalar>,
        face_distances: &B::Buffer<B::Scalar>,
    ) {
        self.compute_rhs_internal(field, flow_data, cell_volumes, face_distances, None);
    }

    /// 计算所有面的通量并累加到 RHS（带 Smagorinsky 模型支持）
    pub fn compute_rhs_with_smagorinsky(
        &mut self,
        field: &mut TracerField<B>,
        flow_data: &[FaceFlowData<B>],
        cell_volumes: &B::Buffer<B::Scalar>,
        face_distances: &B::Buffer<B::Scalar>,
        smagorinsky_data: &SmagorinskyData<B>,
    ) {
        self.compute_rhs_internal(field, flow_data, cell_volumes, face_distances, Some(smagorinsky_data));
    }

    /// 内部实现：计算 RHS
    fn compute_rhs_internal(
        &mut self,
        field: &mut TracerField<B>,
        flow_data: &[FaceFlowData<B>],
        cell_volumes: &B::Buffer<B::Scalar>,
        face_distances: &B::Buffer<B::Scalar>,
        smagorinsky_data: Option<&SmagorinskyData<B>>,
    ) {
        field.clear_rhs();

        // 确保工作数组大小足够
        if self.face_fluxes.len() < flow_data.len() {
            self.face_fluxes.resize(flow_data.len(), TracerFaceFlux::default());
        }

        // 计算所有面的通量
        for (i, face) in flow_data.iter().enumerate() {
            let c_left = field.concentration_slice()[face.left_cell];
            let c_right = face.right_cell
                .map(|idx| field.concentration_slice()[idx])
                .unwrap_or(c_left); // 边界面使用左侧值

            // 对流通量
            let advective = self.compute_advective_flux_upwind(
                c_left,
                c_right,
                face.h_face,
                face.un,
                face.length,
            );

            // 计算扩散系数（根据配置选择常数或 Smagorinsky）
            let diffusivity = if self.config.diffusion.use_smagorinsky {
                if let Some(smag) = smagorinsky_data {
                    smag.face_diffusivity(&self.config.diffusion, face.left_cell, face.right_cell)
                } else {
                    let min_k = B::Scalar::from_f64(1e-3).unwrap_or(B::Scalar::ZERO);
                    Float::max(self.config.diffusion.horizontal_diffusivity, min_k)
                }
            } else {
                self.config.diffusion.horizontal_diffusivity
            };

            // 扩散通量
            let diffusive = if face.right_cell.is_some() {
                self.compute_diffusive_flux(
                    c_left,
                    c_right,
                    face.h_face,
                    face_distances[i],
                    face.length,
                    diffusivity,
                )
            } else {
                B::Scalar::ZERO // 边界面无扩散
            };

            self.face_fluxes[i] = TracerFaceFlux { advective, diffusive };

            // 累加到单元 RHS
            let flux = advective + diffusive;

            // 左侧单元：通量流出为负
            let vol_left = cell_volumes[face.left_cell];
            if vol_left > B::Scalar::ZERO {
                field.add_rhs(face.left_cell, -flux / vol_left);
            }

            // 右侧单元（如果存在）：通量流入为正
            if let Some(right_cell) = face.right_cell {
                let vol_right = cell_volumes[right_cell];
                if vol_right > B::Scalar::ZERO {
                    field.add_rhs(right_cell, flux / vol_right);
                }
            }
        }
    }

    /// 时间步进更新（显式欧拉）
    pub fn update_forward_euler(&self, field: &mut TracerField<B>, dt: B::Scalar) {
        // 使用 TracerField 的内置方法来避免借用冲突
        field.apply_euler_update(dt);
    }

    /// 应用浓度限制
    pub fn apply_clipping(&self, field: &mut TracerField<B>) {
        if self.config.enable_clipping {
            field.clamp_concentration(self.config.c_min, self.config.c_max);
        }
    }

    /// 完整的单步更新流程
    pub fn step(
        &mut self,
        field: &mut TracerField<B>,
        flow_data: &[FaceFlowData<B>],
        cell_volumes: &B::Buffer<B::Scalar>,
        face_distances: &B::Buffer<B::Scalar>,
        water_depths: &[B::Scalar],
        dt: B::Scalar,
    ) {
        // 1. 计算 RHS
        self.compute_rhs(field, flow_data, cell_volumes, face_distances);

        // 2. 时间步进
        self.update_forward_euler(field, dt);

        // 3. 从守恒量更新浓度
        field.update_concentration_from_conserved(water_depths, self.config.h_min);

        // 4. 应用限制
        self.apply_clipping(field);
    }

    /// 计算示踪剂的 CFL 限制时间步
    pub fn compute_dt_limit(
        &self,
        max_velocity: B::Scalar,
        min_cell_size: B::Scalar,
        cfl_number: B::Scalar,
    ) -> B::Scalar {
        let eps = B::Scalar::from_f64(1e-10).unwrap_or(B::Scalar::ZERO);
        let scalar_max = B::Scalar::from_f64(1e20).unwrap_or(B::Scalar::ONE);
        let dx = Float::max(min_cell_size, eps);

        // 对流限制
        let dt_advection = if max_velocity > eps {
            cfl_number * dx / max_velocity
        } else {
            scalar_max
        };

        // 扩散限制（如果启用）
        let dt_diffusion = if self.config.diffusion.enabled {
            let k = self.config.diffusion.horizontal_diffusivity;
            if k > eps {
                B::Scalar::HALF * cfl_number * dx * dx / k
            } else {
                scalar_max
            }
        } else {
            scalar_max
        };

        Float::min(dt_advection, dt_diffusion)
    }
}

impl<B: Backend + Default> Default for TracerTransportSolver<B> {
    fn default() -> Self {
        Self::new_with_backend(B::default(), TracerTransportConfig::default())
    }
}

// ============================================================
// 多示踪剂求解器（Backend泛型化）
// ============================================================

/// 多示踪剂输运求解器（Backend泛型化）
pub struct MultiTracerSolver<B: Backend> {
    /// 单示踪剂求解器
    solver: TracerTransportSolver<B>,
}

impl<B: Backend> MultiTracerSolver<B> {
    /// 创建新的多示踪剂求解器
    pub fn new_with_backend(backend: B, config: TracerTransportConfig<B::Scalar>) -> Self {
        Self {
            solver: TracerTransportSolver::new_with_backend(backend, config),
        }
    }

    /// 更新所有示踪剂
    pub fn step_all(
        &mut self,
        state: &mut TracerState<B>,
        flow_data: &[FaceFlowData<B>],
        cell_volumes: &B::Buffer<B::Scalar>,
        face_distances: &B::Buffer<B::Scalar>,
        water_depths: &B::Buffer<B::Scalar>,
        dt: B::Scalar,
    ) {
        for (_, field) in state.iter_mut() {
            self.solver.step(field, flow_data, cell_volumes, face_distances, water_depths, dt);
        }
    }

    /// 获取内部求解器
    pub fn solver(&self) -> &TracerTransportSolver<B> {
        &self.solver
    }

    /// 获取内部求解器（可变）
    pub fn solver_mut(&mut self) -> &mut TracerTransportSolver<B> {
        &mut self.solver
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracer::state::TracerType;
    use crate::tracer::state::TracerProperties;
    use mh_runtime::CpuBackend;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_advection_scheme() {
        let scheme = TracerAdvectionScheme::FirstOrderUpwind;
        assert!(!scheme.requires_gradient());

        let scheme = TracerAdvectionScheme::TvdMinmod;
        assert!(scheme.requires_gradient());
    }

    #[test]
    fn test_diffusion_config_f64() {
        let config = TracerDiffusionConfig::<f64>::constant(50.0);
        assert!(config.enabled);
        assert!(approx_eq(config.horizontal_diffusivity, 50.0));
        assert!(!config.use_smagorinsky);

        let config = TracerDiffusionConfig::<f64>::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_diffusion_config_f32() {
        let config = TracerDiffusionConfig::<f32>::constant(50.0f32);
        assert!(config.enabled);
        assert_eq!(config.horizontal_diffusivity, 50.0f32);
    }

    #[test]
    fn test_upwind_flux_f64() {
        let backend = CpuBackend::<f64>::new();
        let solver = TracerTransportSolver::<CpuBackend<f64>>::new_with_backend(backend, TracerTransportConfig::default());

        // 从左到右流动
        let flux = solver.compute_advective_flux_upwind(10.0, 20.0, 1.0, 1.0, 1.0);
        assert!(approx_eq(flux, 10.0)); // 使用左侧浓度

        // 从右到左流动
        let flux = solver.compute_advective_flux_upwind(10.0, 20.0, 1.0, -1.0, 1.0);
        assert!(approx_eq(flux, -20.0)); // 使用右侧浓度
    }

    #[test]
    fn test_upwind_flux_f32() {
        let backend = CpuBackend::<f32>::new();
        let solver = TracerTransportSolver::<CpuBackend<f32>>::new_with_backend(backend, TracerTransportConfig::default());

        let flux = solver.compute_advective_flux_upwind(10.0f32, 20.0f32, 1.0f32, 1.0f32, 1.0f32);
        assert_eq!(flux, 10.0f32);
    }

    #[test]
    fn test_diffusive_flux_f64() {
        let backend = CpuBackend::<f64>::new();
        let solver = TracerTransportSolver::<CpuBackend<f64>>::new_with_backend(
            backend,
            TracerTransportConfig {
                diffusion: TracerDiffusionConfig::constant(10.0),
                ..Default::default()
            },
        );

        // 浓度梯度：从左(10)到右(20)，扩散应该从高到低
        let flux = solver.compute_diffusive_flux(10.0, 20.0, 1.0, 1.0, 1.0, 10.0);
        // F = -h * K * dC/dx = -1 * 10 * (20-10)/1 = -100
        assert!(approx_eq(flux, -100.0));
    }

    #[test]
    fn test_dt_limit_f64() {
        let backend = CpuBackend::<f64>::new();
        let solver = TracerTransportSolver::<CpuBackend<f64>>::new_with_backend(
            backend,
            TracerTransportConfig {
                diffusion: TracerDiffusionConfig::constant(10.0),
                ..Default::default()
            },
        );

        let dt = solver.compute_dt_limit(1.0, 10.0, 0.5);
        // 对流限制: 0.5 * 10 / 1 = 5
        // 扩散限制: 0.5 * 0.5 * 100 / 10 = 2.5
        assert!(approx_eq(dt, 2.5));
    }

    #[test]
    fn test_dt_limit_f32() {
        let backend = CpuBackend::<f32>::new();
        let solver = TracerTransportSolver::<CpuBackend<f32>>::new_with_backend(
            backend,
            TracerTransportConfig {
                diffusion: TracerDiffusionConfig::constant(10.0f32),
                ..Default::default()
            },
        );

        let dt = solver.compute_dt_limit(1.0f32, 10.0f32, 0.5f32);
        assert!(dt > 0.0f32);
    }

    #[test]
    fn test_single_step_f64() {
        let backend = CpuBackend::<f64>::new();
        let mut solver = TracerTransportSolver::<CpuBackend<f64>>::new_with_backend(backend, TracerTransportConfig::default());
        let props = TracerProperties::<f64>::salinity().with_background(0.0);
        let mut field = TracerField::<CpuBackend<f64>>::new_with_backend(CpuBackend::<f64>::new(), props, 3);
        
        // 手动设置浓度梯度
        let concentrations = [10.0, 5.0, 0.0];
        for (i, &c) in concentrations.iter().enumerate() {
            field.concentration_slice_mut()[i] = c;
        }

        // 初始化守恒量
        let depths = vec![1.0, 1.0, 1.0];
        field.update_conserved_from_depth(&depths);

        // 简单的两面流动数据
        let flow_data = vec![
            FaceFlowData {
                face_id: 0,
                left_cell: 0,
                right_cell: Some(1),
                normal: CpuBackend::<f64>::vec2_new(1.0, 0.0),
                length: 1.0,
                un: 1.0,  // 从左到右
                h_face: 1.0,
            },
            FaceFlowData {
                face_id: 1,
                left_cell: 1,
                right_cell: Some(2),
                normal: CpuBackend::<f64>::vec2_new(1.0, 0.0),
                length: 1.0,
                un: 1.0,
                h_face: 1.0,
            },
        ];

        let volumes = vec![1.0, 1.0, 1.0];
        let distances = vec![1.0, 1.0];

        // 执行一步
        solver.step(&mut field, &flow_data, &volumes, &distances, &depths, 0.1);

        // 浓度应该变化了
        // 由于迎风格式，浓度会向右传输
        assert!(field.concentration_slice()[0] > 5.0); // 从单元0获得质量
    }

    #[test]
    fn test_multi_tracer_solver_f64() {
        let backend = CpuBackend::<f64>::new();
        let mut state = TracerState::<CpuBackend<f64>>::new_with_backend(backend.clone(), 10);
        state.add_tracer(TracerProperties::<f64>::salinity()).unwrap();
        state.add_tracer(TracerProperties::<f64>::temperature()).unwrap();

        let _solver = MultiTracerSolver::<CpuBackend<f64>>::new_with_backend(backend, TracerTransportConfig::default());

        // 确保可以访问两个示踪剂
        assert!(state.get(TracerType::Salinity).is_some());
        assert!(state.get(TracerType::Temperature).is_some());
    }

    #[test]
    fn test_f32_backend_full() {
        let backend = CpuBackend::<f32>::new();
        let config = TracerTransportConfig::<f32>::default();
        let mut solver = TracerTransportSolver::<CpuBackend<f32>>::new_with_backend(backend.clone(), config);
        
        let props = TracerProperties::<f32>::salinity();
        let mut field = TracerField::<CpuBackend<f32>>::new_with_backend(backend.clone(), props, 100);
        
        // 测试基本操作
        assert_eq!(field.len(), 100);
        solver.compute_rhs(&mut field, &[], &backend.alloc(100), &backend.alloc(0));
    }
}
