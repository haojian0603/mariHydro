// crates/mh_physics/src/tracer/transport.rs

//! 示踪剂输运求解器
//!
//! 本模块提供示踪剂对流-扩散方程的求解功能：
//! - TracerTransportSolver: 主求解器
//! - TracerAdvectionScheme: 对流格式
//! - TracerDiffusionConfig: 扩散配置
//!
//! # 基本方程
//!
//! 示踪剂输运方程（二维深度平均）：
//!
//! $$\frac{\partial (hC)}{\partial t} + \nabla \cdot (hC\vec{u}) = \nabla \cdot (hK\nabla C) + S$$
//!
//! 其中：
//! - $C$: 示踪剂浓度
//! - $h$: 水深
//! - $\vec{u}$: 深度平均流速
//! - $K$: 扩散系数张量
//! - $S$: 源汇项
//!
//! # 迁移说明
//!
//! 从 legacy_src/tracer/tracer_transport.rs 迁移，改进：
//! - 使用 trait 抽象网格访问
//! - 支持多种对流格式
//! - 与新架构的时间积分器集成

use glam::DVec2;
use serde::{Deserialize, Serialize};
use super::state::{TracerField, TracerState};

// ============================================================
// 对流格式
// ============================================================

/// 对流格式类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TracerAdvectionScheme {
    /// 一阶迎风格式
    ///
    /// 简单稳定，但数值扩散较大。
    #[default]
    FirstOrderUpwind,

    /// 二阶 Lax-Wendroff 格式
    ///
    /// 精度更高，但可能产生振荡。
    LaxWendroff,

    /// 二阶 TVD 格式（MinMod 限制器）
    ///
    /// 平衡精度和稳定性。
    TvdMinmod,

    /// 二阶 TVD 格式（Superbee 限制器）
    ///
    /// 更尖锐的间断，但可能过度压缩。
    TvdSuperbee,

    /// 二阶 TVD 格式（Van Leer 限制器）
    ///
    /// 平滑的限制，适用于一般情况。
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
// 扩散配置
// ============================================================

/// 扩散计算配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerDiffusionConfig {
    /// 是否启用扩散
    pub enabled: bool,

    /// 水平扩散系数 [m²/s]
    ///
    /// 可以是常数或基于网格尺度的 Smagorinsky 公式。
    pub horizontal_diffusivity: f64,

    /// Smagorinsky 系数（用于自适应扩散）
    ///
    /// K = C_s * dx² * |S|，其中 |S| 是应变率。
    pub smagorinsky_coefficient: f64,

    /// 是否使用 Smagorinsky 模型
    pub use_smagorinsky: bool,
}

impl Default for TracerDiffusionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: 10.0, // 典型值 1-100 m²/s
            smagorinsky_coefficient: 0.2,
            use_smagorinsky: false,
        }
    }
}

impl TracerDiffusionConfig {
    /// 仅使用常数扩散
    pub fn constant(diffusivity: f64) -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: diffusivity,
            use_smagorinsky: false,
            ..Default::default()
        }
    }

    /// 使用 Smagorinsky 模型
    pub fn smagorinsky(coefficient: f64) -> Self {
        Self {
            enabled: true,
            horizontal_diffusivity: 0.0,
            smagorinsky_coefficient: coefficient,
            use_smagorinsky: true,
        }
    }

    /// 禁用扩散
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }
}

// ============================================================
// 求解器配置
// ============================================================

/// 示踪剂输运求解器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracerTransportConfig {
    /// 对流格式
    pub advection_scheme: TracerAdvectionScheme,

    /// 扩散配置
    pub diffusion: TracerDiffusionConfig,

    /// 最小水深阈值 [m]
    ///
    /// 水深小于此值的单元不计算示踪剂。
    pub h_min: f64,

    /// 浓度限制器
    ///
    /// 防止产生负浓度或超过物理范围的浓度。
    pub enable_clipping: bool,

    /// 最小浓度
    pub c_min: f64,

    /// 最大浓度（可选）
    pub c_max: Option<f64>,
}

impl Default for TracerTransportConfig {
    fn default() -> Self {
        Self {
            advection_scheme: TracerAdvectionScheme::default(),
            diffusion: TracerDiffusionConfig::default(),
            h_min: 1e-6,
            enable_clipping: true,
            c_min: 0.0,
            c_max: None,
        }
    }
}

// ============================================================
// 面通量数据
// ============================================================

/// 面的流动数据（用于计算示踪剂通量）
#[derive(Debug, Clone, Copy)]
pub struct FaceFlowData {
    /// 面索引
    pub face_id: usize,
    /// 左侧单元索引
    pub left_cell: usize,
    /// 右侧单元索引（边界面为 None）
    pub right_cell: Option<usize>,
    /// 面法向量（从左到右）
    pub normal: DVec2,
    /// 面长度 [m]
    pub length: f64,
    /// 面上的法向流速 [m/s]
    pub un: f64,
    /// 面上的水深 [m]
    pub h_face: f64,
}

/// 示踪剂面通量
#[derive(Debug, Clone, Copy, Default)]
pub struct TracerFaceFlux {
    /// 对流通量 [单位/s]
    pub advective: f64,
    /// 扩散通量 [单位/s]
    pub diffusive: f64,
}

impl TracerFaceFlux {
    /// 总通量
    pub fn total(&self) -> f64 {
        self.advective + self.diffusive
    }
}

// ============================================================
// 示踪剂输运求解器
// ============================================================

/// 示踪剂输运求解器
///
/// 负责计算示踪剂的对流和扩散通量，并更新浓度场。
///
/// # 使用流程
///
/// 1. 创建求解器实例
/// 2. 准备面流动数据（从水动力求解器获取）
/// 3. 计算通量并更新 RHS
/// 4. 使用时间积分器更新守恒量
/// 5. 从守恒量反算浓度
///
/// # 示例
///
/// ```ignore
/// use mh_physics::tracer::{TracerTransportSolver, TracerTransportConfig};
///
/// let solver = TracerTransportSolver::new(TracerTransportConfig::default());
/// ```
pub struct TracerTransportSolver {
    config: TracerTransportConfig,
    
    /// 临时工作数组：面通量
    face_fluxes: Vec<TracerFaceFlux>,
}

impl TracerTransportSolver {
    /// 创建新的求解器
    pub fn new(config: TracerTransportConfig) -> Self {
        Self {
            config,
            face_fluxes: Vec::new(),
        }
    }

    /// 获取配置引用
    pub fn config(&self) -> &TracerTransportConfig {
        &self.config
    }

    /// 设置配置
    pub fn set_config(&mut self, config: TracerTransportConfig) {
        self.config = config;
    }

    /// 计算单个面的对流通量（一阶迎风）
    ///
    /// # 参数
    /// - `c_left`: 左侧单元浓度
    /// - `c_right`: 右侧单元浓度
    /// - `h_face`: 面上水深
    /// - `un`: 面法向速度（正值从左到右）
    /// - `face_length`: 面长度
    ///
    /// # 返回
    /// 对流通量（正值表示从左到右输送）
    pub fn compute_advective_flux_upwind(
        &self,
        c_left: f64,
        c_right: f64,
        h_face: f64,
        un: f64,
        face_length: f64,
    ) -> f64 {
        // 迎风选择
        let c_upwind = if un >= 0.0 { c_left } else { c_right };
        h_face * un * c_upwind * face_length
    }

    /// 计算单个面的扩散通量
    ///
    /// # 参数
    /// - `c_left`: 左侧单元浓度
    /// - `c_right`: 右侧单元浓度
    /// - `h_face`: 面上水深
    /// - `distance`: 单元中心间距
    /// - `face_length`: 面长度
    /// - `diffusivity`: 扩散系数 [m²/s]
    ///
    /// # 返回
    /// 扩散通量（正值表示从左到右输送）
    pub fn compute_diffusive_flux(
        &self,
        c_left: f64,
        c_right: f64,
        h_face: f64,
        distance: f64,
        face_length: f64,
        diffusivity: f64,
    ) -> f64 {
        if !self.config.diffusion.enabled || diffusivity <= 0.0 {
            return 0.0;
        }

        // 扩散通量: F = -h * K * dC/dx
        let dc_dx = (c_right - c_left) / distance.max(1e-10);
        -h_face * diffusivity * dc_dx * face_length
    }

    /// 计算所有面的通量并累加到 RHS
    ///
    /// # 参数
    /// - `field`: 示踪剂场
    /// - `flow_data`: 面流动数据
    /// - `cell_volumes`: 单元体积
    /// - `face_distances`: 面对应的单元中心间距
    pub fn compute_rhs(
        &mut self,
        field: &mut TracerField,
        flow_data: &[FaceFlowData],
        cell_volumes: &[f64],
        face_distances: &[f64],
    ) {
        field.clear_rhs();

        let diffusivity = if self.config.diffusion.use_smagorinsky {
            // TODO: 计算 Smagorinsky 扩散系数
            self.config.diffusion.horizontal_diffusivity
        } else {
            self.config.diffusion.horizontal_diffusivity
        };

        // 确保工作数组大小足够
        if self.face_fluxes.len() < flow_data.len() {
            self.face_fluxes.resize(flow_data.len(), TracerFaceFlux::default());
        }

        // 计算所有面的通量
        for (i, face) in flow_data.iter().enumerate() {
            let c_left = field.concentration(face.left_cell);
            let c_right = face.right_cell
                .map(|idx| field.concentration(idx))
                .unwrap_or(c_left); // 边界面使用左侧值

            // 对流通量
            let advective = self.compute_advective_flux_upwind(
                c_left,
                c_right,
                face.h_face,
                face.un,
                face.length,
            );

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
                0.0 // 边界面无扩散
            };

            self.face_fluxes[i] = TracerFaceFlux { advective, diffusive };

            // 累加到单元 RHS
            let flux = advective + diffusive;

            // 左侧单元：通量流出为负
            let vol_left = cell_volumes[face.left_cell];
            if vol_left > 0.0 {
                field.add_rhs(face.left_cell, -flux / vol_left);
            }

            // 右侧单元（如果存在）：通量流入为正
            if let Some(right_cell) = face.right_cell {
                let vol_right = cell_volumes[right_cell];
                if vol_right > 0.0 {
                    field.add_rhs(right_cell, flux / vol_right);
                }
            }
        }
    }

    /// 时间步进更新
    ///
    /// 使用显式欧拉格式更新守恒量。
    ///
    /// # 参数
    /// - `field`: 示踪剂场
    /// - `dt`: 时间步长 [s]
    pub fn update_forward_euler(&self, field: &mut TracerField, dt: f64) {
        field.apply_euler_update(dt);
    }

    /// 应用浓度限制
    pub fn apply_clipping(&self, field: &mut TracerField) {
        if self.config.enable_clipping {
            field.clamp_concentration(self.config.c_min, self.config.c_max);
        }
    }

    /// 完整的单步更新流程
    ///
    /// # 参数
    /// - `field`: 示踪剂场
    /// - `flow_data`: 面流动数据
    /// - `cell_volumes`: 单元体积
    /// - `face_distances`: 面对应的单元中心间距
    /// - `water_depths`: 水深数组（用于更新浓度）
    /// - `dt`: 时间步长
    pub fn step(
        &mut self,
        field: &mut TracerField,
        flow_data: &[FaceFlowData],
        cell_volumes: &[f64],
        face_distances: &[f64],
        water_depths: &[f64],
        dt: f64,
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
    ///
    /// # 参数
    /// - `max_velocity`: 最大流速 [m/s]
    /// - `min_cell_size`: 最小单元尺寸 [m]
    /// - `cfl_number`: CFL 数（默认 0.5）
    ///
    /// # 返回
    /// 建议的最大时间步长 [s]
    pub fn compute_dt_limit(
        &self,
        max_velocity: f64,
        min_cell_size: f64,
        cfl_number: f64,
    ) -> f64 {
        let dx = min_cell_size.max(1e-10);

        // 对流限制
        let dt_advection = if max_velocity > 1e-10 {
            cfl_number * dx / max_velocity
        } else {
            f64::MAX
        };

        // 扩散限制（如果启用）
        let dt_diffusion = if self.config.diffusion.enabled {
            let k = self.config.diffusion.horizontal_diffusivity;
            if k > 1e-10 {
                0.5 * cfl_number * dx * dx / k
            } else {
                f64::MAX
            }
        } else {
            f64::MAX
        };

        dt_advection.min(dt_diffusion)
    }
}

impl Default for TracerTransportSolver {
    fn default() -> Self {
        Self::new(TracerTransportConfig::default())
    }
}

// ============================================================
// 多示踪剂求解器
// ============================================================

/// 多示踪剂输运求解器
///
/// 包装 TracerTransportSolver，支持同时处理多个示踪剂。
pub struct MultiTracerSolver {
    /// 单示踪剂求解器
    solver: TracerTransportSolver,
}

impl MultiTracerSolver {
    /// 创建新的多示踪剂求解器
    pub fn new(config: TracerTransportConfig) -> Self {
        Self {
            solver: TracerTransportSolver::new(config),
        }
    }

    /// 更新所有示踪剂
    pub fn step_all(
        &mut self,
        state: &mut TracerState,
        flow_data: &[FaceFlowData],
        cell_volumes: &[f64],
        face_distances: &[f64],
        water_depths: &[f64],
        dt: f64,
    ) {
        for (_, field) in state.iter_mut() {
            self.solver.step(field, flow_data, cell_volumes, face_distances, water_depths, dt);
        }
    }

    /// 获取内部求解器
    pub fn solver(&self) -> &TracerTransportSolver {
        &self.solver
    }

    /// 获取内部求解器（可变）
    pub fn solver_mut(&mut self) -> &mut TracerTransportSolver {
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
    use super::super::state::TracerProperties;

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
    fn test_diffusion_config() {
        let config = TracerDiffusionConfig::constant(50.0);
        assert!(config.enabled);
        assert!(approx_eq(config.horizontal_diffusivity, 50.0));
        assert!(!config.use_smagorinsky);

        let config = TracerDiffusionConfig::disabled();
        assert!(!config.enabled);
    }

    #[test]
    fn test_upwind_flux() {
        let solver = TracerTransportSolver::default();

        // 从左到右流动
        let flux = solver.compute_advective_flux_upwind(10.0, 20.0, 1.0, 1.0, 1.0);
        assert!(approx_eq(flux, 10.0)); // 使用左侧浓度

        // 从右到左流动
        let flux = solver.compute_advective_flux_upwind(10.0, 20.0, 1.0, -1.0, 1.0);
        assert!(approx_eq(flux, -20.0)); // 使用右侧浓度
    }

    #[test]
    fn test_diffusive_flux() {
        let solver = TracerTransportSolver::new(TracerTransportConfig {
            diffusion: TracerDiffusionConfig::constant(10.0),
            ..Default::default()
        });

        // 浓度梯度：从左(10)到右(20)，扩散应该从高到低
        let flux = solver.compute_diffusive_flux(10.0, 20.0, 1.0, 1.0, 1.0, 10.0);
        // F = -h * K * dC/dx = -1 * 10 * (20-10)/1 = -100
        assert!(approx_eq(flux, -100.0));
    }

    #[test]
    fn test_dt_limit() {
        let solver = TracerTransportSolver::new(TracerTransportConfig {
            diffusion: TracerDiffusionConfig::constant(10.0),
            ..Default::default()
        });

        let dt = solver.compute_dt_limit(1.0, 10.0, 0.5);
        // 对流限制: 0.5 * 10 / 1 = 5
        // 扩散限制: 0.5 * 0.5 * 100 / 10 = 2.5
        assert!(approx_eq(dt, 2.5));
    }

    #[test]
    fn test_single_step() {
        let mut solver = TracerTransportSolver::default();
        let props = TracerProperties::salinity().with_background(0.0);
        let mut field = TracerField::from_concentration(
            props,
            vec![10.0, 5.0, 0.0], // 浓度梯度
        );

        // 初始化守恒量
        let depths = vec![1.0, 1.0, 1.0];
        field.update_conserved_from_depth(&depths);

        // 简单的两面流动数据
        let flow_data = vec![
            FaceFlowData {
                face_id: 0,
                left_cell: 0,
                right_cell: Some(1),
                normal: DVec2::new(1.0, 0.0),
                length: 1.0,
                un: 1.0,  // 从左到右
                h_face: 1.0,
            },
            FaceFlowData {
                face_id: 1,
                left_cell: 1,
                right_cell: Some(2),
                normal: DVec2::new(1.0, 0.0),
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
        assert!(field.concentration(1) > 5.0); // 从单元0获得质量
    }

    #[test]
    fn test_multi_tracer_solver() {
        let mut state = TracerState::new(10);
        state.add_tracer(TracerProperties::salinity()).unwrap();
        state.add_tracer(TracerProperties::temperature()).unwrap();

        let _solver = MultiTracerSolver::new(TracerTransportConfig::default());

        // 确保可以访问两个示踪剂
        assert!(state.get(TracerType::Salinity).is_some());
        assert!(state.get(TracerType::Temperature).is_some());
    }
}
