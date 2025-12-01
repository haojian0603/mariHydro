// src-tauri/src/marihydro/core/traits/source.rs

//! 源项接口 (枚举化重构版本)
//!
//! 使用枚举分发替代 dyn Trait，解决 E0038 (trait不是dyn兼容) 问题。
//! 
//! # 设计原则
//! 1. 已知源项类型使用枚举分发
//! 2. 使用具体类型 (UnstructuredMesh, ShallowWaterState) 而非泛型
//! 3. 通过 Custom 变体支持扩展
//! 4. 所有类型均满足 Send + Sync

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::{CellIndex, FaceIndex, NumericalParams};
use crate::marihydro::domain::mesh::UnstructuredMesh;
use crate::marihydro::domain::state::ShallowWaterState;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::traits::state::StateAccess;
use glam::DVec2;
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

// ========== 源项贡献结构体 ==========

/// 源项贡献
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceContribution {
    /// 质量源 [m/s]
    pub s_h: f64,
    /// x动量源 [m²/s²]
    pub s_hu: f64,
    /// y动量源 [m²/s²]
    pub s_hv: f64,
}

impl SourceContribution {
    pub const ZERO: Self = Self {
        s_h: 0.0,
        s_hu: 0.0,
        s_hv: 0.0,
    };

    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            s_h: self.s_h + other.s_h,
            s_hu: self.s_hu + other.s_hu,
            s_hv: self.s_hv + other.s_hv,
        }
    }

    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        self.s_h += other.s_h;
        self.s_hu += other.s_hu;
        self.s_hv += other.s_hv;
    }

    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            s_h: self.s_h * factor,
            s_hu: self.s_hu * factor,
            s_hv: self.s_hv * factor,
        }
    }
}

// ========== 源项计算上下文 ==========

/// 源项计算上下文
///
/// 包含源项计算所需的所有外部信息
pub struct SourceContext<'a> {
    /// 当前模拟时间 [s]
    pub time: f64,
    /// 时间步长 [s]
    pub dt: f64,
    /// 数值参数
    pub params: &'a NumericalParams,
}

impl<'a> SourceContext<'a> {
    /// 创建新的源项上下文
    pub fn new(time: f64, dt: f64, params: &'a NumericalParams) -> Self {
        Self { time, dt, params }
    }
}

// ========== 源项配置结构体 ==========

/// 风应力源项配置
#[derive(Debug, Clone)]
pub struct WindStressConfig {
    pub enabled: bool,
    pub rho_air: f64,
    pub rho_water: f64,
    /// 预计算 rho_air / rho_water
    stress_factor: f64,
    /// 风场 (u, v)
    pub wind_u: Vec<f64>,
    pub wind_v: Vec<f64>,
}

impl WindStressConfig {
    pub fn new(n_cells: usize, rho_air: f64, rho_water: f64) -> Self {
        Self {
            enabled: true,
            rho_air,
            rho_water,
            stress_factor: rho_air / rho_water,
            wind_u: vec![0.0; n_cells],
            wind_v: vec![0.0; n_cells],
        }
    }

    pub fn set_uniform_wind(&mut self, u: f64, v: f64) {
        self.wind_u.fill(u);
        self.wind_v.fill(v);
    }

    pub fn set_wind_field(&mut self, u: Vec<f64>, v: Vec<f64>) {
        self.wind_u = u;
        self.wind_v = v;
    }
}

/// 科氏力源项配置
#[derive(Debug, Clone)]
pub struct CoriolisConfig {
    pub enabled: bool,
    /// 科氏参数 f = 2*omega*sin(lat)
    pub f: f64,
    /// 是否使用精确旋转（否则线性近似）
    pub use_exact_rotation: bool,
}

impl CoriolisConfig {
    pub fn new(f: f64) -> Self {
        Self {
            enabled: true,
            f,
            use_exact_rotation: true,
        }
    }

    pub fn from_latitude(lat_deg: f64) -> Self {
        let omega = 7.2921e-5;
        let f = 2.0 * omega * (lat_deg * PI / 180.0).sin();
        Self::new(f)
    }
}

/// Manning 摩擦源项配置
#[derive(Debug, Clone)]
pub struct ManningFrictionConfig {
    pub enabled: bool,
    pub g: f64,
    /// 预计算 g * n^2 (均匀场时)
    pub precomputed_gn2: Option<f64>,
    pub manning_n: Vec<f64>,
}

impl ManningFrictionConfig {
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> Self {
        let gn2 = g * default_n * default_n;
        Self {
            enabled: true,
            g,
            precomputed_gn2: Some(gn2),
            manning_n: vec![default_n; n_cells],
        }
    }

    pub fn with_field(g: f64, manning_n: Vec<f64>) -> Self {
        Self {
            enabled: true,
            g,
            precomputed_gn2: None,
            manning_n,
        }
    }
}

/// Chezy 摩擦源项配置
#[derive(Debug, Clone)]
pub struct ChezyFrictionConfig {
    pub enabled: bool,
    pub g: f64,
    pub chezy_c: f64,
    /// 预计算 cf = g / C^2
    pub cf: f64,
}

impl ChezyFrictionConfig {
    pub fn new(g: f64, chezy_c: f64) -> Self {
        let cf = g / (chezy_c * chezy_c);
        Self {
            enabled: true,
            g,
            chezy_c,
            cf,
        }
    }
}

/// Smagorinsky 湍流源项配置
#[derive(Debug, Clone)]
pub struct SmagorinskyConfig {
    pub enabled: bool,
    pub cs: f64,
    pub nu_min: f64,
    pub nu_max: f64,
}

impl SmagorinskyConfig {
    pub fn new(cs: f64) -> Self {
        Self {
            enabled: true,
            cs,
            nu_min: 1e-6,
            nu_max: 1000.0,
        }
    }

    pub fn with_limits(mut self, nu_min: f64, nu_max: f64) -> Self {
        self.nu_min = nu_min;
        self.nu_max = nu_max;
        self
    }
}

/// 压力梯度源项配置 (使用RwLock替代RefCell以满足Sync)
#[derive(Debug)]
pub struct PressureGradientConfig {
    pub enabled: bool,
    pub rho_water: f64,
    /// 预计算 -1.0 / rho_water
    inv_rho: f64,
    pub pressure: RwLock<Vec<f64>>,
}

impl Clone for PressureGradientConfig {
    fn clone(&self) -> Self {
        Self {
            enabled: self.enabled,
            rho_water: self.rho_water,
            inv_rho: self.inv_rho,
            pressure: RwLock::new(self.pressure.read().unwrap().clone()),
        }
    }
}

impl PressureGradientConfig {
    pub fn new(n_cells: usize, rho_water: f64) -> Self {
        Self {
            enabled: true,
            rho_water,
            inv_rho: -1.0 / rho_water,
            pressure: RwLock::new(vec![101325.0; n_cells]),
        }
    }

    pub fn set_pressure_field(&self, p: Vec<f64>) {
        *self.pressure.write().unwrap() = p;
    }
}

/// 自定义源项回调类型
pub type CustomSourceFn = Arc<
    dyn Fn(&UnstructuredMesh, &ShallowWaterState, CellIndex, &SourceContext) -> SourceContribution
        + Send
        + Sync
>;

// ========== 核心枚举 ==========

/// 源项类型枚举 - 替代 Box<dyn SourceTerm>
#[derive(Clone)]
pub enum SourceTermKind {
    WindStress(WindStressConfig),
    Coriolis(CoriolisConfig),
    ManningFriction(ManningFrictionConfig),
    ChezyFriction(ChezyFrictionConfig),
    Smagorinsky(SmagorinskyConfig),
    PressureGradient(PressureGradientConfig),
    Custom {
        name: String,
        enabled: bool,
        compute: CustomSourceFn,
    },
}

impl SourceTermKind {
    /// 获取源项名称
    pub fn name(&self) -> &str {
        match self {
            Self::WindStress(_) => "WindStress",
            Self::Coriolis(_) => "Coriolis",
            Self::ManningFriction(_) => "ManningFriction",
            Self::ChezyFriction(_) => "ChezyFriction",
            Self::Smagorinsky(_) => "Smagorinsky",
            Self::PressureGradient(_) => "PressureGradient",
            Self::Custom { name, .. } => name,
        }
    }

    /// 是否启用
    pub fn is_enabled(&self) -> bool {
        match self {
            Self::WindStress(c) => c.enabled,
            Self::Coriolis(c) => c.enabled,
            Self::ManningFriction(c) => c.enabled,
            Self::ChezyFriction(c) => c.enabled,
            Self::Smagorinsky(c) => c.enabled,
            Self::PressureGradient(c) => c.enabled,
            Self::Custom { enabled, .. } => *enabled,
        }
    }

    /// 计算单个单元的源项贡献
    pub fn compute_cell(
        &self,
        mesh: &UnstructuredMesh,
        state: &ShallowWaterState,
        cell: CellIndex,
        ctx: &SourceContext,
    ) -> SourceContribution {
        if !self.is_enabled() {
            return SourceContribution::ZERO;
        }

        match self {
            Self::WindStress(cfg) => compute_wind_stress(cfg, state, cell, ctx),
            Self::Coriolis(cfg) => compute_coriolis(cfg, state, cell, ctx),
            Self::ManningFriction(cfg) => compute_manning_friction(cfg, state, cell, ctx),
            Self::ChezyFriction(cfg) => compute_chezy_friction(cfg, state, cell, ctx),
            Self::Smagorinsky(cfg) => compute_smagorinsky_cell(cfg, mesh, state, cell, ctx),
            Self::PressureGradient(cfg) => compute_pressure_gradient_cell(cfg, mesh, state, cell, ctx),
            Self::Custom { compute, .. } => compute(mesh, state, cell, ctx),
        }
    }

    /// 批量计算所有单元的源项（累加到输出缓冲区）
    pub fn compute_all(
        &self,
        mesh: &UnstructuredMesh,
        state: &ShallowWaterState,
        ctx: &SourceContext,
        output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) -> MhResult<()> {
        if !self.is_enabled() {
            return Ok(());
        }

        let n_cells = mesh.n_cells();
        if output_h.len() != n_cells || output_hu.len() != n_cells || output_hv.len() != n_cells {
            return Err(crate::marihydro::core::MhError::size_mismatch(
                "source output arrays",
                n_cells,
                output_h.len(),
            ));
        }

        // 特殊处理需要全局计算的源项
        match self {
            Self::Smagorinsky(cfg) => {
                compute_smagorinsky_all(cfg, mesh, state, ctx, output_hu, output_hv)?;
            }
            Self::PressureGradient(cfg) => {
                compute_pressure_gradient_all(cfg, mesh, state, ctx, output_hu, output_hv)?;
            }
            _ => {
                // 其他源项逐单元计算
                for i in 0..n_cells {
                    let contrib = self.compute_cell(mesh, state, CellIndex(i), ctx);
                    output_h[i] += contrib.s_h;
                    output_hu[i] += contrib.s_hu;
                    output_hv[i] += contrib.s_hv;
                }
            }
        }

        Ok(())
    }

    /// 源项是否显式（需要CFL限制）
    pub fn is_explicit(&self) -> bool {
        !matches!(self, Self::ManningFriction(_) | Self::ChezyFriction(_))
    }

    /// 是否需要隐式处理
    pub fn requires_implicit_treatment(&self) -> bool {
        matches!(self, Self::ManningFriction(_) | Self::ChezyFriction(_))
    }
}

// ========== 计算函数实现 ==========

const MAX_WIND_SPEED: f64 = 100.0;

/// Large and Pond (1981) 风阻系数
#[inline(always)]
fn wind_drag_coefficient_lp81(wind_speed: f64) -> f64 {
    let w = wind_speed.abs().min(MAX_WIND_SPEED);
    if w < 11.0 {
        1.2e-3
    } else if w < 25.0 {
        (0.49 + 0.065 * w) * 1e-3
    } else {
        2.11e-3
    }
}

fn compute_wind_stress(
    cfg: &WindStressConfig,
    state: &ShallowWaterState,
    cell: CellIndex,
    ctx: &SourceContext,
) -> SourceContribution {
    let h = state.h(cell);
    if ctx.params.is_dry(h) {
        return SourceContribution::ZERO;
    }

    let wu = cfg.wind_u.get(cell.0).copied().unwrap_or(0.0);
    let wv = cfg.wind_v.get(cell.0).copied().unwrap_or(0.0);

    if h < 1e-6 {
        return SourceContribution::ZERO;
    }

    let mag = (wu * wu + wv * wv).sqrt();
    if mag < 1e-8 {
        return SourceContribution::ZERO;
    }

    let cd = wind_drag_coefficient_lp81(mag);
    let factor = cfg.stress_factor * cd * mag / h;

    SourceContribution {
        s_h: 0.0,
        s_hu: h * factor * wu,
        s_hv: h * factor * wv,
    }
}

fn compute_coriolis(
    cfg: &CoriolisConfig,
    state: &ShallowWaterState,
    cell: CellIndex,
    ctx: &SourceContext,
) -> SourceContribution {
    let h = state.h(cell);
    if ctx.params.is_dry(h) {
        return SourceContribution::ZERO;
    }

    let hu = state.hu(cell);
    let hv = state.hv(cell);
    let dt = ctx.dt;

    let (hu_new, hv_new) = if cfg.use_exact_rotation {
        // 精确旋转
        let theta = cfg.f * dt;
        let (sin_t, cos_t) = if theta.abs() < 1e-3 {
            let t2 = theta * theta;
            (theta * (1.0 - t2 / 6.0), 1.0 - t2 * 0.5)
        } else {
            theta.sin_cos()
        };
        (hu * cos_t + hv * sin_t, -hu * sin_t + hv * cos_t)
    } else {
        // 线性近似
        let dhu = cfg.f * hv * dt;
        let dhv = -cfg.f * hu * dt;
        (hu + dhu, hv + dhv)
    };

    SourceContribution {
        s_h: 0.0,
        s_hu: (hu_new - hu) / dt,
        s_hv: (hv_new - hv) / dt,
    }
}

fn compute_manning_friction(
    cfg: &ManningFrictionConfig,
    state: &ShallowWaterState,
    cell: CellIndex,
    ctx: &SourceContext,
) -> SourceContribution {
    let h = state.h(cell);
    let hu = state.hu(cell);
    let hv = state.hv(cell);
    let dt = ctx.dt;

    if ctx.params.is_dry(h) {
        return SourceContribution {
            s_h: 0.0,
            s_hu: -hu / dt,
            s_hv: -hv / dt,
        };
    }

    // 使用安全速度计算
    let vel = ctx.params.safe_velocity(hu, hv, h);
    let speed_sq = vel.speed_squared();
    if speed_sq < 1e-20 {
        return SourceContribution::ZERO;
    }

    let h_safe = h.max(1e-4);
    let cf = if let Some(gn2) = cfg.precomputed_gn2 {
        gn2 / h_safe.cbrt()
    } else {
        let n = cfg.manning_n.get(cell.0).copied().unwrap_or(0.025);
        cfg.g * n * n / h_safe.cbrt()
    };

    let speed = speed_sq.sqrt();
    let decay = 1.0 / (1.0 + dt * cf * speed);
    let factor = (decay - 1.0) / dt;

    SourceContribution {
        s_h: 0.0,
        s_hu: hu * factor,
        s_hv: hv * factor,
    }
}

fn compute_chezy_friction(
    cfg: &ChezyFrictionConfig,
    state: &ShallowWaterState,
    cell: CellIndex,
    ctx: &SourceContext,
) -> SourceContribution {
    let h = state.h(cell);
    let hu = state.hu(cell);
    let hv = state.hv(cell);
    let dt = ctx.dt;

    if ctx.params.is_dry(h) {
        return SourceContribution {
            s_h: 0.0,
            s_hu: -hu / dt,
            s_hv: -hv / dt,
        };
    }

    let vel = ctx.params.safe_velocity(hu, hv, h);
    let speed_sq = vel.speed_squared();
    if speed_sq < 1e-20 {
        return SourceContribution::ZERO;
    }

    let speed = speed_sq.sqrt();
    let decay = 1.0 / (1.0 + dt * cfg.cf * speed);
    let factor = (decay - 1.0) / dt;

    SourceContribution {
        s_h: 0.0,
        s_hu: hu * factor,
        s_hv: hv * factor,
    }
}

fn compute_smagorinsky_cell(
    _cfg: &SmagorinskyConfig,
    _mesh: &UnstructuredMesh,
    _state: &ShallowWaterState,
    _cell: CellIndex,
    _ctx: &SourceContext,
) -> SourceContribution {
    // Smagorinsky需要全局梯度计算，单元计算返回零
    SourceContribution::ZERO
}

fn compute_smagorinsky_all(
    cfg: &SmagorinskyConfig,
    mesh: &UnstructuredMesh,
    state: &ShallowWaterState,
    ctx: &SourceContext,
    output_hu: &mut [f64],
    output_hv: &mut [f64],
) -> MhResult<()> {
    let n = mesh.n_cells();
    let params = ctx.params;

    // 计算速度场
    let mut velocities: Vec<DVec2> = vec![DVec2::ZERO; n];
    for i in 0..n {
        let h = state.h(CellIndex(i));
        if params.is_dry(h) {
            velocities[i] = DVec2::ZERO;
        } else {
            velocities[i] = DVec2::new(
                state.hu(CellIndex(i)) / h,
                state.hv(CellIndex(i)) / h,
            );
        }
    }

    // 计算速度梯度 (Green-Gauss)
    let mut du_dx = vec![0.0; n];
    let mut du_dy = vec![0.0; n];
    let mut dv_dx = vec![0.0; n];
    let mut dv_dy = vec![0.0; n];

    for i in 0..n {
        let cell = CellIndex(i);
        let area = mesh.cell_area(cell);
        if area < 1e-14 {
            continue;
        }

        let mut gu = DVec2::ZERO;
        let mut gv = DVec2::ZERO;

        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            let sign = if i == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let vel_face = if !neighbor.is_valid() {
                velocities[i]
            } else {
                let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                (velocities[i] + velocities[o]) * 0.5
            };

            gu += ds * vel_face.x;
            gv += ds * vel_face.y;
        }

        du_dx[i] = gu.x / area;
        du_dy[i] = gu.y / area;
        dv_dx[i] = gv.x / area;
        dv_dy[i] = gv.y / area;
    }

    // 计算涡粘系数
    let mut nu_t: Vec<f64> = vec![cfg.nu_min; n];
    for i in 0..n {
        let area = mesh.cell_area(CellIndex(i));
        if area < 1e-14 {
            continue;
        }

        let delta = area.sqrt();
        let s11 = du_dx[i];
        let s22 = dv_dy[i];
        let s12 = 0.5 * (du_dy[i] + dv_dx[i]);
        let s_mag = (2.0 * (s11 * s11 + s22 * s22 + 2.0 * s12 * s12)).sqrt();
        let nu = (cfg.cs * delta).powi(2) * s_mag;
        nu_t[i] = nu.clamp(cfg.nu_min, cfg.nu_max);
    }

    // 计算扩散通量
    let n_faces = mesh.n_faces();
    for face_idx in 0..n_faces {
        let face = crate::marihydro::core::types::FaceIndex(face_idx);
        let owner = mesh.face_owner(face);
        let neighbor = mesh.face_neighbor(face);

        if !neighbor.is_valid() {
            continue;
        }

        let h_o = state.h(owner);
        let h_n = state.h(neighbor);
        if params.is_dry(h_o) && params.is_dry(h_n) {
            continue;
        }

        let u_o = if params.is_dry(h_o) { 0.0 } else { state.hu(owner) / h_o };
        let v_o = if params.is_dry(h_o) { 0.0 } else { state.hv(owner) / h_o };
        let u_n = if params.is_dry(h_n) { 0.0 } else { state.hu(neighbor) / h_n };
        let v_n = if params.is_dry(h_n) { 0.0 } else { state.hv(neighbor) / h_n };

        let dist = (mesh.cell_centroid(neighbor) - mesh.cell_centroid(owner)).length();
        if dist < 1e-14 {
            continue;
        }

        // 调和平均涡粘系数
        let nu_o = nu_t[owner.0];
        let nu_n = nu_t[neighbor.0];
        let nu_face = if nu_o + nu_n > 1e-14 {
            2.0 * nu_o * nu_n / (nu_o + nu_n)
        } else {
            0.0
        };

        let h_face = 0.5 * (h_o + h_n);
        let length = mesh.face_length(face);
        let flux_u = nu_face * h_face * (u_n - u_o) / dist * length;
        let flux_v = nu_face * h_face * (v_n - v_o) / dist * length;

        output_hu[owner.0] += flux_u;
        output_hv[owner.0] += flux_v;
        output_hu[neighbor.0] -= flux_u;
        output_hv[neighbor.0] -= flux_v;
    }

    Ok(())
}

fn compute_pressure_gradient_cell(
    _cfg: &PressureGradientConfig,
    _mesh: &UnstructuredMesh,
    _state: &ShallowWaterState,
    _cell: CellIndex,
    _ctx: &SourceContext,
) -> SourceContribution {
    // 压力梯度需要全局计算，单元返回零
    SourceContribution::ZERO
}

fn compute_pressure_gradient_all(
    cfg: &PressureGradientConfig,
    mesh: &UnstructuredMesh,
    state: &ShallowWaterState,
    ctx: &SourceContext,
    output_hu: &mut [f64],
    output_hv: &mut [f64],
) -> MhResult<()> {
    let n = mesh.n_cells();
    let pressure = cfg.pressure.read().unwrap();
    let factor = cfg.inv_rho;
    let h_dry = ctx.params.h_dry;

    // 计算压力梯度 (Green-Gauss)
    for i in 0..n {
        let h = state.h(CellIndex(i));
        if h < h_dry {
            continue;
        }

        let cell = CellIndex(i);
        let area = mesh.cell_area(cell);
        if area < 1e-14 {
            continue;
        }

        let mut grad = DVec2::ZERO;
        for &face in mesh.cell_faces(cell) {
            let owner = mesh.face_owner(face);
            let neighbor = mesh.face_neighbor(face);
            let normal = mesh.face_normal(face);
            let length = mesh.face_length(face);
            let sign = if i == owner.0 { 1.0 } else { -1.0 };
            let ds = normal * length * sign;

            let phi = if !neighbor.is_valid() {
                pressure[i]
            } else {
                let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                0.5 * (pressure[i] + pressure[o])
            };

            grad += ds * phi;
        }

        grad /= area;
        output_hu[i] += h * grad.x * factor;
        output_hv[i] += h * grad.y * factor;
    }

    Ok(())
}

// ========== 源项管理器 ==========

/// 源项管理器 - 替代 SourceTermAggregator<Vec<Box<dyn SourceTerm>>>
#[derive(Clone, Default)]
pub struct SourceTermManager {
    sources: Vec<SourceTermKind>,
}

impl SourceTermManager {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// 添加源项
    pub fn add(&mut self, source: SourceTermKind) -> &mut Self {
        self.sources.push(source);
        self
    }

    /// 链式添加风应力
    pub fn with_wind_stress(mut self, cfg: WindStressConfig) -> Self {
        self.sources.push(SourceTermKind::WindStress(cfg));
        self
    }

    /// 链式添加科氏力
    pub fn with_coriolis(mut self, cfg: CoriolisConfig) -> Self {
        self.sources.push(SourceTermKind::Coriolis(cfg));
        self
    }

    /// 链式添加 Manning 摩擦
    pub fn with_manning_friction(mut self, cfg: ManningFrictionConfig) -> Self {
        self.sources.push(SourceTermKind::ManningFriction(cfg));
        self
    }

    /// 链式添加 Chezy 摩擦
    pub fn with_chezy_friction(mut self, cfg: ChezyFrictionConfig) -> Self {
        self.sources.push(SourceTermKind::ChezyFriction(cfg));
        self
    }

    /// 链式添加 Smagorinsky 湍流
    pub fn with_smagorinsky(mut self, cfg: SmagorinskyConfig) -> Self {
        self.sources.push(SourceTermKind::Smagorinsky(cfg));
        self
    }

    /// 链式添加压力梯度
    pub fn with_pressure_gradient(mut self, cfg: PressureGradientConfig) -> Self {
        self.sources.push(SourceTermKind::PressureGradient(cfg));
        self
    }

    /// 批量计算所有源项
    pub fn compute_all(
        &self,
        mesh: &UnstructuredMesh,
        state: &ShallowWaterState,
        ctx: &SourceContext,
        output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) -> MhResult<()> {
        // 清零输出
        output_h.fill(0.0);
        output_hu.fill(0.0);
        output_hv.fill(0.0);

        // 累加所有启用的源项
        for source in &self.sources {
            if source.is_enabled() {
                source.compute_all(mesh, state, ctx, output_h, output_hu, output_hv)?;
            }
        }

        Ok(())
    }

    /// 获取启用的源项名称
    pub fn enabled_names(&self) -> Vec<&str> {
        self.sources
            .iter()
            .filter(|s| s.is_enabled())
            .map(|s| s.name())
            .collect()
    }

    /// 源项数量
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

// ========== 兼容性别名 ==========

/// 向后兼容的类型别名
pub type SourceTermAggregator = SourceTermManager;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_contribution() {
        let s1 = SourceContribution {
            s_h: 1.0,
            s_hu: 2.0,
            s_hv: 3.0,
        };
        let s2 = SourceContribution {
            s_h: 0.5,
            s_hu: 1.0,
            s_hv: 1.5,
        };

        let sum = s1.add(&s2);
        assert!((sum.s_h - 1.5).abs() < 1e-10);
        assert!((sum.s_hu - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_source_manager() {
        let manager = SourceTermManager::new()
            .with_coriolis(CoriolisConfig::from_latitude(30.0))
            .with_manning_friction(ManningFrictionConfig::new(9.81, 100, 0.025));

        assert_eq!(manager.len(), 2);
        let names = manager.enabled_names();
        assert!(names.contains(&"Coriolis"));
        assert!(names.contains(&"ManningFriction"));
    }
}
