// crates/mh_physics/src/state.rs

//! 浅水方程状态管理
//!
//! 本模块提供浅水方程求解所需的状态管理，包括：
//! - ShallowWaterState: 守恒变量状态 (h, hu, hv, z)
//! - GradientState: 梯度状态 (grad_h, grad_hu, grad_hv)
//! - Flux: 数值通量
//! - RhsBuffers: 右端项缓冲区
//!
//! # 布局设计
//!
//! 采用 SoA (Structure of Arrays) 布局以优化缓存性能：
//! ```text
//! h:  [h_0,  h_1,  h_2,  ...]
//! hu: [hu_0, hu_1, hu_2, ...]
//! hv: [hv_0, hv_1, hv_2, ...]
//! z:  [z_0,  z_1,  z_2,  ...]
//! ```
//!
//! # 迁移说明
//!
//! 从 legacy_src/domain/state/shallow_water.rs 迁移，保持算法不变。

use glam::DVec2;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::types::{CellIndex, NumericalParams, SafeVelocity};

// ============================================================
// 守恒状态
// ============================================================

/// 单个单元的守恒状态
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConservedState {
    /// 水深 [m]
    pub h: f64,
    /// x 方向动量 [m²/s]
    pub hu: f64,
    /// y 方向动量 [m²/s]
    pub hv: f64,
}

impl ConservedState {
    /// 创建新的守恒状态
    #[inline]
    pub const fn new(h: f64, hu: f64, hv: f64) -> Self {
        Self { h, hu, hv }
    }

    /// 零状态
    pub const ZERO: Self = Self {
        h: 0.0,
        hu: 0.0,
        hv: 0.0,
    };

    /// 从原始变量创建
    #[inline]
    pub fn from_primitive(h: f64, u: f64, v: f64) -> Self {
        Self {
            h,
            hu: h * u,
            hv: h * v,
        }
    }

    /// 获取速度 (使用安全除法)
    #[inline]
    pub fn velocity(&self, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu, self.hv, self.h)
    }

    /// 状态是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.h.is_finite() && self.hu.is_finite() && self.hv.is_finite() && self.h >= 0.0
    }
}

impl Add for ConservedState {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            h: self.h + rhs.h,
            hu: self.hu + rhs.hu,
            hv: self.hv + rhs.hv,
        }
    }
}

impl Sub for ConservedState {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            h: self.h - rhs.h,
            hu: self.hu - rhs.hu,
            hv: self.hv - rhs.hv,
        }
    }
}

impl Mul<f64> for ConservedState {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        Self {
            h: self.h * rhs,
            hu: self.hu * rhs,
            hv: self.hv * rhs,
        }
    }
}

// ============================================================
// 浅水方程状态 (SoA 布局)
// ============================================================

/// 浅水方程守恒状态（SoA 布局）
///
/// 存储整个网格的状态变量，采用 SoA 布局优化缓存访问。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShallowWaterState {
    /// 单元数量
    n_cells: usize,

    /// 水深 [m]
    pub h: Vec<f64>,
    /// x 方向动量 [m²/s]
    pub hu: Vec<f64>,
    /// y 方向动量 [m²/s]
    pub hv: Vec<f64>,
    /// 底床高程 [m]
    pub z: Vec<f64>,

    /// 可选的标量浓度场
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hc: Option<Vec<f64>>,
}

impl ShallowWaterState {
    /// 创建新状态
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z: vec![0.0; n_cells],
            hc: None,
        }
    }

    /// 创建带标量的状态
    pub fn with_scalar(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z: vec![0.0; n_cells],
            hc: Some(vec![0.0; n_cells]),
        }
    }

    /// 从初始水位和底床创建（冷启动）
    pub fn cold_start(initial_eta: f64, z_bed: &[f64]) -> Self {
        let n_cells = z_bed.len();

        let h: Vec<f64> = z_bed.iter().map(|&z| (initial_eta - z).max(0.0)).collect();

        Self {
            n_cells,
            h,
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            z: z_bed.to_vec(),
            hc: None,
        }
    }

    /// 克隆结构（不复制数据，创建零初始化的状态）
    pub fn clone_structure(&self) -> Self {
        Self {
            n_cells: self.n_cells,
            h: vec![0.0; self.n_cells],
            hu: vec![0.0; self.n_cells],
            hv: vec![0.0; self.n_cells],
            z: self.z.clone(),
            hc: self.hc.as_ref().map(|_| vec![0.0; self.n_cells]),
        }
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    // ========== 状态访问 ==========

    /// 获取单元的守恒状态
    #[inline]
    pub fn get(&self, idx: usize) -> ConservedState {
        ConservedState::new(self.h[idx], self.hu[idx], self.hv[idx])
    }

    /// 获取单元的守恒状态（使用 CellIndex）
    #[inline]
    pub fn get_by_index(&self, cell: CellIndex) -> ConservedState {
        self.get(cell.get())
    }

    /// 获取原始变量 (h, u, v)
    #[inline]
    pub fn primitive(&self, idx: usize, params: &NumericalParams) -> (f64, f64, f64) {
        let h = self.h[idx];
        let vel = params.safe_velocity(self.hu[idx], self.hv[idx], h);
        (h, vel.u, vel.v)
    }

    /// 获取速度
    #[inline]
    pub fn velocity(&self, idx: usize, params: &NumericalParams) -> SafeVelocity {
        params.safe_velocity(self.hu[idx], self.hv[idx], self.h[idx])
    }

    /// 获取速度（使用阈值）
    #[inline]
    pub fn velocity_with_eps(&self, idx: usize, eps: f64) -> DVec2 {
        let h = self.h[idx];
        if h > eps {
            DVec2::new(self.hu[idx] / h, self.hv[idx] / h)
        } else {
            DVec2::ZERO
        }
    }

    /// 获取水位 (eta = h + z)
    #[inline]
    pub fn water_level(&self, idx: usize) -> f64 {
        self.h[idx] + self.z[idx]
    }

    // ========== 状态修改 ==========

    /// 设置守恒变量
    #[inline]
    pub fn set(&mut self, idx: usize, h: f64, hu: f64, hv: f64) {
        self.h[idx] = h;
        self.hu[idx] = hu;
        self.hv[idx] = hv;
    }

    /// 设置守恒状态
    #[inline]
    pub fn set_state(&mut self, idx: usize, state: ConservedState) {
        self.h[idx] = state.h;
        self.hu[idx] = state.hu;
        self.hv[idx] = state.hv;
    }

    /// 从原始变量设置
    #[inline]
    pub fn set_from_primitive(&mut self, idx: usize, h: f64, u: f64, v: f64) {
        self.h[idx] = h;
        self.hu[idx] = h * u;
        self.hv[idx] = h * v;
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.h.fill(0.0);
        self.hu.fill(0.0);
        self.hv.fill(0.0);
        if let Some(ref mut hc) = self.hc {
            hc.fill(0.0);
        }
    }

    // ========== 切片访问 ==========

    /// 获取水深切片
    #[inline]
    pub fn h_slice(&self) -> &[f64] {
        &self.h
    }

    /// 获取 x 动量切片
    #[inline]
    pub fn hu_slice(&self) -> &[f64] {
        &self.hu
    }

    /// 获取 y 动量切片
    #[inline]
    pub fn hv_slice(&self) -> &[f64] {
        &self.hv
    }

    /// 获取底床高程切片
    #[inline]
    pub fn z_slice(&self) -> &[f64] {
        &self.z
    }

    /// 获取可变水深切片
    #[inline]
    pub fn h_slice_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    /// 获取可变 x 动量切片
    #[inline]
    pub fn hu_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hu
    }

    /// 获取可变 y 动量切片
    #[inline]
    pub fn hv_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hv
    }

    /// 获取可变底床高程切片
    #[inline]
    pub fn z_slice_mut(&mut self) -> &mut [f64] {
        &mut self.z
    }

    // ========== 积分计算 ==========

    /// 计算总质量
    pub fn total_mass(&self, cell_areas: &[f64]) -> f64 {
        self.h.iter().zip(cell_areas).map(|(h, a)| h * a).sum()
    }

    /// 计算总动量
    pub fn total_momentum(&self, cell_areas: &[f64]) -> DVec2 {
        let hux: f64 = self.hu.iter().zip(cell_areas).map(|(hu, a)| hu * a).sum();
        let hvx: f64 = self.hv.iter().zip(cell_areas).map(|(hv, a)| hv * a).sum();
        DVec2::new(hux, hvx)
    }

    // ========== 时间积分支持 ==========

    /// 从另一个状态复制数据
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());
        self.h.copy_from_slice(&other.h);
        self.hu.copy_from_slice(&other.hu);
        self.hv.copy_from_slice(&other.hv);
        if let (Some(ref mut dst), Some(ref src)) = (&mut self.hc, &other.hc) {
            dst.copy_from_slice(src);
        }
    }

    /// 添加缩放的 RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &RhsBuffers, scale: f64) {
        for i in 0..self.n_cells {
            self.h[i] += scale * rhs.dh_dt[i];
            self.hu[i] += scale * rhs.dhu_dt[i];
            self.hv[i] += scale * rhs.dhv_dt[i];
        }
    }

    /// 二元线性组合: self = a*A + b*B
    pub fn linear_combine(&mut self, a: f64, state_a: &Self, b: f64, state_b: &Self) {
        debug_assert_eq!(self.n_cells(), state_a.n_cells());
        debug_assert_eq!(self.n_cells(), state_b.n_cells());

        for i in 0..self.n_cells {
            self.h[i] = a * state_a.h[i] + b * state_b.h[i];
            self.hu[i] = a * state_a.hu[i] + b * state_b.hu[i];
            self.hv[i] = a * state_a.hv[i] + b * state_b.hv[i];
        }

        if let (Some(ref mut dst), Some(ref src_a), Some(ref src_b)) =
            (&mut self.hc, &state_a.hc, &state_b.hc)
        {
            for i in 0..self.n_cells {
                dst[i] = a * src_a[i] + b * src_b[i];
            }
        }
    }

    /// 自线性组合: self = a * self + b * other
    pub fn axpy(&mut self, a: f64, b: f64, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());

        for i in 0..self.n_cells {
            self.h[i] = a * self.h[i] + b * other.h[i];
            self.hu[i] = a * self.hu[i] + b * other.hu[i];
            self.hv[i] = a * self.hv[i] + b * other.hv[i];
        }

        if let (Some(ref mut dst), Some(ref src_b)) = (&mut self.hc, &other.hc) {
            for i in 0..self.n_cells {
                dst[i] = a * dst[i] + b * src_b[i];
            }
        }
    }

    /// 强制正性约束
    pub fn enforce_positivity(&mut self) {
        for h in self.h.iter_mut() {
            if *h < 0.0 {
                *h = 0.0;
            }
        }

        if let Some(ref mut hc) = self.hc {
            for c in hc.iter_mut() {
                if *c < 0.0 {
                    *c = 0.0;
                }
            }
        }
    }

    // ========== 验证 ==========

    /// 验证状态有效性
    pub fn validate(&self, time: f64, params: &NumericalParams) -> Result<(), StateError> {
        for idx in 0..self.n_cells {
            // 检查 NaN/Inf
            if !self.h[idx].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "h",
                    cell: idx,
                    value: self.h[idx],
                    time,
                });
            }

            if !self.hu[idx].is_finite() || !self.hv[idx].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "momentum",
                    cell: idx,
                    value: if !self.hu[idx].is_finite() {
                        self.hu[idx]
                    } else {
                        self.hv[idx]
                    },
                    time,
                });
            }

            // 检查负水深
            if self.h[idx] < 0.0 {
                return Err(StateError::NegativeDepth {
                    cell: idx,
                    value: self.h[idx],
                    time,
                });
            }

            // 检查速度
            if !params.is_dry(self.h[idx]) {
                let vel = self.velocity(idx, params);
                if params.is_velocity_excessive(vel.speed()) {
                    return Err(StateError::ExcessiveVelocity {
                        cell: idx,
                        speed: vel.speed(),
                        max_speed: params.vel_max,
                        time,
                    });
                }
            }
        }

        Ok(())
    }
}

// ============================================================
// 右端项缓冲区
// ============================================================

/// 右端项缓冲区 (用于时间积分)
#[derive(Debug, Clone)]
pub struct RhsBuffers {
    /// 水深变化率 [m/s]
    pub dh_dt: Vec<f64>,
    /// x 动量变化率 [m²/s²]
    pub dhu_dt: Vec<f64>,
    /// y 动量变化率 [m²/s²]
    pub dhv_dt: Vec<f64>,
    /// 标量示踪剂变化率（可选）
    pub tracer_rhs: Vec<Vec<f64>>,
}

impl RhsBuffers {
    /// 创建新的 RHS 缓冲区
    pub fn new(n_cells: usize) -> Self {
        Self {
            dh_dt: vec![0.0; n_cells],
            dhu_dt: vec![0.0; n_cells],
            dhv_dt: vec![0.0; n_cells],
            tracer_rhs: Vec::new(),
        }
    }

    /// 创建带有示踪剂的 RHS 缓冲区
    pub fn with_tracers(n_cells: usize, n_tracers: usize) -> Self {
        Self {
            dh_dt: vec![0.0; n_cells],
            dhu_dt: vec![0.0; n_cells],
            dhv_dt: vec![0.0; n_cells],
            tracer_rhs: (0..n_tracers).map(|_| vec![0.0; n_cells]).collect(),
        }
    }

    /// 获取单元数量
    pub fn n_cells(&self) -> usize {
        self.dh_dt.len()
    }

    /// 获取示踪剂数量
    pub fn n_tracers(&self) -> usize {
        self.tracer_rhs.len()
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.dh_dt.fill(0.0);
        self.dhu_dt.fill(0.0);
        self.dhv_dt.fill(0.0);
        for tracer in &mut self.tracer_rhs {
            tracer.fill(0.0);
        }
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize, n_tracers: usize) {
        self.dh_dt.resize(n_cells, 0.0);
        self.dhu_dt.resize(n_cells, 0.0);
        self.dhv_dt.resize(n_cells, 0.0);
        self.tracer_rhs.resize_with(n_tracers, || vec![0.0; n_cells]);
        for tracer in &mut self.tracer_rhs {
            tracer.resize(n_cells, 0.0);
        }
    }

    /// 添加通量贡献
    #[inline]
    pub fn add_flux(&mut self, cell: usize, flux: Flux, area_inv: f64) {
        self.dh_dt[cell] += flux.mass * area_inv;
        self.dhu_dt[cell] += flux.mom_x * area_inv;
        self.dhv_dt[cell] += flux.mom_y * area_inv;
    }

    /// 添加源项贡献
    #[inline]
    pub fn add_source(&mut self, cell: usize, source: ConservedState) {
        self.dh_dt[cell] += source.h;
        self.dhu_dt[cell] += source.hu;
        self.dhv_dt[cell] += source.hv;
    }
}

// ============================================================
// 梯度状态
// ============================================================

/// 梯度状态 (用于二阶重构)
#[derive(Debug, Clone)]
pub struct GradientState {
    /// 水深梯度
    pub grad_h: Vec<DVec2>,
    /// x 动量梯度
    pub grad_hu: Vec<DVec2>,
    /// y 动量梯度
    pub grad_hv: Vec<DVec2>,
}

impl GradientState {
    /// 创建新的梯度状态
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_h: vec![DVec2::ZERO; n_cells],
            grad_hu: vec![DVec2::ZERO; n_cells],
            grad_hv: vec![DVec2::ZERO; n_cells],
        }
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.grad_h.fill(DVec2::ZERO);
        self.grad_hu.fill(DVec2::ZERO);
        self.grad_hv.fill(DVec2::ZERO);
    }

    /// 获取单元梯度
    #[inline]
    pub fn get(&self, cell: usize) -> (DVec2, DVec2, DVec2) {
        (self.grad_h[cell], self.grad_hu[cell], self.grad_hv[cell])
    }

    /// 设置单元梯度
    #[inline]
    pub fn set(&mut self, cell: usize, grad_h: DVec2, grad_hu: DVec2, grad_hv: DVec2) {
        self.grad_h[cell] = grad_h;
        self.grad_hu[cell] = grad_hu;
        self.grad_hv[cell] = grad_hv;
    }
}

// ============================================================
// 数值通量
// ============================================================

/// 数值通量
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Flux {
    /// 质量通量 [m²/s]
    pub mass: f64,
    /// x 动量通量 [m³/s²]
    pub mom_x: f64,
    /// y 动量通量 [m³/s²]
    pub mom_y: f64,
}

impl Flux {
    /// 创建新通量
    #[inline]
    pub const fn new(mass: f64, mom_x: f64, mom_y: f64) -> Self {
        Self { mass, mom_x, mom_y }
    }

    /// 零通量
    pub const ZERO: Self = Self {
        mass: 0.0,
        mom_x: 0.0,
        mom_y: 0.0,
    };

    /// 缩放通量
    #[inline]
    pub fn scale(self, factor: f64) -> Self {
        Self {
            mass: self.mass * factor,
            mom_x: self.mom_x * factor,
            mom_y: self.mom_y * factor,
        }
    }

    /// 通量大小
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.mass * self.mass + self.mom_x * self.mom_x + self.mom_y * self.mom_y).sqrt()
    }

    /// 检查通量是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.mass.is_finite() && self.mom_x.is_finite() && self.mom_y.is_finite()
    }
}

impl Add for Flux {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            mass: self.mass + rhs.mass,
            mom_x: self.mom_x + rhs.mom_x,
            mom_y: self.mom_y + rhs.mom_y,
        }
    }
}

impl AddAssign for Flux {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.mass += rhs.mass;
        self.mom_x += rhs.mom_x;
        self.mom_y += rhs.mom_y;
    }
}

impl Sub for Flux {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            mass: self.mass - rhs.mass,
            mom_x: self.mom_x - rhs.mom_x,
            mom_y: self.mom_y - rhs.mom_y,
        }
    }
}

impl SubAssign for Flux {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.mass -= rhs.mass;
        self.mom_x -= rhs.mom_x;
        self.mom_y -= rhs.mom_y;
    }
}

impl Neg for Flux {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            mass: -self.mass,
            mom_x: -self.mom_x,
            mom_y: -self.mom_y,
        }
    }
}

impl Mul<f64> for Flux {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl Mul<Flux> for f64 {
    type Output = Flux;
    #[inline]
    fn mul(self, rhs: Flux) -> Flux {
        rhs.scale(self)
    }
}

// ============================================================
// 错误类型
// ============================================================

/// 状态错误
#[derive(Debug, Clone)]
pub enum StateError {
    /// 无效值 (NaN/Inf)
    InvalidValue {
        field: &'static str,
        cell: usize,
        value: f64,
        time: f64,
    },
    /// 负水深
    NegativeDepth {
        cell: usize,
        value: f64,
        time: f64,
    },
    /// 速度过大
    ExcessiveVelocity {
        cell: usize,
        speed: f64,
        max_speed: f64,
        time: f64,
    },
    /// 尺寸不匹配
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidValue {
                field,
                cell,
                value,
                time,
            } => {
                write!(
                    f,
                    "Invalid {} at cell {} (value={}, time={})",
                    field, cell, value, time
                )
            }
            Self::NegativeDepth { cell, value, time } => {
                write!(
                    f,
                    "Negative depth at cell {} (h={}, time={})",
                    cell, value, time
                )
            }
            Self::ExcessiveVelocity {
                cell,
                speed,
                max_speed,
                time,
            } => {
                write!(
                    f,
                    "Excessive velocity at cell {} (speed={} > max={}, time={})",
                    cell, speed, max_speed, time
                )
            }
            Self::SizeMismatch { expected, actual } => {
                write!(
                    f,
                    "Size mismatch: expected {} cells, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::error::Error for StateError {}

// ============================================================
// StateAccess / StateAccessMut trait 实现
// ============================================================

use crate::traits::{StateAccess, StateAccessMut};

impl StateAccess for ShallowWaterState {
    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn get(&self, cell: usize) -> ConservedState {
        ConservedState::new(self.h[cell], self.hu[cell], self.hv[cell])
    }

    #[inline]
    fn h(&self, cell: usize) -> f64 {
        self.h[cell]
    }

    #[inline]
    fn hu(&self, cell: usize) -> f64 {
        self.hu[cell]
    }

    #[inline]
    fn hv(&self, cell: usize) -> f64 {
        self.hv[cell]
    }

    #[inline]
    fn z(&self, cell: usize) -> f64 {
        self.z[cell]
    }

    #[inline]
    fn h_slice(&self) -> &[f64] {
        &self.h
    }

    #[inline]
    fn hu_slice(&self) -> &[f64] {
        &self.hu
    }

    #[inline]
    fn hv_slice(&self) -> &[f64] {
        &self.hv
    }

    #[inline]
    fn z_slice(&self) -> &[f64] {
        &self.z
    }
}

impl StateAccessMut for ShallowWaterState {
    #[inline]
    fn set(&mut self, cell: usize, state: ConservedState) {
        self.h[cell] = state.h;
        self.hu[cell] = state.hu;
        self.hv[cell] = state.hv;
    }

    #[inline]
    fn set_h(&mut self, cell: usize, value: f64) {
        self.h[cell] = value;
    }

    #[inline]
    fn set_hu(&mut self, cell: usize, value: f64) {
        self.hu[cell] = value;
    }

    #[inline]
    fn set_hv(&mut self, cell: usize, value: f64) {
        self.hv[cell] = value;
    }

    #[inline]
    fn set_z(&mut self, cell: usize, value: f64) {
        self.z[cell] = value;
    }

    #[inline]
    fn h_slice_mut(&mut self) -> &mut [f64] {
        &mut self.h
    }

    #[inline]
    fn hu_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hu
    }

    #[inline]
    fn hv_slice_mut(&mut self) -> &mut [f64] {
        &mut self.hv
    }

    #[inline]
    fn z_slice_mut(&mut self) -> &mut [f64] {
        &mut self.z
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = ShallowWaterState::new(100);
        assert_eq!(state.n_cells(), 100);
        assert_eq!(state.h.len(), 100);
        assert_eq!(state.hu.len(), 100);
        assert_eq!(state.hv.len(), 100);
        assert_eq!(state.z.len(), 100);
    }

    #[test]
    fn test_cold_start() {
        let z_bed = vec![-10.0, -5.0, 0.0, 5.0];
        let state = ShallowWaterState::cold_start(0.0, &z_bed);

        assert_eq!(state.h[0], 10.0);
        assert_eq!(state.h[1], 5.0);
        assert_eq!(state.h[2], 0.0);
        assert_eq!(state.h[3], 0.0);
    }

    #[test]
    fn test_velocity_calculation() {
        let mut state = ShallowWaterState::new(1);
        state.h[0] = 2.0;
        state.hu[0] = 4.0;
        state.hv[0] = 6.0;

        let params = NumericalParams::default();
        let vel = state.velocity(0, &params);

        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_water_level() {
        let mut state = ShallowWaterState::new(1);
        state.h[0] = 3.0;
        state.z[0] = -5.0;

        assert_eq!(state.water_level(0), -2.0);
    }

    #[test]
    fn test_flux_operations() {
        let f1 = Flux::new(1.0, 2.0, 3.0);
        let f2 = Flux::new(0.5, 1.0, 1.5);

        let sum = f1 + f2;
        assert_eq!(sum.mass, 1.5);
        assert_eq!(sum.mom_x, 3.0);
        assert_eq!(sum.mom_y, 4.5);

        let scaled = f1 * 2.0;
        assert_eq!(scaled.mass, 2.0);
        assert_eq!(scaled.mom_x, 4.0);
        assert_eq!(scaled.mom_y, 6.0);

        let neg = -f1;
        assert_eq!(neg.mass, -1.0);
    }

    #[test]
    fn test_rhs_buffers() {
        let mut rhs = RhsBuffers::new(10);
        assert_eq!(rhs.dh_dt.len(), 10);

        rhs.add_flux(0, Flux::new(1.0, 2.0, 3.0), 0.5);
        assert_eq!(rhs.dh_dt[0], 0.5);
        assert_eq!(rhs.dhu_dt[0], 1.0);
        assert_eq!(rhs.dhv_dt[0], 1.5);
    }

    #[test]
    fn test_gradient_state() {
        let mut grad = GradientState::new(5);
        grad.set(0, DVec2::new(1.0, 2.0), DVec2::new(3.0, 4.0), DVec2::new(5.0, 6.0));

        let (gh, ghu, ghv) = grad.get(0);
        assert_eq!(gh, DVec2::new(1.0, 2.0));
        assert_eq!(ghu, DVec2::new(3.0, 4.0));
        assert_eq!(ghv, DVec2::new(5.0, 6.0));
    }

    #[test]
    fn test_state_linear_combine() {
        let mut result = ShallowWaterState::new(2);
        let mut a = ShallowWaterState::new(2);
        let mut b = ShallowWaterState::new(2);

        a.h[0] = 1.0;
        a.h[1] = 2.0;
        b.h[0] = 3.0;
        b.h[1] = 4.0;

        result.linear_combine(0.5, &a, 0.5, &b);

        assert_eq!(result.h[0], 2.0);
        assert_eq!(result.h[1], 3.0);
    }

    #[test]
    fn test_state_axpy() {
        let mut state = ShallowWaterState::new(2);
        let other = ShallowWaterState::cold_start(10.0, &[0.0, 5.0]);

        state.h[0] = 1.0;
        state.h[1] = 2.0;

        state.axpy(0.5, 0.5, &other);

        assert_eq!(state.h[0], 5.5); // 0.5 * 1.0 + 0.5 * 10.0
        assert_eq!(state.h[1], 3.5); // 0.5 * 2.0 + 0.5 * 5.0
    }

    #[test]
    fn test_conserved_state() {
        let state = ConservedState::new(2.0, 4.0, 6.0);
        let params = NumericalParams::default();
        let vel = state.velocity(&params);

        assert!((vel.u - 2.0).abs() < 1e-10);
        assert!((vel.v - 3.0).abs() < 1e-10);

        let from_prim = ConservedState::from_primitive(2.0, 2.0, 3.0);
        assert_eq!(from_prim, state);
    }
}
