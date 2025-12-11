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


use glam::DVec2;
use mh_foundation::memory::AlignedVec;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use crate::fields::{FieldMeta, FieldRegistry};
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
// 动态标量场（示踪剂等）
// ============================================================

/// 动态标量场集合，按名称管理示踪剂等扩展字段
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicScalars {
    /// 单元数量
    #[serde(default)]
    len: usize,
    /// 字段名称列表（顺序即存储顺序）
    #[serde(default)]
    names: Vec<String>,
    /// 数据存储
    #[serde(default)]
    data: Vec<AlignedVec<f64>>,
}

impl DynamicScalars {
    /// 创建空集合
    pub fn new(len: usize) -> Self {
        Self {
            len,
            names: Vec::new(),
            data: Vec::new(),
        }
    }

    /// 创建指定数量的匿名示踪剂字段（名称为 tracer_i）
    pub fn with_count(len: usize, count: usize) -> Self {
        let mut scalars = Self::new(len);
        for i in 0..count {
            scalars.register(format!("tracer_{i}"));
        }
        scalars
    }

    /// 当前单元数量
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 字段数量
    #[inline]
    pub fn count(&self) -> usize {
        self.data.len()
    }

    /// 字段名称列表
    #[inline]
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// 注册一个新字段，如已存在则直接返回索引
    pub fn register(&mut self, name: impl Into<String>) -> usize {
        let name = name.into();
        if let Some(pos) = self.names.iter().position(|n| n == &name) {
            // 确保长度一致
            self.data[pos].resize(self.len);
            return pos;
        }

        self.names.push(name);
        self.data.push(AlignedVec::zeros(self.len));
        self.data.len() - 1
    }

    /// 按索引获取只读切片
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&[f64]> {
        self.data.get(idx).map(|v| v.as_slice())
    }

    /// 按索引获取可变切片
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut [f64]> {
        self.data.get_mut(idx).map(|v| v.as_mut_slice())
    }

    /// 按名称获取只读切片
    pub fn get_by_name(&self, name: &str) -> Option<&[f64]> {
        self.names.iter().position(|n| n == name).and_then(|i| self.get(i))
    }

    /// 按名称获取可变切片
    pub fn get_mut_by_name(&mut self, name: &str) -> Option<&mut [f64]> {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return self.get_mut(pos);
        }
        None
    }

    /// 将所有字段清零
    pub fn clear_all(&mut self) {
        for field in &mut self.data {
            field.as_mut_slice().fill(0.0);
        }
    }

    /// 调整单元长度并保持已有数据（新增部分填零）
    pub fn resize_len(&mut self, len: usize) {
        self.len = len;
        for field in &mut self.data {
            field.resize(len);
        }
    }

    /// 按另一个集合的布局对齐（名称、数量、长度），但不复制数据
    pub fn match_layout(&mut self, other: &Self) {
        if self.len != other.len || self.names != other.names {
            self.len = other.len;
            self.names = other.names.clone();
            self.data = other
                .data
                .iter()
                .map(|_| AlignedVec::zeros(other.len))
                .collect();
        } else {
            self.resize_len(other.len);
        }
    }

    /// 复制数据并对齐布局
    pub fn copy_from(&mut self, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            dst.as_mut_slice().copy_from_slice(src.as_slice());
        }
    }

    /// self += scale * rhs
    pub fn add_scaled(&mut self, rhs: &Self, scale: f64) {
        self.match_layout(rhs);
        for (dst, src) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
                *d += scale * s;
            }
        }
    }

    /// 设置字段数量，多余的截断，不足的以 tracer_i 填充
    pub fn set_count(&mut self, count: usize) {
        self.names.truncate(count);
        self.data.truncate(count);
        while self.data.len() < count {
            let idx = self.data.len();
            self.names.push(format!("tracer_{idx}"));
            self.data.push(AlignedVec::zeros(self.len));
        }
    }

    /// self = a * A + b * B
    pub fn linear_combine(&mut self, a: f64, state_a: &Self, b: f64, state_b: &Self) {
        debug_assert_eq!(state_a.names, state_b.names, "示踪剂字段布局不一致");
        self.match_layout(state_a);
        for ((dst, sa), sb) in self
            .data
            .iter_mut()
            .zip(state_a.data.iter())
            .zip(state_b.data.iter())
        {
            for ((d, a_val), b_val) in dst
                .as_mut_slice()
                .iter_mut()
                .zip(sa.as_slice())
                .zip(sb.as_slice())
            {
                *d = a * a_val + b * b_val;
            }
        }
    }

    /// self = a * self + b * other
    pub fn axpy(&mut self, a: f64, b: f64, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice()) {
                *d = a * *d + b * s;
            }
        }
    }

    /// 迭代所有字段的可变存储
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut AlignedVec<f64>> {
        self.data.iter_mut()
    }
}

// ============================================================
// 浅水方程状态 (SoA 布局)
// ============================================================

/// 浅水方程守恒状态（SoA 布局）
///
/// 存储整个网格的状态变量，采用 SoA 布局优化缓存访问。
/// 
/// 速度场通过 `velocity()` 方法从动量和水深实时计算，
/// 避免存储冗余数据并确保数据一致性。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShallowWaterState {
    /// 单元数量
    n_cells: usize,

    /// 水深 [m]
    pub h: AlignedVec<f64>,
    /// x 方向动量 [m²/s]
    pub hu: AlignedVec<f64>,
    /// y 方向动量 [m²/s]
    pub hv: AlignedVec<f64>,
    /// 底床高程 [m]
    pub z: AlignedVec<f64>,

    /// 动态示踪剂字段
    #[serde(default)]
    pub tracers: DynamicScalars,

    /// 字段注册表（元数据）
    #[serde(default = "FieldRegistry::shallow_water")]
    pub field_registry: FieldRegistry,
}

impl ShallowWaterState {
    /// 创建新状态
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: AlignedVec::zeros(n_cells),
            hu: AlignedVec::zeros(n_cells),
            hv: AlignedVec::zeros(n_cells),
            z: AlignedVec::zeros(n_cells),
            tracers: DynamicScalars::new(n_cells),
            field_registry: FieldRegistry::shallow_water(),
        }
    }

    /// 创建带标量的状态
    pub fn with_scalar(n_cells: usize) -> Self {
        let mut state = Self::new(n_cells);
        state.register_tracer("tracer_0", "");
        state
    }

    /// 从初始水位和底床创建（冷启动）
    pub fn cold_start(initial_eta: f64, z_bed: &[f64]) -> Self {
        let n_cells = z_bed.len();

        let h: Vec<f64> = z_bed.iter().map(|&z| (initial_eta - z).max(0.0)).collect();

        Self {
            n_cells,
            h: AlignedVec::from_vec(h),
            hu: AlignedVec::zeros(n_cells),
            hv: AlignedVec::zeros(n_cells),
            z: AlignedVec::from_vec(z_bed.to_vec()),
            tracers: DynamicScalars::new(n_cells),
            field_registry: FieldRegistry::shallow_water(),
        }
    }

    /// 克隆结构（不复制数据，创建零初始化的状态）
    pub fn clone_structure(&self) -> Self {
        let mut tracers = DynamicScalars::new(self.n_cells);
        tracers.match_layout(&self.tracers);
        Self {
            n_cells: self.n_cells,
            h: AlignedVec::zeros(self.n_cells),
            hu: AlignedVec::zeros(self.n_cells),
            hv: AlignedVec::zeros(self.n_cells),
            z: self.z.clone(),
            tracers,
            field_registry: self.field_registry.clone(),
        }
    }

    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    /// 注册一个新的示踪剂字段，若已存在则返回其索引
    pub fn register_tracer(&mut self, name: impl Into<String>, unit: impl Into<String>) -> usize {
        let name = name.into();
        let idx = self.tracers.register(name.clone());
        if !self.field_registry.contains(&name) {
            self.field_registry
                .register(FieldMeta::cell_scalar(name.clone(), unit.into()).with_desc("示踪剂标量"));
        }
        idx
    }

    /// 获取示踪剂数量
    #[inline]
    pub fn tracer_count(&self) -> usize {
        self.tracers.count()
    }

    /// 获取所有示踪剂名称
    #[inline]
    pub fn tracer_names(&self) -> &[String] {
        self.tracers.names()
    }

    /// 按索引获取示踪剂切片
    #[inline]
    pub fn tracer_slice(&self, idx: usize) -> Option<&[f64]> {
        self.tracers.get(idx)
    }

    /// 按索引获取可变示踪剂切片
    #[inline]
    pub fn tracer_slice_mut(&mut self, idx: usize) -> Option<&mut [f64]> {
        self.tracers.get_mut(idx)
    }

    /// 按名称获取示踪剂切片
    #[inline]
    pub fn tracer_by_name(&self, name: &str) -> Option<&[f64]> {
        self.tracers.get_by_name(name)
    }

    /// 按名称获取可变示踪剂切片
    #[inline]
    pub fn tracer_by_name_mut(&mut self, name: &str) -> Option<&mut [f64]> {
        self.tracers.get_mut_by_name(name)
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
        self.tracers.clear_all();
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
        self.tracers.copy_from(&other.tracers);
    }

    /// 添加缩放的 RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &RhsBuffers, scale: f64) {
        for i in 0..self.n_cells {
            self.h[i] += scale * rhs.dh_dt[i];
            self.hu[i] += scale * rhs.dhu_dt[i];
            self.hv[i] += scale * rhs.dhv_dt[i];
        }
        self.tracers.add_scaled(&rhs.tracer_rhs, scale);
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
        self.tracers.linear_combine(a, &state_a.tracers, b, &state_b.tracers);
    }

    /// 自线性组合: self = a * self + b * other
    pub fn axpy(&mut self, a: f64, b: f64, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());

        for i in 0..self.n_cells {
            self.h[i] = a * self.h[i] + b * other.h[i];
            self.hu[i] = a * self.hu[i] + b * other.hu[i];
            self.hv[i] = a * self.hv[i] + b * other.hv[i];
        }
        self.tracers.axpy(a, b, &other.tracers);
    }

    /// 强制正性约束
    pub fn enforce_positivity(&mut self) {
        for h in self.h.iter_mut() {
            if *h < 0.0 {
                *h = 0.0;
            }
        }

        for tracer in self.tracers.iter_mut() {
            for v in tracer.as_mut_slice() {
                if *v < 0.0 {
                    *v = 0.0;
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
    pub dh_dt: AlignedVec<f64>,
    /// x 动量变化率 [m²/s²]
    pub dhu_dt: AlignedVec<f64>,
    /// y 动量变化率 [m²/s²]
    pub dhv_dt: AlignedVec<f64>,
    /// 标量示踪剂变化率（可选）
    pub tracer_rhs: DynamicScalars,
}

impl RhsBuffers {
    /// 创建新的 RHS 缓冲区
    pub fn new(n_cells: usize) -> Self {
        Self {
            dh_dt: AlignedVec::zeros(n_cells),
            dhu_dt: AlignedVec::zeros(n_cells),
            dhv_dt: AlignedVec::zeros(n_cells),
            tracer_rhs: DynamicScalars::new(n_cells),
        }
    }

    /// 创建带有示踪剂的 RHS 缓冲区
    pub fn with_tracers(n_cells: usize, n_tracers: usize) -> Self {
        let mut rhs = Self::new(n_cells);
        rhs.tracer_rhs.set_count(n_tracers);
        rhs
    }

    /// 获取单元数量
    pub fn n_cells(&self) -> usize {
        self.dh_dt.len()
    }

    /// 获取示踪剂数量
    pub fn n_tracers(&self) -> usize {
        self.tracer_rhs.count()
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.dh_dt.fill(0.0);
        self.dhu_dt.fill(0.0);
        self.dhv_dt.fill(0.0);
        self.tracer_rhs.clear_all();
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize, n_tracers: usize) {
        self.dh_dt.resize(n_cells);
        self.dhu_dt.resize(n_cells);
        self.dhv_dt.resize(n_cells);
        self.tracer_rhs.resize_len(n_cells);
        self.tracer_rhs.set_count(n_tracers);
    }

    /// 将示踪剂布局对齐到给定状态
    pub fn match_tracers(&mut self, layout: &DynamicScalars) {
        self.tracer_rhs.match_layout(layout);
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

// ============================================================
// 泛型浅水状态 (Backend 抽象)
// ============================================================

use crate::core::{Backend, CpuBackend, DeviceBuffer, Scalar};

/// 泛型浅水状态
///
/// 使用 Backend trait 抽象存储，支持 CPU/GPU 后端。
/// 永远只有4个核心字段：h, hu, hv, z。
///
/// # 设计说明
///
/// 状态持有 Backend 实例的克隆，用于后续的缓冲区操作。
/// 由于 CpuBackend 是零大小类型，这不会带来额外开销。
#[derive(Debug, Clone)]
pub struct ShallowWaterStateGeneric<B: Backend> {
    /// 单元数量
    n_cells: usize,
    /// 水深 [m]
    pub h: B::Buffer<B::Scalar>,
    /// x 方向动量 [m²/s]
    pub hu: B::Buffer<B::Scalar>,
    /// y 方向动量 [m²/s]
    pub hv: B::Buffer<B::Scalar>,
    /// 底床高程 [m]
    pub z: B::Buffer<B::Scalar>,
    /// 后端实例
    backend: B,
}

impl<B: Backend> ShallowWaterStateGeneric<B> {
    /// 使用后端实例创建新状态
    pub fn new_with_backend(backend: B, n_cells: usize) -> Self {
        Self {
            n_cells,
            h: backend.alloc(n_cells),
            hu: backend.alloc(n_cells),
            hv: backend.alloc(n_cells),
            z: backend.alloc(n_cells),
            backend,
        }
    }
    
    /// 单元数量
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 重置为零
    pub fn reset(&mut self) {
        self.h.fill(<B::Scalar as Scalar>::from_f64(0.0));
        self.hu.fill(<B::Scalar as Scalar>::from_f64(0.0));
        self.hv.fill(<B::Scalar as Scalar>::from_f64(0.0));
    }
    
    /// 验证状态有效性
    pub fn is_valid(&self) -> bool {
        if let Some(h) = self.h.as_slice() {
            h.iter().all(|&x| x.to_f64().is_finite() && x.to_f64() >= 0.0)
        } else {
            // GPU 缓冲区需要同步检查
            true
        }
    }
}

/// CPU f64 后端的便捷方法
impl ShallowWaterStateGeneric<CpuBackend<f64>> {
    /// 使用默认 CPU f64 后端创建（向后兼容）
    pub fn new(n_cells: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), n_cells)
    }
    
    /// 从传统 ShallowWaterState 创建
    pub fn from_legacy(state: &ShallowWaterState) -> Self {
        let n = state.n_cells();
        let mut new_state = Self::new(n);
        
        new_state.h.copy_from_slice(state.h.as_slice());
        new_state.hu.copy_from_slice(state.hu.as_slice());
        new_state.hv.copy_from_slice(state.hv.as_slice());
        new_state.z.copy_from_slice(state.z.as_slice());
        
        new_state
    }
    
    /// 转换回传统 ShallowWaterState
    pub fn to_legacy(&self) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(self.n_cells);
        
        state.h.as_mut_slice().copy_from_slice(&self.h);
        state.hu.as_mut_slice().copy_from_slice(&self.hu);
        state.hv.as_mut_slice().copy_from_slice(&self.hv);
        state.z.as_mut_slice().copy_from_slice(&self.z);
        
        state
    }
}

/// CPU f32 后端的便捷方法
impl ShallowWaterStateGeneric<CpuBackend<f32>> {
    /// 使用 CPU f32 后端创建
    pub fn new_f32(n_cells: usize) -> Self {
        Self::new_with_backend(CpuBackend::<f32>::new(), n_cells)
    }
}

/// 类型别名：默认后端的状态
pub type ShallowWaterStateDefault = ShallowWaterStateGeneric<CpuBackend<f64>>;

// ============================================================
// 泛型状态的统计计算
// ============================================================

/// 状态统计信息
#[derive(Debug, Clone, Copy, Default)]
pub struct StateStatisticsData<S> {
    /// 最大水深
    pub h_max: S,
    /// 最小水深（非零）
    pub h_min: S,
    /// 平均水深
    pub h_mean: S,
    /// 最大速度
    pub velocity_max: S,
    /// 水体总体积
    pub total_volume: S,
    /// 湿单元数量
    pub wet_cells: usize,
}

impl<B: Backend> ShallowWaterStateGeneric<B> {
    /// 计算状态统计信息（仅 CPU 后端有效）
    pub fn compute_statistics(&self, cell_areas: &[B::Scalar], h_dry: B::Scalar) -> Option<StateStatisticsData<B::Scalar>> {
        let h_slice = self.h.as_slice()?;
        let hu_slice = self.hu.as_slice()?;
        let hv_slice = self.hv.as_slice()?;
        
        let mut stats = StateStatisticsData {
            h_max: <B::Scalar as Scalar>::from_f64(0.0),
            h_min: <B::Scalar as Scalar>::from_f64(f64::MAX),
            h_mean: <B::Scalar as Scalar>::from_f64(0.0),
            velocity_max: <B::Scalar as Scalar>::from_f64(0.0),
            total_volume: <B::Scalar as Scalar>::from_f64(0.0),
            wet_cells: 0,
        };
        
        for i in 0..self.n_cells {
            let h = h_slice[i];
            
            if h > h_dry {
                stats.wet_cells += 1;
                stats.h_max = Float::max(stats.h_max, h);
                stats.h_min = Float::min(stats.h_min, h);
                
                // 计算速度
                let u = hu_slice[i] / h;
                let v = hv_slice[i] / h;
                let speed = Float::sqrt(u * u + v * v);
                stats.velocity_max = Float::max(stats.velocity_max, speed);
                
                // 累加体积
                if i < cell_areas.len() {
                    stats.total_volume = stats.total_volume + h * cell_areas[i];
                }
            }
        }
        
        if stats.wet_cells > 0 {
            stats.h_mean = stats.total_volume / <B::Scalar as Scalar>::from_f64(stats.wet_cells as f64);
        }
        
        Some(stats)
    }
    
    /// 复制状态数据到另一个状态
    pub fn copy_to(&self, other: &mut Self) {
        debug_assert_eq!(self.n_cells, other.n_cells, "状态复制: 单元数量不匹配");
        
        // CPU 后端直接复制
        if let (Some(src_h), Some(dst_h)) = (self.h.as_slice(), other.h.as_slice_mut()) {
            dst_h.copy_from_slice(src_h);
        }
        if let (Some(src_hu), Some(dst_hu)) = (self.hu.as_slice(), other.hu.as_slice_mut()) {
            dst_hu.copy_from_slice(src_hu);
        }
        if let (Some(src_hv), Some(dst_hv)) = (self.hv.as_slice(), other.hv.as_slice_mut()) {
            dst_hv.copy_from_slice(src_hv);
        }
        if let (Some(src_z), Some(dst_z)) = (self.z.as_slice(), other.z.as_slice_mut()) {
            dst_z.copy_from_slice(src_z);
        }
    }
}

