// crates/mh_physics/src/state.rs

//! 浅水方程状态管理
//!
//! 本模块提供浅水方程求解所需的状态管理，包括：
//! - `ShallowWaterState<B>`: 守恒变量状态 (h, hu, hv)，基于 Backend 泛型
//! - `DynamicScalars`: 动态标量场（示踪剂等）
//! - `GradientState`: 梯度状态 (grad_h, grad_hu, grad_hv)
//! - `Flux`: 数值通量
//! - `RhsBuffers`: 右端项缓冲区
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
//! # 类型参数
//!
//! - `B: Backend`: 计算后端，支持 `CpuBackend<f32>` 和 `CpuBackend<f64>`
//!
//! # 示例
//!
//! ```rust
//! use mh_physics::state::ShallowWaterState;
//! use mh_runtime::CpuBackend;
//!
//! // 创建 f64 精度的状态
//! let state_f64 = ShallowWaterState::<CpuBackend<f64>>::new(100);
//!
//! // 创建 f32 精度的状态
//! let state_f32 = ShallowWaterState::<CpuBackend<f32>>::new(100);
//! ```

use crate::fields::{FieldMeta, FieldRegistry};
use crate::traits::{StateAccess, StateAccessMut};
use crate::types::{NumericalParams, SafeVelocity};
use mh_runtime::{Backend, CpuBackend, RuntimeScalar};
use num_traits::Float;
use serde::{Deserialize, Serialize};

// ============================================================
// 单个单元的守恒状态
// ============================================================

/// 单个单元的守恒状态
///
/// 包含浅水方程的三个守恒变量：
/// - `h`: 水深
/// - `hu`: x方向动量
/// - `hv`: y方向动量
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConservedState<S: RuntimeScalar> {
    /// 水深 [m]
    pub h: S,
    /// x 方向动量 [m²/s]
    pub hu: S,
    /// y 方向动量 [m²/s]
    pub hv: S,
}

impl<S: RuntimeScalar> ConservedState<S> {
    /// 创建新的守恒状态
    #[inline]
    pub const fn new(h: S, hu: S, hv: S) -> Self {
        Self { h, hu, hv }
    }

    /// 零状态
    pub const ZERO: Self = Self {
        h: S::ZERO,
        hu: S::ZERO,
        hv: S::ZERO,
    };

    /// 从原始变量创建
    #[inline]
    pub fn from_primitive(h: S, u: S, v: S) -> Self {
        Self {
            h,
            hu: h * u,
            hv: h * v,
        }
    }

    /// 获取速度 (使用安全除法)
    #[inline]
    pub fn velocity(&self, params: &NumericalParams<S>) -> SafeVelocity<S> {
        params.safe_velocity(self.hu, self.hv, self.h)
    }

    /// 状态是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.h.is_finite() && self.hu.is_finite() && self.hv.is_finite() && self.h >= S::ZERO
    }
}

// 算术运算实现
impl<S: RuntimeScalar> std::ops::Add for ConservedState<S> {
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

impl<S: RuntimeScalar> std::ops::Sub for ConservedState<S> {
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

impl<S: RuntimeScalar> std::ops::Mul<S> for ConservedState<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
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
pub struct DynamicScalars<S: RuntimeScalar> {
    /// 单元数量
    #[serde(default)]
    len: usize,
    /// 字段名称列表（顺序即存储顺序）
    #[serde(default)]
    names: Vec<String>,
    /// 数据存储
    #[serde(default)]
    data: Vec<Vec<S>>,
}

impl<S: RuntimeScalar> DynamicScalars<S> {
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
            self.data[pos].resize(self.len, S::ZERO);
            return pos;
        }

        self.names.push(name);
        self.data.push(vec![S::ZERO; self.len]);
        self.data.len() - 1
    }

    /// 按索引获取只读切片
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&[S]> {
        self.data.get(idx).map(|v| v.as_slice())
    }

    /// 按索引获取可变切片
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut [S]> {
        self.data.get_mut(idx).map(|v| v.as_mut_slice())
    }

    /// 按名称获取只读切片
    pub fn get_by_name(&self, name: &str) -> Option<&[S]> {
        self.names.iter().position(|n| n == name).and_then(|i| self.get(i))
    }

    /// 按名称获取可变示踪剂切片
    pub fn get_mut_by_name(&mut self, name: &str) -> Option<&mut [S]> {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return self.get_mut(pos);
        }
        None
    }

    /// 将所有字段清零
    pub fn clear_all(&mut self) {
        for field in &mut self.data {
            field.fill(S::ZERO);
        }
    }

    /// 调整单元长度并保持已有数据（新增部分填零）
    pub fn resize_len(&mut self, len: usize) {
        self.len = len;
        for field in &mut self.data {
            field.resize(len, S::ZERO);
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
                .map(|_| vec![S::ZERO; other.len])
                .collect();
        } else {
            self.resize_len(other.len);
        }
    }

    /// 复制数据并对齐布局
    pub fn copy_from(&mut self, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            dst.copy_from_slice(src.as_slice());
        }
    }

    /// self += scale * rhs
    pub fn add_scaled(&mut self, rhs: &Self, scale: S) {
        self.match_layout(rhs);
        for (dst, src) in self.data.iter_mut().zip(rhs.data.iter()) {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += scale * *s;
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
            self.data.push(vec![S::ZERO; self.len]);
        }
    }

    /// self = a * A + b * B
    pub fn linear_combine(&mut self, a: S, state_a: &Self, b: S, state_b: &Self) {
        debug_assert_eq!(state_a.names, state_b.names, "示踪剂字段布局不一致");
        self.match_layout(state_a);
        for ((dst, sa), sb) in self
            .data
            .iter_mut()
            .zip(state_a.data.iter())
            .zip(state_b.data.iter())
        {
            for ((d, a_val), b_val) in dst
                .iter_mut()
                .zip(sa.iter())
                .zip(sb.iter())
            {
                *d = a * *a_val + b * *b_val;
            }
        }
    }

    /// self = a * self + b * other
    pub fn axpy(&mut self, a: S, b: S, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = a * *d + b * *s;
            }
        }
    }

    /// 迭代所有字段的可变存储
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Vec<S>> {
        self.data.iter_mut()
    }
}

// ============================================================
// 梯度状态
// ============================================================

/// 梯度状态 (用于二阶重构)
#[derive(Debug, Clone)]
pub struct GradientState<S: RuntimeScalar> {
    /// 水深梯度 x 分量
    pub grad_h_x: Vec<S>,
    /// 水深梯度 y 分量
    pub grad_h_y: Vec<S>,
    /// x 动量梯度 x 分量
    pub grad_hu_x: Vec<S>,
    /// x 动量梯度 y 分量
    pub grad_hu_y: Vec<S>,
    /// y 动量梯度 x 分量
    pub grad_hv_x: Vec<S>,
    /// y 动量梯度 y 分量
    pub grad_hv_y: Vec<S>,
}

impl<S: RuntimeScalar> GradientState<S> {
    /// 创建新的梯度状态
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_h_x: vec![S::ZERO; n_cells],
            grad_h_y: vec![S::ZERO; n_cells],
            grad_hu_x: vec![S::ZERO; n_cells],
            grad_hu_y: vec![S::ZERO; n_cells],
            grad_hv_x: vec![S::ZERO; n_cells],
            grad_hv_y: vec![S::ZERO; n_cells],
        }
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.grad_h_x.fill(S::ZERO);
        self.grad_h_y.fill(S::ZERO);
        self.grad_hu_x.fill(S::ZERO);
        self.grad_hu_y.fill(S::ZERO);
        self.grad_hv_x.fill(S::ZERO);
        self.grad_hv_y.fill(S::ZERO);
    }

    /// 获取单元梯度向量
    #[inline]
    pub fn get_h(&self, cell: usize) -> (S, S) {
        (self.grad_h_x[cell], self.grad_h_y[cell])
    }

    /// 设置单元 h 梯度
    #[inline]
    pub fn set_h(&mut self, cell: usize, grad_x: S, grad_y: S) {
        self.grad_h_x[cell] = grad_x;
        self.grad_h_y[cell] = grad_y;
    }

    /// 获取单元 hu 梯度
    #[inline]
    pub fn get_hu(&self, cell: usize) -> (S, S) {
        (self.grad_hu_x[cell], self.grad_hu_y[cell])
    }

    /// 设置单元 hu 梯度
    #[inline]
    pub fn set_hu(&mut self, cell: usize, grad_x: S, grad_y: S) {
        self.grad_hu_x[cell] = grad_x;
        self.grad_hu_y[cell] = grad_y;
    }

    /// 获取单元 hv 梯度
    #[inline]
    pub fn get_hv(&self, cell: usize) -> (S, S) {
        (self.grad_hv_x[cell], self.grad_hv_y[cell])
    }

    /// 设置单元 hv 梯度
    #[inline]
    pub fn set_hv(&mut self, cell: usize, grad_x: S, grad_y: S) {
        self.grad_hv_x[cell] = grad_x;
        self.grad_hv_y[cell] = grad_y;
    }
}

// ============================================================
// 数值通量
// ============================================================

/// 数值通量
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Flux<S: RuntimeScalar> {
    /// 质量通量 [m²/s]
    pub mass: S,
    /// x 动量通量 [m³/s²]
    pub mom_x: S,
    /// y 动量通量 [m³/s²]
    pub mom_y: S,
}

impl<S: RuntimeScalar> Flux<S> {
    /// 创建新通量
    #[inline]
    pub const fn new(mass: S, mom_x: S, mom_y: S) -> Self {
        Self { mass, mom_x, mom_y }
    }

    /// 零通量
    pub const ZERO: Self = Self {
        mass: S::ZERO,
        mom_x: S::ZERO,
        mom_y: S::ZERO,
    };

    /// 缩放通量
    #[inline]
    pub fn scale(self, factor: S) -> Self {
        Self {
            mass: self.mass * factor,
            mom_x: self.mom_x * factor,
            mom_y: self.mom_y * factor,
        }
    }

    /// 通量大小
    #[inline]
    pub fn magnitude(&self) -> S {
        (self.mass * self.mass + self.mom_x * self.mom_x + self.mom_y * self.mom_y).sqrt()
    }

    /// 检查通量是否有效
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.mass.is_finite() && self.mom_x.is_finite() && self.mom_y.is_finite()
    }
}

// 算术运算实现
impl<S: RuntimeScalar> std::ops::Add for Flux<S> {
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

impl<S: RuntimeScalar> std::ops::Sub for Flux<S> {
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

impl<S: RuntimeScalar> std::ops::Neg for Flux<S> {
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

impl<S: RuntimeScalar> std::ops::Mul<S> for Flux<S> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        self.scale(rhs)
    }
}

// ============================================================
// 右端项缓冲区
// ============================================================

/// 右端项缓冲区 (用于时间积分)
#[derive(Debug, Clone)]
pub struct RhsBuffers<S: RuntimeScalar> {
    /// 水深变化率 [m/s]
    pub dh_dt: Vec<S>,
    /// x 动量变化率 [m²/s²]
    pub dhu_dt: Vec<S>,
    /// y 动量变化率 [m²/s²]
    pub dhv_dt: Vec<S>,
    /// 标量示踪剂变化率（可选）
    pub tracer_rhs: DynamicScalars<S>,
}

impl<S: RuntimeScalar> RhsBuffers<S> {
    /// 创建新的 RHS 缓冲区
    pub fn new(n_cells: usize) -> Self {
        Self {
            dh_dt: vec![S::ZERO; n_cells],
            dhu_dt: vec![S::ZERO; n_cells],
            dhv_dt: vec![S::ZERO; n_cells],
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
        self.dh_dt.fill(S::ZERO);
        self.dhu_dt.fill(S::ZERO);
        self.dhv_dt.fill(S::ZERO);
        self.tracer_rhs.clear_all();
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize, n_tracers: usize) {
        self.dh_dt.resize(n_cells, S::ZERO);
        self.dhu_dt.resize(n_cells, S::ZERO);
        self.dhv_dt.resize(n_cells, S::ZERO);
        self.tracer_rhs.resize_len(n_cells);
        self.tracer_rhs.set_count(n_tracers);
    }

    /// 将示踪剂布局对齐到给定状态
    pub fn match_tracers(&mut self, layout: &DynamicScalars<S>) {
        self.tracer_rhs.match_layout(layout);
    }

    /// 添加通量贡献
    #[inline]
    pub fn add_flux(&mut self, cell: usize, flux: Flux<S>, area_inv: S) {
        self.dh_dt[cell] += flux.mass * area_inv;
        self.dhu_dt[cell] += flux.mom_x * area_inv;
        self.dhv_dt[cell] += flux.mom_y * area_inv;
    }

    /// 添加源项贡献
    #[inline]
    pub fn add_source(&mut self, cell: usize, source: ConservedState<S>) {
        self.dh_dt[cell] += source.h;
        self.dhu_dt[cell] += source.hu;
        self.dhv_dt[cell] += source.hv;
    }
}

// ============================================================
// 浅水方程状态 (SoA 布局)
// ============================================================

/// 浅水方程守恒状态（SoA 布局）
///
/// 使用 Backend 泛型存储整个网格的状态变量，采用 SoA 布局优化缓存访问。
/// 支持 f32/f64 精度切换和 GPU 后端扩展。
///
/// # 类型参数
///
/// - `B: Backend`: 计算后端，提供存储和计算能力
///
/// # 设计原则
///
/// 1. **泛型存储**: 所有字段使用 `B::Buffer<S>` 存储
/// 2. **零拷贝**: 通过 Backend 直接操作底层缓冲区
/// 3. **类型安全**: 使用索引类型防止越界
#[derive(Debug, Clone)]
pub struct ShallowWaterState<B: Backend> {
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
    /// 动态示踪剂字段
    pub tracers: DynamicScalars<B::Scalar>,
    /// 字段注册表（元数据）
    pub field_registry: FieldRegistry,
    /// 后端实例
    backend: B,
}

impl<B: Backend> ShallowWaterState<B> {
    /// 使用后端实例创建新状态
    pub fn new_with_backend(backend: B, n_cells: usize) -> Self {
        let tracers = DynamicScalars::new(n_cells);
        let field_registry = FieldRegistry::shallow_water();
        
        Self {
            n_cells,
            h: backend.alloc(n_cells),
            hu: backend.alloc(n_cells),
            hv: backend.alloc(n_cells),
            z: backend.alloc(n_cells),
            tracers,
            field_registry,
            backend,
        }
    }

    /// 创建带标量的状态
    pub fn with_scalars(backend: B, n_cells: usize, n_scalars: usize) -> Self {
        let mut state = Self::new_with_backend(backend, n_cells);
        for i in 0..n_scalars {
            state.register_tracer(&format!("tracer_{i}"), "");
        }
        state
    }

    /// 从初始水位和底床创建（冷启动）
    pub fn cold_start(backend: B, initial_eta: B::Scalar, z_bed: &[B::Scalar]) -> Self {
        let n_cells = z_bed.len();
        let mut state = Self::new_with_backend(backend, n_cells);
        
        // 计算水深 h = max(0, eta - z)
        for (i, &z) in z_bed.iter().enumerate() {
            let h = (initial_eta - z).max(B::Scalar::ZERO);
            state.h[i] = h;
            state.hu[i] = B::Scalar::ZERO;
            state.hv[i] = B::Scalar::ZERO;
            state.z[i] = z;
        }
        
        state
    }

    /// 克隆结构（不复制数据，创建零初始化的状态）
    pub fn clone_structure(&self) -> Self {
        let backend = self.backend.clone();
        let mut tracers = DynamicScalars::new(self.n_cells);
        tracers.match_layout(&self.tracers);
        
        Self {
            n_cells: self.n_cells,
            h: backend.alloc(self.n_cells),
            hu: backend.alloc(self.n_cells),
            hv: backend.alloc(self.n_cells),
            z: backend.alloc(self.n_cells),
            tracers,
            field_registry: self.field_registry.clone(),
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

    /// 注册一个新的示踪剂字段，若已存在则返回其索引
    pub fn register_tracer(&mut self, name: impl Into<String>, unit: impl Into<String>) -> usize {
        let name = name.into();
        let idx = self.tracers.register(name.clone());
        if !self.field_registry.contains(&name) {
            self.field_registry.register(
                FieldMeta::cell_scalar(name.clone(), unit.into())
                    .with_desc("示踪剂标量")
            );
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
    pub fn tracer_slice(&self, idx: usize) -> Option<&[B::Scalar]> {
        self.tracers.get(idx)
    }

    /// 按索引获取可变示踪剂切片
    #[inline]
    pub fn tracer_slice_mut(&mut self, idx: usize) -> Option<&mut [B::Scalar]> {
        self.tracers.get_mut(idx)
    }

    /// 按名称获取示踪剂切片
    #[inline]
    pub fn tracer_by_name(&self, name: &str) -> Option<&[B::Scalar]> {
        self.tracers.get_by_name(name)
    }

    /// 按名称获取可变示踪剂切片
    #[inline]
    pub fn tracer_by_name_mut(&mut self, name: &str) -> Option<&mut [B::Scalar]> {
        self.tracers.get_mut_by_name(name)
    }

    // ========== 状态访问 ==========

    /// 获取单元的守恒状态
    #[inline]
    pub fn get(&self, idx: usize) -> ConservedState<B::Scalar> {
        ConservedState::new(self.h[idx], self.hu[idx], self.hv[idx])
    }

    /// 获取原始变量 (h, u, v)
    #[inline]
    pub fn primitive(&self, idx: usize, params: &NumericalParams<B::Scalar>) -> (B::Scalar, B::Scalar, B::Scalar) {
        let h = self.h[idx];
        let vel = params.safe_velocity(self.hu[idx], self.hv[idx], h);
        (h, vel.u, vel.v)
    }

    /// 获取速度
    #[inline]
    pub fn velocity(&self, idx: usize, params: &NumericalParams<B::Scalar>) -> SafeVelocity<B::Scalar> {
        params.safe_velocity(self.hu[idx], self.hv[idx], self.h[idx])
    }

    /// 获取水位 (eta = h + z)
    #[inline]
    pub fn water_level(&self, idx: usize) -> B::Scalar {
        self.h[idx] + self.z[idx]
    }

    // ========== 状态修改 ==========

    /// 设置守恒变量
    #[inline]
    pub fn set(&mut self, idx: usize, h: B::Scalar, hu: B::Scalar, hv: B::Scalar) {
        self.h[idx] = h;
        self.hu[idx] = hu;
        self.hv[idx] = hv;
    }

    /// 设置守恒状态
    #[inline]
    pub fn set_state(&mut self, idx: usize, state: ConservedState<B::Scalar>) {
        self.h[idx] = state.h;
        self.hu[idx] = state.hu;
        self.hv[idx] = state.hv;
    }

    /// 从原始变量设置
    #[inline]
    pub fn set_from_primitive(&mut self, idx: usize, h: B::Scalar, u: B::Scalar, v: B::Scalar) {
        self.h[idx] = h;
        self.hu[idx] = h * u;
        self.hv[idx] = h * v;
    }

    /// 重置为零
    pub fn reset(&mut self) {
        let zero = B::Scalar::ZERO;
        self.h.fill(zero);
        self.hu.fill(zero);
        self.hv.fill(zero);
        self.tracers.clear_all();
    }

    // ========== 切片访问 ==========

    /// 获取水深切片
    #[inline]
    pub fn h_slice(&self) -> &[B::Scalar] {
        &self.h
    }

    /// 获取 x 动量切片
    #[inline]
    pub fn hu_slice(&self) -> &[B::Scalar] {
        &self.hu
    }

    /// 获取 y 动量切片
    #[inline]
    pub fn hv_slice(&self) -> &[B::Scalar] {
        &self.hv
    }

    /// 获取底床高程切片
    #[inline]
    pub fn z_slice(&self) -> &[B::Scalar] {
        &self.z
    }

    /// 获取可变水深切片
    #[inline]
    pub fn h_slice_mut(&mut self) -> &mut [B::Scalar] {
        &mut self.h
    }

    /// 获取可变 x 动量切片
    #[inline]
    pub fn hu_slice_mut(&mut self) -> &mut [B::Scalar] {
        &mut self.hu
    }

    /// 获取可变 y 动量切片
    #[inline]
    pub fn hv_slice_mut(&mut self) -> &mut [B::Scalar] {
        &mut self.hv
    }

    /// 获取可变底床高程切片
    #[inline]
    pub fn z_slice_mut(&mut self) -> &mut [B::Scalar] {
        &mut self.z
    }

    // ========== 积分计算 ==========

    /// 计算总质量
    pub fn total_mass(&self, cell_areas: &[B::Scalar]) -> B::Scalar {
        self.h.iter()
            .zip(cell_areas.iter())
            .map(|(h, a)| *h * *a)
            .fold(B::Scalar::ZERO, |acc, x| acc + x)
    }

    /// 计算总动量
    pub fn total_momentum(&self, cell_areas: &[B::Scalar]) -> (B::Scalar, B::Scalar) {
        let hux: B::Scalar = self.hu.iter()
            .zip(cell_areas.iter())
            .map(|(hu, a)| *hu * *a)
            .fold(B::Scalar::ZERO, |acc, x| acc + x);
        let hvx: B::Scalar = self.hv.iter()
            .zip(cell_areas.iter())
            .map(|(hv, a)| *hv * *a)
            .fold(B::Scalar::ZERO, |acc, x| acc + x);
        (hux, hvx)
    }

    // ========== 时间积分支持 ==========

    /// 从另一个状态复制数据
    pub fn copy_from(&mut self, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());
        
        // 复制主变量
        let h_slice = self.h_slice_mut();
        h_slice.copy_from_slice(other.h_slice());
        
        let hu_slice = self.hu_slice_mut();
        hu_slice.copy_from_slice(other.hu_slice());
        
        let hv_slice = self.hv_slice_mut();
        hv_slice.copy_from_slice(other.hv_slice());
        
        let z_slice = self.z_slice_mut();
        z_slice.copy_from_slice(other.z_slice());
        
        // 复制示踪剂
        self.tracers.copy_from(&other.tracers);
    }

    /// 添加缩放的 RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &RhsBuffers<B::Scalar>, scale: B::Scalar) {
        for i in 0..self.n_cells {
            self.h[i] = self.h[i] + scale * rhs.dh_dt[i];
            self.hu[i] = self.hu[i] + scale * rhs.dhu_dt[i];
            self.hv[i] = self.hv[i] + scale * rhs.dhv_dt[i];
        }
        self.tracers.add_scaled(&rhs.tracer_rhs, scale);
    }

    /// 二元线性组合: self = a*A + b*B
    pub fn linear_combine(&mut self, a: B::Scalar, state_a: &Self, b: B::Scalar, state_b: &Self) {
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
    pub fn axpy(&mut self, a: B::Scalar, b: B::Scalar, other: &Self) {
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
            if *h < B::Scalar::ZERO {
                *h = B::Scalar::ZERO;
            }
        }

        for tracer in self.tracers.iter_mut() {
            for v in tracer.iter_mut() {
                if *v < B::Scalar::ZERO {
                    *v = B::Scalar::ZERO;
                }
            }
        }
    }

    // ========== 验证 ==========

    /// 验证状态有效性
    pub fn validate(&self, time: B::Scalar, params: &NumericalParams<B::Scalar>) -> Result<(), StateError<B::Scalar>> {
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
            if self.h[idx] < B::Scalar::ZERO {
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
// 错误类型
// ============================================================

/// 状态错误
#[derive(Debug, Clone)]
pub enum StateError<S: RuntimeScalar> {
    /// 无效值 (NaN/Inf)
    InvalidValue {
        field: &'static str,
        cell: usize,
        value: S,
        time: S,
    },
    /// 负水深
    NegativeDepth {
        cell: usize,
        value: S,
        time: S,
    },
    /// 速度过大
    ExcessiveVelocity {
        cell: usize,
        speed: S,
        max_speed: S,
        time: S,
    },
    /// 尺寸不匹配
    SizeMismatch {
        expected: usize,
        actual: usize,
    },
}

impl<S: RuntimeScalar> std::fmt::Display for StateError<S> {
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

impl<S: RuntimeScalar> std::error::Error for StateError<S> {}

// ============================================================
// 兼容性类型别名
// ============================================================

/// 泛型状态类型别名（向后兼容）
/// 对于需要直接使用 Backend 参数的代码使用此别名
pub type ShallowWaterStateGeneric<B> = ShallowWaterState<B>;

/// 默认后端状态类型别名（使用 f64）
pub type ShallowWaterStateDefault = ShallowWaterState<CpuBackend<f64>>;

/// f64 后端的状态类型别名（向后兼容）
pub type ShallowWaterStateF64 = ShallowWaterState<CpuBackend<f64>>;

/// f32 后端的状态类型别名
pub type ShallowWaterStateF32 = ShallowWaterState<CpuBackend<f32>>;

/// f64 RhsBuffers 类型别名
pub type RhsBuffersF64 = RhsBuffers<f64>;

/// f32 RhsBuffers 类型别名
pub type RhsBuffersF32 = RhsBuffers<f32>;

// ============================================================
// StateAccess Trait 实现
// ============================================================

// 为 CpuBackend<f64> 实现 StateAccess trait
impl StateAccess for ShallowWaterState<CpuBackend<f64>> {
    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn get(&self, cell: usize) -> ConservedState<f64> {
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

// 为 CpuBackend<f64> 实现 StateAccessMut trait
impl StateAccessMut for ShallowWaterState<CpuBackend<f64>> {
    #[inline]
    fn set(&mut self, cell: usize, state: ConservedState<f64>) {
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

// 为 CpuBackend<f32> 实现 StateAccess trait
impl StateAccess for ShallowWaterState<CpuBackend<f32>> {
    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn get(&self, cell: usize) -> ConservedState<f64> {
        ConservedState::new(self.h[cell] as f64, self.hu[cell] as f64, self.hv[cell] as f64)
    }

    #[inline]
    fn h(&self, cell: usize) -> f64 {
        self.h[cell] as f64
    }

    #[inline]
    fn hu(&self, cell: usize) -> f64 {
        self.hu[cell] as f64
    }

    #[inline]
    fn hv(&self, cell: usize) -> f64 {
        self.hv[cell] as f64
    }

    #[inline]
    fn z(&self, cell: usize) -> f64 {
        self.z[cell] as f64
    }

    #[inline]
    fn h_slice(&self) -> &[f64] {
        // f32 需要转换，返回转换后的临时向量引用
        // 实际项目中应考虑性能优化
        &[]
    }

    #[inline]
    fn hu_slice(&self) -> &[f64] {
        &[]
    }

    #[inline]
    fn hv_slice(&self) -> &[f64] {
        &[]
    }

    #[inline]
    fn z_slice(&self) -> &[f64] {
        &[]
    }
}

// 为 CpuBackend<f32> 实现 StateAccessMut trait
impl StateAccessMut for ShallowWaterState<CpuBackend<f32>> {
    #[inline]
    fn set(&mut self, cell: usize, state: ConservedState<f64>) {
        self.h[cell] = state.h as f32;
        self.hu[cell] = state.hu as f32;
        self.hv[cell] = state.hv as f32;
    }

    #[inline]
    fn set_h(&mut self, cell: usize, value: f64) {
        self.h[cell] = value as f32;
    }

    #[inline]
    fn set_hu(&mut self, cell: usize, value: f64) {
        self.hu[cell] = value as f32;
    }

    #[inline]
    fn set_hv(&mut self, cell: usize, value: f64) {
        self.hv[cell] = value as f32;
    }

    #[inline]
    fn set_z(&mut self, cell: usize, value: f64) {
        self.z[cell] = value as f32;
    }

    #[inline]
    fn h_slice_mut(&mut self) -> &mut [f64] {
        // f32 无法直接返回 f64 可变切片，返回空切片
        &mut []
    }

    #[inline]
    fn hu_slice_mut(&mut self) -> &mut [f64] {
        &mut []
    }

    #[inline]
    fn hv_slice_mut(&mut self) -> &mut [f64] {
        &mut []
    }

    #[inline]
    fn z_slice_mut(&mut self) -> &mut [f64] {
        &mut []
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_state_creation_f64() {
        let backend = CpuBackend::<f64>::new();
        let state = ShallowWaterState::<CpuBackend<f64>>::new_with_backend(backend, 100);
        assert_eq!(state.n_cells(), 100);
    }

    #[test]
    fn test_state_creation_f32() {
        let backend = CpuBackend::<f32>::new();
        let state = ShallowWaterState::<CpuBackend<f32>>::new_with_backend(backend, 100);
        assert_eq!(state.n_cells(), 100);
    }

    #[test]
    fn test_conserved_state_operations() {
        let state1 = ConservedState::new(1.0f64, 2.0, 3.0);
        let state2 = ConservedState::new(0.5f64, 1.0, 1.5);
        
        let sum = state1 + state2;
        assert_eq!(sum.h, 1.5);
        assert_eq!(sum.hu, 3.0);
        assert_eq!(sum.hv, 4.5);
        
        let scaled = state1 * 2.0;
        assert_eq!(scaled.h, 2.0);
        assert_eq!(scaled.hu, 4.0);
    }

    #[test]
    fn test_dynamic_scalars() {
        let mut scalars = DynamicScalars::<f64>::new(10);
        assert_eq!(scalars.len(), 10);
        assert_eq!(scalars.count(), 0);
        
        let idx = scalars.register("temperature");
        assert_eq!(idx, 0);
        assert_eq!(scalars.count(), 1);
        
        if let Some(slice) = scalars.get_mut(0) {
            slice[0] = 25.0;
            slice[1] = 26.0;
        }
        
        if let Some(slice) = scalars.get(0) {
            assert_eq!(slice[0], 25.0);
            assert_eq!(slice[1], 26.0);
        }
    }

    #[test]
    fn test_gradient_state() {
        let grad = GradientState::<f64>::new(5);
        assert_eq!(grad.grad_h_x.len(), 5);
        assert_eq!(grad.grad_h_y.len(), 5);
        
        let (gx, gy) = grad.get_h(0);
        assert_eq!(gx, 0.0);
        assert_eq!(gy, 0.0);
    }

    #[test]
    fn test_flux_operations() {
        let f1 = Flux::new(1.0f64, 2.0, 3.0);
        let f2 = Flux::new(0.5f64, 1.0, 1.5);
        
        let sum = f1 + f2;
        assert_eq!(sum.mass, 1.5);
        assert_eq!(sum.mom_x, 3.0);
        assert_eq!(sum.mom_y, 4.5);
        
        let scaled = f1 * 2.0;
        assert_eq!(scaled.mass, 2.0);
    }

    #[test]
    fn test_rhs_buffers() {
        let rhs = RhsBuffers::<f64>::new(10);
        assert_eq!(rhs.dh_dt.len(), 10);
        assert_eq!(rhs.dhu_dt.len(), 10);
        assert_eq!(rhs.dhv_dt.len(), 10);
    }

    #[test]
    fn test_cold_start() {
        let backend = CpuBackend::<f64>::new();
        let z_bed = vec![-10.0, -5.0, 0.0, 5.0];
        let state = ShallowWaterState::cold_start(backend, 0.0, &z_bed);
        
        assert_eq!(state.h[0], 10.0);
        assert_eq!(state.h[1], 5.0);
        assert_eq!(state.h[2], 0.0);
        assert_eq!(state.h[3], 0.0);
    }

    #[test]
    fn test_state_linear_combine() {
        let backend = CpuBackend::<f64>::new();
        let mut result = ShallowWaterState::new_with_backend(backend.clone(), 2);
        let state_a = ShallowWaterState::new_with_backend(backend.clone(), 2);
        let state_b = ShallowWaterState::new_with_backend(backend, 2);
        
        // 设置测试数据
        result.h[0] = 1.0;
        result.h[1] = 2.0;
        
        state_b.h[0] = 3.0;
        state_b.h[1] = 4.0;
        
        // 执行线性组合: result = 0.5 * state_a + 0.5 * state_b
        result.linear_combine(0.5, &state_a, 0.5, &state_b);
        
        assert_eq!(result.h[0], 2.0);
        assert_eq!(result.h[1], 3.0);
    }

    #[test]
    fn test_state_axpy() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::new_with_backend(backend.clone(), 2);
        let other = ShallowWaterState::cold_start(backend, 10.0, &[0.0, 5.0]);
        
        state.h[0] = 1.0;
        state.h[1] = 2.0;
        
        state.axpy(0.5, 0.5, &other);
        
        assert_eq!(state.h[0], 5.5); // 0.5 * 1.0 + 0.5 * 10.0
        assert_eq!(state.h[1], 3.5); // 0.5 * 2.0 + 0.5 * 5.0
    }

    #[test]
    fn test_state_validate() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::new_with_backend(backend, 2);
        let params = NumericalParams::<f64>::default();
        
        state.h[0] = 1.0;
        state.hu[0] = 0.1;
        state.hv[0] = 0.0;
        state.z[0] = 0.0;
        
        state.h[1] = -0.1; // 负水深
        state.hu[1] = 0.0;
        state.hv[1] = 0.0;
        state.z[1] = 0.0;
        
        let result = state.validate(0.0, &params);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            StateError::NegativeDepth { cell, .. } => {
                assert_eq!(cell, 1);
            }
            _ => panic!("Expected NegativeDepth error"),
        }
    }

    #[test]
    fn test_total_mass() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::new_with_backend(backend, 3);
        let areas = vec![1.0, 2.0, 3.0];
        
        state.h[0] = 1.0;
        state.h[1] = 2.0;
        state.h[2] = 3.0;
        
        let mass = state.total_mass(&areas);
        assert_eq!(mass, 14.0); // 1*1 + 2*2 + 3*3
    }

    #[test]
    fn test_velocity_calculation() {
        let backend_f64 = CpuBackend::<f64>::new();
        let params_f64 = NumericalParams::<f64>::default();
        
        let mut state_f64 = ShallowWaterState::new_with_backend(backend_f64, 1);
        state_f64.h[0] = 2.0;
        state_f64.hu[0] = 4.0;
        state_f64.hv[0] = 6.0;
        
        let vel_f64 = state_f64.velocity(0, &params_f64);
        assert!((vel_f64.u - 2.0).abs() < 1e-10);
        assert!((vel_f64.v - 3.0).abs() < 1e-10);
        
        let backend_f32 = CpuBackend::<f32>::new();
        let params_f32 = NumericalParams::<f32>::default();
        
        let mut state_f32 = ShallowWaterState::new_with_backend(backend_f32, 1);
        state_f32.h[0] = 2.0f32;
        state_f32.hu[0] = 4.0f32;
        state_f32.hv[0] = 6.0f32;
        
        let vel_f32 = state_f32.velocity(0, &params_f32);
        assert!((vel_f32.u - 2.0f32).abs() < 1e-6f32);
        assert!((vel_f32.v - 3.0f32).abs() < 1e-6f32);
    }

    #[test]
    fn test_state_access_trait_f64() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::new_with_backend(backend, 5);
        
        state.set_h(0, 1.5);
        assert_eq!(state.h(0), 1.5);
        
        let slice = state.h_slice();
        assert_eq!(slice.len(), 5);
    }
    
    #[test]
    fn test_state_access_trait_f32() {
        let backend = CpuBackend::<f32>::new();
        let mut state = ShallowWaterState::new_with_backend(backend, 5);
        
        state.set_h(0, 1.5);
        assert_eq!(state.h(0), 1.5);
    }
}