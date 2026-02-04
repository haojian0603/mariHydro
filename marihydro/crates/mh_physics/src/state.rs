// crates/mh_physics/src/state.rs

//! 浅水方程状态管理
//!
//! 本模块提供浅水方程求解所需的状态管理，基于 [`Backend`] 泛型设计。
//! 支持 f32/f64 精度切换和 GPU 后端扩展，采用 SoA 布局优化缓存性能。
//!
//! # 核心类型
//!
//! - [`ShallowWaterState`]：泛型状态存储
//! - [`ConservedState`]：单元状态
//! - [`ShallowWaterState`]：泛型状态存储
//!
//! # 设计原则
//!
//! 1. **单轨泛型**：所有接口基于 [`RuntimeScalar`]，无 Legacy f64 别名
//! 2. **Backend 无关**：使用 `[S; 2]` 元组表示向量

use crate::fields::{FieldMeta, FieldRegistry};
use crate::traits::{StateAccess, StateAccessMut};
use crate::types::{NumericalParams, SafeVelocity};
use mh_runtime::{Backend, DeviceBuffer};
use num_traits::{Float, Zero};
use mh_runtime::RuntimeScalar;

// ============================================
// 🔥 类型别名仅在 mh_physics::lib.rs 统一导出
// ============================================

/// 单个单元的守恒状态
/// 
/// 包含浅水方程的三个守恒变量，使用泛型支持不同精度。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConservedState<S: RuntimeScalar> {
    /// 水深 [m]
    pub h: S,
    /// x 方向动量 [m²/s]
    pub hu: S,
    /// y 方向动量 [m²/s]  
    pub hv: S,
}

impl<S> ConservedState<S>
where
    S: RuntimeScalar,
{
    /// 创建新的守恒状态
    /// 
    /// 🔥 注意：移除 const 关键字，RuntimeScalar 不保证 const 支持
    #[inline]
    pub fn new(h: S, hu: S, hv: S) -> Self {
        Self { h, hu, hv }
    }

    /// 零状态
    pub fn zero() -> Self {
        Self {
            h: S::ZERO,
            hu: S::ZERO,
            hv: S::ZERO,
        }
    }

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
impl<S> std::ops::Add for ConservedState<S>
where
    S: RuntimeScalar,
{
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

impl<S> std::ops::Sub for ConservedState<S>
where
    S: RuntimeScalar,
{
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

impl<S> std::ops::Mul<S> for ConservedState<S>
where
    S: RuntimeScalar,
{
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

/// 动态标量场集合，按名称管理示踪剂等扩展字段（Backend 感知）
#[derive(Debug, Clone)]
pub struct DynamicScalars<B: Backend> {
    /// 计算后端实例
    backend: B,
    /// 单元数量
    len: usize,
    /// 字段名称列表
    names: Vec<String>,
    /// 数据存储（Backend 缓冲区）
    data: Vec<B::Buffer<B::Scalar>>,
}

impl<B> DynamicScalars<B>
where
    B: Backend,
    B::Scalar: Float + RuntimeScalar,
{
    /// 创建空集合
    pub fn new(backend: B, len: usize) -> Self {
        Self {
            backend,
            len,
            names: Vec::new(),
            data: Vec::new(),
        }
    }

    /// 创建指定数量的匿名示踪剂字段
    pub fn with_count(backend: B, len: usize, count: usize) -> Self {
        let mut scalars = Self::new(backend, len);
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
            self.data[pos].resize(self.len, B::Scalar::ZERO);
            return pos;
        }

        self.names.push(name);
        let mut buf = self.backend.alloc(self.len);
        buf.fill(B::Scalar::ZERO);
        self.data.push(buf);
        self.data.len() - 1
    }

    /// 按索引获取只读缓冲区
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&B::Buffer<B::Scalar>> {
        self.data.get(idx)
    }

    /// 按索引获取可变缓冲区
    #[inline]
    pub fn get_mut(&mut self, idx: usize) -> Option<&mut B::Buffer<B::Scalar>> {
        self.data.get_mut(idx)
    }

    /// 按索引获取只读切片（仅 CPU 可用）
    #[inline]
    pub fn get_slice(&self, idx: usize) -> Option<&[B::Scalar]> {
        self.data.get(idx).and_then(|v| v.try_as_slice())
    }

    /// 按索引获取可变切片（仅 CPU 可用）
    #[inline]
    pub fn get_slice_mut(&mut self, idx: usize) -> Option<&mut [B::Scalar]> {
        self.data.get_mut(idx).and_then(|v| v.try_as_slice_mut())
    }

    /// 按名称获取只读切片（仅 CPU 可用）
    pub fn get_by_name(&self, name: &str) -> Option<&[B::Scalar]> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|i| self.get_slice(i))
    }

    /// 按名称获取可变示踪剂切片（仅 CPU 可用）
    pub fn get_mut_by_name(&mut self, name: &str) -> Option<&mut [B::Scalar]> {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return self.get_slice_mut(pos);
        }
        None
    }

    /// 将所有字段清零
    pub fn clear_all(&mut self) {
        for field in &mut self.data {
            field.fill(B::Scalar::ZERO);
        }
    }

    /// 调整单元长度并保持已有数据
    pub fn resize_len(&mut self, len: usize) {
        self.len = len;
        for field in &mut self.data {
            field.resize(len, B::Scalar::ZERO);
        }
    }

    /// 按另一个集合的布局对齐
    pub fn match_layout(&mut self, other: &Self) {
        if self.len != other.len || self.names != other.names {
            self.len = other.len;
            self.names = other.names.clone();
            self.data = other
                .data
                .iter()
                .map(|_| {
                    let mut buf = self.backend.alloc(other.len);
                    buf.fill(B::Scalar::ZERO);
                    buf
                })
                .collect();
        } else {
            self.resize_len(other.len);
        }
    }

    /// 复制数据并对齐布局
    pub fn copy_from(&mut self, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            self.backend.copy(src, dst);
        }
    }

    /// self += scale * rhs
    pub fn add_scaled(&mut self, rhs: &Self, scale: B::Scalar) {
        self.match_layout(rhs);
        for (dst, src) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.backend.axpy(scale, src, dst);
        }
    }

    /// 设置字段数量
    pub fn set_count(&mut self, count: usize) {
        self.names.truncate(count);
        self.data.truncate(count);
        while self.data.len() < count {
            let idx = self.data.len();
            self.names.push(format!("tracer_{idx}"));
            let mut buf = self.backend.alloc(self.len);
            buf.fill(B::Scalar::ZERO);
            self.data.push(buf);
        }
    }

    /// self = a * A + b * B
    pub fn linear_combine(&mut self, a: B::Scalar, state_a: &Self, b: B::Scalar, state_b: &Self) {
        debug_assert_eq!(state_a.names, state_b.names, "示踪剂字段布局不一致");
        self.match_layout(state_a);
        for ((dst, sa), sb) in self
            .data
            .iter_mut()
            .zip(state_a.data.iter())
            .zip(state_b.data.iter())
        {
            self.backend.copy(sa, dst);
            self.backend.scale(a, dst);
            self.backend.axpy(b, sb, dst);
        }
    }

    /// self = a * self + b * other
    pub fn axpy(&mut self, a: B::Scalar, b: B::Scalar, other: &Self) {
        self.match_layout(other);
        for (dst, src) in self.data.iter_mut().zip(other.data.iter()) {
            self.backend.scale(a, dst);
            self.backend.axpy(b, src, dst);
        }
    }

    /// 迭代所有字段的可变存储
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut B::Buffer<B::Scalar>> {
        self.data.iter_mut()
    }

    /// 迭代所有字段的只读存储
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &B::Buffer<B::Scalar>> {
        self.data.iter()
    }
}

/// 梯度状态 (用于二阶重构)
#[derive(Debug, Clone)]
pub struct GradientState<B: Backend> {
    /// 水深梯度 x 分量
    pub grad_h_x: B::Buffer<B::Scalar>,
    /// 水深梯度 y 分量
    pub grad_h_y: B::Buffer<B::Scalar>,
    /// x 动量梯度 x 分量
    pub grad_hu_x: B::Buffer<B::Scalar>,
    /// x 动量梯度 y 分量
    pub grad_hu_y: B::Buffer<B::Scalar>,
    /// y 动量梯度 x 分量
    pub grad_hv_x: B::Buffer<B::Scalar>,
    /// y 动量梯度 y 分量
    pub grad_hv_y: B::Buffer<B::Scalar>,
}

impl<B: Backend> GradientState<B> {
    /// 创建新的梯度状态
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            grad_h_x: backend.alloc(n_cells),
            grad_h_y: backend.alloc(n_cells),
            grad_hu_x: backend.alloc(n_cells),
            grad_hu_y: backend.alloc(n_cells),
            grad_hv_x: backend.alloc(n_cells),
            grad_hv_y: backend.alloc(n_cells),
        }
    }

    /// 重置为零
    pub fn reset(&mut self) {
        self.grad_h_x.fill(B::Scalar::ZERO);
        self.grad_h_y.fill(B::Scalar::ZERO);
        self.grad_hu_x.fill(B::Scalar::ZERO);
        self.grad_hu_y.fill(B::Scalar::ZERO);
        self.grad_hv_x.fill(B::Scalar::ZERO);
        self.grad_hv_y.fill(B::Scalar::ZERO);
    }

    /// 获取单元梯度向量
    #[inline]
    pub fn get_h(&self, cell: usize) -> (B::Scalar, B::Scalar) {
        (self.grad_h_x[cell], self.grad_h_y[cell])
    }

    /// 设置单元 h 梯度
    #[inline]
    pub fn set_h(&mut self, cell: usize, grad_x: B::Scalar, grad_y: B::Scalar) {
        self.grad_h_x[cell] = grad_x;
        self.grad_h_y[cell] = grad_y;
    }

    /// 获取单元 hu 梯度
    #[inline]
    pub fn get_hu(&self, cell: usize) -> (B::Scalar, B::Scalar) {
        (self.grad_hu_x[cell], self.grad_hu_y[cell])
    }

    /// 设置单元 hu 梯度
    #[inline]
    pub fn set_hu(&mut self, cell: usize, grad_x: B::Scalar, grad_y: B::Scalar) {
        self.grad_hu_x[cell] = grad_x;
        self.grad_hu_y[cell] = grad_y;
    }

    /// 获取单元 hv 梯度
    #[inline]
    pub fn get_hv(&self, cell: usize) -> (B::Scalar, B::Scalar) {
        (self.grad_hv_x[cell], self.grad_hv_y[cell])
    }

    /// 设置单元 hv 梯度
    #[inline]
    pub fn set_hv(&mut self, cell: usize, grad_x: B::Scalar, grad_y: B::Scalar) {
        self.grad_hv_x[cell] = grad_x;
        self.grad_hv_y[cell] = grad_y;
    }
}

/// 数值通量
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Flux<S> {
    /// 质量通量 [m²/s]
    pub mass: S,
    /// x 动量通量 [m³/s²]
    pub mom_x: S,
    /// y 动量通量 [m³/s²]
    pub mom_y: S,
}

impl<S> Flux<S>
where
    S: RuntimeScalar
{
    /// 创建新通量
    #[inline]
    pub const fn new(mass: S, mom_x: S, mom_y: S) -> Self {
        Self { mass, mom_x, mom_y }
    }

    /// 零通量
    pub fn zero() -> Self {
        Self {
            mass: S::ZERO,
            mom_x: S::ZERO,
            mom_y: S::ZERO,
        }
    }

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
impl<S> std::ops::Add for Flux<S>
where
    S: std::ops::Add<Output = S>,
{
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

impl<S> std::ops::Sub for Flux<S>
where
    S: std::ops::Sub<Output = S>,
{
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

impl<S> std::ops::Neg for Flux<S>
where
    S: std::ops::Neg<Output = S>,
{
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

impl<S> std::ops::Mul<S> for Flux<S>
where
    S: RuntimeScalar,
{
    type Output = Self;
    #[inline]
    fn mul(self, rhs: S) -> Self {
        self.scale(rhs)
    }
}

/// 右端项缓冲区 (用于时间积分)
#[derive(Debug, Clone)]
pub struct RhsBuffers<B: Backend> {
    /// 水深变化率 [m/s]
    pub dh_dt: B::Buffer<B::Scalar>,
    /// x 动量变化率 [m²/s²]
    pub dhu_dt: B::Buffer<B::Scalar>,
    /// y 动量变化率 [m²/s²]
    pub dhv_dt: B::Buffer<B::Scalar>,
    /// 标量示踪剂变化率（可选）
    pub tracer_rhs: DynamicScalars<B>,
}

impl<B: Backend> RhsBuffers<B> {
    /// 创建新的 RHS 缓冲区
    pub fn new(backend: B, n_cells: usize) -> Self {
        Self {
            dh_dt: backend.alloc(n_cells),
            dhu_dt: backend.alloc(n_cells),
            dhv_dt: backend.alloc(n_cells),
            tracer_rhs: DynamicScalars::new(backend, n_cells),
        }
    }

    /// 创建带有示踪剂的 RHS 缓冲区
    pub fn with_tracers(backend: B, n_cells: usize, n_tracers: usize) -> Self {
        let mut rhs = Self::new(backend, n_cells);
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
        self.dh_dt.fill(B::Scalar::ZERO);
        self.dhu_dt.fill(B::Scalar::ZERO);
        self.dhv_dt.fill(B::Scalar::ZERO);
        self.tracer_rhs.clear_all();
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize, n_tracers: usize) {
        self.dh_dt.resize(n_cells, B::Scalar::ZERO);
        self.dhu_dt.resize(n_cells, B::Scalar::ZERO);
        self.dhv_dt.resize(n_cells, B::Scalar::ZERO);
        self.tracer_rhs.resize_len(n_cells);
        self.tracer_rhs.set_count(n_tracers);
    }

    /// 将示踪剂布局对齐到给定状态
    pub fn match_tracers(&mut self, layout: &DynamicScalars<B>) {
        self.tracer_rhs.match_layout(layout);
    }

    /// 添加通量贡献
    #[inline]
    pub fn add_flux(&mut self, cell: usize, flux: Flux<B::Scalar>, area_inv: B::Scalar) {
        self.dh_dt[cell] += flux.mass * area_inv;
        self.dhu_dt[cell] += flux.mom_x * area_inv;
        self.dhv_dt[cell] += flux.mom_y * area_inv;
    }

    /// 添加源项贡献
    #[inline]
    pub fn add_source(&mut self, cell: usize, source: ConservedState<B::Scalar>) {
        self.dh_dt[cell] += source.h;
        self.dhu_dt[cell] += source.hu;
        self.dhv_dt[cell] += source.hv;
    }
}

/// 浅水方程守恒状态（SoA 布局，Backend泛型）
/// 
/// 使用 Backend 泛型存储整个网格的状态变量，采用 SoA 布局优化缓存访问。
/// 支持 f32/f64 精度切换和 GPU 后端扩展。
/// 
/// # 类型参数
/// 
/// - `B: Backend`: 计算后端，提供存储和计算能力
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
    pub tracers: DynamicScalars<B>,
    /// 字段注册表（元数据）
    pub field_registry: FieldRegistry,
    /// 后端实例
    backend: B,
}

impl<B: Backend> ShallowWaterState<B> {
    /// 使用后端实例创建新状态
    pub fn new_with_backend(backend: B, n_cells: usize) -> Self {
        let tracers = DynamicScalars::new(backend.clone(), n_cells);
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

    /// 从数据切片创建状态
    /// 
    /// 提供类型安全的方式来构造不可变状态，推荐用于测试和初始化。
    /// 自动验证所有输入切片的长度一致性。
    pub fn from_data(
        backend: B,
        h: impl Into<Vec<B::Scalar>>,
        hu: impl Into<Vec<B::Scalar>>,
        hv: impl Into<Vec<B::Scalar>>,
        z: impl Into<Vec<B::Scalar>>,
    ) -> Result<Self, StateError<B::Scalar>> {
        let h_vec = h.into();
        let n_cells = h_vec.len();
        let hu_vec = hu.into();
        let hv_vec = hv.into();
        let z_vec = z.into();

        if hu_vec.len() != n_cells {
            return Err(StateError::SizeMismatch {
                expected: n_cells,
                actual: hu_vec.len(),
            });
        }
        if hv_vec.len() != n_cells {
            return Err(StateError::SizeMismatch {
                expected: n_cells,
                actual: hv_vec.len(),
            });
        }
        if z_vec.len() != n_cells {
            return Err(StateError::SizeMismatch {
                expected: n_cells,
                actual: z_vec.len(),
            });
        }

        for i in 0..n_cells {
            let h_val = h_vec[i];
            if !h_val.is_finite() {
                return Err(StateError::InvalidValue {
                    field: "h",
                    cell: i,
                    value: h_val,
                    time: B::Scalar::ZERO,
                });
            }
            if h_val < B::Scalar::ZERO {
                return Err(StateError::NegativeDepth {
                    cell: i,
                    value: h_val,
                    time: B::Scalar::ZERO,
                });
            }

            let hu_val = hu_vec[i];
            let hv_val = hv_vec[i];
            let z_val = z_vec[i];

            if !hu_val.is_finite() {
                return Err(StateError::InvalidValue {
                    field: "hu",
                    cell: i,
                    value: hu_val,
                    time: B::Scalar::ZERO,
                });
            }
            if !hv_val.is_finite() {
                return Err(StateError::InvalidValue {
                    field: "hv",
                    cell: i,
                    value: hv_val,
                    time: B::Scalar::ZERO,
                });
            }
            if !z_val.is_finite() {
                return Err(StateError::InvalidValue {
                    field: "z",
                    cell: i,
                    value: z_val,
                    time: B::Scalar::ZERO,
                });
            }
        }

        let mut state = Self::new_with_backend(backend, n_cells);
        state.h.copy_from_slice(&h_vec);
        state.hu.copy_from_slice(&hu_vec);
        state.hv.copy_from_slice(&hv_vec);
        state.z.copy_from_slice(&z_vec);
        Ok(state)
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
            let h = (initial_eta - z).max(B::Scalar::zero());
            state.h[i] = h;
            state.hu[i] = B::Scalar::zero();
            state.hv[i] = B::Scalar::zero();
            state.z[i] = z;
        }

        state
    }

    /// 克隆结构（不复制数据，创建零初始化的状态）
    pub fn clone_structure(&self) -> Self {
        let backend = self.backend.clone();
        let mut tracers = DynamicScalars::new(backend.clone(), self.n_cells);
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
            let _ = self.field_registry
                .register(FieldMeta::cell_scalar(name.clone(), unit.into()).with_desc("示踪剂标量"));
        }
        idx
    }

    /// 获取示踪剂数量
    #[inline]
    pub fn tracer_count(&self) -> usize {
        self.tracers.count()
    }

    /// 同步示踪剂布局（名称与长度）
    ///
    /// 用于时间积分等内部复制前的布局对齐。
    pub fn sync_tracer_layout_from(&mut self, other: &Self) {
        self.tracers.match_layout(&other.tracers);
    }

    /// 获取所有示踪剂名称
    #[inline]
    pub fn tracer_names(&self) -> &[String] {
        self.tracers.names()
    }

    /// 按索引获取示踪剂切片
    #[inline]
    pub fn tracer_slice(&self, idx: usize) -> Option<&[B::Scalar]> {
        self.tracers.get_slice(idx)
    }

    /// 按索引获取可变示踪剂切片
    #[inline]
    pub fn tracer_slice_mut(&mut self, idx: usize) -> Option<&mut [B::Scalar]> {
        self.tracers.get_slice_mut(idx)
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

    /// 获取单元的守恒状态
    #[inline]
    pub fn get(&self, idx: usize) -> ConservedState<B::Scalar> {
        ConservedState::new(self.h[idx], self.hu[idx], self.hv[idx])
    }

    /// 获取原始变量 (h, u, v)
    #[inline]
    pub fn primitive(
        &self,
        idx: usize,
        params: &NumericalParams<B::Scalar>,
    ) -> (B::Scalar, B::Scalar, B::Scalar) {
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
        let zero = B::Scalar::zero();
        self.h.fill(zero);
        self.hu.fill(zero);
        self.hv.fill(zero);
        self.tracers.clear_all();
    }

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

    /// 计算总质量
    pub fn total_mass(&self, cell_areas: &B::Buffer<B::Scalar>) -> B::Scalar {
        let areas = cell_areas.try_as_slice().unwrap_or(&[]);
        self.h
            .iter()
            .zip(areas.iter())
            .map(|(h, a)| *h * *a)
            .fold(B::Scalar::zero(), |acc, x| acc + x)
    }

    /// 计算总动量
    pub fn total_momentum(&self, cell_areas: &B::Buffer<B::Scalar>) -> (B::Scalar, B::Scalar) {
        let areas = cell_areas.try_as_slice().unwrap_or(&[]);
        let hux: B::Scalar = self
            .hu
            .iter()
            .zip(areas.iter())
            .map(|(hu, a)| *hu * *a)
            .fold(B::Scalar::zero(), |acc, x| acc + x);
        let hvx: B::Scalar = self
            .hv
            .iter()
            .zip(areas.iter())
            .map(|(hv, a)| *hv * *a)
            .fold(B::Scalar::zero(), |acc, x| acc + x);
        (hux, hvx)
    }

    /// 从另一个状态复制数据
    ///
    /// 返回错误用于上层统一处理。
    pub fn copy_from(&mut self, other: &Self) -> Result<(), StateError<B::Scalar>> {
        self.try_copy_from(other)
    }

    /// 从另一个状态复制数据（跳过数值校验）
    ///
    /// 仅在需要容忍中间态包含 NaN/Inf 的情况下使用。
    /// 仍会检查尺寸与示踪剂布局一致性。
    pub fn copy_from_unchecked(&mut self, other: &Self) -> Result<(), StateError<B::Scalar>> {
        if self.n_cells() != other.n_cells() {
            return Err(StateError::SizeMismatch {
                expected: self.n_cells(),
                actual: other.n_cells(),
            });
        }

        if self.tracers.names != other.tracers.names {
            self.tracers.match_layout(&other.tracers);
        }

        self.backend.copy(&other.h, &mut self.h);
        self.backend.copy(&other.hu, &mut self.hu);
        self.backend.copy(&other.hv, &mut self.hv);
        self.backend.copy(&other.z, &mut self.z);

        self.tracers.copy_from(&other.tracers);

        Ok(())
    }

    /// 从另一个状态复制数据（带校验）
    pub fn try_copy_from(&mut self, other: &Self) -> Result<(), StateError<B::Scalar>> {
        if self.n_cells() != other.n_cells() {
            return Err(StateError::SizeMismatch {
                expected: self.n_cells(),
                actual: other.n_cells(),
            });
        }

        let time = B::Scalar::ZERO;
        for i in 0..other.n_cells {
            if !other.h[i].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "h",
                    cell: i,
                    value: other.h[i],
                    time,
                });
            }
            if other.h[i] < B::Scalar::ZERO {
                return Err(StateError::NegativeDepth {
                    cell: i,
                    value: other.h[i],
                    time,
                });
            }
            if !other.hu[i].is_finite() || !other.hv[i].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "momentum",
                    cell: i,
                    value: if !other.hu[i].is_finite() {
                        other.hu[i]
                    } else {
                        other.hv[i]
                    },
                    time,
                });
            }
            if !other.z[i].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "z",
                    cell: i,
                    value: other.z[i],
                    time,
                });
            }
        }

        if self.tracers.names != other.tracers.names {
            return Err(StateError::LayoutMismatch {
                field: "tracers",
            });
        }

        // 复制主变量
        self.backend.copy(&other.h, &mut self.h);
        self.backend.copy(&other.hu, &mut self.hu);
        self.backend.copy(&other.hv, &mut self.hv);
        self.backend.copy(&other.z, &mut self.z);

        // 复制示踪剂
        self.tracers.copy_from(&other.tracers);

        Ok(())
    }

    /// 添加缩放的 RHS: self += scale * rhs
    pub fn add_scaled_rhs(&mut self, rhs: &RhsBuffers<B>, scale: B::Scalar) {
        self.backend.axpy(scale, &rhs.dh_dt, &mut self.h);
        self.backend.axpy(scale, &rhs.dhu_dt, &mut self.hu);
        self.backend.axpy(scale, &rhs.dhv_dt, &mut self.hv);
        self.tracers.add_scaled(&rhs.tracer_rhs, scale);
    }

    /// 二元线性组合: self = a*A + b*B
    pub fn linear_combine(&mut self, a: B::Scalar, state_a: &Self, b: B::Scalar, state_b: &Self) {
        debug_assert_eq!(self.n_cells(), state_a.n_cells());
        debug_assert_eq!(self.n_cells(), state_b.n_cells());
        self.backend.copy(&state_a.h, &mut self.h);
        self.backend.scale(a, &mut self.h);
        self.backend.axpy(b, &state_b.h, &mut self.h);

        self.backend.copy(&state_a.hu, &mut self.hu);
        self.backend.scale(a, &mut self.hu);
        self.backend.axpy(b, &state_b.hu, &mut self.hu);

        self.backend.copy(&state_a.hv, &mut self.hv);
        self.backend.scale(a, &mut self.hv);
        self.backend.axpy(b, &state_b.hv, &mut self.hv);
        self.tracers
            .linear_combine(a, &state_a.tracers, b, &state_b.tracers);
    }

    /// 自线性组合: self = a * self + b * other
    pub fn axpy(&mut self, a: B::Scalar, b: B::Scalar, other: &Self) {
        debug_assert_eq!(self.n_cells(), other.n_cells());
        self.backend.scale(a, &mut self.h);
        self.backend.axpy(b, &other.h, &mut self.h);

        self.backend.scale(a, &mut self.hu);
        self.backend.axpy(b, &other.hu, &mut self.hu);

        self.backend.scale(a, &mut self.hv);
        self.backend.axpy(b, &other.hv, &mut self.hv);
        self.tracers.axpy(a, b, &other.tracers);
    }

    /// 强制正性约束
    pub fn enforce_positivity(&mut self) {
        self.backend.enforce_positivity(&mut self.h, B::Scalar::zero());
        for tracer in self.tracers.iter_mut() {
            if let Some(slice) = tracer.try_as_slice_mut() {
                for v in slice.iter_mut() {
                    if *v < B::Scalar::zero() {
                        *v = B::Scalar::zero();
                    }
                }
            }
        }
    }

    /// 验证状态有效性
    pub fn validate(
        &self,
        time: B::Scalar,
        params: &NumericalParams<B::Scalar>,
    ) -> Result<(), StateError<B::Scalar>> {
        if self.tracers.len() != self.n_cells {
            return Err(StateError::SizeMismatch {
                expected: self.n_cells,
                actual: self.tracers.len(),
            });
        }

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

            if !self.z[idx].is_finite() {
                return Err(StateError::InvalidValue {
                    field: "z",
                    cell: idx,
                    value: self.z[idx],
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
            if self.h[idx] < B::Scalar::zero() {
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

        for tracer in self.tracers.iter() {
            if tracer.len() != self.n_cells {
                return Err(StateError::SizeMismatch {
                    expected: self.n_cells,
                    actual: tracer.len(),
                });
            }
            let slice = tracer.try_as_slice().ok_or(StateError::BackendAccess { field: "tracer" })?;
            for (cell_idx, &value) in slice.iter().enumerate() {
                if !value.is_finite() {
                    return Err(StateError::InvalidValue {
                        field: "tracer",
                        cell: cell_idx,
                        value,
                        time,
                    });
                }

                if value < B::Scalar::ZERO {
                    return Err(StateError::InvalidValue {
                        field: "tracer",
                        cell: cell_idx,
                        value,
                        time,
                    });
                }
            }
        }

        Ok(())
    }
}

/// 状态错误类型
#[derive(Debug, Clone)]
pub enum StateError<S> {
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
    /// 布局不匹配
    LayoutMismatch {
        field: &'static str,
    },
    /// 后端缓冲区不可直接访问
    BackendAccess {
        field: &'static str,
    },
}

impl<S> std::fmt::Display for StateError<S>
where
    S: std::fmt::Display,
{
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
            Self::LayoutMismatch { field } => {
                write!(f, "Layout mismatch: {field}")
            }
            Self::BackendAccess { field } => {
                write!(f, "Backend buffer not accessible: {field}")
            }
        }
    }
}

impl<S> std::error::Error for StateError<S> where S: std::fmt::Debug + std::fmt::Display {}

// StateAccess Trait 实现（泛型版本）
impl<B> StateAccess for ShallowWaterState<B>
where
    B: Backend,
{
    type Scalar = B::Scalar;

    #[inline]
    fn n_cells(&self) -> usize {
        self.n_cells
    }

    #[inline]
    fn get(&self, cell: usize) -> ConservedState<Self::Scalar> {
        ConservedState::new(self.h[cell], self.hu[cell], self.hv[cell])
    }

    #[inline]
    fn h(&self, cell: usize) -> Self::Scalar {
        self.h[cell]
    }

    #[inline]
    fn hu(&self, cell: usize) -> Self::Scalar {
        self.hu[cell]
    }

    #[inline]
    fn hv(&self, cell: usize) -> Self::Scalar {
        self.hv[cell]
    }

    #[inline]
    fn z(&self, cell: usize) -> Self::Scalar {
        self.z[cell]
    }

    #[inline]
    fn h_slice(&self) -> &[Self::Scalar] {
        self.h_slice()
    }

    #[inline]
    fn hu_slice(&self) -> &[Self::Scalar] {
        self.hu_slice()
    }

    #[inline]
    fn hv_slice(&self) -> &[Self::Scalar] {
        self.hv_slice()
    }

    #[inline]
    fn z_slice(&self) -> &[Self::Scalar] {
        self.z_slice()
    }
}

// StateAccessMut trait 实现
impl<B> StateAccessMut for ShallowWaterState<B>
where
    B: Backend,
{
    #[inline]
    fn set(&mut self, cell: usize, state: ConservedState<Self::Scalar>) {
        self.set_state(cell, state);
    }

    #[inline]
    fn set_h(&mut self, cell: usize, value: Self::Scalar) {
        self.h[cell] = value;
    }

    #[inline]
    fn set_hu(&mut self, cell: usize, value: Self::Scalar) {
        self.hu[cell] = value;
    }

    #[inline]
    fn set_hv(&mut self, cell: usize, value: Self::Scalar) {
        self.hv[cell] = value;
    }

    #[inline]
    fn set_z(&mut self, cell: usize, value: Self::Scalar) {
        self.z[cell] = value;
    }

    #[inline]
    fn h_slice_mut(&mut self) -> &mut [Self::Scalar] {
        self.h_slice_mut()
    }

    #[inline]
    fn hu_slice_mut(&mut self) -> &mut [Self::Scalar] {
        self.hu_slice_mut()
    }

    #[inline]
    fn hv_slice_mut(&mut self) -> &mut [Self::Scalar] {
        self.hv_slice_mut()
    }

    #[inline]
    fn z_slice_mut(&mut self) -> &mut [Self::Scalar] {
        self.z_slice_mut()
    }

    fn apply_flux_update(
        &mut self,
        dt: Self::Scalar,
        areas: &[Self::Scalar],
        flux_h: &[Self::Scalar],
        flux_hu: &[Self::Scalar],
        flux_hv: &[Self::Scalar],
    ) {
        let n = self.n_cells();
        debug_assert_eq!(areas.len(), n);
        debug_assert_eq!(flux_h.len(), n);
        debug_assert_eq!(flux_hu.len(), n);
        debug_assert_eq!(flux_hv.len(), n);

        for i in 0..n {
            let inv_area = Self::Scalar::ONE / areas[i];
            let h_new = self.h[i] + dt * flux_h[i] * inv_area;
            let hu_new = self.hu[i] + dt * flux_hu[i] * inv_area;
            let hv_new = self.hv[i] + dt * flux_hv[i] * inv_area;
            self.h[i] = h_new;
            self.hu[i] = hu_new;
            self.hv[i] = hv_new;
        }
    }

    fn apply_source_update(
        &mut self,
        dt: Self::Scalar,
        source_h: &[Self::Scalar],
        source_hu: &[Self::Scalar],
        source_hv: &[Self::Scalar],
    ) {
        let n = self.n_cells();
        for i in 0..n {
            self.h[i] = self.h[i] + dt * source_h[i];
            self.hu[i] = self.hu[i] + dt * source_hu[i];
            self.hv[i] = self.hv[i] + dt * source_hv[i];
        }
    }

    fn enforce_non_negative_depth(&mut self, h_min: Self::Scalar) {
        let h = self.h_slice_mut();
        for value in h.iter_mut() {
            if *value < h_min {
                *value = Self::Scalar::zero();
            }
        }
    }

    fn copy_from<S2: StateAccess<Scalar = Self::Scalar>>(
        &mut self,
        other: &S2,
    ) -> Result<(), &'static str> {
        if self.n_cells() != other.n_cells() {
            return Err("单元数量不匹配");
        }
        for i in 0..self.n_cells() {
            self.set_h(i, other.h(i));
            self.set_hu(i, other.hu(i));
            self.set_hv(i, other.hv(i));
            self.set_z(i, other.z(i));
        }
        Ok(())
    }
}

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
        let backend = CpuBackend::<f64>::new();
        let mut scalars = DynamicScalars::new(backend, 10);
        assert_eq!(scalars.len(), 10);
        assert_eq!(scalars.count(), 0);

        let idx = scalars.register("temperature");
        assert_eq!(idx, 0);
        assert_eq!(scalars.count(), 1);

        if let Some(slice) = scalars.get_slice_mut(0) {
            slice[0] = 25.0;
            slice[1] = 26.0;
        }

        if let Some(slice) = scalars.get_slice(0) {
            assert_eq!(slice[0], 25.0);
            assert_eq!(slice[1], 26.0);
        }
    }

    #[test]
    fn test_gradient_state() {
        let backend = CpuBackend::<f64>::new();
        let grad = GradientState::<CpuBackend<f64>>::new(backend, 5);
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
        let backend = CpuBackend::<f64>::new();
        let rhs = RhsBuffers::new(backend, 10);
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
    fn test_state_from_data() {
        let backend = CpuBackend::<f64>::new();
        let state = ShallowWaterState::from_data(
            backend,
            vec![1.0, 2.0, 3.0],
            vec![0.1, 0.2, 0.3],
            vec![0.0, 0.0, 0.0],
            vec![-1.0, -2.0, -3.0],
        ).unwrap();

        assert_eq!(state.n_cells(), 3);
        assert_eq!(state.h[0], 1.0);
        assert_eq!(state.h[2], 3.0);
        assert_eq!(state.hu[1], 0.2);
        assert_eq!(state.z[0], -1.0);
    }

    #[test]
    fn test_state_linear_combine() {
        let backend = CpuBackend::<f64>::new();
        
        // 使用 from_data 构造不可变输入状态，确保类型安全
        let state_a = ShallowWaterState::from_data(
            backend.clone(),
            vec![1.0, 2.0],  
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ).unwrap();
        let state_b = ShallowWaterState::from_data(
            backend.clone(),
            vec![3.0, 4.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ).unwrap();
        let mut result = ShallowWaterState::new_with_backend(backend, 2);

        // 执行线性组合: result = 0.5 * state_a + 0.5 * state_b
        result.linear_combine(0.5, &state_a, 0.5, &state_b);

        // 验证结果: (1.0+3.0)/2 = 2.0, (2.0+4.0)/2 = 3.0
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

        state.set_h(0, 1.0);
        state.set_hu(0, 0.1);
        state.set_hv(0, 0.0);
        state.set_z(0, 0.0);

        state.set_h(1, -0.1); // 负水深
        state.set_hu(1, 0.0);
        state.set_hv(1, 0.0);
        state.set_z(1, 0.0);

        let result = state.validate(0.0, &params);
        assert!(result.is_err());

        assert!(matches!(
            result.unwrap_err(),
            StateError::NegativeDepth { cell: 1, .. }
        ));
    }

    #[test]
    fn test_total_mass() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::new_with_backend(backend, 3);
        let areas = vec![1.0, 2.0, 3.0];

        state.set_h(0, 1.0);
        state.set_h(1, 2.0);
        state.set_h(2, 3.0);

        let mass = state.total_mass(&areas);
        assert_eq!(mass, 14.0); // 1*1 + 2*2 + 3*3
    }

    #[test]
    fn test_state_access_trait() {
        let backend = CpuBackend::<f64>::new();
        let mut state = ShallowWaterState::new_with_backend(backend, 5);

        state.set_h(0, 1.5);
        assert_eq!(state.h(0), 1.5);
        assert_eq!(state.n_cells(), 5);

        let slice = state.h_slice();
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[0], 1.5);
    }
}