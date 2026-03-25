// crates/mh_physics/src/sources/traits.rs

//! 源项 trait 定义。
//!
//! 该模块定义主链源项接口、源项上下文、源项贡献结构和测试辅助工具。

use crate::core::Backend;
use crate::state::ShallowWaterState;
use mh_runtime::RuntimeScalar as Scalar;
use std::marker::PhantomData;

// =============================================================================

/// 源项刚性分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceStiffness {
    /// 显式处理：源项较为平缓，可以显式积分
    Explicit,
    /// 局部隐式：源项可能较刚性（如摩擦），需要局部隐式处理
    /// 使用 1/(1 + dt*γ) 形式的隐式因子
    LocallyImplicit,
    /// 全隐式：需要在全局隐式求解器中处理
    FullyImplicit,
}

/// 泛型源项贡献
#[derive(Debug, Clone, Copy)]
pub struct SourceContributionGeneric<S: Scalar> {
    /// 质量源 [m/s]
    pub s_h: S,
    /// x 方向动量源 [m²/s²]
    pub s_hu: S,
    /// y 方向动量源 [m²/s²]
    pub s_hv: S,
}

impl<S: Scalar> Default for SourceContributionGeneric<S> {
    fn default() -> Self {
        Self {
            s_h: S::ZERO,
            s_hu: S::ZERO,
            s_hv: S::ZERO,
        }
    }
}

impl<S: Scalar> SourceContributionGeneric<S> {
    /// 零贡献
    #[inline]
    pub fn zero() -> Self {
        Self {
            s_h: S::ZERO,
            s_hu: S::ZERO,
            s_hv: S::ZERO,
        }
    }

    /// 创建新的源项贡献
    #[inline]
    pub fn new(s_h: S, s_hu: S, s_hv: S) -> Self {
        Self { s_h, s_hu, s_hv }
    }

    /// 创建仅动量贡献
    #[inline]
    pub fn momentum(s_hu: S, s_hv: S) -> Self {
        Self {
            s_h: S::ZERO,
            s_hu,
            s_hv,
        }
    }

    /// 创建仅质量贡献
    #[inline]
    pub fn mass(s_h: S) -> Self {
        Self {
            s_h,
            s_hu: S::ZERO,
            s_hv: S::ZERO,
        }
    }

    /// 原地加法
    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        self.s_h += other.s_h;
        self.s_hu += other.s_hu;
        self.s_hv += other.s_hv;
    }
}

/// 泛型源项计算上下文
#[derive(Debug, Clone)]
pub struct SourceContextGeneric<S: Scalar> {
    /// 当前模拟时间 [s]
    pub time: f64, // ALLOW_F64: 时间参数与模拟进度配合
    /// 时间步长 [s]
    pub dt: S,
    /// 重力加速度 [m/s²]
    pub gravity: S,
    /// 干单元阈值 [m]
    pub h_dry: S,
    /// 湿单元阈值 [m]
    pub h_wet: S,
}

impl<S: Scalar> SourceContextGeneric<S> {
    /// 创建新的源项上下文
    // ALLOW_F64: 时间参数与模拟进度配合
    pub fn new(time: f64, dt: S, gravity: S, h_dry: S, h_wet: S) -> Self {
        Self {
            time,
            dt,
            gravity,
            h_dry,
            h_wet,
        }
    }

    /// 使用默认物理参数创建
    // ALLOW_F64: 时间参数与模拟进度配合
    pub fn with_defaults<B: Backend<Scalar = S>>(backend: &B, time: f64, dt: S) -> Self {
        Self {
            time,
            dt,
            gravity: backend.config_scalar(9.81, "SourceContextGeneric.gravity"),
            h_dry: backend.config_scalar(1e-6, "SourceContextGeneric.h_dry"),
            h_wet: backend.config_scalar(1e-4, "SourceContextGeneric.h_wet"),
        }
    }

    /// 检查水深是否为干
    #[inline]
    pub fn is_dry(&self, h: S) -> bool {
        h < self.h_dry
    }

    /// 检查水深是否为湿
    #[inline]
    pub fn is_wet(&self, h: S) -> bool {
        h >= self.h_wet
    }
}

/// 泛型源项 Trait
pub trait SourceTermGeneric<B: Backend>: Send + Sync {
    /// 获取源项名称
    fn name(&self) -> &'static str;

    /// 获取源项刚性分类
    fn stiffness(&self) -> SourceStiffness;

    /// 源项是否启用
    fn is_enabled(&self) -> bool {
        true
    }

    /// 计算单个单元的源项贡献
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar>;

    /// 批量计算所有单元的源项
    fn compute_batch(
        &self,
        state: &ShallowWaterState<B>,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.is_enabled() {
            return;
        }
        for cell in 0..state.n_cells() {
            contributions[cell] = self.compute_cell(cell, state, ctx);
        }
    }

    /// 累加源项到右端项缓冲区
    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
}

/// 空源项（静态分发占位）
#[derive(Debug, Clone, Copy, Default)]
pub struct NoSource<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> NoSource<B> {
    #[inline]
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B: Backend> SourceTermGeneric<B> for NoSource<B> {
    fn name(&self) -> &'static str {
        "NoSource"
    }

    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::Explicit
    }

    fn is_enabled(&self) -> bool {
        false
    }

    fn compute_cell(
        &self,
        _cell: usize,
        _state: &ShallowWaterState<B>,
        _ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        SourceContributionGeneric::zero()
    }

    fn compute_batch(
        &self,
        state: &ShallowWaterState<B>,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        _ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        for cell in 0..state.n_cells() {
            contributions[cell] = SourceContributionGeneric::zero();
        }
    }

    fn accumulate(
        &self,
        _state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        _rhs_hu: &mut B::Buffer<B::Scalar>,
        _rhs_hv: &mut B::Buffer<B::Scalar>,
        _ctx: &SourceContextGeneric<B::Scalar>,
    ) {
    }
}

/// 源项注册中心（静态分发）
pub struct SourceRegistryGeneric<B: Backend, S: SourceTermGeneric<B>> {
    /// 注册的源项列表
    sources: Vec<S>,
    /// 后端标记
    _marker: PhantomData<B>,
}

impl<B: Backend, S: SourceTermGeneric<B>> SourceRegistryGeneric<B, S> {
    /// 创建空的注册中心
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// 注册新的源项
    pub fn register(&mut self, source: S) {
        self.sources.push(source);
    }

    /// 获取已注册的源项数量
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// 检查是否为空
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// 获取所有源项的名称
    pub fn names(&self) -> Vec<&'static str> {
        self.sources.iter().map(|s| s.name()).collect()
    }

    /// 累加所有源项到右端项缓冲区
    pub fn accumulate_all(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        for source in &self.sources {
            if !source.is_enabled() {
                continue;
            }
            source.accumulate(state, rhs_h, rhs_hu, rhs_hv, ctx);
        }
    }
}

impl<B: Backend, S: SourceTermGeneric<B>> Default for SourceRegistryGeneric<B, S> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::{SourceContextGeneric, SourceStiffness, SourceTermGeneric};
    use mh_runtime::CpuBackend;

    pub(crate) type TestBackend = CpuBackend<f64>;

    pub(crate) fn test_backend() -> TestBackend {
        CpuBackend::<f64>::new()
    }

    pub(crate) fn test_context(time: f64, dt: f64) -> SourceContextGeneric<f64> {
        let backend = test_backend();
        SourceContextGeneric::with_defaults(&backend, time, dt)
    }

    pub(crate) fn assert_source_metadata<T>(
        source: &T,
        expected_name: &'static str,
        expected_stiffness: SourceStiffness,
    ) where
        T: SourceTermGeneric<TestBackend>,
    {
        assert_eq!(source.name(), expected_name);
        assert_eq!(source.stiffness(), expected_stiffness);
    }

    pub(crate) fn assert_source_enabled<T>(source: &T, expected_enabled: bool)
    where
        T: SourceTermGeneric<TestBackend>,
    {
        assert_eq!(source.is_enabled(), expected_enabled);
    }
}
