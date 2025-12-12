// crates/mh_physics/src/sources/traits.rs

//! 源项 Trait 定义
//!
//! 定义源项的核心接口和数据结构。

use crate::core::{Backend, CpuBackend};
use mh_core::Scalar;
use crate::state::{ShallowWaterState, ShallowWaterStateGeneric};
use crate::types::NumericalParams;
use std::marker::PhantomData;

/// 源项贡献
///
/// 表示单个单元的源项贡献，包括质量和动量变化率。
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
    /// 零贡献常量
    pub const ZERO: Self = Self {
        s_h: 0.0,
        s_hu: 0.0,
        s_hv: 0.0,
    };

    /// 创建新的源项贡献
    #[inline]
    pub fn new(s_h: f64, s_hu: f64, s_hv: f64) -> Self {
        Self { s_h, s_hu, s_hv }
    }

    /// 创建仅动量贡献
    #[inline]
    pub fn momentum(s_hu: f64, s_hv: f64) -> Self {
        Self { s_h: 0.0, s_hu, s_hv }
    }

    /// 创建仅质量贡献
    #[inline]
    pub fn mass(s_h: f64) -> Self {
        Self { s_h, s_hu: 0.0, s_hv: 0.0 }
    }

    /// 加法
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            s_h: self.s_h + other.s_h,
            s_hu: self.s_hu + other.s_hu,
            s_hv: self.s_hv + other.s_hv,
        }
    }

    /// 原地加法
    #[inline]
    pub fn add_assign(&mut self, other: &Self) {
        self.s_h += other.s_h;
        self.s_hu += other.s_hu;
        self.s_hv += other.s_hv;
    }

    /// 缩放
    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            s_h: self.s_h * factor,
            s_hu: self.s_hu * factor,
            s_hv: self.s_hv * factor,
        }
    }

    /// 检查是否有效（所有分量都是有限数）
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.s_h.is_finite() && self.s_hu.is_finite() && self.s_hv.is_finite()
    }

    /// 钳位到安全范围
    #[inline]
    pub fn clamp(&self, max_abs: f64) -> Self {
        Self {
            s_h: self.s_h.clamp(-max_abs, max_abs),
            s_hu: self.s_hu.clamp(-max_abs, max_abs),
            s_hv: self.s_hv.clamp(-max_abs, max_abs),
        }
    }
}

impl std::ops::Add for SourceContribution {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            s_h: self.s_h + rhs.s_h,
            s_hu: self.s_hu + rhs.s_hu,
            s_hv: self.s_hv + rhs.s_hv,
        }
    }
}

impl std::ops::AddAssign for SourceContribution {
    fn add_assign(&mut self, rhs: Self) {
        self.s_h += rhs.s_h;
        self.s_hu += rhs.s_hu;
        self.s_hv += rhs.s_hv;
    }
}

impl std::ops::Mul<f64> for SourceContribution {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        self.scale(rhs)
    }
}

/// 源项计算上下文
///
/// 包含源项计算所需的时间和参数信息。
#[derive(Debug, Clone)]
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

    /// 检查单元是否干燥
    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.params.h_dry
    }

    /// 检查单元是否湿润
    #[inline]
    pub fn is_wet(&self, h: f64) -> bool {
        h >= self.params.h_wet
    }
}

/// 源项 Trait
///
/// 定义源项计算的统一接口。
pub trait SourceTerm: Send + Sync {
    /// 获取源项名称
    fn name(&self) -> &'static str;

    /// 是否启用
    fn is_enabled(&self) -> bool;

    /// 计算单个单元的源项贡献
    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution;

    /// 批量计算所有单元的源项
    ///
    /// 默认实现逐单元调用 `compute_cell`。
    /// 子类可以覆盖以提供优化的批量计算。
    fn compute_all(
        &self,
        state: &ShallowWaterState,
        ctx: &SourceContext,
        output_h: &mut [f64],
        output_hu: &mut [f64],
        output_hv: &mut [f64],
    ) {
        if !self.is_enabled() {
            return;
        }

        let n_cells = state.h.len();
        for i in 0..n_cells {
            let contrib = self.compute_cell(state, i, ctx);
            output_h[i] += contrib.s_h;
            output_hu[i] += contrib.s_hu;
            output_hv[i] += contrib.s_hv;
        }
    }

    /// 源项是否显式（需要CFL限制）
    fn is_explicit(&self) -> bool {
        true
    }

    /// 源项是否使用局部隐式处理
    /// 
    /// 局部隐式意味着源项内部处理刚性（如摩擦的 1/(1+dt*γ)），
    /// 而非需要全局隐式求解器。
    fn is_locally_implicit(&self) -> bool {
        false
    }

    /// 是否需要隐式处理（已废弃，使用 is_locally_implicit）
    #[deprecated(since = "0.5.0", note = "use is_locally_implicit() instead")]
    fn requires_implicit_treatment(&self) -> bool {
        self.is_locally_implicit()
    }

    // ========== 半隐式分裂方法 ==========

    /// 计算预测步源项贡献
    ///
    /// 用于半隐式方法的预测阶段。默认返回完整源项。
    fn compute_prediction(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        ctx: &SourceContext,
    ) -> SourceContribution {
        self.compute_cell(state, cell, ctx)
    }

    /// 计算校正步源项贡献
    ///
    /// 用于半隐式方法的校正阶段。默认返回零贡献。
    fn compute_correction(
        &self,
        _state: &ShallowWaterState,
        _cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        SourceContribution::ZERO
    }

    /// 校正步是否需要此源项
    fn requires_correction(&self) -> bool {
        false
    }

    /// 获取隐式因子
    ///
    /// 返回 0.0 表示完全显式，1.0 表示完全隐式。
    /// 用于时间步长控制和稳定性分析。
    fn implicit_factor(&self) -> f64 {
        if self.is_locally_implicit() {
            1.0
        } else {
            0.0
        }
    }

    /// 获取稳定性限制时间步长
    ///
    /// 返回 None 表示无限制，Some(dt) 表示最大允许时间步长。
    fn stability_limit(&self, _state: &ShallowWaterState, _ctx: &SourceContext) -> Option<f64> {
        None
    }
}

/// 源项辅助函数
pub struct SourceHelpers;

impl SourceHelpers {
    /// 安全累加（忽略无效值）
    #[inline]
    pub fn safe_accumulate(acc: &mut f64, val: f64) {
        if val.is_finite() {
            *acc += val;
        }
    }

    /// 验证贡献值并钳位
    #[inline]
    pub fn validate_contribution(val: f64, max_abs: f64) -> f64 {
        if !val.is_finite() {
            return 0.0;
        }
        val.clamp(-max_abs, max_abs)
    }

    /// 光滑过渡函数 (干湿过渡)
    ///
    /// 返回 0.0 (完全干) 到 1.0 (完全湿) 之间的值
    #[inline]
    pub fn smooth_transition(h: f64, h_dry: f64, h_wet: f64) -> f64 {
        if h <= h_dry {
            0.0
        } else if h >= h_wet {
            1.0
        } else {
            (h - h_dry) / (h_wet - h_dry)
        }
    }

    /// 计算安全速度（避免除以零）
    #[inline]
    pub fn safe_velocity(hu: f64, hv: f64, h: f64, h_min: f64) -> (f64, f64) {
        let h_safe = h.max(h_min);
        (hu / h_safe, hv / h_safe)
    }
}

// =============================================================================
// 泛型版本（推荐使用）
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
        Self { s_h: S::ZERO, s_hu: S::ZERO, s_hv: S::ZERO }
    }
    
    /// 创建新的源项贡献
    #[inline]
    pub fn new(s_h: S, s_hu: S, s_hv: S) -> Self {
        Self { s_h, s_hu, s_hv }
    }
    
    /// 创建仅动量贡献
    #[inline]
    pub fn momentum(s_hu: S, s_hv: S) -> Self {
        Self { s_h: S::ZERO, s_hu, s_hv }
    }
    
    /// 创建仅质量贡献
    #[inline]
    pub fn mass(s_h: S) -> Self {
        Self { s_h, s_hu: S::ZERO, s_hv: S::ZERO }
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
    pub time: f64,
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
    pub fn new(time: f64, dt: S, gravity: S, h_dry: S, h_wet: S) -> Self {
        Self { time, dt, gravity, h_dry, h_wet }
    }
    
    /// 使用默认物理参数创建
    pub fn with_defaults(time: f64, dt: S) -> Self {
        Self {
            time,
            dt,
            gravity: <S as Scalar>::from_f64_lossless(9.81),
            h_dry: <S as Scalar>::from_f64_lossless(1e-6),
            h_wet: <S as Scalar>::from_f64_lossless(1e-4),
        }
    }
    
    /// 检查水深是否为干
    #[inline]
    pub fn is_dry(&self, h: S) -> bool { h < self.h_dry }
    
    /// 检查水深是否为湿
    #[inline]
    pub fn is_wet(&self, h: S) -> bool { h >= self.h_wet }
}

/// 泛型源项 Trait
pub trait SourceTermGeneric<B: Backend>: Send + Sync {
    /// 获取源项名称
    fn name(&self) -> &'static str;
    
    /// 获取源项刚性分类
    fn stiffness(&self) -> SourceStiffness;
    
    /// 源项是否启用
    fn is_enabled(&self) -> bool { true }
    
    /// 计算单个单元的源项贡献
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar>;
    
    /// 批量计算所有单元的源项
    fn compute_batch(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        contributions: &mut [SourceContributionGeneric<B::Scalar>],
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.is_enabled() { return; }
        for cell in 0..state.n_cells() {
            contributions[cell] = self.compute_cell(cell, state, ctx);
        }
    }
    
    /// 累加源项到右端项缓冲区
    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
}

/// 源项注册中心（兼容旧版）
pub struct SourceRegistryGeneric<B: Backend> {
    /// 注册的源项列表
    sources: Vec<Box<dyn SourceTermGeneric<B>>>,
    /// 工作缓冲区（用于批量计算）
    contributions: Vec<SourceContributionGeneric<B::Scalar>>,
    /// 后端标记
    _marker: PhantomData<B>,
}

impl<B: Backend> SourceRegistryGeneric<B> {
    /// 创建空的注册中心
    pub fn new() -> Self {
        Self { sources: Vec::new(), contributions: Vec::new(), _marker: PhantomData }
    }
    
    /// 注册新的源项
    pub fn register(&mut self, source: Box<dyn SourceTermGeneric<B>>) { self.sources.push(source); }
    
    /// 获取已注册的源项数量
    pub fn len(&self) -> usize { self.sources.len() }
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool { self.sources.is_empty() }
    
    /// 获取所有源项的名称
    pub fn names(&self) -> Vec<&'static str> { self.sources.iter().map(|s| s.name()).collect() }
    
    /// 确保工作缓冲区容量
    fn ensure_capacity(&mut self, n_cells: usize) {
        if self.contributions.len() < n_cells {
            self.contributions.resize(n_cells, SourceContributionGeneric::default());
        }
    }
}

impl<B: Backend> Default for SourceRegistryGeneric<B> {
    fn default() -> Self { Self::new() }
}

/// CPU f64 后端的源项注册中心特化实现
impl SourceRegistryGeneric<CpuBackend<f64>> {
    /// 累加所有源项到右端项缓冲区
    pub fn accumulate_all(
        &mut self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        let n_cells = state.n_cells();
        self.ensure_capacity(n_cells);
        for source in &self.sources {
            if !source.is_enabled() { continue; }
            for c in self.contributions[..n_cells].iter_mut() { *c = SourceContributionGeneric::default(); }
            source.compute_batch(state, &mut self.contributions[..n_cells], ctx);
            for (i, c) in self.contributions[..n_cells].iter().enumerate() {
                rhs_h[i] += c.s_h;
                rhs_hu[i] += c.s_hu;
                rhs_hv[i] += c.s_hv;
            }
        }
    }
}

/// 向后兼容别名
#[deprecated(since = "0.4.0", note = "Use SourceTermGeneric<CpuBackend<f64>> instead")]
pub type SourceTermF64 = dyn SourceTermGeneric<CpuBackend<f64>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_contribution_zero() {
        let c = SourceContribution::ZERO;
        assert_eq!(c.s_h, 0.0);
        assert_eq!(c.s_hu, 0.0);
        assert_eq!(c.s_hv, 0.0);
    }

    #[test]
    fn test_source_contribution_add() {
        let c1 = SourceContribution::new(1.0, 2.0, 3.0);
        let c2 = SourceContribution::new(0.5, 1.0, 1.5);
        let c3 = c1.add(&c2);
        assert_eq!(c3.s_h, 1.5);
        assert_eq!(c3.s_hu, 3.0);
        assert_eq!(c3.s_hv, 4.5);
    }

    #[test]
    fn test_source_contribution_scale() {
        let c = SourceContribution::new(1.0, 2.0, 3.0);
        let scaled = c.scale(2.0);
        assert_eq!(scaled.s_h, 2.0);
        assert_eq!(scaled.s_hu, 4.0);
        assert_eq!(scaled.s_hv, 6.0);
    }

    #[test]
    fn test_source_contribution_operators() {
        let c1 = SourceContribution::new(1.0, 2.0, 3.0);
        let c2 = SourceContribution::new(0.5, 1.0, 1.5);
        
        let c3 = c1 + c2;
        assert_eq!(c3.s_h, 1.5);
        
        let c4 = c1 * 2.0;
        assert_eq!(c4.s_hu, 4.0);
    }

    #[test]
    fn test_source_contribution_validity() {
        let valid = SourceContribution::new(1.0, 2.0, 3.0);
        assert!(valid.is_valid());

        let invalid = SourceContribution::new(f64::NAN, 2.0, 3.0);
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_source_contribution_clamp() {
        let c = SourceContribution::new(100.0, -200.0, 50.0);
        let clamped = c.clamp(75.0);
        assert_eq!(clamped.s_h, 75.0);
        assert_eq!(clamped.s_hu, -75.0);
        assert_eq!(clamped.s_hv, 50.0);
    }

    #[test]
    fn test_smooth_transition() {
        assert_eq!(SourceHelpers::smooth_transition(0.0, 0.01, 0.1), 0.0);
        assert_eq!(SourceHelpers::smooth_transition(0.1, 0.01, 0.1), 1.0);
        
        let mid = SourceHelpers::smooth_transition(0.055, 0.01, 0.1);
        assert!((mid - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_source_context() {
        let params = NumericalParams::default();
        let ctx = SourceContext::new(10.0, 0.1, &params);
        
        assert_eq!(ctx.time, 10.0);
        assert_eq!(ctx.dt, 0.1);
        // 默认 h_dry = 1e-6，所以 1e-7 是干的，1e-5 不是
        assert!(ctx.is_dry(1e-7));
        assert!(!ctx.is_dry(1e-5));
        assert!(ctx.is_wet(0.1));
    }
}
