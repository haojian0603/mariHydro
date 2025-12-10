// crates/mh_physics/src/sources/traits_generic.rs
//! 泛型源项 Trait 定义
//!
//! 该模块定义了后端无关的源项计算接口，支持 CPU 和 GPU 后端。
//!
//! # 设计原则
//!
//! 1. **后端无关**: 所有类型都通过 `Backend` trait 参数化
//! 2. **可扩展**: 通过注册机制支持动态添加源项
//! 3. **高性能**: 支持批量计算和 GPU 加速

use crate::core::{Backend, CpuBackend, Scalar};
use crate::state::ShallowWaterStateGeneric;
use std::marker::PhantomData;

/// 源项刚性分类
/// 
/// 用于确定源项的时间积分策略。
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
/// 
/// 表示单个单元的源项贡献，包括质量和动量变化率。
/// 
/// # 类型参数
/// 
/// - `S`: 标量类型（f32 或 f64）
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
    pub fn scale(&self, factor: S) -> Self {
        Self {
            s_h: self.s_h * factor,
            s_hu: self.s_hu * factor,
            s_hv: self.s_hv * factor,
        }
    }
}

/// 泛型源项计算上下文
/// 
/// 包含源项计算所需的时间、参数和物理常数。
/// 
/// # 类型参数
/// 
/// - `S`: 标量类型
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
            gravity: S::GRAVITY,
            h_dry: Scalar::from_f64(1e-6),
            h_wet: Scalar::from_f64(1e-4),
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
/// 
/// 定义后端无关的源项计算接口。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
/// 
/// # 实现说明
/// 
/// 实现者需要提供 `compute_cell` 方法进行逐单元计算。
/// 可以选择性地覆盖 `compute_batch` 方法以提供优化的批量计算（如 GPU 并行）。
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
    /// 
    /// # 参数
    /// 
    /// - `cell`: 单元索引
    /// - `state`: 当前状态
    /// - `ctx`: 源项上下文
    /// 
    /// # 返回
    /// 
    /// 返回该单元的源项贡献
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar>;
    
    /// 批量计算所有单元的源项
    /// 
    /// 默认实现逐单元调用 `compute_cell`。
    /// GPU 后端可以覆盖此方法提供并行实现。
    /// 
    /// # 参数
    /// 
    /// - `state`: 当前状态
    /// - `contributions`: 输出的贡献数组（将被写入）
    /// - `ctx`: 源项上下文
    fn compute_batch(
        &self,
        state: &ShallowWaterStateGeneric<B>,
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
    /// 
    /// # 参数
    /// 
    /// - `state`: 当前状态
    /// - `rhs_h`: 质量右端项
    /// - `rhs_hu`: x动量右端项
    /// - `rhs_hv`: y动量右端项
    /// - `ctx`: 源项上下文
    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    );
}

/// 源项注册中心
/// 
/// 管理多个源项，提供统一的累加接口。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
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
        Self {
            sources: Vec::new(),
            contributions: Vec::new(),
            _marker: PhantomData,
        }
    }
    
    /// 注册新的源项
    pub fn register(&mut self, source: Box<dyn SourceTermGeneric<B>>) {
        self.sources.push(source);
    }
    
    /// 移除所有源项
    pub fn clear(&mut self) {
        self.sources.clear();
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
    
    /// 确保工作缓冲区容量
    fn ensure_capacity(&mut self, n_cells: usize) {
        if self.contributions.len() < n_cells {
            self.contributions.resize(n_cells, SourceContributionGeneric::default());
        }
    }
}

impl<B: Backend> Default for SourceRegistryGeneric<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU f64 后端的源项注册中心特化实现
impl SourceRegistryGeneric<CpuBackend<f64>> {
    /// 累加所有源项到右端项缓冲区
    /// 
    /// # 参数
    /// 
    /// - `state`: 当前状态
    /// - `rhs_h`: 质量右端项
    /// - `rhs_hu`: x动量右端项
    /// - `rhs_hv`: y动量右端项
    /// - `ctx`: 源项上下文
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
            if !source.is_enabled() {
                continue;
            }
            
            // 重置贡献缓冲区
            for c in self.contributions[..n_cells].iter_mut() {
                *c = SourceContributionGeneric::default();
            }
            
            // 计算源项贡献
            source.compute_batch(state, &mut self.contributions[..n_cells], ctx);
            
            // 累加到右端项
            for (i, c) in self.contributions[..n_cells].iter().enumerate() {
                rhs_h[i] += c.s_h;
                rhs_hu[i] += c.s_hu;
                rhs_hv[i] += c.s_hv;
            }
        }
    }
    
    /// 仅累加显式源项
    pub fn accumulate_explicit(
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
            if !source.is_enabled() || source.stiffness() != SourceStiffness::Explicit {
                continue;
            }
            
            for c in self.contributions[..n_cells].iter_mut() {
                *c = SourceContributionGeneric::default();
            }
            
            source.compute_batch(state, &mut self.contributions[..n_cells], ctx);
            
            for (i, c) in self.contributions[..n_cells].iter().enumerate() {
                rhs_h[i] += c.s_h;
                rhs_hu[i] += c.s_hu;
                rhs_hv[i] += c.s_hv;
            }
        }
    }
}
