// crates/mh_physics/src/sources/friction.rs

//! 摩擦源项
//!
//! 实现 Manning 和 Chezy 摩擦模型，采用隐式半解析方法处理。
//!
//! # 算法
//!
//! Manning 摩擦:
//! ```text
//! S_f = g n² |V| / h^(4/3)
//! decay = 1 / (1 + dt * c_f * |V|)
//! ```
//!
//! Chezy 摩擦:
//! ```text
//! S_f = g |V| / C²
//! decay = 1 / (1 + dt * c_f * |V|)
//! ```
//!
//! 使用隐式处理避免大摩擦系数时的数值不稳定。
use super::traits::{

    SourceContributionGeneric,

    SourceContextGeneric,
    SourceStiffness,

    SourceTermGeneric,
};
use crate::prelude::*;
use crate::state::ShallowWaterState;

/// Manning 摩擦配置
#[derive(Debug, Clone)]
pub struct ManningFrictionConfig {
    /// 是否启用
    pub enabled: bool,
    /// 重力加速度 [m/s²]
    pub g: f64, // ALLOW_F64: Layer 4 配置参数
    /// 预计算 g * n² (均匀场时)
    precomputed_gn2: Option<f64>, // ALLOW_F64: Layer 4 配置参数
    /// Manning 系数场 [s/m^(1/3)]
    pub manning_n: Vec<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 摩擦计算的最小水深
    pub h_friction_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl ManningFrictionConfig {
    /// 创建均匀 Manning 系数配置
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> Self {
        let gn2 = g * default_n * default_n;
        Self {
            enabled: true,
            g,
            precomputed_gn2: Some(gn2),
            manning_n: vec![default_n; n_cells],
            h_friction_min: 1e-4,
        }
    }

    /// 创建空间变化 Manning 系数配置
    pub fn with_field(g: f64, manning_n: Vec<f64>) -> Self {
        Self {
            enabled: true,
            g,
            precomputed_gn2: None,
            manning_n,
            h_friction_min: 1e-4,
        }
    }

    /// 创建默认配置 (g=9.81, n=0.025)
    pub fn default_config(n_cells: usize) -> Self {
        Self::new(9.81, n_cells, 0.025)
    }

    /// 设置最小摩擦水深
    pub fn with_min_depth(mut self, h_min: f64) -> Self {
        self.h_friction_min = h_min;
        self
    }

    /// 计算摩擦系数 c_f = g n² / h^(1/3)
    #[inline]
    fn compute_cf(&self, h: f64, cell: usize) -> f64 {
        let h_safe = h.max(self.h_friction_min);
        if let Some(gn2) = self.precomputed_gn2 {
            gn2 / h_safe.cbrt()
        } else {
            let n = self.manning_n.get(cell).copied().unwrap_or(0.025);
            self.g * n * n / h_safe.cbrt()
        }
    }
}

impl<B> SourceTermGeneric<B> for ManningFrictionConfig
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str { "ManningFriction" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::LocallyImplicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let backend = state.backend();
        let h = state.h[cell];
        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;

        if !dt.is_finite() || dt <= B::Scalar::ZERO || !h.is_finite() {
            return SourceContributionGeneric::default();
        }

        if ctx.is_dry(h) {
            return SourceContributionGeneric::momentum(-hu / dt, -hv / dt);
        }

        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < backend.config_scalar(1e-20, "ManningFrictionConfig.speed_sq_floor") {
            return SourceContributionGeneric::default();
        }

        let cf = backend.config_scalar(
            self.compute_cf(h.to_f64_lossy(), cell),
            "ManningFrictionConfig.cf",
        );
        let speed = speed_sq.sqrt();
        let decay = B::Scalar::ONE / (B::Scalar::ONE + dt * cf * speed);
        let factor = (decay - B::Scalar::ONE) / dt;

        SourceContributionGeneric::momentum(hu * factor, hv * factor)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.enabled {
            return;
        }

        let dt = ctx.dt;
        if !dt.is_finite() || dt <= B::Scalar::ZERO {
            return;
        }

        let n_cells = state.h.len();
        if rhs_hu.len() < n_cells {
            rhs_hu.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n_cells {
            rhs_hv.resize(n_cells, B::Scalar::ZERO);
        }

        for i in 0..n_cells {
            let contrib = self.compute_cell(i, state, ctx);
            rhs_hu[i] += contrib.s_hu;
            rhs_hv[i] += contrib.s_hv;
        }
    }
}
pub struct ChezyFrictionConfig {
    /// 是否启用
    pub enabled: bool,
    /// 重力加速度 [m/s²]
    pub g: f64,
    /// Chezy 系数 [m^(1/2)/s]
    pub chezy_c: f64,
    /// 预计算 cf = g / C²
    cf: f64,
}

impl ChezyFrictionConfig {
    /// 创建新的 Chezy 摩擦配置
    
    pub fn new(g: f64, chezy_c: f64) -> Self {
        let cf = g / (chezy_c * chezy_c);
        Self {
            enabled: true,
            g,
            chezy_c,
            cf,
        }
    }

    /// 创建默认配置 (g=9.81, C=50)
    pub fn default_config() -> Self {
        Self::new(9.81, 50.0)
    }
}

impl<B> SourceTermGeneric<B> for ChezyFrictionConfig
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str { "ChezyFriction" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::LocallyImplicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let backend = state.backend();
        let h = state.h[cell];
        let hu = state.hu[cell];
        let hv = state.hv[cell];
        let dt = ctx.dt;

        if !dt.is_finite() || dt <= B::Scalar::ZERO || !h.is_finite() {
            return SourceContributionGeneric::default();
        }

        if ctx.is_dry(h) {
            return SourceContributionGeneric::momentum(-hu / dt, -hv / dt);
        }

        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < backend.config_scalar(1e-20, "ChezyFrictionConfig.speed_sq_floor") {
            return SourceContributionGeneric::default();
        }

        let speed = speed_sq.sqrt();
        let cf = backend.config_scalar(self.cf, "ChezyFrictionConfig.cf");
        let decay = B::Scalar::ONE / (B::Scalar::ONE + dt * cf * speed);
        let factor = (decay - B::Scalar::ONE) / dt;

        SourceContributionGeneric::momentum(hu * factor, hv * factor)
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        _rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.enabled {
            return;
        }

        let n_cells = state.h.len();
        if rhs_hu.len() < n_cells {
            rhs_hu.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n_cells {
            rhs_hv.resize(n_cells, B::Scalar::ZERO);
        }

        for i in 0..n_cells {
            let contrib = self.compute_cell(i, state, ctx);
            rhs_hu[i] += contrib.s_hu;
            rhs_hv[i] += contrib.s_hv;
        }
    }
}
pub struct FrictionCalculator {
    g: f64, 
    h_min: f64, 
    h_friction: f64, 
}

impl FrictionCalculator {
    /// 创建新的计算器
    
    pub fn new(g: f64, h_min: f64, h_friction: f64) -> Self {
        Self { g, h_min, h_friction }
    }

    /// 从数值参数创建
    
    pub fn from_params(g: f64, params: &crate::types::NumericalParams<f64>) -> Self {
        Self::new(g, params.h_dry, params.h_dry)
    }

    /// 计算 Manning 摩擦系数
    #[inline]
    pub fn manning_cf(&self, h: f64, n: f64) -> f64 {
        let h_safe = h.max(self.h_friction);
        self.g * n * n / h_safe.cbrt()
    }

    /// 计算 Chezy 摩擦系数
    #[inline]
    pub fn chezy_cf(&self, chezy_c: f64) -> f64 {
        self.g / (chezy_c * chezy_c)
    }

    /// 计算衰减因子
    #[inline]
    pub fn decay_factor(&self, cf: f64, speed: f64, dt: f64) -> f64 {
        1.0 / (1.0 + dt * cf * speed)
    }

    /// 应用隐式摩擦
    #[inline]
    pub fn apply_implicit(&self, hu: f64, hv: f64, h: f64, cf: f64, dt: f64) -> (f64, f64) {
        if h < self.h_min {
            return (0.0, 0.0);
        }

        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 {
            return (hu, hv);
        }

        let speed = speed_sq.sqrt();
        let decay = self.decay_factor(cf, speed, dt);
        (hu * decay, hv * decay)
    }
}

/// Manning 摩擦便捷构造器
pub struct ManningFriction;

impl ManningFriction {
    /// 创建均匀 Manning 系数配置
    
    pub fn new(g: f64, n_cells: usize, default_n: f64) -> ManningFrictionConfig {
        ManningFrictionConfig::new(g, n_cells, default_n)
    }

    /// 创建空间变化 Manning 系数配置
    
    pub fn with_field(g: f64, manning_n: Vec<f64>) -> ManningFrictionConfig {
        ManningFrictionConfig::with_field(g, manning_n)
    }

    /// 创建默认配置 (g=9.81, n=0.025)
    pub fn default_config(n_cells: usize) -> ManningFrictionConfig {
        ManningFrictionConfig::default_config(n_cells)
    }
}

/// Chezy 摩擦便捷构造器
pub struct ChezyFriction;

impl ChezyFriction {
    /// 创建 Chezy 摩擦配置
    
    pub fn new(g: f64, chezy_c: f64) -> ChezyFrictionConfig {
        ChezyFrictionConfig::new(g, chezy_c)
    }

    /// 创建默认配置 (g=9.81, C=50)
    pub fn default_config() -> ChezyFrictionConfig {
        ChezyFrictionConfig::default_config()
    }
}

// =============================================================================
// 泛型摩擦源项（后端无关）
// =============================================================================

/// Manning 摩擦配置（泛型）
#[derive(Debug, Clone)]
pub struct ManningFrictionConfigGeneric<B: Backend> {
    /// 重力加速度 [m/s²]
    pub gravity: B::Scalar,
    /// 每个单元的 Manning 系数 [s/m^{1/3}]
    pub manning_n: B::Buffer<B::Scalar>,
    /// 最小水深（用于避免除零）[m]
    pub min_depth: B::Scalar,
    /// 最大摩擦系数（用于稳定性）
    pub max_cf: B::Scalar,
}

impl<B: Backend> ManningFrictionConfigGeneric<B> {
    /// 创建均匀 Manning 系数配置
    pub fn uniform(backend: &B, n_cells: usize, manning_n: B::Scalar) -> Self {
        Self {
            gravity: backend.config_scalar(9.81, "ManningFrictionConfigGeneric.gravity"),
            manning_n: backend.alloc_init(n_cells, manning_n),
            min_depth: backend.config_scalar(1e-6, "ManningFrictionConfigGeneric.min_depth"),
            max_cf: backend.config_scalar(100.0, "ManningFrictionConfigGeneric.max_cf"),
        }
    }

    /// 从 Manning 系数数组创建
    pub fn from_array(backend: &B, manning_n: &[B::Scalar]) -> Self {
        let mut buffer = backend.alloc_init(manning_n.len(), B::Scalar::ZERO);
        buffer.as_slice_mut().copy_from_slice(manning_n);
        Self {
            gravity: backend.config_scalar(9.81, "ManningFrictionConfigGeneric.gravity"),
            manning_n: buffer,
            min_depth: backend.config_scalar(1e-6, "ManningFrictionConfigGeneric.min_depth"),
            max_cf: backend.config_scalar(100.0, "ManningFrictionConfigGeneric.max_cf"),
        }
    }
}

/// 泛型 Manning 摩擦源项（后端无关）
pub struct ManningFrictionGeneric<B: Backend> {
    config: ManningFrictionConfigGeneric<B>,
    backend: B,
    enabled: bool,
}

impl<B: Backend> ManningFrictionGeneric<B> {
    /// 创建新的 Manning 摩擦源项
    pub fn new(backend: B, config: ManningFrictionConfigGeneric<B>) -> Self {
        Self { config, backend, enabled: true }
    }

    /// 创建均匀 Manning 系数的摩擦源项
    pub fn uniform(backend: B, n_cells: usize, manning_n: B::Scalar) -> Self {
        let config = ManningFrictionConfigGeneric::uniform(&backend, n_cells, manning_n);
        Self::new(backend, config)
    }

    /// 获取后端引用
    pub fn backend(&self) -> &B { &self.backend }

    /// 获取配置引用
    pub fn config(&self) -> &ManningFrictionConfigGeneric<B> { &self.config }

    /// 设置启用状态
    pub fn set_enabled(&mut self, enabled: bool) { self.enabled = enabled; }

    /// 更新单个单元的 Manning 系数
    pub fn set_manning_n(&mut self, cell: usize, n: B::Scalar) {
        if cell < self.config.manning_n.len() {
            self.config.manning_n[cell] = n;
        }
    }
}

// =============================================================================
// generic main-path implementation
// =============================================================================

impl<B> SourceTermGeneric<B> for ManningFrictionGeneric<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str { "ManningFriction" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::LocallyImplicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let h = state.h[cell];
        let hu = state.hu[cell];
        let hv = state.hv[cell];

        if !h.is_finite() || !ctx.dt.is_finite() || ctx.dt <= B::Scalar::ZERO {
            return SourceContributionGeneric::default();
        }

        if h < self.config.min_depth {
            return SourceContributionGeneric::default();
        }

        let n = self
            .config
            .manning_n
            .get(cell)
            .copied()
            .unwrap_or_else(|| state.backend().config_scalar(0.03, "ManningFrictionGeneric.default_n"));
        let g = self.config.gravity;

        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();
        if speed < state.backend().config_scalar(1e-10, "ManningFrictionGeneric.min_speed") {
            return SourceContributionGeneric::default();
        }

        let h_pow = h.powf(state.backend().config_scalar(1.0 / 3.0, "ManningFrictionGeneric.one_third"));
        let cf = (g * n * n / h_pow).min(self.config.max_cf);
        let gamma = cf * speed / h;
        let decay = B::Scalar::ONE / (B::Scalar::ONE + ctx.dt * gamma);
        let factor = (decay - B::Scalar::ONE) / ctx.dt;

        SourceContributionGeneric {
            s_h: B::Scalar::ZERO,
            s_hu: hu * factor,
            s_hv: hv * factor,
        }
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.enabled {
            return;
        }

        let n_cells = state.n_cells();
        if rhs_h.len() < n_cells {
            rhs_h.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hu.len() < n_cells {
            rhs_hu.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n_cells {
            rhs_hv.resize(n_cells, B::Scalar::ZERO);
        }
        for cell in 0..n_cells {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

/// Chezy 摩擦配置（泛型）
#[derive(Debug, Clone)]
pub struct ChezyFrictionConfigGeneric<B: Backend> {
    /// 重力加速度 [m/s²]
    pub gravity: B::Scalar,
    /// 每个单元的 Chezy 系数 [m^{1/2}/s]
    pub chezy_c: B::Buffer<B::Scalar>,
    /// 最小水深 [m]
    pub min_depth: B::Scalar,
}

impl<B: Backend> ChezyFrictionConfigGeneric<B> {
    /// 创建均匀 Chezy 系数配置
    pub fn uniform(backend: &B, n_cells: usize, chezy_c: B::Scalar) -> Self {
        Self {
            gravity: backend.config_scalar(9.81, "ChezyFrictionConfigGeneric.gravity"),
            chezy_c: backend.alloc_init(n_cells, chezy_c),
            min_depth: backend.config_scalar(1e-6, "ChezyFrictionConfigGeneric.min_depth"),
        }
    }
}

/// 泛型 Chezy 摩擦源项
pub struct ChezyFrictionGeneric<B: Backend> {
    config: ChezyFrictionConfigGeneric<B>,
    #[allow(dead_code)]
    backend: B,
    enabled: bool,
}

impl<B: Backend> ChezyFrictionGeneric<B> {
    pub fn new(backend: B, config: ChezyFrictionConfigGeneric<B>) -> Self {
        Self { config, backend, enabled: true }
    }

    pub fn uniform(backend: B, n_cells: usize, chezy_c: B::Scalar) -> Self {
        let config = ChezyFrictionConfigGeneric::uniform(&backend, n_cells, chezy_c);
        Self::new(backend, config)
    }
}

// =============================================================================
// generic main-path implementation
// =============================================================================

impl<B> SourceTermGeneric<B> for ChezyFrictionGeneric<B>
where
    B: Backend,
    B::Scalar: RuntimeScalar,
{
    fn name(&self) -> &'static str { "ChezyFriction" }

    fn stiffness(&self) -> SourceStiffness { SourceStiffness::LocallyImplicit }

    fn is_enabled(&self) -> bool { self.enabled }

    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterState<B>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) -> SourceContributionGeneric<B::Scalar> {
        let h = state.h[cell];
        let hu = state.hu[cell];
        let hv = state.hv[cell];

        if !h.is_finite() || !ctx.dt.is_finite() || ctx.dt <= B::Scalar::ZERO {
            return SourceContributionGeneric::default();
        }

        if h < self.config.min_depth {
            return SourceContributionGeneric::default();
        }

        let c = self
            .config
            .chezy_c
            .get(cell)
            .copied()
            .unwrap_or_else(|| state.backend().config_scalar(50.0, "ChezyFrictionGeneric.default_c"));
        let g = self.config.gravity;

        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();
        if speed < state.backend().config_scalar(1e-10, "ChezyFrictionGeneric.min_speed") {
            return SourceContributionGeneric::default();
        }

        let cf = g / (c * c);
        let gamma = cf * speed / h;
        let decay = B::Scalar::ONE / (B::Scalar::ONE + ctx.dt * gamma);
        let factor = (decay - B::Scalar::ONE) / ctx.dt;

        SourceContributionGeneric {
            s_h: B::Scalar::ZERO,
            s_hu: hu * factor,
            s_hv: hv * factor,
        }
    }

    fn accumulate(
        &self,
        state: &ShallowWaterState<B>,
        rhs_h: &mut B::Buffer<B::Scalar>,
        rhs_hu: &mut B::Buffer<B::Scalar>,
        rhs_hv: &mut B::Buffer<B::Scalar>,
        ctx: &SourceContextGeneric<B::Scalar>,
    ) {
        if !self.enabled {
            return;
        }

        let n_cells = state.n_cells();
        if rhs_h.len() < n_cells {
            rhs_h.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hu.len() < n_cells {
            rhs_hu.resize(n_cells, B::Scalar::ZERO);
        }
        if rhs_hv.len() < n_cells {
            rhs_hv.resize(n_cells, B::Scalar::ZERO);
        }
        for cell in 0..n_cells {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sources::traits::test_support::{
        assert_source_metadata,
        test_backend,
        test_context,
        TestBackend,
    };

    fn create_test_state(n_cells: usize, h: f64, u: f64, v: f64) -> ShallowWaterState<TestBackend> {
        let mut state = ShallowWaterState::<TestBackend>::new_with_backend(test_backend(), n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
            state.hu[i] = h * u;
            state.hv[i] = h * v;
            state.z[i] = 0.0;
        }
        state
    }

    #[test]
    fn test_manning_config_creation() {
        let config = ManningFrictionConfig::new(9.81, 100, 0.03);
        assert!(config.enabled);
        assert_eq!(config.g, 9.81);
        assert!(config.precomputed_gn2.is_some());
    }

    #[test]
    fn test_manning_dry_cell() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 0.0001, 1.0, 1.0);
        let ctx = test_context(0.0, 0.1);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        // 干单元应该衰减动量
        assert!(contrib.s_h.abs() < 1e-10);
        assert!(contrib.s_hu < 0.0); // 负的源项，减少动量
        assert!(contrib.s_hv < 0.0);
    }

    #[test]
    fn test_manning_still_water() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 1.0, 0.0, 0.0);
        let ctx = test_context(0.0, 0.1);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        // 静水应该没有摩擦
        assert_eq!(contrib.s_h, 0.0);
        assert_eq!(contrib.s_hu, 0.0);
        assert_eq!(contrib.s_hv, 0.0);
    }

    #[test]
    fn test_manning_flowing_water() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let ctx = test_context(0.0, 0.1);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        // 流动水应该有摩擦减速
        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu < 0.0); // x方向减速
        assert!(contrib.s_hv.abs() < 1e-10); // y方向无速度
    }

    #[test]
    fn test_manning_implicit() {
        // 验证隐式处理不会产生负动量
        let config = ManningFrictionConfig::new(9.81, 10, 0.1); // 高摩擦
        let state = create_test_state(10, 0.1, 1.0, 0.5); // 浅水
        let ctx = test_context(0.0, 1.0); // 大时间步

        let contrib = config.compute_cell(0, &state, &ctx);
        
        // 隐式处理应该给出有限的源项
        assert!(contrib.s_h.is_finite() && contrib.s_hu.is_finite() && contrib.s_hv.is_finite());
        assert!(contrib.s_hu < 0.0);
    }

    #[test]
    fn test_chezy_config_creation() {
        let config = ChezyFrictionConfig::new(9.81, 50.0);
        assert!(config.enabled);
        assert_eq!(config.chezy_c, 50.0);
        assert!((config.cf - 9.81 / 2500.0).abs() < 1e-10);
    }

    #[test]
    fn test_chezy_flowing_water() {
        let config = ChezyFrictionConfig::new(9.81, 50.0);
        let state = create_test_state(10, 1.0, 1.0, 0.0);
        let ctx = test_context(0.0, 0.1);

        let contrib = config.compute_cell(0, &state, &ctx);
        
        assert_eq!(contrib.s_h, 0.0);
        assert!(contrib.s_hu < 0.0);
    }

    #[test]
    fn test_friction_calculator() {
        let calc = FrictionCalculator::new(9.81, 0.001, 0.001);
        
        let cf = calc.manning_cf(1.0, 0.03);
        assert!(cf > 0.0);
        
        let decay = calc.decay_factor(cf, 1.0, 0.1);
        assert!(decay > 0.0 && decay < 1.0);
    }

    #[test]
    fn test_friction_calculator_apply() {
        let calc = FrictionCalculator::new(9.81, 0.001, 0.001);
        
        let (hu_new, hv_new) = calc.apply_implicit(1.0, 0.5, 1.0, 0.01, 0.1);
        
        // 应该减少但不变号
        assert!(hu_new > 0.0 && hu_new < 1.0);
        assert!(hv_new > 0.0 && hv_new < 0.5);
    }

    #[test]
    fn test_manning_batch_compute() {
        let config = ManningFrictionConfig::new(9.81, 10, 0.03);
        let state = create_test_state(10, 1.0, 1.0, 0.5);
        let ctx = test_context(0.0, 0.1);

        let mut out_h = vec![0.0; 10];
        let mut out_hu = vec![0.0; 10];
        let mut out_hv = vec![0.0; 10];

        config.accumulate(&state, &mut out_h, &mut out_hu, &mut out_hv, &ctx);

        // 所有单元应该有相同的负源项
        for i in 0..10 {
            assert!(out_h[i].abs() < 1e-10);
            assert!(out_hu[i] < 0.0);
            assert!(out_hv[i] < 0.0);
        }
    }

    #[test]
    fn test_source_term_trait() {
        let manning = ManningFrictionConfig::new(9.81, 10, 0.03);
        let chezy = ChezyFrictionConfig::new(9.81, 50.0);

        assert_source_metadata(&manning, "ManningFriction", SourceStiffness::LocallyImplicit);
        assert_source_metadata(&chezy, "ChezyFriction", SourceStiffness::LocallyImplicit);
    }

    #[test]
    fn test_convenience_constructors() {
        let manning = ManningFriction::new(9.81, 100, 0.025);
        assert!(manning.enabled);

        let chezy = ChezyFriction::default_config();
        assert_eq!(chezy.chezy_c, 50.0);
    }
}
