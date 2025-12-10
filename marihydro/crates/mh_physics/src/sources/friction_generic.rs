// crates/mh_physics/src/sources/friction_generic.rs
//! 泛型摩擦源项
//!
//! 该模块实现了后端无关的 Manning 摩擦源项。
//!
//! # Manning 公式
//!
//! Manning 摩擦应力为：
//! $$\tau_b = \rho g n^2 |u| u / h^{1/3}$$
//!
//! 其中：
//! - $n$ 是 Manning 糙率系数 [s/m^{1/3}]
//! - $h$ 是水深 [m]
//! - $u$ 是流速 [m/s]
//!
//! # 隐式处理
//!
//! 摩擦项是刚性的，需要隐式处理以避免数值不稳定。
//! 使用局部隐式因子：$\gamma = c_f |u| / h$
//! 更新公式：$u^{n+1} = u^n / (1 + dt \cdot \gamma)$

use crate::core::{Backend, CpuBackend, Scalar};
use crate::state::ShallowWaterStateGeneric;
use super::traits_generic::{
    SourceTermGeneric, SourceContributionGeneric, SourceContextGeneric, SourceStiffness
};

/// Manning 摩擦配置
#[derive(Debug, Clone)]
pub struct ManningFrictionConfigGeneric<S: Scalar> {
    /// 重力加速度 [m/s²]
    pub gravity: S,
    /// 每个单元的 Manning 系数 [s/m^{1/3}]
    pub manning_n: Vec<S>,
    /// 最小水深（用于避免除零）[m]
    pub min_depth: S,
    /// 最大摩擦系数（用于稳定性）
    pub max_cf: S,
}

impl<S: Scalar> ManningFrictionConfigGeneric<S> {
    /// 创建均匀 Manning 系数配置
    pub fn uniform(n_cells: usize, manning_n: S) -> Self {
        Self {
            gravity: S::GRAVITY,
            manning_n: vec![manning_n; n_cells],
            min_depth: Scalar::from_f64(1e-6),
            max_cf: Scalar::from_f64(100.0),
        }
    }
    
    /// 从 Manning 系数数组创建
    pub fn from_array(manning_n: Vec<S>) -> Self {
        Self {
            gravity: S::GRAVITY,
            manning_n,
            min_depth: Scalar::from_f64(1e-6),
            max_cf: Scalar::from_f64(100.0),
        }
    }
}

/// 泛型 Manning 摩擦源项
/// 
/// 使用 Manning 公式计算床面摩擦力。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
pub struct ManningFrictionGeneric<B: Backend> {
    /// 配置
    config: ManningFrictionConfigGeneric<B::Scalar>,
    /// 计算后端
    backend: B,
    /// 是否启用
    enabled: bool,
}

impl<B: Backend> ManningFrictionGeneric<B> {
    /// 创建新的 Manning 摩擦源项
    pub fn new(backend: B, config: ManningFrictionConfigGeneric<B::Scalar>) -> Self {
        Self {
            config,
            backend,
            enabled: true,
        }
    }
    
    /// 创建均匀 Manning 系数的摩擦源项
    pub fn uniform(backend: B, n_cells: usize, manning_n: B::Scalar) -> Self {
        Self::new(backend, ManningFrictionConfigGeneric::uniform(n_cells, manning_n))
    }
    
    /// 获取后端引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取配置引用
    pub fn config(&self) -> &ManningFrictionConfigGeneric<B::Scalar> {
        &self.config
    }
    
    /// 设置启用状态
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// 更新单个单元的 Manning 系数
    pub fn set_manning_n(&mut self, cell: usize, n: B::Scalar) {
        if cell < self.config.manning_n.len() {
            self.config.manning_n[cell] = n;
        }
    }
}

/// CPU f64 后端的 Manning 摩擦实现
impl SourceTermGeneric<CpuBackend<f64>> for ManningFrictionGeneric<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "Manning 摩擦"
    }
    
    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::LocallyImplicit
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];
        let hu = state.hu[cell];
        let hv = state.hv[cell];
        
        // 干单元不计算摩擦
        if h < self.config.min_depth {
            return SourceContributionGeneric::default();
        }
        
        // 获取 Manning 系数
        let n = if cell < self.config.manning_n.len() {
            self.config.manning_n[cell]
        } else {
            0.03  // 默认值
        };
        
        let g = self.config.gravity;
        
        // 计算速度
        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();
        
        // 避免速度过小时的数值问题
        if speed < 1e-10 {
            return SourceContributionGeneric::default();
        }
        
        // Manning 摩擦系数
        // c_f = g * n² / h^(1/3)
        let h_pow = h.powf(1.0 / 3.0);
        let cf = (g * n * n / h_pow).min(self.config.max_cf);
        
        // 隐式处理因子
        // γ = c_f * |u| / h
        let gamma = cf * speed / h;
        let factor = 1.0 / (1.0 + ctx.dt * gamma);
        
        // 摩擦力 = -c_f * |u| * u
        // 使用隐式因子处理
        // 注意：这里返回的是隐式处理后的等效显式源项
        // 实际应用时，可能需要在更新步骤中直接使用隐式因子
        SourceContributionGeneric {
            s_h: 0.0,
            s_hu: -cf * speed * u * factor,
            s_hv: -cf * speed * v * factor,
        }
    }
    
    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }
        
        let n_cells = state.n_cells();
        for cell in 0..n_cells {
            let contrib = self.compute_cell(cell, state, ctx);
            rhs_h[cell] += contrib.s_h;
            rhs_hu[cell] += contrib.s_hu;
            rhs_hv[cell] += contrib.s_hv;
        }
    }
}

/// Chezy 摩擦配置
#[derive(Debug, Clone)]
pub struct ChezyFrictionConfigGeneric<S: Scalar> {
    /// 重力加速度 [m/s²]
    pub gravity: S,
    /// 每个单元的 Chezy 系数 [m^{1/2}/s]
    pub chezy_c: Vec<S>,
    /// 最小水深 [m]
    pub min_depth: S,
}

impl<S: Scalar> ChezyFrictionConfigGeneric<S> {
    /// 创建均匀 Chezy 系数配置
    pub fn uniform(n_cells: usize, chezy_c: S) -> Self {
        Self {
            gravity: S::GRAVITY,
            chezy_c: vec![chezy_c; n_cells],
            min_depth: Scalar::from_f64(1e-6),
        }
    }
}

/// 泛型 Chezy 摩擦源项
pub struct ChezyFrictionGeneric<B: Backend> {
    config: ChezyFrictionConfigGeneric<B::Scalar>,
    #[allow(dead_code)]
    backend: B,
    enabled: bool,
}

impl<B: Backend> ChezyFrictionGeneric<B> {
    /// 创建新的 Chezy 摩擦源项
    pub fn new(backend: B, config: ChezyFrictionConfigGeneric<B::Scalar>) -> Self {
        Self {
            config,
            backend,
            enabled: true,
        }
    }
    
    /// 创建均匀 Chezy 系数的摩擦源项
    pub fn uniform(backend: B, n_cells: usize, chezy_c: B::Scalar) -> Self {
        Self::new(backend, ChezyFrictionConfigGeneric::uniform(n_cells, chezy_c))
    }
}

impl SourceTermGeneric<CpuBackend<f64>> for ChezyFrictionGeneric<CpuBackend<f64>> {
    fn name(&self) -> &'static str {
        "Chezy 摩擦"
    }
    
    fn stiffness(&self) -> SourceStiffness {
        SourceStiffness::LocallyImplicit
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    fn compute_cell(
        &self,
        cell: usize,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        ctx: &SourceContextGeneric<f64>,
    ) -> SourceContributionGeneric<f64> {
        let h = state.h[cell];
        let hu = state.hu[cell];
        let hv = state.hv[cell];
        
        if h < self.config.min_depth {
            return SourceContributionGeneric::default();
        }
        
        let c = if cell < self.config.chezy_c.len() {
            self.config.chezy_c[cell]
        } else {
            50.0  // 默认值
        };
        
        let g = self.config.gravity;
        
        let u = hu / h;
        let v = hv / h;
        let speed = (u * u + v * v).sqrt();
        
        if speed < 1e-10 {
            return SourceContributionGeneric::default();
        }
        
        // Chezy 摩擦系数
        // c_f = g / C²
        let cf = g / (c * c);
        
        // 隐式因子
        let gamma = cf * speed / h;
        let factor = 1.0 / (1.0 + ctx.dt * gamma);
        
        SourceContributionGeneric {
            s_h: 0.0,
            s_hu: -cf * speed * u * factor,
            s_hv: -cf * speed * v * factor,
        }
    }
    
    fn accumulate(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        rhs_h: &mut Vec<f64>,
        rhs_hu: &mut Vec<f64>,
        rhs_hv: &mut Vec<f64>,
        ctx: &SourceContextGeneric<f64>,
    ) {
        if !self.enabled {
            return;
        }
        
        let n_cells = state.n_cells();
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
    
    #[test]
    fn test_manning_friction_dry_cell() {
        let backend = CpuBackend::<f64>::new();
        let friction = ManningFrictionGeneric::uniform(backend, 10, 0.03);
        
        let mut state = ShallowWaterStateGeneric::new_with_backend(
            CpuBackend::<f64>::new(), 10
        );
        
        // 干单元
        state.h[0] = 1e-8;
        state.hu[0] = 1.0;
        state.hv[0] = 1.0;
        
        let ctx = SourceContextGeneric::with_defaults(0.0, 0.01);
        let contrib = friction.compute_cell(0, &state, &ctx);
        
        // 干单元应返回零贡献
        assert!((contrib.s_h).abs() < 1e-14);
        assert!((contrib.s_hu).abs() < 1e-14);
        assert!((contrib.s_hv).abs() < 1e-14);
    }
    
    #[test]
    fn test_manning_friction_wet_cell() {
        let backend = CpuBackend::<f64>::new();
        let friction = ManningFrictionGeneric::uniform(backend, 10, 0.03);
        
        let mut state = ShallowWaterStateGeneric::new_with_backend(
            CpuBackend::<f64>::new(), 10
        );
        
        // 湿单元
        state.h[0] = 1.0;
        state.hu[0] = 1.0;  // u = 1 m/s
        state.hv[0] = 0.0;
        
        let ctx = SourceContextGeneric::with_defaults(0.0, 0.01);
        let contrib = friction.compute_cell(0, &state, &ctx);
        
        // 摩擦力应为负（减速）
        assert!(contrib.s_hu < 0.0, "x方向摩擦力应为负");
        assert!((contrib.s_hv).abs() < 1e-10, "y方向摩擦力应接近零");
    }
}
