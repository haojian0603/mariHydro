// crates/mh_physics/src/sediment/manager.rs
//! 泥沙系统统一管理器
//!
//! 该模块实现了泥沙输运的统一管理，包括：
//! - 床面泥沙质量管理
//! - 悬沙浓度（深度平均）
//! - 侵蚀/沉降交换通量计算
//! - 质量守恒校验
//!
//! # 设计原则
//!
//! 1. **后端无关**: 所有计算通过 Backend trait 抽象
//! 2. **质量守恒**: 严格保证泥沙质量守恒
//! 3. **可扩展**: 支持多种泥沙粒径和分层

use crate::core::{Backend, CpuBackend};
use mh_runtime::RuntimeScalar as Scalar;
use crate::state::ShallowWaterStateGeneric;
use std::marker::PhantomData;

/// 泥沙系统错误
#[derive(Debug, Clone)]
pub enum SedimentError {
    /// 质量守恒违反
    ConservationViolation {
        expected: f64,
        actual: f64,
        relative_error: f64,
    },
    /// 负质量
    NegativeMass {
        cell: usize,
        value: f64,
    },
    /// 无效参数
    InvalidParameter(String),
}

impl std::fmt::Display for SedimentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SedimentError::ConservationViolation { expected, actual, relative_error } => {
                write!(f, "质量守恒违反：期望 {:.6e}，实际 {:.6e}，相对误差 {:.6e}",
                    expected, actual, relative_error)
            }
            SedimentError::NegativeMass { cell, value } => {
                write!(f, "单元 {} 出现负质量：{:.6e}", cell, value)
            }
            SedimentError::InvalidParameter(msg) => {
                write!(f, "无效参数：{}", msg)
            }
        }
    }
}

impl std::error::Error for SedimentError {}

/// 泥沙系统配置
#[derive(Debug, Clone)]
pub struct SedimentConfigGeneric<S: Scalar> {
    /// 临界剪切应力 [Pa]
    pub tau_critical: S,
    /// 侵蚀系数 [kg/m²/s/Pa]
    pub erosion_rate: S,
    /// 沉降速度 [m/s]
    pub settling_velocity: S,
    /// 泥沙密度 [kg/m³]
    pub sediment_density: S,
    /// 水密度 [kg/m³]
    pub water_density: S,
    /// 孔隙率
    pub porosity: S,
    /// 守恒误差容限
    pub conservation_tolerance: S,
    /// 最小水深（泥沙计算阈值）[m]
    pub min_depth: S,
}

impl<S: Scalar> Default for SedimentConfigGeneric<S> {
    fn default() -> Self {
        Self {
            tau_critical: S::from_config(0.1).unwrap_or(S::ZERO),
            erosion_rate: S::from_config(1e-4).unwrap_or(S::ZERO),
            settling_velocity: S::from_config(0.001).unwrap_or(S::ZERO),
            sediment_density: S::from_config(2650.0).unwrap_or(S::ZERO),
            water_density: S::from_config(998.2).unwrap_or(S::ZERO),
            porosity: S::from_config(0.4).unwrap_or(S::ZERO),
            conservation_tolerance: S::from_config(1e-10).unwrap_or(S::ZERO),
            min_depth: S::from_config(1e-4).unwrap_or(S::ZERO),
        }
    }
}

/// 泥沙交换通量统计
#[derive(Debug, Clone, Default)]
pub struct SedimentFluxStats<S: Scalar> {
    /// 总侵蚀量 [kg]
    pub total_erosion: S,
    /// 总沉降量 [kg]
    pub total_deposition: S,
    /// 净交换量 [kg]
    pub net_exchange: S,
    /// 最大侵蚀单元
    pub max_erosion_cell: usize,
    /// 最大沉降单元
    pub max_deposition_cell: usize,
}

/// 泥沙状态（深度平均）
/// 
/// 存储悬沙浓度和床面泥沙质量。
pub struct SedimentStateGeneric<B: Backend> {
    /// 悬沙浓度 [kg/m³]
    pub concentration: B::Buffer<B::Scalar>,
    /// 床面泥沙质量 [kg/m²]
    pub bed_mass: B::Buffer<B::Scalar>,
    /// 守恒量 (h * C) [kg/m²]
    pub conserved: B::Buffer<B::Scalar>,
    /// 单元数量
    n_cells: usize,
    /// 后端标记
    _marker: PhantomData<B>,
}

impl<B: Backend> SedimentStateGeneric<B> {
    /// 创建新的泥沙状态
    pub fn new_with_backend(backend: &B, n_cells: usize) -> Self {
        Self {
            concentration: backend.alloc(n_cells),
            bed_mass: backend.alloc(n_cells),
            conserved: backend.alloc(n_cells),
            n_cells,
            _marker: PhantomData,
        }
    }
    
    /// 获取单元数量
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }
}

/// 泥沙系统管理器
/// 
/// 负责：
/// - 床面泥沙质量管理
/// - 悬沙浓度（深度平均）
/// - 侵蚀/沉降交换通量
/// - 质量守恒校验
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
pub struct SedimentManagerGeneric<B: Backend> {
    /// 配置
    config: SedimentConfigGeneric<B::Scalar>,
    
    /// 泥沙状态
    state: SedimentStateGeneric<B>,
    
    /// 床面侵蚀/沉降交换通量 [kg/m²/s]
    /// 正值 = 侵蚀（床面→悬浮），负值 = 沉降（悬浮→床面）
    exchange_flux: B::Buffer<B::Scalar>,
    
    /// 床面剪切应力（计算用）[Pa]
    tau_bed: B::Buffer<B::Scalar>,
    
    /// 初始总质量（守恒校验用）
    initial_total_mass: B::Scalar,
    
    /// 是否已初始化
    initialized: bool,
    
    /// 计算后端
    backend: B,
}

impl<B: Backend + Clone> SedimentManagerGeneric<B> {
    /// 创建新的泥沙管理器
    pub fn new_with_backend(backend: B, n_cells: usize, config: SedimentConfigGeneric<B::Scalar>) -> Self {
        Self {
            state: SedimentStateGeneric::new_with_backend(&backend, n_cells),
            exchange_flux: backend.alloc(n_cells),
            tau_bed: backend.alloc(n_cells),
            config,
            initial_total_mass: B::Scalar::ZERO,
            initialized: false,
            backend,
        }
    }
    
    /// 获取后端引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取配置引用
    pub fn config(&self) -> &SedimentConfigGeneric<B::Scalar> {
        &self.config
    }
    
    /// 获取泥沙状态引用
    pub fn state(&self) -> &SedimentStateGeneric<B> {
        &self.state
    }
    
    /// 获取泥沙状态可变引用
    pub fn state_mut(&mut self) -> &mut SedimentStateGeneric<B> {
        &mut self.state
    }
    
    /// 获取交换通量引用
    pub fn exchange_flux(&self) -> &B::Buffer<B::Scalar> {
        &self.exchange_flux
    }
}

/// CPU f64 后端的泥沙管理器实现
impl SedimentManagerGeneric<CpuBackend<f64>> {
    /// 使用默认后端创建
    pub fn new(n_cells: usize, config: SedimentConfigGeneric<f64>) -> Self {
        Self::new_with_backend(CpuBackend::<f64>::new(), n_cells, config)
    }
    
    /// 设置初始床面质量
    pub fn set_initial_bed_mass(&mut self, mass: &[f64]) {
        if mass.len() != self.state.n_cells {
            return;
        }
        for (i, &m) in mass.iter().enumerate() {
            self.state.bed_mass[i] = m;
        }
        self.compute_initial_mass();
    }
    
    /// 设置初始悬沙浓度
    pub fn set_initial_concentration(&mut self, conc: &[f64]) {
        if conc.len() != self.state.n_cells {
            return;
        }
        for (i, &c) in conc.iter().enumerate() {
            self.state.concentration[i] = c;
        }
    }
    
    /// 从水动力状态更新守恒量
    pub fn update_conserved(&mut self, state: &ShallowWaterStateGeneric<CpuBackend<f64>>) {
        let n_cells = self.state.n_cells;
        for i in 0..n_cells {
            self.state.conserved[i] = state.h[i] * self.state.concentration[i];
        }
    }
    
    /// 计算初始总质量
    fn compute_initial_mass(&mut self) {
        let mut total = 0.0;
        for i in 0..self.state.n_cells {
            total += self.state.bed_mass[i];
        }
        self.initial_total_mass = total;
        self.initialized = true;
    }
    
    /// 计算床面剪切应力
    /// 
    /// 使用 Manning 公式：τ_b = ρ g n² |u|² / h^(1/3)
    /// 
    /// # 参数
    /// 
    /// - `state`: 水动力状态
    /// - `manning_n`: Manning 系数数组
    pub fn compute_bed_shear_stress(
        &mut self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        manning_n: &[f64],
    ) {
        let n_cells = self.state.n_cells;
        let g = self.config.water_density.to_f64() * 9.81;  // ρg
        let h_min = self.config.min_depth;
        
        for i in 0..n_cells {
            let h = state.h[i];
            if h < h_min {
                self.tau_bed[i] = 0.0;
                continue;
            }
            
            let hu = state.hu[i];
            let hv = state.hv[i];
            let u = hu / h;
            let v = hv / h;
            let speed_sq = u * u + v * v;
            
            let n = if i < manning_n.len() { manning_n[i] } else { 0.03 };
            let h_pow = h.powf(1.0 / 3.0);
            
            // τ = ρ g n² |u|² / h^(1/3)
            self.tau_bed[i] = g * n * n * speed_sq / h_pow;
        }
    }
    
    /// 计算侵蚀/沉降交换通量
    /// 
    /// 侵蚀：E = M (τ - τ_c) / τ_c  当 τ > τ_c
    /// 沉降：D = w_s * C
    /// 净通量：F = E - D
    /// 
    /// # 参数
    /// 
    /// - `state`: 水动力状态
    pub fn compute_exchange_flux(
        &mut self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
    ) {
        let n_cells = self.state.n_cells;
        let tau_c = self.config.tau_critical;
        let m = self.config.erosion_rate;
        let ws = self.config.settling_velocity;
        let h_min = self.config.min_depth;
        
        for i in 0..n_cells {
            let h = state.h[i];
            if h < h_min {
                self.exchange_flux[i] = 0.0;
                continue;
            }
            
            let tau = self.tau_bed[i];
            let c = self.state.concentration[i];
            
            // 侵蚀
            let erosion = if tau > tau_c && self.state.bed_mass[i] > 0.0 {
                m * (tau - tau_c) / tau_c
            } else {
                0.0
            };
            
            // 沉降
            let deposition = ws * c;
            
            // 净通量：正值表示侵蚀，负值表示沉降
            self.exchange_flux[i] = erosion - deposition;
        }
    }
    
    /// 单步更新泥沙系统
    /// 
    /// # 参数
    /// 
    /// - `state`: 水动力状态
    /// - `cell_areas`: 单元面积数组
    /// - `dt`: 时间步长
    /// 
    /// # 返回
    /// 
    /// 返回交换通量统计和可能的错误
    pub fn step(
        &mut self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        cell_areas: &[f64],
        dt: f64,
    ) -> Result<SedimentFluxStats<f64>, SedimentError> {
        let n_cells = self.state.n_cells;
        let h_min = self.config.min_depth;
        
        let mut stats = SedimentFluxStats::default();
        let mut max_erosion = 0.0f64;
        let mut max_deposition = 0.0f64;
        
        for i in 0..n_cells {
            let h = state.h[i];
            let area = if i < cell_areas.len() { cell_areas[i] } else { 1.0 };
            let flux = self.exchange_flux[i];
            
            // 质量变化 [kg/m²]
            let delta_mass = flux * dt;
            
            // 更新床面质量
            let new_bed = self.state.bed_mass[i] - delta_mass;
            
            // 检查负质量
            if new_bed < 0.0 {
                // 限制侵蚀量，不能超过床面存量
                let max_erosion_flux = self.state.bed_mass[i] / dt;
                self.exchange_flux[i] = self.exchange_flux[i].min(max_erosion_flux);
                self.state.bed_mass[i] = 0.0;
            } else {
                self.state.bed_mass[i] = new_bed;
            }
            
            // 更新悬沙浓度
            if h > h_min {
                // dC/dt = F/h（简化，忽略对流扩散）
                let dc = self.exchange_flux[i] / h;
                let new_c = (self.state.concentration[i] + dc * dt).max(0.0);
                self.state.concentration[i] = new_c;
            }
            
            // 统计
            let mass_change = self.exchange_flux[i] * area;
            if mass_change > 0.0 {
                stats.total_erosion += mass_change;
                if mass_change > max_erosion {
                    max_erosion = mass_change;
                    stats.max_erosion_cell = i;
                }
            } else {
                stats.total_deposition -= mass_change;
                if -mass_change > max_deposition {
                    max_deposition = -mass_change;
                    stats.max_deposition_cell = i;
                }
            }
        }
        
        stats.net_exchange = stats.total_erosion - stats.total_deposition;
        
        // 更新守恒量
        self.update_conserved(state);
        
        Ok(stats)
    }
    
    /// 验证质量守恒
    pub fn verify_conservation(
        &self,
        state: &ShallowWaterStateGeneric<CpuBackend<f64>>,
        cell_areas: &[f64],
    ) -> Result<(), SedimentError> {
        if !self.initialized {
            return Ok(());
        }
        
        let n_cells = self.state.n_cells;
        let mut total_bed = 0.0;
        let mut total_suspended = 0.0;
        
        for i in 0..n_cells {
            let area = if i < cell_areas.len() { cell_areas[i] } else { 1.0 };
            total_bed += self.state.bed_mass[i] * area;
            total_suspended += state.h[i] * self.state.concentration[i] * area;
        }
        
        let total_current = total_bed + total_suspended;
        let error = (total_current - self.initial_total_mass).abs();
        let relative_error = if self.initial_total_mass.abs() > 1e-14 {
            error / self.initial_total_mass
        } else {
            error
        };
        
        if relative_error > self.config.conservation_tolerance {
            return Err(SedimentError::ConservationViolation {
                expected: self.initial_total_mass,
                actual: total_current,
                relative_error,
            });
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sediment_manager_creation() {
        let config = SedimentConfigGeneric::default();
        let manager = SedimentManagerGeneric::new(100, config);
        
        assert_eq!(manager.state().n_cells(), 100);
    }
    
    #[test]
    fn test_erosion_deposition_balance() {
        let mut config = SedimentConfigGeneric::default();
        config.tau_critical = 0.1;
        config.erosion_rate = 1e-4;
        config.settling_velocity = 0.001;
        
        let mut manager = SedimentManagerGeneric::new(10, config);
        
        // 设置初始床面质量
        let bed_mass = vec![100.0; 10];
        manager.set_initial_bed_mass(&bed_mass);
        
        // 设置初始浓度
        let conc = vec![0.1; 10];
        manager.set_initial_concentration(&conc);
        
        // 创建水动力状态
        let mut state = ShallowWaterStateGeneric::new_with_backend(
            CpuBackend::<f64>::new(), 10
        );
        for i in 0..10 {
            state.h[i] = 1.0;  // 1m 水深
        }
        
        manager.update_conserved(&state);
        
        // 初始总质量应为床面 + 悬浮
        let total_bed: f64 = manager.state().bed_mass.iter().sum();
        assert!((total_bed - 1000.0).abs() < 1e-6);
    }
}
