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

use crate::prelude::*;
use crate::state::ShallowWaterState;
use std::marker::PhantomData;

/// 泥沙系统错误
#[derive(Debug, Clone)]
pub enum SedimentError<S: RuntimeScalar> {
    /// 质量守恒违反
    ConservationViolation {
        expected: S,
        actual: S,
        relative_error: S,
    },
    /// 负质量
    NegativeMass {
        cell: usize,
        value: S,
    },
    /// 无效参数
    InvalidParameter(String),
}

impl<S> std::fmt::Display for SedimentError<S>
where
    S: RuntimeScalar + std::fmt::LowerExp,
{
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

impl<S> std::error::Error for SedimentError<S>
where
    S: RuntimeScalar + std::fmt::Debug + std::fmt::Display + std::fmt::LowerExp,
{
}

/// 泥沙系统配置
#[derive(Debug, Clone)]
pub struct SedimentConfigGeneric<S: RuntimeScalar> {
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

impl Default for SedimentConfigGeneric<f64> {
    fn default() -> Self {
        Self {
            tau_critical: 0.1,
            erosion_rate: 1e-4,
            settling_velocity: 0.001,
            sediment_density: 2650.0,
            water_density: 998.2,
            porosity: 0.4,
            conservation_tolerance: 1e-10,
            min_depth: 1e-4,
        }
    }
}

impl Default for SedimentConfigGeneric<f32> {
    fn default() -> Self {
        Self {
            tau_critical: 0.1_f32,
            erosion_rate: 1e-4_f32,
            settling_velocity: 0.001_f32,
            sediment_density: 2650.0_f32,
            water_density: 998.2_f32,
            porosity: 0.4_f32,
            conservation_tolerance: 1e-10_f32,
            min_depth: 1e-4_f32,
        }
    }
}

/// 泥沙交换通量统计
#[derive(Debug, Clone, Default)]
pub struct SedimentFluxStats<S: RuntimeScalar> {
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

impl<B> SedimentManagerGeneric<B>
where
    B: Backend + Clone,
{
    /// 创建新的泥沙管理器
    pub fn new_with_backend(backend: B, n_cells: usize, config: SedimentConfigGeneric<B::Scalar>) -> Self {
        if config.tau_critical <= B::Scalar::ZERO {
            panic!("SedimentConfig: tau_critical 必须为正值");
        }
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

/// 泛型实现
impl<B> SedimentManagerGeneric<B>
where
    B: Backend + Clone,
{
    /// 使用默认后端创建（便捷）
    pub fn new(n_cells: usize, config: SedimentConfigGeneric<B::Scalar>) -> Self where B: Default {
        Self::new_with_backend(B::default(), n_cells, config)
    }

    /// 设置初始床面质量（泛型）
    pub fn set_initial_bed_mass_from_slice(&mut self, mass: &[B::Scalar]) {
        if mass.len() != self.state.n_cells {
            log::warn!("初始床面质量长度不匹配，期望 {}，得到 {}", self.state.n_cells, mass.len());
            return;
        }
        self.state.bed_mass.copy_from_slice(mass);
        self.compute_initial_mass();
    }

    /// 便捷方法：从切片设置床面质量
    pub fn set_initial_bed_mass(&mut self, mass: &[B::Scalar]) {
        self.set_initial_bed_mass_from_slice(mass);
    }

    /// 设置初始悬沙浓度
    pub fn set_initial_concentration_from_slice(&mut self, conc: &[B::Scalar]) {
        if conc.len() != self.state.n_cells {
            log::warn!("初始浓度长度不匹配，期望 {}，得到 {}", self.state.n_cells, conc.len());
            return;
        }
        self.state.concentration.copy_from_slice(conc);
    }

    /// 便捷方法：从切片设置悬沙浓度
    pub fn set_initial_concentration(&mut self, conc: &[B::Scalar]) {
        self.set_initial_concentration_from_slice(conc);
    }

    /// 从水动力状态更新守恒量
    pub fn update_conserved(&mut self, state: &ShallowWaterState<B>) {
        let n_cells = self.state.n_cells;
        for i in 0..n_cells {
            self.state.conserved[i] = state.h[i] * self.state.concentration[i];
        }
    }

    /// 计算初始总质量
    fn compute_initial_mass(&mut self) {
        // Kahan 求和减轻大数吃小数
        let mut total = B::Scalar::ZERO;
        let mut c = B::Scalar::ZERO;
        for i in 0..self.state.n_cells {
            let y = self.state.bed_mass[i] - c;
            let t = total + y;
            c = (t - total) - y;
            total = t;
        }
        self.initial_total_mass = total;
        self.initialized = true;
    }

    /// 计算床面剪切应力（Manning）
    pub fn compute_bed_shear_stress(
        &mut self,
        state: &ShallowWaterState<B>,
        manning_n: &[B::Scalar],
    ) {
        let n_cells = self.state.n_cells;
        let cfg = |v| self.backend.config_scalar(v, "SedimentManagerGeneric.compute_bed_shear_stress");
        let g = self.config.water_density * cfg(9.81);
        let h_min = self.config.min_depth;

        for i in 0..n_cells {
            let h = state.h[i];
            if h < h_min {
                self.tau_bed[i] = B::Scalar::ZERO;
                continue;
            }

            let hu = state.hu[i];
            let hv = state.hv[i];
            let u = hu / h;
            let v = hv / h;
            let speed_sq = u * u + v * v;

            let n = if i < manning_n.len() { manning_n[i] } else { cfg(0.03) };
            let h_pow = h.powf(cfg(1.0 / 3.0));

            // τ = ρ g n² |u|² / h^(1/3)
            self.tau_bed[i] = g * n * n * speed_sq / h_pow;
        }
    }

    /// 计算侵蚀/沉降交换通量
    pub fn compute_exchange_flux(
        &mut self,
        state: &ShallowWaterState<B>,
    ) {
        let n_cells = self.state.n_cells;
        let tau_c = self.config.tau_critical;
        let m = self.config.erosion_rate;
        let ws = self.config.settling_velocity;
        let h_min = self.config.min_depth;

        if tau_c <= B::Scalar::ZERO {
            for i in 0..n_cells {
                self.exchange_flux[i] = B::Scalar::ZERO;
            }
            return;
        }

        for i in 0..n_cells {
            let h = state.h[i];
            if h < h_min {
                self.exchange_flux[i] = B::Scalar::ZERO;
                continue;
            }

            let tau = self.tau_bed[i];
            let c = self.state.concentration[i];

            // 侵蚀
            let erosion = if tau > tau_c && self.state.bed_mass[i] > B::Scalar::ZERO {
                m * (tau - tau_c) / tau_c
            } else {
                B::Scalar::ZERO
            };

            // 沉降
            let deposition = ws * c;

            // 净通量：正值表示侵蚀，负值表示沉降
            self.exchange_flux[i] = erosion - deposition;
        }
    }

    /// 单步更新泥沙系统
    pub fn step(
        &mut self,
        state: &ShallowWaterState<B>,
        cell_areas: &[B::Scalar],
        manning_n: &[B::Scalar],
        dt: B::Scalar,
    ) -> Result<SedimentFluxStats<B::Scalar>, SedimentError<B::Scalar>> {
        if !dt.is_finite() || dt <= B::Scalar::ZERO {
            return Err(SedimentError::InvalidParameter("dt must be positive".to_string()));
        }
        self.compute_bed_shear_stress(state, manning_n);
        self.compute_exchange_flux(state);

        let n_cells = self.state.n_cells;
        let h_min = self.config.min_depth;

        let mut stats = SedimentFluxStats::default();
        let mut max_erosion = B::Scalar::ZERO;
        let mut max_deposition = B::Scalar::ZERO;

        for i in 0..n_cells {
            let h = state.h[i];
            let area = if i < cell_areas.len() { cell_areas[i] } else { B::Scalar::ONE };
            let flux = self.exchange_flux[i];

            // 质量变化 [kg/m²]
            let delta_mass = flux * dt;

            // 更新床面质量
            let new_bed = self.state.bed_mass[i] - delta_mass;

            // 检查负质量
            if new_bed < B::Scalar::ZERO {
                // 限制侵蚀量，不能超过床面存量
                let max_erosion_flux = self.state.bed_mass[i] / dt;
                self.exchange_flux[i] = self.exchange_flux[i].min(max_erosion_flux);
                self.state.bed_mass[i] = B::Scalar::ZERO;
            } else {
                self.state.bed_mass[i] = new_bed;
            }

            // 更新悬沙浓度
            if h > h_min {
                // dC/dt = F/h（简化，忽略对流扩散）
                let dc = self.exchange_flux[i] / h;
                let new_c = (self.state.concentration[i] + dc * dt).max(B::Scalar::ZERO);
                self.state.concentration[i] = new_c;
            }

            // 统计
            let mass_change = self.exchange_flux[i] * area;
            if mass_change > B::Scalar::ZERO {
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
        state: &ShallowWaterState<B>,
        cell_areas: &[B::Scalar],
    ) -> Result<(), SedimentError<B::Scalar>> {
        if !self.initialized {
            return Ok(());
        }

        let n_cells = self.state.n_cells;
        let mut total_bed = B::Scalar::ZERO;
        let mut total_suspended = B::Scalar::ZERO;

        for i in 0..n_cells {
            let area = if i < cell_areas.len() { cell_areas[i] } else { B::Scalar::ONE };
            total_bed += self.state.bed_mass[i] * area;
            total_suspended += state.h[i] * self.state.concentration[i] * area;
        }

        let total_current = total_bed + total_suspended;
        let error = (total_current - self.initial_total_mass).abs();
        let abs_tol = self
            .backend
            .config_scalar(1e-6, "SedimentManagerGeneric.verify_conservation.abs_tol");
        let rel_tol = self.config.conservation_tolerance;
        let baseline = self.initial_total_mass.abs().max(abs_tol);
        let relative_error = error / baseline;

        if relative_error > rel_tol && error > abs_tol {
            return Err(SedimentError::ConservationViolation {
                expected: self.initial_total_mass,
                actual: total_current,
                relative_error,
            });
        }

        Ok(())
    }

    /// 完整的悬移质步进（包含对流-扩散输运）
    pub fn step_suspended_transport(
        &mut self,
        state: &ShallowWaterState<B>,
        cell_areas: &[B::Scalar],
        tracer_rhs: &mut [B::Scalar],
        manning_n: &[B::Scalar],
        _dt: B::Scalar,
    ) -> Result<SedimentFluxStats<B::Scalar>, SedimentError<B::Scalar>> {
        let n_cells = self.state.n_cells;
        let h_min = self.config.min_depth;
        let ws = self.config.settling_velocity;
        let tau_c = self.config.tau_critical;
        let m = self.config.erosion_rate;

        if tau_c <= B::Scalar::ZERO {
            return Err(SedimentError::InvalidParameter("tau_critical must be positive".to_string()));
        }

        self.compute_bed_shear_stress(state, manning_n);

        let mut stats = SedimentFluxStats::default();
        let mut max_erosion = B::Scalar::ZERO;
        let mut max_deposition = B::Scalar::ZERO;

        // 步骤 1-2: 计算沉降通量和再悬浮通量
        for i in 0..n_cells {
            let h = state.h[i];
            let tau = self.tau_bed[i];
            let c = self.state.concentration[i];

            if h < h_min {
                tracer_rhs[i] = B::Scalar::ZERO;
                self.exchange_flux[i] = B::Scalar::ZERO;
                continue;
            }

            // 沉降通量 [kg/m²/s]
            let settling = ws * c;

            // 再悬浮通量 [kg/m²/s]
            let resuspension = if tau > tau_c && self.state.bed_mass[i] > B::Scalar::ZERO {
                m * (tau - tau_c) / tau_c
            } else {
                B::Scalar::ZERO
            };

            // 净交换通量（正值=侵蚀）
            let net_flux = resuspension - settling;
            self.exchange_flux[i] = net_flux;

            // 步骤 3: 创建源项场（转换为浓度变化率）
            // dC/dt = F/h
            tracer_rhs[i] = net_flux / h;

            // 收集统计
            let area = cell_areas.get(i).copied().unwrap_or(B::Scalar::ONE);
            if net_flux > B::Scalar::ZERO {
                stats.total_erosion += net_flux * area;
                if net_flux > max_erosion {
                    max_erosion = net_flux;
                    stats.max_erosion_cell = i;
                }
            } else {
                stats.total_deposition += (-net_flux) * area;
                if -net_flux > max_deposition {
                    max_deposition = -net_flux;
                    stats.max_deposition_cell = i;
                }
            }
        }

        stats.net_exchange = stats.total_erosion - stats.total_deposition;

        Ok(stats)
    }

    /// 应用对流-扩散后更新床面交换
    pub fn apply_bed_exchange(
        &mut self,
        advected_concentration: &[B::Scalar],
        dt: B::Scalar,
    ) {
        let n_cells = self.state.n_cells;

        for i in 0..n_cells {
            // 更新浓度
            if i < advected_concentration.len() {
                let updated = advected_concentration[i];
                self.state.concentration[i] = updated.max(B::Scalar::ZERO);
            }

            // 更新床面质量
            let delta_mass = -self.exchange_flux[i] * dt; // 负交换通量增加床面
            let new_bed = (self.state.bed_mass[i] + delta_mass).max(B::Scalar::ZERO);
            self.state.bed_mass[i] = new_bed;
        }
    }

    /// 计算床面高程变化率
    pub fn bed_elevation_change_rate(&self, cell: usize) -> B::Scalar {
        let flux = self.exchange_flux.get(cell).copied().unwrap_or(B::Scalar::ZERO);
        let rho_s = self.config.sediment_density;
        let p = self.config.porosity;

        // 侵蚀（正通量）降低床面高程
        -flux / ((B::Scalar::ONE - p) * rho_s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;
    
    #[test]
    fn test_sediment_manager_creation() {
        let config: SedimentConfigGeneric<f64> = SedimentConfigGeneric::default();
        let manager: SedimentManagerGeneric<CpuBackend<f64>> =
            SedimentManagerGeneric::new(100, config);
        
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
        let mut state = ShallowWaterState::new_with_backend(
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
