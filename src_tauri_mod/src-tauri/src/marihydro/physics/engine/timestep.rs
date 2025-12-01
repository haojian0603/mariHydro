// src-tauri/src/marihydro/physics/engine/timestep.rs
//! 优化版时间步长控制模块
//! 
//! 性能改进：
//! - 预计算 dx_min，避免每步重复计算
//! - 并行波速计算使用原子操作
//! - 可选的自适应时间步长增长
//! - 缓存优化

use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use crate::marihydro::domain::state::ShallowWaterState;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// 优化版CFL计算器
/// 
/// 主要优化：预计算网格最小特征长度
#[derive(Clone, Debug)]
pub struct OptimizedCflCalculator {
    g: f64,
    cfl: f64,
    dt_min: f64,
    dt_max: f64,
    /// 预计算的最小特征长度
    cached_dx_min: Option<f64>,
    /// 最小波速阈值（低于此值视为静止）
    min_wave_speed: f64,
}

impl OptimizedCflCalculator {
    /// 创建计算器（不含缓存）
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self {
            g,
            cfl: params.cfl,
            dt_min: params.dt_min,
            dt_max: params.dt_max,
            cached_dx_min: None,
            min_wave_speed: params.min_wave_speed,
        }
    }
    
    /// 预计算网格最小特征长度
    /// 
    /// 应在网格加载后调用一次
    pub fn precompute_dx_min<M: MeshAccess + Sync>(&mut self, mesh: &M) {
        self.cached_dx_min = Some(self.compute_min_char_length(mesh));
    }
    
    /// 获取缓存的 dx_min
    pub fn dx_min(&self) -> Option<f64> {
        self.cached_dx_min
    }
    
    /// 计算时间步长
    pub fn compute_dt<M: MeshAccess + Sync>(
        &self,
        state: &ShallowWaterState,
        mesh: &M,
        params: &NumericalParams,
    ) -> f64 {
        let n_cells = mesh.n_cells();
        if n_cells == 0 {
            return self.dt_max;
        }
        
        // 使用预计算的 dx_min 或现场计算
        let min_length = self.cached_dx_min
            .unwrap_or_else(|| self.compute_min_char_length(mesh));
        
        // 并行计算最大波速
        let max_speed = self.compute_max_wave_speed_parallel(state, params);
        
        if max_speed < self.min_wave_speed {
            return self.dt_max;
        }
        
        let dt = self.cfl * min_length / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }
    
    /// 从已知最大波速计算时间步长
    /// 
    /// 当通量计算已得到最大波速时使用此方法，避免重复计算
    pub fn compute_from_max_speed(&self, max_speed: f64) -> f64 {
        let min_length = self.cached_dx_min.unwrap_or(1.0);
        
        if max_speed < self.min_wave_speed {
            return self.dt_max;
        }
        
        let dt = self.cfl * min_length / max_speed;
        dt.clamp(self.dt_min, self.dt_max)
    }
    
    /// 并行计算最大波速（使用原子操作）
    fn compute_max_wave_speed_parallel(
        &self,
        state: &ShallowWaterState,
        params: &NumericalParams,
    ) -> f64 {
        let n = state.h.len();
        if n == 0 {
            return 0.0;
        }
        
        // 使用原子操作收集最大值
        let max_speed = AtomicU64::new(0u64);
        
        (0..n).into_par_iter().for_each(|i| {
            let h = state.h[i];
            if params.is_dry(h) {
                return;
            }
            
            let vel = params.safe_velocity(state.hu[i], state.hv[i], h);
            let speed = (vel.u * vel.u + vel.v * vel.v).sqrt();
            let c = (self.g * h).sqrt();
            let wave_speed = speed + c;
            
            // 原子更新最大值
            let bits = wave_speed.to_bits();
            max_speed.fetch_max(bits, Ordering::Relaxed);
        });
        
        f64::from_bits(max_speed.load(Ordering::Relaxed))
    }
    
    /// 计算最小特征长度
    fn compute_min_char_length<M: MeshAccess + Sync>(&self, mesh: &M) -> f64 {
        let n = mesh.n_cells();
        if n == 0 {
            return f64::MAX;
        }
        
        // 使用原子操作收集最小值
        let min_dx = AtomicU64::new(f64::MAX.to_bits());
        
        (0..n).into_par_iter().for_each(|i| {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            let faces = mesh.cell_faces(cell);
            
            let perimeter: f64 = faces.iter()
                .map(|&f| mesh.face_length(f))
                .sum();
            
            if perimeter < 1e-14 {
                return;
            }
            
            // 水力直径近似
            let dx = 2.0 * area / perimeter;
            
            // 原子更新最小值
            let bits = dx.to_bits();
            min_dx.fetch_min(bits, Ordering::Relaxed);
        });
        
        f64::from_bits(min_dx.load(Ordering::Relaxed))
    }
}

/// 优化版时间步长控制器
/// 
/// 特性：
/// - 预计算 dx_min
/// - 自适应增长/收缩因子
/// - 时间步长历史追踪
pub struct OptimizedTimeStepController {
    calculator: OptimizedCflCalculator,
    /// 当前时间步长
    current_dt: f64,
    /// 增长因子
    growth_factor: f64,
    /// 收缩因子
    shrink_factor: f64,
    /// 最大允许增长因子
    max_growth_factor: f64,
    /// 连续稳定步数
    stable_steps: usize,
    /// 稳定增长阈值
    stable_growth_threshold: usize,
    /// 是否启用自适应增长
    adaptive_growth: bool,
}

impl OptimizedTimeStepController {
    /// 创建控制器
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self {
            calculator: OptimizedCflCalculator::new(g, params),
            current_dt: params.dt_max,
            growth_factor: 1.1,
            shrink_factor: 0.5,
            max_growth_factor: 1.5,
            stable_steps: 0,
            stable_growth_threshold: 10,
            adaptive_growth: true,
        }
    }
    
    /// 预计算网格特征
    pub fn precompute_mesh_characteristics<M: MeshAccess + Sync>(&mut self, mesh: &M) {
        self.calculator.precompute_dx_min(mesh);
    }
    
    /// 获取预计算的 dx_min
    pub fn dx_min(&self) -> Option<f64> {
        self.calculator.dx_min()
    }
    
    /// 更新时间步长
    pub fn update<M: MeshAccess + Sync>(
        &mut self,
        state: &ShallowWaterState,
        mesh: &M,
        params: &NumericalParams,
    ) -> f64 {
        let suggested = self.calculator.compute_dt(state, mesh, params);
        
        // 计算增长因子
        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };
        
        let grown = self.current_dt * growth;
        let new_dt = suggested.min(grown);
        
        // 更新稳定步数
        if new_dt >= self.current_dt * 0.95 {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }
        
        self.current_dt = new_dt;
        self.current_dt
    }
    
    /// 从已知最大波速更新时间步长
    pub fn update_from_max_speed(&mut self, max_speed: f64) -> f64 {
        let suggested = self.calculator.compute_from_max_speed(max_speed);
        
        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };
        
        let grown = self.current_dt * growth;
        let new_dt = suggested.min(grown);
        
        if new_dt >= self.current_dt * 0.95 {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }
        
        self.current_dt = new_dt;
        self.current_dt
    }
    
    /// 计算自适应增长因子
    fn compute_adaptive_growth(&self) -> f64 {
        if self.stable_steps >= self.stable_growth_threshold {
            // 长期稳定，允许更大增长
            self.growth_factor.min(self.max_growth_factor)
        } else if self.stable_steps >= self.stable_growth_threshold / 2 {
            // 中等稳定
            self.growth_factor
        } else {
            // 不稳定，保守增长
            1.0 + (self.growth_factor - 1.0) * 0.5
        }
    }
    
    /// 收缩时间步长（遇到问题时调用）
    pub fn shrink(&mut self) {
        self.current_dt *= self.shrink_factor;
        self.current_dt = self.current_dt.max(self.calculator.dt_min);
        self.stable_steps = 0;
    }
    
    /// 强制收缩（严重问题时）
    pub fn force_shrink(&mut self, factor: f64) {
        self.current_dt *= factor;
        self.current_dt = self.current_dt.max(self.calculator.dt_min);
        self.stable_steps = 0;
    }
    
    /// 获取当前时间步长
    pub fn current_dt(&self) -> f64 {
        self.current_dt
    }
    
    /// 设置时间步长（手动覆盖）
    pub fn set_dt(&mut self, dt: f64) {
        self.current_dt = dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.stable_steps = 0;
    }
    
    /// 设置增长因子
    pub fn set_growth_factor(&mut self, factor: f64) {
        self.growth_factor = factor.max(1.0);
    }
    
    /// 设置收缩因子
    pub fn set_shrink_factor(&mut self, factor: f64) {
        self.shrink_factor = factor.clamp(0.1, 0.9);
    }
    
    /// 启用/禁用自适应增长
    pub fn set_adaptive_growth(&mut self, enabled: bool) {
        self.adaptive_growth = enabled;
    }
    
    /// 获取统计信息
    pub fn stats(&self) -> TimeStepStats {
        TimeStepStats {
            current_dt: self.current_dt,
            dx_min: self.calculator.cached_dx_min,
            stable_steps: self.stable_steps,
            adaptive_growth_enabled: self.adaptive_growth,
        }
    }
}

/// 时间步长统计
#[derive(Clone, Debug)]
pub struct TimeStepStats {
    pub current_dt: f64,
    pub dx_min: Option<f64>,
    pub stable_steps: usize,
    pub adaptive_growth_enabled: bool,
}

/// 构建器
pub struct TimeStepControllerBuilder {
    g: f64,
    cfl: f64,
    dt_min: f64,
    dt_max: f64,
    growth_factor: f64,
    shrink_factor: f64,
    adaptive_growth: bool,
}

impl TimeStepControllerBuilder {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            cfl: 0.5,
            dt_min: 1e-6,
            dt_max: 1.0,
            growth_factor: 1.1,
            shrink_factor: 0.5,
            adaptive_growth: true,
        }
    }
    
    pub fn with_cfl(mut self, cfl: f64) -> Self {
        self.cfl = cfl;
        self
    }
    
    pub fn with_dt_limits(mut self, dt_min: f64, dt_max: f64) -> Self {
        self.dt_min = dt_min;
        self.dt_max = dt_max;
        self
    }
    
    pub fn with_growth_factor(mut self, factor: f64) -> Self {
        self.growth_factor = factor;
        self
    }
    
    pub fn with_shrink_factor(mut self, factor: f64) -> Self {
        self.shrink_factor = factor;
        self
    }
    
    pub fn with_adaptive_growth(mut self, enabled: bool) -> Self {
        self.adaptive_growth = enabled;
        self
    }
    
    pub fn build(self) -> OptimizedTimeStepController {
        let params = NumericalParams {
            cfl: self.cfl,
            dt_min: self.dt_min,
            dt_max: self.dt_max,
            ..Default::default()
        };
        
        let mut controller = OptimizedTimeStepController::new(self.g, &params);
        controller.growth_factor = self.growth_factor;
        controller.shrink_factor = self.shrink_factor;
        controller.adaptive_growth = self.adaptive_growth;
        
        controller
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculator_creation() {
        let params = NumericalParams::default();
        let calc = OptimizedCflCalculator::new(9.81, &params);
        assert!(calc.cached_dx_min.is_none());
    }
    
    #[test]
    fn test_controller_adaptive_growth() {
        let params = NumericalParams::default();
        let mut controller = OptimizedTimeStepController::new(9.81, &params);
        
        // 模拟稳定步
        for _ in 0..15 {
            controller.stable_steps += 1;
        }
        
        let growth = controller.compute_adaptive_growth();
        assert!(growth >= controller.growth_factor);
    }
    
    #[test]
    fn test_shrink() {
        let params = NumericalParams::default();
        let mut controller = OptimizedTimeStepController::new(9.81, &params);
        controller.current_dt = 0.1;
        
        controller.shrink();
        assert!(controller.current_dt < 0.1);
        assert_eq!(controller.stable_steps, 0);
    }
    
    #[test]
    fn test_builder() {
        let controller = TimeStepControllerBuilder::new(9.81)
            .with_cfl(0.3)
            .with_dt_limits(1e-8, 0.5)
            .with_adaptive_growth(false)
            .build();
        
        assert!(!controller.adaptive_growth);
    }
}
