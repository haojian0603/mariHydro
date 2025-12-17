// crates/mh_physics/src/engine/timestep.rs

//! 时间步长控制模块
//!
//! 提供基于 CFL 条件的自适应时间步长控制。
//!
//! ## CFL 条件
//!
//! 时间步长需满足 CFL 条件：
//!
//! $$ \Delta t \leq C \cdot \min_i \frac{\Delta x_i}{|u_i| + \sqrt{gh_i}} $$
//!
//! 其中 $C$ 通常取 0.4-0.8（取决于空间格式阶数）。
//!
//! ## 特性
//!
//! - 预计算 dx_min，避免每步重复计算
//! - 并行波速计算使用原子操作
//! - 可选的自适应时间步长增长
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/engine/timestep.rs 迁移。

use crate::adapter::PhysicsMesh;
use crate::state::{ShallowWaterStateF64, ShallowWaterStateGeneric as ShallowWaterState};
use crate::types::NumericalParamsF64;
use mh_runtime::{Backend, CellIndex};
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// CFL 时间步计算器
///
/// 主要优化：预计算网格最小特征长度
// ALLOW_F64: Layer 4 时间步长配置
#[derive(Clone, Debug)]
pub struct CflCalculator<B: Backend> {
    /// 重力加速度
    g: f64, // ALLOW_F64: Layer 4 配置参数
    /// CFL 数
    cfl: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小时间步长
    dt_min: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最大时间步长
    dt_max: f64, // ALLOW_F64: Layer 4 配置参数
    /// 预计算的最小特征长度
    cached_dx_min: Option<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 最小波速阈值（低于此值视为静止）
    min_wave_speed: f64, // ALLOW_F64: Layer 4 配置参数
    /// 标记后端类型（占位以保持泛型签名）
    marker: PhantomData<B>,
}

impl<B: Backend> CflCalculator<B> {
    /// 创建计算器
    pub fn new(g: f64, params: &NumericalParamsF64) -> Self { // ALLOW_F64: 时间步长配置/物理参数
        Self {
            g,
            cfl: params.cfl,
            dt_min: params.dt_min,
            dt_max: params.dt_max,
            cached_dx_min: None,
            min_wave_speed: params.min_wave_speed,
            marker: PhantomData,
        }
    }

    /// 预计算网格最小特征长度
    ///
    /// 应在网格加载后调用一次
    pub fn precompute_dx_min(&mut self, mesh: &PhysicsMesh) {
        self.cached_dx_min = Some(self.compute_min_char_length(mesh));
    }

    /// 获取缓存的 dx_min
    pub fn dx_min(&self) -> Option<f64> {
        self.cached_dx_min
    }

    /// 计算时间步长
    pub fn compute_dt(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        params: &NumericalParamsF64,
    ) -> f64 {
        let n_cells = mesh.n_cells();
        if n_cells == 0 {
            return self.dt_max;
        }

        // 使用预计算的 dx_min 或现场计算
        let min_length = self
            .cached_dx_min
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
    pub fn compute_from_max_speed(&self, max_speed: f64) -> f64 { // ALLOW_F64: 时间步长配置/物理参数
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
        state: &ShallowWaterState<B>,
        params: &NumericalParamsF64,
    ) -> f64 {
        let n = state.h.len();
        if n == 0 {
            return 0.0;
        }

        // 使用原子操作收集最大值
        let max_speed = AtomicU64::new(0u64);

        (0..n).into_par_iter().for_each(|i| {
            // 将 B::Scalar 转换为 f64 进行计算
            let h = state.h[i].to_f64().unwrap_or(0.0);
            if params.is_dry(h) {
                return;
            }

            let hu = state.hu[i].to_f64().unwrap_or(0.0);
            let hv = state.hv[i].to_f64().unwrap_or(0.0);
            let (u, v) = params.safe_velocity_components(hu, hv, h);
            let speed = (u * u + v * v).sqrt();
            let c = (self.g * h).sqrt();
            let wave_speed = speed + c;

            // 原子更新最大值
            let bits = wave_speed.to_bits();
            max_speed.fetch_max(bits, Ordering::Relaxed);
        });

        f64::from_bits(max_speed.load(Ordering::Relaxed))
    }

    /// 计算最小特征长度
    fn compute_min_char_length(&self, mesh: &PhysicsMesh) -> f64 {
        let n = mesh.n_cells();
        if n == 0 {
            return f64::MAX;
        }

        // 使用原子操作收集最小值
        let min_dx = AtomicU64::new(f64::MAX.to_bits());

        (0..n).into_par_iter().for_each(|i| {
            let area = mesh.cell_area(CellIndex(i)).unwrap_or(0.0);
            let perimeter = mesh.cell_perimeter(CellIndex(i)).unwrap_or(0.0);

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

/// 时间步长控制器
///
/// 提供自适应时间步长控制，支持：
/// - 预计算 dx_min
/// - 自适应增长/收缩因子
/// - 时间步长历史追踪
// ALLOW_F64: Layer 4 时间步长配置
pub struct TimeStepController<B: Backend> {
    calculator: CflCalculator<B>,
    /// 当前时间步长
    current_dt: f64, // ALLOW_F64: Layer 4 配置参数
    /// 增长因子
    growth_factor: f64, // ALLOW_F64: Layer 4 配置参数
    /// 收缩因子
    shrink_factor: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最大允许增长因子
    max_growth_factor: f64, // ALLOW_F64: Layer 4 配置参数
    /// 连续稳定步数
    stable_steps: usize,
    /// 稳定增长阈值
    stable_growth_threshold: usize,
    /// 是否启用自适应增长
    adaptive_growth: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> TimeStepController<B> {
    /// 创建控制器
    pub fn new(g: f64, params: &NumericalParamsF64) -> Self {
        Self {
            calculator: CflCalculator::new(g, params),
            current_dt: params.dt_max,
            growth_factor: 1.1,
            shrink_factor: 0.5,
            max_growth_factor: 1.5,
            stable_steps: 0,
            stable_growth_threshold: 10,
            adaptive_growth: true,
            marker: PhantomData,
        }
    }

    /// 预计算网格特征
    pub fn precompute_mesh_characteristics(&mut self, mesh: &PhysicsMesh) {
        self.calculator.precompute_dx_min(mesh);
    }

    /// 获取预计算的 dx_min
    pub fn dx_min(&self) -> Option<f64> {
        self.calculator.dx_min()
    }

    /// 更新时间步长
    pub fn update(
        &mut self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        params: &NumericalParamsF64,
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
    pub fn force_shrink(&mut self, factor: f64) { // ALLOW_F64: 时间步长配置/物理参数
        self.current_dt *= factor;
        self.current_dt = self.current_dt.max(self.calculator.dt_min);
        self.stable_steps = 0;
    }

    /// 获取当前时间步长
    pub fn current_dt(&self) -> f64 {
        self.current_dt
    }

    /// 设置时间步长（手动覆盖）
    pub fn set_dt(&mut self, dt: f64) { // ALLOW_F64: 时间步长配置/物理参数
        self.current_dt = dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.stable_steps = 0;
    }

    /// 设置增长因子
    pub fn set_growth_factor(&mut self, factor: f64) { // ALLOW_F64: 时间步长配置/物理参数
        self.growth_factor = factor.max(1.0);
    }

    /// 设置收缩因子
    pub fn set_shrink_factor(&mut self, factor: f64) { // ALLOW_F64: 时间步长配置/物理参数
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

    /// 半隐式方法迭代次数自适应
    ///
    /// 根据压力求解器迭代次数调整时间步长。
    ///
    /// # 参数
    ///
    /// - `iterations`: 实际迭代次数
    /// - `target_iterations`: 目标迭代次数（通常为求解器最大迭代的 50%）
    pub fn adapt_from_iterations(
        &mut self,
        iterations: usize,
        target_iterations: usize,
    ) -> f64 {
        let ratio = iterations as f64 / target_iterations.max(1) as f64; // ALLOW_F64: 时间步长配置/物理参数

        if ratio < 0.5 {
            // 收敛太快，可以增大时间步长
            let growth = (1.0 + (1.0 - ratio * 2.0) * 0.2).min(self.max_growth_factor);
            self.current_dt *= growth;
            self.stable_steps += 1;
        } else if ratio > 1.5 {
            // 收敛太慢，减小时间步长
            let shrink = (1.0 - (ratio - 1.5) * 0.3).max(0.5);
            self.current_dt *= shrink;
            self.stable_steps = 0;
        } else if ratio > 1.0 {
            // 接近边界，保守增长
            self.stable_steps = self.stable_steps.saturating_sub(1);
        }

        self.current_dt = self.current_dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.current_dt
    }

    /// 应用源项稳定性限制
    ///
    /// 将所有源项的稳定性限制应用于时间步长。
    ///
    /// # 参数
    ///
    /// - `limits`: 各源项返回的稳定性限制时间步长
    pub fn apply_source_limits(&mut self, limits: &[Option<f64>]) -> f64 {
        let mut min_dt = self.current_dt;

        for &limit in limits {
            if let Some(dt_limit) = limit {
                min_dt = min_dt.min(dt_limit);
            }
        }

        if min_dt < self.current_dt * 0.9 {
            self.stable_steps = 0;
        }

        self.current_dt = min_dt.clamp(self.calculator.dt_min, self.calculator.dt_max);
        self.current_dt
    }

    /// 计算科氏力稳定性限制
    ///
    /// 返回 dt < 2π / |f| 以保证惯性振荡稳定
    pub fn coriolis_stability_limit(&self, f: f64) -> Option<f64> { // ALLOW_F64: 时间步长配置/物理参数
        if f.abs() < 1e-14 {
            None
        } else {
            Some(std::f64::consts::PI / f.abs()) // ALLOW_F64: 时间步长配置/物理参数
        }
    }

    /// 计算摩擦稳定性限制
    ///
    /// 对于曼宁公式的隐式摩擦
    pub fn friction_stability_limit(&self, max_cf: f64) -> Option<f64> { // ALLOW_F64: 时间步长配置/物理参数
        if max_cf < 1e-14 {
            None
        } else {
            // 显式稳定性限制
            Some(2.0 / max_cf)
        }
    }

    /// 获取 CFL 数
    pub fn cfl(&self) -> f64 {
        self.calculator.cfl
    }

    /// 设置 CFL 数
    pub fn set_cfl(&mut self, cfl: f64) { // ALLOW_F64: 时间步长配置/物理参数
        self.calculator.cfl = cfl.clamp(0.1, 1.0);
    }
}

/// 时间步长统计
// ALLOW_F64: Layer 4 时间步长配置
#[derive(Clone, Debug)]
pub struct TimeStepStats {
    /// 当前时间步长
    pub current_dt: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小特征长度
    pub dx_min: Option<f64>, // ALLOW_F64: Layer 4 配置参数
    /// 连续稳定步数
    pub stable_steps: usize,
    /// 是否启用自适应增长
    pub adaptive_growth_enabled: bool,
}

/// 时间步长控制器构建器
// ALLOW_F64: Layer 4 时间步长配置
pub struct TimeStepControllerBuilder<B: Backend> {
    g: f64, // ALLOW_F64: Layer 4 配置参数
    cfl: f64, // ALLOW_F64: Layer 4 配置参数
    dt_min: f64, // ALLOW_F64: Layer 4 配置参数
    dt_max: f64, // ALLOW_F64: Layer 4 配置参数
    growth_factor: f64, // ALLOW_F64: Layer 4 配置参数
    shrink_factor: f64, // ALLOW_F64: Layer 4 配置参数
    adaptive_growth: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> TimeStepControllerBuilder<B> {
    /// 创建构建器
    pub fn new(g: f64) -> Self { // ALLOW_F64: 时间步长配置/物理参数
        Self {
            g,
            cfl: 0.5,
            dt_min: 1e-6,
            dt_max: 1.0,
            growth_factor: 1.1,
            shrink_factor: 0.5,
            adaptive_growth: true,
            marker: PhantomData,
        }
    }

    /// 设置 CFL 数
    pub fn with_cfl(mut self, cfl: f64) -> Self { // ALLOW_F64: 时间步长配置/物理参数
        self.cfl = cfl;
        self
    }

    /// 设置时间步限制
    pub fn with_dt_limits(mut self, dt_min: f64, dt_max: f64) -> Self { // ALLOW_F64: 时间步长配置/物理参数
        self.dt_min = dt_min;
        self.dt_max = dt_max;
        self
    }

    /// 设置增长因子
    pub fn with_growth_factor(mut self, factor: f64) -> Self { // ALLOW_F64: 时间步长配置/物理参数
        self.growth_factor = factor;
        self
    }

    /// 设置收缩因子
    pub fn with_shrink_factor(mut self, factor: f64) -> Self { // ALLOW_F64: 时间步长配置/物理参数
        self.shrink_factor = factor;
        self
    }

    /// 设置自适应增长
    pub fn with_adaptive_growth(mut self, enabled: bool) -> Self {
        self.adaptive_growth = enabled;
        self
    }

    /// 构建控制器
    pub fn build(self) -> TimeStepController<B> {
        let params = NumericalParamsF64 {
            cfl: self.cfl,
            dt_min: self.dt_min,
            dt_max: self.dt_max,
            ..Default::default()
        };

        let mut controller = TimeStepController::<B>::new(self.g, &params);
        controller.growth_factor = self.growth_factor;
        controller.shrink_factor = self.shrink_factor;
        controller.adaptive_growth = self.adaptive_growth;

        controller
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_cfl_calculator_creation() {
        let params = NumericalParamsF64::default();
        let calc: CflCalculator<CpuBackend<f64>> = CflCalculator::new(9.81, &params);
        assert!(calc.cached_dx_min.is_none());
        assert!((calc.g - 9.81).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculator_from_max_speed() {
        let params = NumericalParamsF64::default();
        let mut calc: CflCalculator<CpuBackend<f64>> = CflCalculator::new(9.81, &params);
        calc.cached_dx_min = Some(1.0);

        let dt = calc.compute_from_max_speed(10.0);
        // dt = cfl * dx_min / max_speed = 0.5 * 1.0 / 10.0 = 0.05
        assert!((dt - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculator_static_water() {
        let params = NumericalParamsF64::default();
        let calc: CflCalculator<CpuBackend<f64>> = CflCalculator::new(9.81, &params);

        // 静水时，max_speed < min_wave_speed，返回dt_max
        let dt = calc.compute_from_max_speed(1e-10);
        assert!((dt - params.dt_max).abs() < 1e-10);
    }

    #[test]
    fn test_controller_creation() {
        let params = NumericalParamsF64::default();
        let controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(9.81, &params);
        assert!(controller.adaptive_growth);
        assert_eq!(controller.stable_steps, 0);
    }

    #[test]
    fn test_controller_adaptive_growth() {
        let params = NumericalParamsF64::default();
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(9.81, &params);

        // 模拟稳定步
        for _ in 0..15 {
            controller.stable_steps += 1;
        }

        let growth = controller.compute_adaptive_growth();
        assert!(growth >= controller.growth_factor);
    }

    #[test]
    fn test_controller_shrink() {
        let params = NumericalParamsF64::default();
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(9.81, &params);
        controller.current_dt = 0.1;

        controller.shrink();
        assert!(controller.current_dt < 0.1);
        assert_eq!(controller.stable_steps, 0);
    }

    #[test]
    fn test_controller_force_shrink() {
        let params = NumericalParamsF64::default();
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(9.81, &params);
        controller.current_dt = 0.1;

        controller.force_shrink(0.1);
        assert!((controller.current_dt - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_builder() {
        let controller: TimeStepController<CpuBackend<f64>> = TimeStepControllerBuilder::new(9.81)
            .with_cfl(0.3)
            .with_dt_limits(1e-8, 0.5)
            .with_adaptive_growth(false)
            .build();

        assert!(!controller.adaptive_growth);
    }

    #[test]
    fn test_stats() {
        let params = NumericalParamsF64::default();
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(9.81, &params);
        controller.stable_steps = 5;

        let stats = controller.stats();
        assert_eq!(stats.stable_steps, 5);
        assert!(stats.adaptive_growth_enabled);
    }

    #[test]
    fn test_set_dt() {
        let params = NumericalParamsF64::default();
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(9.81, &params);
        controller.set_dt(0.5);
        assert!((controller.current_dt() - 0.5).abs() < 1e-10);
        assert_eq!(controller.stable_steps, 0);
    }
}
