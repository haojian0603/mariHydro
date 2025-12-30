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
//! # 架构改造
//!
//! **Phase 3 泛型化**：完全支持 `CpuBackend<f32>` 和 `CpuBackend<f64>`。

use crate::adapter::PhysicsMesh;
use crate::state::ShallowWaterStateGeneric as ShallowWaterState;
use crate::types::{NumericalParams};
use mh_runtime::{Backend, CellIndex, RuntimeScalar};
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

/// CFL 时间步计算器（Backend 泛型化）
///
/// 主要优化：预计算网格最小特征长度
#[derive(Clone, Debug)]
pub struct CflCalculator<B: Backend> {
    /// 重力加速度（Backend 标量）
    g: B::Scalar,
    /// CFL 数（Backend 标量）
    cfl: B::Scalar,
    /// 最小时间步长（Backend 标量）
    dt_min: B::Scalar,
    /// 最大时间步长（Backend 标量）
    dt_max: B::Scalar,
    /// 预计算的最小特征长度（Backend 标量）
    cached_dx_min: Option<B::Scalar>,
    /// 最小波速阈值（Backend 标量）
    min_wave_speed: B::Scalar,
    /// 标记后端类型
    marker: PhantomData<B>,
}

impl<B: Backend> CflCalculator<B>
where
    B::Scalar: RuntimeScalar + FromPrimitive + ToPrimitive,
{
    /// 创建计算器
    pub fn new(g: B::Scalar, params: &NumericalParams<B::Scalar>) -> Self {
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
    pub fn precompute_dx_min(&mut self, mesh: &PhysicsMesh) {
        self.cached_dx_min = Some(self.compute_min_char_length(mesh));
    }

    /// 获取缓存的 dx_min
    pub fn dx_min(&self) -> Option<B::Scalar> {
        self.cached_dx_min
    }

    /// 计算时间步长
    pub fn compute_dt(
        &self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        params: &NumericalParams<B::Scalar>,
    ) -> B::Scalar {
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
        dt.min(self.dt_max).max(self.dt_min)
    }

    /// 从已知最大波速计算时间步长
    pub fn compute_from_max_speed(&self, max_speed: B::Scalar) -> B::Scalar {
        let min_length = self.cached_dx_min.unwrap_or_else(|| {
            B::Scalar::from_f64(1.0).unwrap_or(B::Scalar::ONE)
        });

        if max_speed < self.min_wave_speed {
            return self.dt_max;
        }

        let dt = self.cfl * min_length / max_speed;
        dt.min(self.dt_max).max(self.dt_min)
    }

    /// 并行计算最大波速（使用原子操作）
    fn compute_max_wave_speed_parallel(
        &self,
        state: &ShallowWaterState<B>,
        params: &NumericalParams<B::Scalar>,
    ) -> B::Scalar {
        let n = state.h.len();
        if n == 0 {
            return B::Scalar::ZERO;
        }

        // 使用原子操作收集最大值
        let max_speed = AtomicU64::new(0u64);

        (0..n).into_par_iter().for_each(|i| {
            let h = state.h[i];
            if params.is_dry(h) {
                return;
            }

            let hu = state.hu[i];
            let hv = state.hv[i];
            let (u, v) = params.safe_velocity_components(hu, hv, h);
            let speed = (u * u + v * v).sqrt();
            let c = (self.g * h.max(B::Scalar::ZERO)).sqrt();
            let wave_speed = speed + c;

            // 原子更新最大值
            if let Some(bits) = wave_speed.to_f64().map(|f| f.to_bits()) {
                max_speed.fetch_max(bits, Ordering::Relaxed);
            }
        });

        B::Scalar::from_f64(f64::from_bits(max_speed.load(Ordering::Relaxed)))
            .unwrap_or(B::Scalar::ZERO)
    }

    /// 计算最小特征长度
    fn compute_min_char_length(&self, mesh: &PhysicsMesh) -> B::Scalar {
        let n = mesh.n_cells();
        if n == 0 {
            return B::Scalar::from_f64(f64::MAX).unwrap_or(B::Scalar::MAX);
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

        B::Scalar::from_f64(f64::from_bits(min_dx.load(Ordering::Relaxed)))
            .unwrap_or(B::Scalar::ONE)
    }
}

/// 时间步长控制器（Backend 泛型化）
#[derive(Debug)]
pub struct TimeStepController<B: Backend> {
    calculator: CflCalculator<B>,
    /// 当前时间步长（Backend 标量）
    current_dt: B::Scalar,
    /// 增长因子（Backend 标量）
    growth_factor: B::Scalar,
    /// 收缩因子（Backend 标量）
    shrink_factor: B::Scalar,
    /// 最大允许增长因子（Backend 标量）
    max_growth_factor: B::Scalar,
    /// 连续稳定步数
    stable_steps: usize,
    /// 稳定增长阈值
    stable_growth_threshold: usize,
    /// 是否启用自适应增长
    adaptive_growth: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> TimeStepController<B>
where
    B::Scalar: RuntimeScalar + FromPrimitive + ToPrimitive,
{
    /// 创建控制器
    pub fn new(gravity: B::Scalar, params: &NumericalParams<B::Scalar>) -> Self {
        Self {
            calculator: CflCalculator::new(gravity, params),
            current_dt: params.dt_max,
            growth_factor: B::Scalar::from_f64(1.1).unwrap_or(B::Scalar::ONE),
            shrink_factor: B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ONE),
            max_growth_factor: B::Scalar::from_f64(1.5).unwrap_or(B::Scalar::ONE),
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
    pub fn dx_min(&self) -> Option<B::Scalar> {
        self.calculator.dx_min()
    }

    /// 更新时间步长（修复类型不匹配）
    pub fn update(
        &mut self,
        state: &ShallowWaterState<B>,
        mesh: &PhysicsMesh,
        params: &NumericalParams<B::Scalar>,
    ) -> B::Scalar {
        let suggested = self.calculator.compute_dt(state, mesh, params);

        // 计算增长因子
        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };

        let grown = self.current_dt * growth;
        let new_dt = if suggested < grown { suggested } else { grown };

        // 更新稳定步数
        let threshold = B::Scalar::from_f64(0.95).unwrap_or(B::Scalar::ONE);
        if new_dt >= self.current_dt * threshold {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }

        self.current_dt = new_dt;
        self.current_dt
    }

    /// 从已知最大波速更新时间步长
    pub fn update_from_max_speed(&mut self, max_speed: B::Scalar) -> B::Scalar {
        let suggested = self.calculator.compute_from_max_speed(max_speed);

        let growth = if self.adaptive_growth {
            self.compute_adaptive_growth()
        } else {
            self.growth_factor
        };

        let grown = self.current_dt * growth;
        let new_dt = if suggested < grown { suggested } else { grown };

        let threshold = B::Scalar::from_f64(0.95).unwrap_or(B::Scalar::ONE);
        if new_dt >= self.current_dt * threshold {
            self.stable_steps += 1;
        } else {
            self.stable_steps = 0;
        }

        self.current_dt = new_dt;
        self.current_dt
    }

    /// 计算自适应增长因子
    fn compute_adaptive_growth(&self) -> B::Scalar {
        if self.stable_steps >= self.stable_growth_threshold {
            // 长期稳定，允许更大增长
            self.growth_factor.min(self.max_growth_factor)
        } else if self.stable_steps >= self.stable_growth_threshold / 2 {
            self.growth_factor
        } else {
            // 不稳定，保守增长
            let one = B::Scalar::ONE;
            let half = B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ZERO);
            one + (self.growth_factor - one) * half
        }
    }

    /// 收缩时间步长
    pub fn shrink(&mut self) {
        self.current_dt = self.current_dt * self.shrink_factor;
        let dt_min = self.calculator.dt_min;
        if self.current_dt < dt_min {
            self.current_dt = dt_min;
        }
        self.stable_steps = 0;
    }

    /// 强制收缩
    pub fn force_shrink(&mut self, factor: B::Scalar) {
        self.current_dt = self.current_dt * factor;
        let dt_min = self.calculator.dt_min;
        if self.current_dt < dt_min {
            self.current_dt = dt_min;
        }
        self.stable_steps = 0;
    }

    /// 获取当前时间步长
    pub fn current_dt(&self) -> B::Scalar {
        self.current_dt
    }

    /// 设置时间步长
    pub fn set_dt(&mut self, dt: B::Scalar) {
        let dt_min = self.calculator.dt_min;
        let dt_max = self.calculator.dt_max;
        self.current_dt = if dt < dt_min { dt_min } else if dt > dt_max { dt_max } else { dt };
        self.stable_steps = 0;
    }

    /// 设置增长因子
    pub fn set_growth_factor(&mut self, factor: B::Scalar) {
        let one = B::Scalar::ONE;
        self.growth_factor = if factor > one { factor } else { one };
    }

    /// 设置收缩因子
    pub fn set_shrink_factor(&mut self, factor: B::Scalar) {
        let zero = B::Scalar::ZERO;
        let one = B::Scalar::ONE;
        self.shrink_factor = if factor < zero { zero } else if factor > one { one } else { factor };
    }

    /// 启用/禁用自适应增长
    pub fn set_adaptive_growth(&mut self, enabled: bool) {
        self.adaptive_growth = enabled;
    }

    /// 半隐式方法迭代次数自适应
    pub fn adapt_from_iterations(&mut self, iterations: usize, target_iterations: usize) -> B::Scalar {
        let ratio = B::Scalar::from_usize(iterations).unwrap_or(B::Scalar::ZERO)
            / B::Scalar::from_usize(target_iterations.max(1)).unwrap_or(B::Scalar::ONE);

        let _zero = B::Scalar::ZERO;
        let one = B::Scalar::ONE;
        let two = B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::ONE);

        if ratio < one / two {
            // 收敛太快，增大时间步长
            let growth = (one + (one - ratio * two) * B::Scalar::from_f64(0.2).unwrap_or(B::Scalar::ZERO))
                .min(self.max_growth_factor);
            self.current_dt = self.current_dt * growth;
            self.stable_steps += 1;
        } else if ratio > one + one / two {
            // 收敛太慢，减小时间步长
            let shrink = (one - (ratio - one - one / two) * B::Scalar::from_f64(0.3).unwrap_or(B::Scalar::ZERO))
                .max(B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ZERO));
            self.current_dt = self.current_dt * shrink;
            self.stable_steps = 0;
        } else if ratio > one {
            // 接近边界，保守增长
            self.stable_steps = self.stable_steps.saturating_sub(1);
        }

        let dt_min = self.calculator.dt_min;
        let dt_max = self.calculator.dt_max;
        self.current_dt = self.current_dt.min(dt_max).max(dt_min);
        self.current_dt
    }

    /// 应用源项稳定性限制
    pub fn apply_source_limits(&mut self, limits: &[Option<B::Scalar>]) -> B::Scalar {
        let mut min_dt = self.current_dt;

        for &limit in limits {
            if let Some(dt_limit) = limit {
                min_dt = min_dt.min(dt_limit);
            }
        }

        let threshold = B::Scalar::from_f64(0.9).unwrap_or(B::Scalar::ONE);
        if min_dt < self.current_dt * threshold {
            self.stable_steps = 0;
        }

        let dt_min = self.calculator.dt_min;
        let dt_max = self.calculator.dt_max;
        self.current_dt = min_dt.min(dt_max).max(dt_min);
        self.current_dt
    }

    /// 计算科氏力稳定性限制
    pub fn coriolis_stability_limit(&self, f: B::Scalar) -> Option<B::Scalar> {
        if f.abs() < B::Scalar::from_f64(1e-14).unwrap_or(B::Scalar::EPSILON) {
            None
        } else {
            // 使用 std::f64::consts::PI，对于 f32 和 f64 转换始终成功
            let pi = B::Scalar::from_f64(std::f64::consts::PI).unwrap_or(B::Scalar::ONE);
            Some(pi / f.abs())
        }
    }

    /// 计算摩擦稳定性限制
    pub fn friction_stability_limit(&self, max_cf: B::Scalar) -> Option<B::Scalar> {
        if max_cf < B::Scalar::from_f64(1e-14).unwrap_or(B::Scalar::EPSILON) {
            None
        } else {
            // 显式稳定性限制
            Some(B::Scalar::from_f64(2.0).unwrap_or(B::Scalar::ONE) / max_cf)
        }
    }

    /// 获取 CFL 数
    pub fn cfl(&self) -> B::Scalar {
        self.calculator.cfl
    }

    /// 设置 CFL 数
    pub fn set_cfl(&mut self, cfl: B::Scalar) {
        let zero = B::Scalar::ZERO;
        let one = B::Scalar::ONE;
        self.calculator.cfl = if cfl <= zero { zero } else if cfl > one { one } else { cfl };
    }
    /// 获取统计信息快照
    pub fn stats(&self) -> TimeStepStats<B> {
        TimeStepStats {
            current_dt: self.current_dt,
            dx_min: self.calculator.dx_min(),
            stable_steps: self.stable_steps,
            adaptive_growth_enabled: self.adaptive_growth,
        }
    }

}

/// 时间步长统计
#[derive(Clone, Debug)]
pub struct TimeStepStats<B: Backend> {
    pub current_dt: B::Scalar,
    pub dx_min: Option<B::Scalar>,
    pub stable_steps: usize,
    pub adaptive_growth_enabled: bool,
}

/// 时间步长控制器构建器
pub struct TimeStepControllerBuilder<B: Backend> {
    g: B::Scalar,
    cfl: B::Scalar,
    dt_min: B::Scalar,
    dt_max: B::Scalar,
    growth_factor: B::Scalar,
    shrink_factor: B::Scalar,
    adaptive_growth: bool,
    marker: PhantomData<B>,
}

impl<B: Backend> TimeStepControllerBuilder<B>
where
    B::Scalar: RuntimeScalar + FromPrimitive,
{
    /// 创建构建器
    pub fn new(g: B::Scalar) -> Self {
        Self {
            g,
            cfl: B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ONE),
            dt_min: B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::EPSILON),
            dt_max: B::Scalar::from_f64(1.0).unwrap_or(B::Scalar::ONE),
            growth_factor: B::Scalar::from_f64(1.1).unwrap_or(B::Scalar::ONE),
            shrink_factor: B::Scalar::from_f64(0.5).unwrap_or(B::Scalar::ONE),
            adaptive_growth: true,
            marker: PhantomData,
        }
    }

    pub fn with_cfl(mut self, cfl: B::Scalar) -> Self {
        self.cfl = cfl;
        self
    }

    pub fn with_dt_limits(mut self, dt_min: B::Scalar, dt_max: B::Scalar) -> Self {
        self.dt_min = dt_min;
        self.dt_max = dt_max;
        self
    }

    pub fn with_growth_factor(mut self, factor: B::Scalar) -> Self {
        self.growth_factor = factor;
        self
    }

    pub fn with_shrink_factor(mut self, factor: B::Scalar) -> Self {
        self.shrink_factor = factor;
        self
    }

    pub fn with_adaptive_growth(mut self, enabled: bool) -> Self {
        self.adaptive_growth = enabled;
        self
    }

    pub fn build(self) -> TimeStepController<B> {
        let mut params = NumericalParams::<B::Scalar>::default();
        params.cfl = self.cfl;
        params.dt_min = self.dt_min;
        params.dt_max = self.dt_max;

        let mut controller = TimeStepController::<B>::new(self.g, &params);
        controller.growth_factor = self.growth_factor;
        controller.shrink_factor = self.shrink_factor;
        controller.adaptive_growth = self.adaptive_growth;

        controller
    }
}

// ============================================================
// 单元测试（已泛型化）
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_cfl_calculator_creation() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let calc: CflCalculator<CpuBackend<f64>> = CflCalculator::new(g, &params);
        assert!(calc.cached_dx_min.is_none());
        assert!((calc.g.to_f64().unwrap() - 9.81).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculator_from_max_speed() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let mut calc: CflCalculator<CpuBackend<f64>> = CflCalculator::new(g, &params);
        calc.cached_dx_min = Some(backend.scalar_from_f64(1.0));

        let dt = calc.compute_from_max_speed(backend.scalar_from_f64(10.0));
        assert!((dt.to_f64().unwrap() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculator_static_water() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let calc: CflCalculator<CpuBackend<f64>> = CflCalculator::new(g, &params);

        let dt = calc.compute_from_max_speed(backend.scalar_from_f64(1e-10));
        assert!((dt.to_f64().unwrap() - params.dt_max.to_f64().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_controller_creation() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(g, &params);
        assert!(controller.adaptive_growth);
        assert_eq!(controller.stable_steps, 0);
    }

    #[test]
    fn test_controller_adaptive_growth() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(g, &params);

        for _ in 0..15 {
            controller.stable_steps += 1;
        }

        let growth = controller.compute_adaptive_growth();
        assert!(growth.to_f64().unwrap() >= controller.growth_factor.to_f64().unwrap());
    }

    #[test]
    fn test_controller_shrink() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(g, &params);
        controller.current_dt = backend.scalar_from_f64(0.1);

        controller.shrink();
        assert!(controller.current_dt.to_f64().unwrap() < 0.1);
        assert_eq!(controller.stable_steps, 0);
    }

    #[test]
    fn test_controller_force_shrink() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(g, &params);
        controller.current_dt = backend.scalar_from_f64(0.1);

        controller.force_shrink(backend.scalar_from_f64(0.1));
        assert!((controller.current_dt.to_f64().unwrap() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_builder() {
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let controller: TimeStepController<CpuBackend<f64>> = TimeStepControllerBuilder::new(g)
            .with_cfl(backend.scalar_from_f64(0.3))
            .with_dt_limits(
                backend.scalar_from_f64(1e-8),
                backend.scalar_from_f64(0.5)
            )
            .with_adaptive_growth(false)
            .build();

        assert!(!controller.adaptive_growth);
    }

    #[test]
    fn test_stats() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(g, &params);
        controller.stable_steps = 5;

        let stats = controller.stats();
        assert_eq!(stats.stable_steps, 5);
        assert!(stats.adaptive_growth_enabled);
    }

    #[test]
    fn test_set_dt() {
        let params = NumericalParams::<f64>::default();
        let backend = CpuBackend::<f64>::new();
        let g = backend.scalar_from_f64(9.81);
        let mut controller: TimeStepController<CpuBackend<f64>> = TimeStepController::new(g, &params);
        controller.set_dt(backend.scalar_from_f64(0.5));
        assert!((controller.current_dt().to_f64().unwrap() - 0.5).abs() < 1e-10);
        assert_eq!(controller.stable_steps, 0);
    }
}