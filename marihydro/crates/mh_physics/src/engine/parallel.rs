// crates/mh_physics/src/engine/parallel.rs

//! 并行通量计算模块
//!
//! 提供多种并行策略用于加速通量计算：
//! - 串行计算（小规模问题）
//! - 收集后累加（先并行计算通量，后串行累加到单元）
//! - 着色并行（使用图着色实现真正无锁并行，TODO）
//!
//! # 迁移说明
//!
//! 从 legacy_src/physics/engine/parallel.rs 简化迁移。
//! 完整的着色并行等高级功能将在后续版本实现。
//!
//! # 技术债务 (TD-5.3.2, TD-5.3.3)
//!
//! 当前实现的"并行"是伪并行：通量计算并行，但累加阶段串行。
//! 对于大规模网格，需要实现真正的着色并行以避免累加瓶颈。

use crate::adapter::PhysicsMesh;
use crate::engine::solver::{BedSlopeCorrection, HydrostaticFaceState, HydrostaticReconstruction};
use crate::schemes::{HllcSolver, RiemannFlux, RiemannSolver};
use crate::schemes::wetting_drying::{WetState, WettingDryingHandler};
use crate::state::ShallowWaterState;
use crate::types::NumericalParams;

use glam::DVec2;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================
// 配置
// ============================================================

/// 并行策略
///
/// # 策略说明
///
/// - `Sequential`: 完全串行执行，适用于小规模问题
/// - `CollectThenAccumulate`: 先并行计算各面通量(真正并行)，
///   然后串行累加到单元(瓶颈)。对于中等规模问题有效。
/// - `Auto`: 根据面数自动选择策略
///
/// # TODO (TD-5.3.3)
///
/// 未来需要添加：
/// - `Colored`: 使用图着色实现真正的无锁并行累加
/// - `Atomic`: 使用原子操作进行并行累加
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    /// 串行执行
    Sequential,
    /// 收集后累加：并行计算通量 → 收集结果 → 串行累加
    ///
    /// 注意：累加阶段是串行的，对于大规模网格可能成为瓶颈
    CollectThenAccumulate,
    /// 自动选择（根据问题规模）
    Auto,
}

impl Default for ParallelStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

/// 并行计算配置
#[derive(Debug, Clone)]
pub struct ParallelFluxConfig {
    /// 数值参数
    pub params: NumericalParams,
    /// 重力加速度
    pub g: f64,
    /// 最小并行面数（低于此值使用串行）
    pub min_parallel_size: usize,
    /// 并行策略
    pub strategy: ParallelStrategy,
    /// 是否启用静水重构
    pub use_hydrostatic_reconstruction: bool,
}

impl Default for ParallelFluxConfig {
    fn default() -> Self {
        Self {
            params: NumericalParams::default(),
            g: 9.81,
            min_parallel_size: 1000,
            strategy: ParallelStrategy::Auto,
            use_hydrostatic_reconstruction: true,
        }
    }
}

impl ParallelFluxConfig {
    /// 创建构建器
    pub fn builder() -> ParallelFluxConfigBuilder {
        ParallelFluxConfigBuilder::default()
    }
}

/// 配置构建器
#[derive(Default)]
pub struct ParallelFluxConfigBuilder {
    config: ParallelFluxConfig,
}

impl ParallelFluxConfigBuilder {
    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.g = g;
        self
    }

    pub fn min_parallel_size(mut self, size: usize) -> Self {
        self.config.min_parallel_size = size;
        self
    }

    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn use_hydrostatic_reconstruction(mut self, enable: bool) -> Self {
        self.config.use_hydrostatic_reconstruction = enable;
        self
    }

    pub fn build(self) -> ParallelFluxConfig {
        self.config
    }
}

// ============================================================
// 性能指标
// ============================================================

/// 性能指标
#[derive(Debug, Clone, Default)]
pub struct FluxComputeMetrics {
    /// 总计算次数
    pub total_calls: usize,
    /// 并行计算次数
    pub parallel_calls: usize,
    /// 串行计算次数
    pub sequential_calls: usize,
    /// 总计算时间
    pub total_duration: Duration,
    /// 处理的面总数
    pub total_faces: usize,
}

impl FluxComputeMetrics {
    /// 记录一次计算
    // TODO(phase5): 添加策略选择的详细日志（如 legacy 的 StrategySelector）
    pub fn record(&mut self, n_faces: usize, is_parallel: bool, duration: Duration) {
        self.total_calls += 1;
        self.total_faces += n_faces;
        self.total_duration += duration;
        if is_parallel {
            self.parallel_calls += 1;
        } else {
            self.sequential_calls += 1;
        }
    }

    /// 重置指标
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// 平均每面计算时间
    pub fn avg_time_per_face(&self) -> Duration {
        if self.total_faces > 0 {
            self.total_duration / self.total_faces as u32
        } else {
            Duration::ZERO
        }
    }
}

// ============================================================
// 并行通量计算器
// ============================================================

/// 并行通量计算器
///
/// 封装通量计算的并行执行逻辑。
///
/// # TODO(phase5)
/// 
/// - 添加 AdaptiveScheduler 支持（参考 legacy）
/// - 添加 StrategySelector 自动选择最优策略
/// - 实现着色并行（TD-5.3.3）
pub struct ParallelFluxCalculator {
    config: ParallelFluxConfig,
    /// 黎曼求解器
    riemann: HllcSolver,
    /// 干湿处理器
    wetting_drying: WettingDryingHandler,
    /// 静水重构
    hydrostatic: HydrostaticReconstruction,
    /// 性能指标
    metrics: FluxComputeMetrics,
}

impl ParallelFluxCalculator {
    /// 创建计算器
    pub fn new(config: ParallelFluxConfig) -> Self {
        Self {
            riemann: HllcSolver::new(&config.params, config.g),
            wetting_drying: WettingDryingHandler::from_params(&config.params),
            hydrostatic: HydrostaticReconstruction::new(&config.params, config.g),
            metrics: FluxComputeMetrics::default(),
            config,
        }
    }

    /// 计算通量（自动选择策略）
    pub fn compute_fluxes(
        &mut self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        let n_faces = mesh.n_faces();
        let start = Instant::now();

        let (max_speed, is_parallel) = match self.config.strategy {
            ParallelStrategy::Sequential => {
                (self.compute_serial(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), false)
            }
            ParallelStrategy::CollectThenAccumulate => {
                (self.compute_parallel(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
            }
            ParallelStrategy::Auto => {
                if n_faces < self.config.min_parallel_size {
                    (self.compute_serial(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), false)
                } else {
                    (self.compute_parallel(state, mesh, flux_h, flux_hu, flux_hv, source_hu, source_hv), true)
                }
            }
        };

        let duration = start.elapsed();
        self.metrics.record(n_faces, is_parallel, duration);

        max_speed
    }

    /// 串行计算
    fn compute_serial(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        // 重置
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        let n_faces = mesh.n_faces();
        let mut max_speed = 0.0f64;

        for face_idx in 0..n_faces {
            let (flux, bed_src, length, owner, neighbor) = 
                self.compute_face(state, mesh, face_idx);

            max_speed = max_speed.max(flux.max_wave_speed);

            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            flux_h[owner] -= fh;
            flux_hu[owner] -= fhu;
            flux_hv[owner] -= fhv;
            source_hu[owner] += bed_src.source_x;
            source_hv[owner] += bed_src.source_y;

            if let Some(neigh) = neighbor {
                flux_h[neigh] += fh;
                flux_hu[neigh] += fhu;
                flux_hv[neigh] += fhv;
                source_hu[neigh] -= bed_src.source_x;
                source_hv[neigh] -= bed_src.source_y;
            }
        }

        max_speed
    }

    /// 并行计算（先并行计算，后串行累加）
    fn compute_parallel(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        flux_h: &mut [f64],
        flux_hu: &mut [f64],
        flux_hv: &mut [f64],
        source_hu: &mut [f64],
        source_hv: &mut [f64],
    ) -> f64 {
        let n_faces = mesh.n_faces();
        let max_speed_atomic = AtomicU64::new(0u64);

        // 并行计算所有面
        let face_results: Vec<_> = (0..n_faces)
            .into_par_iter()
            .map(|face_idx| {
                let (flux, bed_src, length, owner, neighbor) = 
                    self.compute_face(state, mesh, face_idx);

                max_speed_atomic.fetch_max(flux.max_wave_speed.to_bits(), Ordering::Relaxed);

                (flux, bed_src, length, owner, neighbor)
            })
            .collect();

        // 串行累加
        flux_h.fill(0.0);
        flux_hu.fill(0.0);
        flux_hv.fill(0.0);
        source_hu.fill(0.0);
        source_hv.fill(0.0);

        for (flux, bed_src, length, owner, neighbor) in face_results {
            let fh = flux.mass * length;
            let fhu = flux.momentum_x * length;
            let fhv = flux.momentum_y * length;

            flux_h[owner] -= fh;
            flux_hu[owner] -= fhu;
            flux_hv[owner] -= fhv;
            source_hu[owner] += bed_src.source_x;
            source_hv[owner] += bed_src.source_y;

            if let Some(neigh) = neighbor {
                flux_h[neigh] += fh;
                flux_hu[neigh] += fhu;
                flux_hv[neigh] += fhv;
                source_hu[neigh] -= bed_src.source_x;
                source_hv[neigh] -= bed_src.source_y;
            }
        }

        f64::from_bits(max_speed_atomic.load(Ordering::Relaxed))
    }

    /// 计算单个面的通量
    fn compute_face(
        &self,
        state: &ShallowWaterState,
        mesh: &PhysicsMesh,
        face_idx: usize,
    ) -> (RiemannFlux, BedSlopeCorrection, f64, usize, Option<usize>) {
        let normal = mesh.face_normal(face_idx);
        let length = mesh.face_length(face_idx);
        let owner = mesh.face_owner(face_idx);
        let neighbor = mesh.face_neighbor(face_idx);

        // 左侧状态
        let h_l = state.h[owner];
        let z_l = state.z[owner];
        let (u_l, v_l) = self.config.params.safe_velocity_components(
            state.hu[owner], state.hv[owner], h_l
        );
        let vel_l = DVec2::new(u_l, v_l);

        // 右侧状态
        let (h_r, vel_r, z_r) = if let Some(neigh) = neighbor {
            let h = state.h[neigh];
            let (u, v) = self.config.params.safe_velocity_components(
                state.hu[neigh], state.hv[neigh], h
            );
            (h, DVec2::new(u, v), state.z[neigh])
        } else {
            let vn = vel_l.dot(normal);
            (h_l, vel_l - 2.0 * vn * normal, z_l)
        };

        // 静水重构
        let recon = if self.config.use_hydrostatic_reconstruction {
            self.hydrostatic.reconstruct_face_simple(h_l, h_r, z_l, z_r, vel_l, vel_r)
        } else {
            HydrostaticFaceState {
                h_left: h_l,
                h_right: h_r,
                vel_left: vel_l,
                vel_right: vel_r,
                z_face: 0.5 * (z_l + z_r),
            }
        };

        // 干湿限制
        let wet_l = self.wetting_drying.get_state(recon.h_left);
        let wet_r = self.wetting_drying.get_state(recon.h_right);
        let flux_limiter = match (wet_l, wet_r) {
            (WetState::Dry, WetState::Dry) => 0.0,
            (WetState::Dry, _) | (_, WetState::Dry) => {
                let h_min = recon.h_left.min(recon.h_right);
                (h_min / self.config.params.h_wet).min(1.0)
            }
            (WetState::PartiallyWet, _) | (_, WetState::PartiallyWet) => {
                let h_min = recon.h_left.min(recon.h_right);
                ((h_min - self.config.params.h_dry)
                    / (self.config.params.h_wet - self.config.params.h_dry)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        };

        // 黎曼通量
        let flux = self.riemann.solve(
            recon.h_left, recon.h_right,
            recon.vel_left, recon.vel_right,
            normal,
        ).unwrap_or(RiemannFlux::ZERO);

        let limited_flux = if flux_limiter < 1.0 {
            flux.scaled(flux_limiter)
        } else {
            flux
        };

        // 床坡源项
        let bed_src = self.hydrostatic.bed_slope_correction(h_l, h_r, z_l, z_r, normal, length);

        (limited_flux, bed_src, length, owner, neighbor)
    }

    // =========================================================================
    // 访问器
    // =========================================================================

    /// 获取配置
    pub fn config(&self) -> &ParallelFluxConfig {
        &self.config
    }

    /// 获取性能指标
    pub fn metrics(&self) -> &FluxComputeMetrics {
        &self.metrics
    }

    /// 重置性能指标
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

// ============================================================
// 构建器
// ============================================================

/// 并行计算器构建器
pub struct ParallelFluxCalculatorBuilder {
    config: ParallelFluxConfig,
}

impl ParallelFluxCalculatorBuilder {
    pub fn new() -> Self {
        Self {
            config: ParallelFluxConfig::default(),
        }
    }

    pub fn config(mut self, config: ParallelFluxConfig) -> Self {
        self.config = config;
        self
    }

    pub fn params(mut self, params: NumericalParams) -> Self {
        self.config.params = params;
        self
    }

    pub fn gravity(mut self, g: f64) -> Self {
        self.config.g = g;
        self
    }

    pub fn strategy(mut self, strategy: ParallelStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn build(self) -> ParallelFluxCalculator {
        ParallelFluxCalculator::new(self.config)
    }
}

impl Default for ParallelFluxCalculatorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// 测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ParallelFluxConfig::default();
        assert!((config.g - 9.81).abs() < 1e-10);
        assert_eq!(config.min_parallel_size, 1000);
        assert_eq!(config.strategy, ParallelStrategy::Auto);
    }

    #[test]
    fn test_config_builder() {
        let config = ParallelFluxConfig::builder()
            .gravity(10.0)
            .min_parallel_size(500)
            .strategy(ParallelStrategy::Sequential)
            .build();

        assert!((config.g - 10.0).abs() < 1e-10);
        assert_eq!(config.min_parallel_size, 500);
        assert_eq!(config.strategy, ParallelStrategy::Sequential);
    }

    #[test]
    fn test_metrics() {
        let mut metrics = FluxComputeMetrics::default();
        metrics.record(1000, true, Duration::from_millis(10));
        metrics.record(500, false, Duration::from_millis(5));

        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.parallel_calls, 1);
        assert_eq!(metrics.sequential_calls, 1);
        assert_eq!(metrics.total_faces, 1500);
    }

    #[test]
    fn test_calculator_builder() {
        let calc = ParallelFluxCalculatorBuilder::new()
            .gravity(10.0)
            .strategy(ParallelStrategy::CollectThenAccumulate)
            .build();

        assert!((calc.config().g - 10.0).abs() < 1e-10);
        assert_eq!(calc.config().strategy, ParallelStrategy::CollectThenAccumulate);
    }
}
