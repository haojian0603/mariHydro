// crates/mh_physics/src/schemes/riemann/adaptive.rs

//! 自适应黎曼求解器（泛型化）
//!
//! T=4 改造：完整泛型化以支持 f32/f64 精度切换
//!
//! 根据局部流动特征自动选择最合适的黎曼求解器。
//!
//! # 选择策略
//!
//! ```text
//! 干湿过渡区 (h < h_transition) → Rusanov
//! 强间断区 (|Δh/h| > threshold) → Rusanov
//! 超临界流 (Fr > Fr_crit) → Rusanov
//! 其他情况 → HLLC
//! ```

use mh_runtime::Vector2D;
use num_traits::real::Real;
use super::hllc::HllcSolver;
use num_traits::FromPrimitive;
use super::rusanov::RusanovSolver;
use super::traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use crate::types::NumericalParams;
use mh_runtime::{Backend, CpuBackend, RuntimeScalar};
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// 自适应配置（泛型化）
// ============================================================================

/// 自适应求解器配置
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig<S: RuntimeScalar> {
    /// 干湿过渡阈值 [m]
    pub transition_depth: S,
    /// 相对水深跳跃阈值
    pub depth_jump_threshold: S,
    /// 临界 Froude 数
    pub froude_critical: S,
    /// 过渡区宽度因子
    pub transition_width: S,
    /// 是否启用混合模式
    pub enable_blending: bool,
    /// 最小混合权重
    pub min_hllc_weight: S,
    /// 速度差阈值
    pub velocity_jump_threshold: S,
}

impl<S: RuntimeScalar> Default for AdaptiveConfig<S> {
    fn default() -> Self {
        Self {
            transition_depth: S::from_f64(0.01).unwrap(),
            depth_jump_threshold: S::from_f64(0.5).unwrap(),
            froude_critical: S::from_f64(1.5).unwrap(),
            transition_width: S::from_f64(0.1).unwrap(),
            enable_blending: true,
            min_hllc_weight: S::ZERO,
            velocity_jump_threshold: S::from_f64(5.0).unwrap(),
        }
    }
}

impl<S: RuntimeScalar> AdaptiveConfig<S> {
    /// 保守配置（倾向于使用 Rusanov）
    pub fn conservative() -> Self {
        Self {
            transition_depth: S::from_f64(0.05).unwrap(),
            depth_jump_threshold: S::from_f64(0.3).unwrap(),
            froude_critical: S::from_f64(1.2).unwrap(),
            transition_width: S::from_f64(0.2).unwrap(),
            enable_blending: true,
            min_hllc_weight: S::ZERO,
            velocity_jump_threshold: S::from_f64(3.0).unwrap(),
        }
    }

    /// 精度优先配置
    pub fn accuracy() -> Self {
        Self {
            transition_depth: S::from_f64(0.001).unwrap(),
            depth_jump_threshold: S::from_f64(0.8).unwrap(),
            froude_critical: S::from_f64(2.0).unwrap(),
            transition_width: S::from_f64(0.05).unwrap(),
            enable_blending: true,
            min_hllc_weight: S::from_f64(0.2).unwrap(),
            velocity_jump_threshold: S::from_f64(10.0).unwrap(),
        }
    }
}

// ============================================================================
// 求解器选择
// ============================================================================

/// 求解器选择结果
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverChoice<S: RuntimeScalar> {
    /// 使用 HLLC
    Hllc,
    /// 使用 Rusanov
    Rusanov,
    /// 混合（指定 HLLC 权重）
    Blended(S),
}

impl<S: RuntimeScalar> SolverChoice<S> {
    /// HLLC 权重
    pub fn hllc_weight(&self) -> S {
        match self {
            SolverChoice::Hllc => S::ONE,
            SolverChoice::Rusanov => S::ZERO,
            SolverChoice::Blended(w) => *w,
        }
    }
}

// ============================================================================
// 统计信息
// ============================================================================

/// 自适应求解器统计
#[derive(Debug, Clone, Default)]
pub struct AdaptiveStats {
    pub hllc_count: u64,
    pub rusanov_count: u64,
    pub blended_count: u64,
    pub dry_wet_rusanov: u64,
    pub jump_rusanov: u64,
    pub supercritical_rusanov: u64,
}

impl AdaptiveStats {
    pub fn total(&self) -> u64 {
        self.hllc_count + self.rusanov_count + self.blended_count
    }

    pub fn hllc_ratio(&self) -> f64 {
        let total = self.total();
        if total > 0 { self.hllc_count as f64 / total as f64 } else { 0.0 }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// 线程安全的统计计数器
#[derive(Debug, Default)]
struct AdaptiveStatsCounters {
    hllc_count: AtomicU64,
    rusanov_count: AtomicU64,
    blended_count: AtomicU64,
    dry_wet_rusanov: AtomicU64,
    jump_rusanov: AtomicU64,
    supercritical_rusanov: AtomicU64,
}

impl AdaptiveStatsCounters {
    fn snapshot(&self) -> AdaptiveStats {
        AdaptiveStats {
            hllc_count: self.hllc_count.load(Ordering::Relaxed),
            rusanov_count: self.rusanov_count.load(Ordering::Relaxed),
            blended_count: self.blended_count.load(Ordering::Relaxed),
            dry_wet_rusanov: self.dry_wet_rusanov.load(Ordering::Relaxed),
            jump_rusanov: self.jump_rusanov.load(Ordering::Relaxed),
            supercritical_rusanov: self.supercritical_rusanov.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.hllc_count.store(0, Ordering::Relaxed);
        self.rusanov_count.store(0, Ordering::Relaxed);
        self.blended_count.store(0, Ordering::Relaxed);
        self.dry_wet_rusanov.store(0, Ordering::Relaxed);
        self.jump_rusanov.store(0, Ordering::Relaxed);
        self.supercritical_rusanov.store(0, Ordering::Relaxed);
    }

    fn update<S: RuntimeScalar>(&self, choice: &SolverChoice<S>, reason: AdaptiveReason) {
        match choice {
            SolverChoice::Hllc => {
                self.hllc_count.fetch_add(1, Ordering::Relaxed);
            }
            SolverChoice::Rusanov => {
                self.rusanov_count.fetch_add(1, Ordering::Relaxed);
                match reason {
                    AdaptiveReason::DryWet => self.dry_wet_rusanov.fetch_add(1, Ordering::Relaxed),
                    AdaptiveReason::DepthJump => self.jump_rusanov.fetch_add(1, Ordering::Relaxed),
                    AdaptiveReason::Supercritical => self.supercritical_rusanov.fetch_add(1, Ordering::Relaxed),
                    _ => 0,
                };
            }
            SolverChoice::Blended(_) => {
                self.blended_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// 选择原因（内部使用）
#[derive(Debug, Clone, Copy)]
enum AdaptiveReason {
    Default,
    DryWet,
    DepthJump,
    Supercritical,
    VelocityJump,
}

// ============================================================================
// 自适应求解器（泛型化）
// ============================================================================

/// 自适应黎曼求解器（泛型化）
pub struct AdaptiveSolver<B: Backend> {
    /// HLLC 求解器
    hllc: HllcSolver<B>,
    /// Rusanov 求解器
    rusanov: RusanovSolver<B>,
    /// 配置
    config: AdaptiveConfig<B::Scalar>,
    /// 统计信息
    stats: AdaptiveStatsCounters,
    /// 基本参数
    params: SolverParams<B::Scalar>,
}

impl<B: Backend> AdaptiveSolver<B> {
    /// 创建新的自适应求解器
    pub fn new(numerical_params: &NumericalParams<B::Scalar>, gravity: B::Scalar) -> Self {
        let params = SolverParams::from_numerical(numerical_params, gravity);
        Self {
            hllc: HllcSolver::new(&params, gravity),
            rusanov: RusanovSolver::from_params(params),
            config: AdaptiveConfig::default(),
            stats: AdaptiveStatsCounters::default(),
            params,
        }
    }

    /// 使用配置创建
    pub fn with_config(
        numerical_params: &NumericalParams<B::Scalar>,
        gravity: B::Scalar,
        config: AdaptiveConfig<B::Scalar>,
    ) -> Self {
        let params = SolverParams::from_numerical(numerical_params, gravity);
        Self {
            hllc: HllcSolver::new(&params, gravity),
            rusanov: RusanovSolver::from_params(params),
            config,
            stats: AdaptiveStatsCounters::default(),
            params,
        }
    }

    /// 获取配置
    pub fn config(&self) -> &AdaptiveConfig<B::Scalar> {
        &self.config
    }

    /// 获取统计信息
    pub fn stats(&self) -> AdaptiveStats {
        self.stats.snapshot()
    }

    /// 重置统计
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    /// 选择求解器
    fn choose_solver(
        &self,
        h_left: B::Scalar,
        h_right: B::Scalar,
        vel_left: B::Vector2D,
        vel_right: B::Vector2D,
    ) -> (SolverChoice<B::Scalar>, AdaptiveReason) {
        let h_max = num_traits::Float::max(h_left, h_right);
        let h_min_val = num_traits::Float::min(h_left, h_right);

        // 1. 干湿检查
        if h_min_val < self.config.transition_depth {
            return (SolverChoice::Rusanov, AdaptiveReason::DryWet);
        }

        // 2. 水深跳跃检查
        let depth_jump = num_traits::Float::abs(h_left - h_right) / h_max;
        if depth_jump > self.config.depth_jump_threshold {
            if self.config.enable_blending {
                let weight = self.compute_blend_weight(depth_jump, self.config.depth_jump_threshold);
                if weight < B::Scalar::from_f64(0.01).unwrap() {
                    return (SolverChoice::Rusanov, AdaptiveReason::DepthJump);
                }
                return (
                    SolverChoice::Blended(Real::max(weight, self.config.min_hllc_weight)),
                    AdaptiveReason::DepthJump,
                );
            }
            return (SolverChoice::Rusanov, AdaptiveReason::DepthJump);
        }

        // 3. Froude 数检查
        let g = self.params.gravity;
        let c_avg = num_traits::Float::sqrt(g * h_max);
        let vel_l_x = vel_left.x();
        let vel_l_y = vel_left.y();
        let vel_r_x = vel_right.x();
        let vel_r_y = vel_right.y();
        let vel_avg_x = (vel_l_x + vel_r_x) * B::Scalar::HALF;
        let vel_avg_y = (vel_l_y + vel_r_y) * B::Scalar::HALF;
        let vel_avg_len = num_traits::Float::sqrt(vel_avg_x * vel_avg_x + vel_avg_y * vel_avg_y);
        let froude = vel_avg_len / num_traits::Float::max(c_avg, B::Scalar::from_f64(1e-10).unwrap());

        if froude > self.config.froude_critical {
            if self.config.enable_blending {
                let weight = self.compute_blend_weight(froude, self.config.froude_critical);
                if weight < B::Scalar::from_f64(0.01).unwrap() {
                    return (SolverChoice::Rusanov, AdaptiveReason::Supercritical);
                }
                return (
                    SolverChoice::Blended(num_traits::Float::max(weight, self.config.min_hllc_weight)),
                    AdaptiveReason::Supercritical,
                );
            }
            return (SolverChoice::Rusanov, AdaptiveReason::Supercritical);
        }

        // 4. 速度跳跃检查
        let vel_diff_x = vel_l_x - vel_r_x;
        let vel_diff_y = vel_l_y - vel_r_y;
        let vel_jump = num_traits::Float::sqrt(vel_diff_x * vel_diff_x + vel_diff_y * vel_diff_y);
        if vel_jump > self.config.velocity_jump_threshold {
            if self.config.enable_blending {
                let weight = self.compute_blend_weight(vel_jump, self.config.velocity_jump_threshold);
                if weight < B::Scalar::from_f64(0.01).unwrap() {
                    return (SolverChoice::Rusanov, AdaptiveReason::VelocityJump);
                }
                return (
                    SolverChoice::Blended(num_traits::Float::max(weight, self.config.min_hllc_weight)),
                    AdaptiveReason::VelocityJump,
                );
            }
            return (SolverChoice::Rusanov, AdaptiveReason::VelocityJump);
        }

        (SolverChoice::Hllc, AdaptiveReason::Default)
    }

    /// 计算混合权重
    fn compute_blend_weight(&self, value: B::Scalar, threshold: B::Scalar) -> B::Scalar {
        if value <= threshold {
            return B::Scalar::ONE;
        }

        let width = threshold * self.config.transition_width;
        let tiny = B::Scalar::from_f64(1e-10).unwrap();
        if width < tiny {
            return B::Scalar::ZERO;
        }

        let x = (value - threshold) / width;
        if x >= B::Scalar::ONE {
            return B::Scalar::ZERO;
        }

        // 简化的线性混合（避免依赖 PI 常量）
        B::Scalar::ONE - x
    }

    /// 混合两个通量
    fn blend_flux(
        flux_hllc: RiemannFlux<B::Scalar>,
        flux_rusanov: RiemannFlux<B::Scalar>,
        hllc_weight: B::Scalar,
    ) -> RiemannFlux<B::Scalar> {
        let w = hllc_weight;
        let w_inv = B::Scalar::ONE - w;

        RiemannFlux {
            mass: w * flux_hllc.mass + w_inv * flux_rusanov.mass,
            momentum_x: w * flux_hllc.momentum_x + w_inv * flux_rusanov.momentum_x,
            momentum_y: w * flux_hllc.momentum_y + w_inv * flux_rusanov.momentum_y,
            max_wave_speed: num_traits::Float::max(flux_hllc.max_wave_speed, flux_rusanov.max_wave_speed),
        }
    }
}

// ============================================================================
// RiemannSolver trait 实现
// ============================================================================

impl<B: Backend> RiemannSolver for AdaptiveSolver<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

    fn name(&self) -> &'static str {
        "Adaptive (HLLC/Rusanov)"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: true,
            order: 2,
            positivity_preserving: true,
        }
    }

    fn solve(
        &self,
        h_left: B::Scalar,
        h_right: B::Scalar,
        vel_left: B::Vector2D,
        vel_right: B::Vector2D,
        normal: B::Vector2D,
    ) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        let (choice, reason) = self.choose_solver(h_left, h_right, vel_left, vel_right);
        self.stats.update(&choice, reason);

        match choice {
            SolverChoice::Hllc => self.hllc.solve(h_left, h_right, vel_left, vel_right, normal),
            SolverChoice::Rusanov => self.rusanov.solve(h_left, h_right, vel_left, vel_right, normal),
            SolverChoice::Blended(w) => {
                let flux_hllc = self.hllc.solve(h_left, h_right, vel_left, vel_right, normal)?;
                let flux_rusanov = self.rusanov.solve(h_left, h_right, vel_left, vel_right, normal)?;
                Ok(Self::blend_flux(flux_hllc, flux_rusanov, w))
            }
        }
    }

    fn gravity(&self) -> B::Scalar {
        self.params.gravity
    }

    fn dry_threshold(&self) -> B::Scalar {
        self.params.h_dry
    }
}

// ============================================================================
// 向后兼容类型别名
// ============================================================================

/// f64 版本的 AdaptiveSolver
pub type AdaptiveSolverF64 = AdaptiveSolver<CpuBackend<f64>>;

/// f32 版本的 AdaptiveSolver
pub type AdaptiveSolverF32 = AdaptiveSolver<CpuBackend<f32>>;

// ============================================================================
// 工厂函数
// ============================================================================

/// 创建默认自适应求解器 (f64)
pub fn create_adaptive_solver(gravity: f64) -> AdaptiveSolverF64 {
    let params = NumericalParams::<f64>::default();
    AdaptiveSolverF64::new(&params, gravity)
}

/// 创建保守自适应求解器 (f64)
pub fn create_conservative_adaptive_solver(gravity: f64) -> AdaptiveSolverF64 {
    let params = NumericalParams::<f64>::default();
    AdaptiveSolverF64::with_config(&params, gravity, AdaptiveConfig::conservative())
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_solver_f64() {
        let solver = create_adaptive_solver(9.81);

        let flux = solver
            .solve(
                1.0, 1.0,
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
            )
            .unwrap();

        assert!(flux.is_valid());
        assert!(flux.mass.abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_solver_f32() {
        let params = NumericalParams::<f32>::default();
        let solver = AdaptiveSolverF32::new(&params, 9.81f32);

        let flux = solver
            .solve(
                1.0f32, 1.0f32,
                CpuBackend::<f32>::vec2_new(0.0, 0.0),
                CpuBackend::<f32>::vec2_new(0.0, 0.0),
                CpuBackend::<f32>::vec2_new(1.0, 0.0),
            )
            .unwrap();

        assert!(flux.is_valid());
    }

    #[test]
    fn test_stats() {
        let solver = create_adaptive_solver(9.81);
        
        // 静水使用 HLLC
        let _ = solver.solve(
            1.0, 1.0,
            CpuBackend::<f64>::vec2_new(0.0, 0.0),
            CpuBackend::<f64>::vec2_new(0.0, 0.0),
            CpuBackend::<f64>::vec2_new(1.0, 0.0),
        );

        let stats = solver.stats();
        assert!(stats.total() > 0);
    }
}
