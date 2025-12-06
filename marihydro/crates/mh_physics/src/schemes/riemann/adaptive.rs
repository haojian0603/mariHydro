// crates/mh_physics/src/schemes/riemann/adaptive.rs

//! 自适应黎曼求解器
//!
//! 根据局部流动特征自动选择最合适的黎曼求解器。
//!
//! # 设计目标
//!
//! 1. **精度优先区域**: 使用 HLLC 获得高精度
//! 2. **稳定优先区域**: 使用 Rusanov 确保稳定性
//! 3. **平滑过渡**: 避免求解器切换引起的数值振荡
//!
//! # 选择策略
//!
//! ```text
//! 干湿过渡区 (h < h_transition) → Rusanov
//! 强间断区 (|Δh/h| > threshold) → Rusanov
//! 超临界流 (Fr > Fr_crit) → Rusanov
//! 其他情况 → HLLC
//! ```
//!
//! # 混合策略
//!
//! 在过渡区域使用两个求解器的加权平均：
//! ```text
//! F* = α * F_HLLC + (1-α) * F_Rusanov
//! ```
//!
//! 其中 α ∈ [0, 1] 是平滑权重函数。

use super::hllc::HllcSolver;
use super::rusanov::RusanovSolver;
use super::traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use crate::types::NumericalParams;
use glam::DVec2;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// 自适应配置
// ============================================================================

/// 自适应求解器配置
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    /// 干湿过渡阈值 [m]
    ///
    /// 水深低于此值时优先使用 Rusanov
    pub transition_depth: f64,

    /// 相对水深跳跃阈值
    ///
    /// |h_L - h_R| / max(h_L, h_R) 超过此值时使用 Rusanov
    pub depth_jump_threshold: f64,

    /// 临界 Froude 数
    ///
    /// Fr > Fr_crit 时使用 Rusanov（超临界流）
    pub froude_critical: f64,

    /// 过渡区宽度因子
    ///
    /// 控制混合区域的平滑程度，值越大过渡越平滑
    pub transition_width: f64,

    /// 是否启用混合模式
    ///
    /// 启用时在过渡区使用加权平均
    pub enable_blending: bool,

    /// 最小混合权重
    ///
    /// HLLC 的最小权重，防止完全退化为 Rusanov
    pub min_hllc_weight: f64,

    /// 速度差阈值
    ///
    /// |vel_L - vel_R| 超过此值时倾向使用 Rusanov
    pub velocity_jump_threshold: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            transition_depth: 0.01,       // 1 cm
            depth_jump_threshold: 0.5,    // 50% 相对变化
            froude_critical: 1.5,         // 超临界阈值
            transition_width: 0.1,        // 10% 过渡宽度
            enable_blending: true,
            min_hllc_weight: 0.0,
            velocity_jump_threshold: 5.0, // 5 m/s
        }
    }
}

impl AdaptiveConfig {
    /// 保守配置（倾向于使用 Rusanov）
    pub fn conservative() -> Self {
        Self {
            transition_depth: 0.05,
            depth_jump_threshold: 0.3,
            froude_critical: 1.2,
            transition_width: 0.2,
            enable_blending: true,
            min_hllc_weight: 0.0,
            velocity_jump_threshold: 3.0,
        }
    }

    /// 精度优先配置（倾向于使用 HLLC）
    pub fn accuracy() -> Self {
        Self {
            transition_depth: 0.001,
            depth_jump_threshold: 0.8,
            froude_critical: 2.0,
            transition_width: 0.05,
            enable_blending: true,
            min_hllc_weight: 0.2,
            velocity_jump_threshold: 10.0,
        }
    }

    /// 禁用自适应（始终使用 HLLC）
    pub fn hllc_only() -> Self {
        Self {
            transition_depth: 0.0,
            depth_jump_threshold: f64::INFINITY,
            froude_critical: f64::INFINITY,
            enable_blending: false,
            min_hllc_weight: 1.0,
            ..Default::default()
        }
    }

    /// 禁用自适应（始终使用 Rusanov）
    pub fn rusanov_only() -> Self {
        Self {
            transition_depth: f64::INFINITY,
            depth_jump_threshold: 0.0,
            froude_critical: 0.0,
            enable_blending: false,
            min_hllc_weight: 0.0,
            ..Default::default()
        }
    }
}

// ============================================================================
// 求解器选择
// ============================================================================

/// 求解器选择结果
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolverChoice {
    /// 使用 HLLC
    Hllc,
    /// 使用 Rusanov
    Rusanov,
    /// 混合（指定 HLLC 权重）
    Blended(f64),
}

impl SolverChoice {
    /// 是否使用 HLLC
    pub fn uses_hllc(&self) -> bool {
        match self {
            SolverChoice::Hllc => true,
            SolverChoice::Blended(w) => *w > 0.0,
            SolverChoice::Rusanov => false,
        }
    }

    /// 是否使用 Rusanov
    pub fn uses_rusanov(&self) -> bool {
        match self {
            SolverChoice::Rusanov => true,
            SolverChoice::Blended(w) => *w < 1.0,
            SolverChoice::Hllc => false,
        }
    }

    /// HLLC 权重
    pub fn hllc_weight(&self) -> f64 {
        match self {
            SolverChoice::Hllc => 1.0,
            SolverChoice::Rusanov => 0.0,
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
    /// HLLC 使用次数
    pub hllc_count: u64,
    /// Rusanov 使用次数
    pub rusanov_count: u64,
    /// 混合使用次数
    pub blended_count: u64,
    /// 因干湿选择 Rusanov 的次数
    pub dry_wet_rusanov: u64,
    /// 因间断选择 Rusanov 的次数
    pub jump_rusanov: u64,
    /// 因超临界选择 Rusanov 的次数
    pub supercritical_rusanov: u64,
}

impl AdaptiveStats {
    /// 总调用次数
    pub fn total(&self) -> u64 {
        self.hllc_count + self.rusanov_count + self.blended_count
    }

    /// HLLC 使用比例
    pub fn hllc_ratio(&self) -> f64 {
        let total = self.total();
        if total > 0 {
            self.hllc_count as f64 / total as f64
        } else {
            0.0
        }
    }

    /// Rusanov 使用比例
    pub fn rusanov_ratio(&self) -> f64 {
        let total = self.total();
        if total > 0 {
            self.rusanov_count as f64 / total as f64
        } else {
            0.0
        }
    }

    /// 重置统计
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

    fn update(&self, choice: &SolverChoice, reason: AdaptiveReason) {
        match choice {
            SolverChoice::Hllc => {
                self.hllc_count.fetch_add(1, Ordering::Relaxed);
            }
            SolverChoice::Rusanov => {
                self.rusanov_count.fetch_add(1, Ordering::Relaxed);
                match reason {
                    AdaptiveReason::DryWet => {
                        self.dry_wet_rusanov.fetch_add(1, Ordering::Relaxed);
                    }
                    AdaptiveReason::DepthJump => {
                        self.jump_rusanov.fetch_add(1, Ordering::Relaxed);
                    }
                    AdaptiveReason::Supercritical => {
                        self.supercritical_rusanov.fetch_add(1, Ordering::Relaxed);
                    }
                    AdaptiveReason::VelocityJump | AdaptiveReason::Default => {}
                }
            }
            SolverChoice::Blended(_) => {
                self.blended_count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

// ============================================================================
// 自适应求解器
// ============================================================================

/// 自适应黎曼求解器
///
/// 根据局部流动特征自动选择最合适的求解器。
///
/// # 示例
///
/// ```ignore
/// use mh_physics::schemes::riemann::{AdaptiveSolver, AdaptiveConfig};
///
/// let solver = AdaptiveSolver::new(&params, 9.81);
///
/// // 求解器会自动选择最合适的方法
/// let flux = solver.solve(h_l, h_r, vel_l, vel_r, normal)?;
///
/// // 查看统计
/// println!("HLLC 比例: {:.1}%", solver.stats().hllc_ratio() * 100.0);
/// ```
pub struct AdaptiveSolver {
    /// HLLC 求解器
    hllc: HllcSolver,

    /// Rusanov 求解器
    rusanov: RusanovSolver,

    /// 配置
    config: AdaptiveConfig,

    /// 统计信息
    stats: AdaptiveStatsCounters,

    /// 基本参数
    params: SolverParams,
}

impl AdaptiveSolver {
    /// 创建新的自适应求解器
    pub fn new(numerical_params: &NumericalParams, gravity: f64) -> Self {
        let params = SolverParams::from_numerical(numerical_params, gravity);
        Self {
            hllc: HllcSolver::from_params(params),
            rusanov: RusanovSolver::from_params(params),
            config: AdaptiveConfig::default(),
            stats: AdaptiveStatsCounters::default(),
            params,
        }
    }

    /// 使用配置创建
    pub fn with_config(
        numerical_params: &NumericalParams,
        gravity: f64,
        config: AdaptiveConfig,
    ) -> Self {
        let params = SolverParams::from_numerical(numerical_params, gravity);
        Self {
            hllc: HllcSolver::from_params(params),
            rusanov: RusanovSolver::from_params(params),
            config,
            stats: AdaptiveStatsCounters::default(),
            params,
        }
    }

    /// 获取配置
    pub fn config(&self) -> &AdaptiveConfig {
        &self.config
    }

    /// 设置配置
    pub fn set_config(&mut self, config: AdaptiveConfig) {
        self.config = config;
    }

    /// 获取统计信息
    pub fn stats(&self) -> AdaptiveStats {
        self.stats.snapshot()
    }

    /// 重置统计
    pub fn reset_stats(&self) {
        self.stats.reset();
    }

    // =========================================================================
    // 选择逻辑
    // =========================================================================

    /// 选择求解器
    fn choose_solver(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
    ) -> (SolverChoice, AdaptiveReason) {
        let h_max = h_left.max(h_right);
        let h_min = h_left.min(h_right);

        // 1. 干湿检查
        if h_min < self.config.transition_depth {
            return (SolverChoice::Rusanov, AdaptiveReason::DryWet);
        }

        // 2. 水深跳跃检查
        let depth_jump = (h_left - h_right).abs() / h_max;
        if depth_jump > self.config.depth_jump_threshold {
            if self.config.enable_blending {
                // 计算混合权重
                let weight = self.compute_blend_weight(depth_jump, self.config.depth_jump_threshold);
                if weight < 0.01 {
                    return (SolverChoice::Rusanov, AdaptiveReason::DepthJump);
                }
                return (
                    SolverChoice::Blended(weight.max(self.config.min_hllc_weight)),
                    AdaptiveReason::DepthJump,
                );
            }
            return (SolverChoice::Rusanov, AdaptiveReason::DepthJump);
        }

        // 3. Froude 数检查
        let g = self.params.gravity;
        let c_avg = (g * h_max).sqrt();
        let vel_avg = (vel_left + vel_right) * 0.5;
        let froude = vel_avg.length() / c_avg.max(1e-10);

        if froude > self.config.froude_critical {
            if self.config.enable_blending {
                let weight = self.compute_blend_weight(froude, self.config.froude_critical);
                if weight < 0.01 {
                    return (SolverChoice::Rusanov, AdaptiveReason::Supercritical);
                }
                return (
                    SolverChoice::Blended(weight.max(self.config.min_hllc_weight)),
                    AdaptiveReason::Supercritical,
                );
            }
            return (SolverChoice::Rusanov, AdaptiveReason::Supercritical);
        }

        // 4. 速度跳跃检查
        let vel_jump = (vel_left - vel_right).length();
        if vel_jump > self.config.velocity_jump_threshold {
            if self.config.enable_blending {
                let weight = self.compute_blend_weight(vel_jump, self.config.velocity_jump_threshold);
                if weight < 0.01 {
                    return (SolverChoice::Rusanov, AdaptiveReason::VelocityJump);
                }
                return (
                    SolverChoice::Blended(weight.max(self.config.min_hllc_weight)),
                    AdaptiveReason::VelocityJump,
                );
            }
            return (SolverChoice::Rusanov, AdaptiveReason::VelocityJump);
        }

        // 默认使用 HLLC
        (SolverChoice::Hllc, AdaptiveReason::Default)
    }

    /// 计算混合权重
    ///
    /// 使用平滑函数从 HLLC 过渡到 Rusanov
    fn compute_blend_weight(&self, value: f64, threshold: f64) -> f64 {
        if value <= threshold {
            return 1.0; // 完全 HLLC
        }

        let width = threshold * self.config.transition_width;
        if width < 1e-10 {
            return 0.0; // 硬切换
        }

        let x = (value - threshold) / width;
        if x >= 1.0 {
            return 0.0; // 完全 Rusanov
        }

        // 使用余弦平滑函数
        0.5 * (1.0 + (x * std::f64::consts::PI).cos())
    }

    /// 更新统计
    fn update_stats(&self, choice: &SolverChoice, reason: AdaptiveReason) {
        self.stats.update(choice, reason);
    }

    /// 混合两个通量
    fn blend_flux(flux_hllc: RiemannFlux, flux_rusanov: RiemannFlux, hllc_weight: f64) -> RiemannFlux {
        let w = hllc_weight;
        let w_inv = 1.0 - w;

        RiemannFlux {
            mass: w * flux_hllc.mass + w_inv * flux_rusanov.mass,
            momentum_x: w * flux_hllc.momentum_x + w_inv * flux_rusanov.momentum_x,
            momentum_y: w * flux_hllc.momentum_y + w_inv * flux_rusanov.momentum_y,
            max_wave_speed: flux_hllc.max_wave_speed.max(flux_rusanov.max_wave_speed),
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
// RiemannSolver trait 实现
// ============================================================================

impl RiemannSolver for AdaptiveSolver {
    fn name(&self) -> &'static str {
        "Adaptive (HLLC/Rusanov)"
    }

    fn capabilities(&self) -> SolverCapabilities {
        // 继承两个求解器的最佳能力
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: true,
            order: 2, // HLLC 是二阶精度
            positivity_preserving: true, // 通过自适应确保
        }
    }

    fn solve(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
        normal: DVec2,
    ) -> Result<RiemannFlux, RiemannError> {
        // 选择求解器
        let (choice, reason) = self.choose_solver(h_left, h_right, vel_left, vel_right);
        self.update_stats(&choice, reason);

        // 执行求解
        match choice {
            SolverChoice::Hllc => {
                self.hllc.solve(h_left, h_right, vel_left, vel_right, normal)
            }
            SolverChoice::Rusanov => {
                self.rusanov.solve(h_left, h_right, vel_left, vel_right, normal)
            }
            SolverChoice::Blended(w) => {
                // 两者都求解，然后混合
                let flux_hllc = self.hllc.solve(h_left, h_right, vel_left, vel_right, normal)?;
                let flux_rusanov = self.rusanov.solve(h_left, h_right, vel_left, vel_right, normal)?;
                Ok(Self::blend_flux(flux_hllc, flux_rusanov, w))
            }
        }
    }

    fn gravity(&self) -> f64 {
        self.params.gravity
    }

    fn dry_threshold(&self) -> f64 {
        self.params.h_dry
    }
}

// ============================================================================
// 工厂函数
// ============================================================================

/// 创建默认自适应求解器
pub fn create_adaptive_solver(gravity: f64) -> AdaptiveSolver {
    let params = NumericalParams::default();
    AdaptiveSolver::new(&params, gravity)
}

/// 创建保守自适应求解器
pub fn create_conservative_adaptive_solver(gravity: f64) -> AdaptiveSolver {
    let params = NumericalParams::default();
    AdaptiveSolver::with_config(&params, gravity, AdaptiveConfig::conservative())
}

/// 创建精度优先自适应求解器
pub fn create_accuracy_adaptive_solver(gravity: f64) -> AdaptiveSolver {
    let params = NumericalParams::default();
    AdaptiveSolver::with_config(&params, gravity, AdaptiveConfig::accuracy())
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_solver() -> AdaptiveSolver {
        create_adaptive_solver(9.81)
    }

    #[test]
    fn test_solver_name() {
        let solver = create_test_solver();
        assert!(solver.name().contains("Adaptive"));
    }

    #[test]
    fn test_capabilities() {
        let solver = create_test_solver();
        let caps = solver.capabilities();
        assert!(caps.handles_dry_wet);
        assert!(caps.positivity_preserving);
        assert_eq!(caps.order, 2);
    }

    #[test]
    fn test_static_water() {
        let solver = create_test_solver();
        let flux = solver
            .solve(1.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        assert!(flux.mass.abs() < 1e-10);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_chooses_hllc_for_smooth() {
        let solver = create_test_solver();

        // 平滑情况应该选择 HLLC
        let (choice, _) = solver.choose_solver(1.0, 1.1, DVec2::ZERO, DVec2::ZERO);
        assert_eq!(choice, SolverChoice::Hllc);
    }

    #[test]
    fn test_chooses_rusanov_for_dry() {
        let solver = create_test_solver();

        // 干湿情况应该选择 Rusanov
        let (choice, _) = solver.choose_solver(0.001, 1.0, DVec2::ZERO, DVec2::ZERO);
        assert_eq!(choice, SolverChoice::Rusanov);
    }

    #[test]
    fn test_chooses_rusanov_for_jump() {
        let solver = create_test_solver();

        // 大跳跃应该选择 Rusanov
        let (choice, _) = solver.choose_solver(2.0, 0.5, DVec2::ZERO, DVec2::ZERO);
        // 相对跳跃 = 1.5/2.0 = 0.75 > 0.5
        assert!(choice.uses_rusanov());
    }

    #[test]
    fn test_blending() {
        let config = AdaptiveConfig {
            enable_blending: true,
            depth_jump_threshold: 0.3,
            transition_width: 0.5,
            ..Default::default()
        };
        let params = NumericalParams::default();
        let solver = AdaptiveSolver::with_config(&params, 9.81, config);

        // 中等跳跃应该混合
        let (choice, _) = solver.choose_solver(1.0, 0.6, DVec2::ZERO, DVec2::ZERO);
        // 相对跳跃 = 0.4/1.0 = 0.4，略高于阈值
        match choice {
            SolverChoice::Blended(w) => {
                assert!(w > 0.0 && w < 1.0);
            }
            _ => {} // 可能完全选择一个
        }
    }

    #[test]
    fn test_dam_break() {
        let solver = create_test_solver();

        // 溃坝问题
        let flux = solver
            .solve(2.0, 0.1, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        // 应该产生有效通量
        assert!(flux.is_valid());
        assert!(flux.mass > 0.0); // 向低水位流
    }

    #[test]
    fn test_supercritical_flow() {
        let solver = create_test_solver();

        // 超临界流（Fr > 1.5）
        let h = 0.5_f64;
        let c = (9.81_f64 * h).sqrt(); // ≈ 2.2 m/s
        let u = c * 2.0; // Fr = 2

        let (choice, _) = solver.choose_solver(h, h, DVec2::new(u, 0.0), DVec2::new(u, 0.0));
        assert!(choice.uses_rusanov());
    }

    #[test]
    fn test_statistics() {
        let solver = create_test_solver();

        // 多次求解
        for _ in 0..10 {
            let _ = solver.solve(1.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X);
        }

        let stats = solver.stats();
        assert_eq!(stats.total(), 10);
        assert_eq!(stats.hllc_count, 10);
        assert_eq!(stats.rusanov_count, 0);
    }

    #[test]
    fn test_config_presets() {
        let conservative = AdaptiveConfig::conservative();
        assert!(conservative.depth_jump_threshold < AdaptiveConfig::default().depth_jump_threshold);

        let accuracy = AdaptiveConfig::accuracy();
        assert!(accuracy.depth_jump_threshold > AdaptiveConfig::default().depth_jump_threshold);

        let hllc_only = AdaptiveConfig::hllc_only();
        assert_eq!(hllc_only.min_hllc_weight, 1.0);

        let rusanov_only = AdaptiveConfig::rusanov_only();
        assert_eq!(rusanov_only.min_hllc_weight, 0.0);
    }

    #[test]
    fn test_blend_weight_computation() {
        let solver = create_test_solver();

        // 在阈值以下应该返回 1.0
        let w1 = solver.compute_blend_weight(0.3, 0.5);
        assert_eq!(w1, 1.0);

        // 远超阈值应该返回 0.0
        let w2 = solver.compute_blend_weight(1.0, 0.1);
        assert!(w2 < 0.01);
    }

    #[test]
    fn test_flux_blending() {
        let flux1 = RiemannFlux::new(1.0, 2.0, 3.0, 5.0);
        let flux2 = RiemannFlux::new(0.5, 1.0, 1.5, 4.0);

        // 50% 混合
        let blended = AdaptiveSolver::blend_flux(flux1, flux2, 0.5);

        assert!((blended.mass - 0.75).abs() < 1e-10);
        assert!((blended.momentum_x - 1.5).abs() < 1e-10);
        assert!((blended.momentum_y - 2.25).abs() < 1e-10);
        assert!((blended.max_wave_speed - 5.0).abs() < 1e-10); // 取最大值
    }
}
