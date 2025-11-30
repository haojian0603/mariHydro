// src-tauri/src/marihydro/physics/schemes/riemann/adaptive.rs

//! 自适应黎曼求解器选择器
//!
//! 根据局部流场特征自动选择最合适的求解器。

use super::hllc::HllcSolver;
use super::rusanov::RusanovSolver;
use super::traits::{RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use super::SolverType;
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;
use std::sync::atomic::{AtomicU64, Ordering};

/// 求解器选择标准
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SelectionCriterion {
    /// 基于 Froude 数
    #[default]
    FroudeNumber,
    /// 基于水深比
    DepthRatio,
    /// 基于速度梯度
    VelocityGradient,
    /// 总是使用 HLLC
    AlwaysHllc,
    /// 总是使用 Rusanov
    AlwaysRusanov,
}

/// 选择器配置
#[derive(Debug, Clone, Copy)]
pub struct SelectorConfig {
    /// 选择标准
    pub criterion: SelectionCriterion,
    /// Froude 数阈值（超过此值使用 Rusanov）
    pub froude_threshold: f64,
    /// 水深比阈值
    pub depth_ratio_threshold: f64,
    /// 速度梯度阈值
    pub velocity_gradient_threshold: f64,
}

impl Default for SelectorConfig {
    fn default() -> Self {
        Self {
            criterion: SelectionCriterion::default(),
            froude_threshold: 0.8,      // Fr > 0.8 使用 Rusanov
            depth_ratio_threshold: 10.0, // h_max/h_min > 10 使用 Rusanov
            velocity_gradient_threshold: 5.0, // |du| > 5 m/s 使用 Rusanov
        }
    }
}

/// 求解器选择器
#[derive(Debug)]
pub struct SolverSelector {
    config: SelectorConfig,
    params: SolverParams,

    // 统计信息
    hllc_count: AtomicU64,
    rusanov_count: AtomicU64,
}

impl SolverSelector {
    /// 创建选择器
    pub fn new(params: SolverParams) -> Self {
        Self {
            config: SelectorConfig::default(),
            params,
            hllc_count: AtomicU64::new(0),
            rusanov_count: AtomicU64::new(0),
        }
    }

    /// 设置配置
    pub fn with_config(mut self, config: SelectorConfig) -> Self {
        self.config = config;
        self
    }

    /// 选择求解器类型
    pub fn select(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
    ) -> SolverType {
        match self.config.criterion {
            SelectionCriterion::AlwaysHllc => SolverType::Hllc,
            SelectionCriterion::AlwaysRusanov => SolverType::Rusanov,
            SelectionCriterion::FroudeNumber => {
                self.select_by_froude(h_left, h_right, vel_left, vel_right)
            }
            SelectionCriterion::DepthRatio => self.select_by_depth_ratio(h_left, h_right),
            SelectionCriterion::VelocityGradient => {
                self.select_by_velocity_gradient(vel_left, vel_right)
            }
        }
    }

    /// 基于 Froude 数选择
    fn select_by_froude(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
    ) -> SolverType {
        let g = self.params.gravity;
        let h_avg = 0.5 * (h_left + h_right).max(self.params.h_min);
        let v_avg = 0.5 * (vel_left.length() + vel_right.length());
        let c = (g * h_avg).sqrt();
        let fr = v_avg / c;

        if fr > self.config.froude_threshold {
            SolverType::Rusanov
        } else {
            SolverType::Hllc
        }
    }

    /// 基于水深比选择
    fn select_by_depth_ratio(&self, h_left: f64, h_right: f64) -> SolverType {
        let h_max = h_left.max(h_right);
        let h_min = h_left.min(h_right).max(self.params.h_min);
        let ratio = h_max / h_min;

        if ratio > self.config.depth_ratio_threshold {
            SolverType::Rusanov
        } else {
            SolverType::Hllc
        }
    }

    /// 基于速度梯度选择
    fn select_by_velocity_gradient(&self, vel_left: DVec2, vel_right: DVec2) -> SolverType {
        let dv = (vel_right - vel_left).length();

        if dv > self.config.velocity_gradient_threshold {
            SolverType::Rusanov
        } else {
            SolverType::Hllc
        }
    }

    /// 获取统计信息
    pub fn statistics(&self) -> (u64, u64) {
        (
            self.hllc_count.load(Ordering::Relaxed),
            self.rusanov_count.load(Ordering::Relaxed),
        )
    }

    /// 重置统计
    pub fn reset_statistics(&self) {
        self.hllc_count.store(0, Ordering::Relaxed);
        self.rusanov_count.store(0, Ordering::Relaxed);
    }

    /// 记录选择
    fn record_selection(&self, solver_type: SolverType) {
        match solver_type {
            SolverType::Hllc => {
                self.hllc_count.fetch_add(1, Ordering::Relaxed);
            }
            SolverType::Rusanov => {
                self.rusanov_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

/// 自适应黎曼求解器
///
/// 根据局部条件在 HLLC 和 Rusanov 之间自动切换
#[derive(Debug)]
pub struct AdaptiveSolver {
    hllc: HllcSolver,
    rusanov: RusanovSolver,
    selector: SolverSelector,
}

impl AdaptiveSolver {
    /// 创建自适应求解器
    pub fn new(numerical_params: NumericalParams, gravity: f64) -> Self {
        let params = SolverParams::from_numerical(&numerical_params, gravity);
        Self {
            hllc: HllcSolver::from_params(params),
            rusanov: RusanovSolver::from_params(params),
            selector: SolverSelector::new(params),
        }
    }

    /// 从参数创建
    pub fn from_params(params: SolverParams) -> Self {
        Self {
            hllc: HllcSolver::from_params(params),
            rusanov: RusanovSolver::from_params(params),
            selector: SolverSelector::new(params),
        }
    }

    /// 设置选择器配置
    pub fn with_selector_config(mut self, config: SelectorConfig) -> Self {
        self.selector = self.selector.with_config(config);
        self
    }

    /// 获取统计信息
    pub fn statistics(&self) -> (u64, u64) {
        self.selector.statistics()
    }

    /// 重置统计
    pub fn reset_statistics(&self) {
        self.selector.reset_statistics();
    }
}

impl RiemannSolver for AdaptiveSolver {
    fn name(&self) -> &'static str {
        "Adaptive"
    }

    fn capabilities(&self) -> SolverCapabilities {
        // 返回较保守的能力（两者的交集）
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: true,
            order: 1, // 使用较低阶
            positivity_preserving: true,
        }
    }

    fn solve(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
        normal: DVec2,
    ) -> MhResult<RiemannFlux> {
        // 选择求解器
        let solver_type = self.selector.select(h_left, h_right, vel_left, vel_right);
        self.selector.record_selection(solver_type);

        // 使用选定的求解器
        match solver_type {
            SolverType::Hllc => self.hllc.solve(h_left, h_right, vel_left, vel_right, normal),
            SolverType::Rusanov => self
                .rusanov
                .solve(h_left, h_right, vel_left, vel_right, normal),
            _ => self.hllc.solve(h_left, h_right, vel_left, vel_right, normal),
        }
    }

    fn gravity(&self) -> f64 {
        self.hllc.gravity()
    }

    fn dry_threshold(&self) -> f64 {
        self.hllc.dry_threshold()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selector_froude() {
        let selector = SolverSelector::new(SolverParams::default());

        // 低 Froude 数（亚临界流）
        let h = 10.0;
        let slow_vel = DVec2::new(1.0, 0.0); // ~0.1 Fr
        assert_eq!(
            selector.select(h, h, slow_vel, slow_vel),
            SolverType::Hllc
        );

        // 高 Froude 数（超临界流）
        let fast_vel = DVec2::new(15.0, 0.0); // ~1.5 Fr
        assert_eq!(
            selector.select(h, h, fast_vel, fast_vel),
            SolverType::Rusanov
        );
    }

    #[test]
    fn test_selector_depth_ratio() {
        let config = SelectorConfig {
            criterion: SelectionCriterion::DepthRatio,
            ..Default::default()
        };
        let selector = SolverSelector::new(SolverParams::default()).with_config(config);

        // 小水深比
        assert_eq!(
            selector.select(5.0, 4.0, DVec2::ZERO, DVec2::ZERO),
            SolverType::Hllc
        );

        // 大水深比（溃坝）
        assert_eq!(
            selector.select(10.0, 0.1, DVec2::ZERO, DVec2::ZERO),
            SolverType::Rusanov
        );
    }

    #[test]
    fn test_adaptive_solver() {
        let solver = AdaptiveSolver::from_params(SolverParams::default());

        // 静水
        let flux = solver
            .solve(5.0, 5.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert!(flux.mass.abs() < 1e-10);

        // 溃坝
        let flux = solver
            .solve(10.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert!(flux.mass > 0.0);

        // 检查统计
        let (hllc, rusanov) = solver.statistics();
        assert!(hllc + rusanov == 2);
    }
}
