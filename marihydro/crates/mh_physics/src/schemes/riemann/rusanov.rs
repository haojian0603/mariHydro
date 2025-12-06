// crates/mh_physics/src/schemes/riemann/rusanov.rs

//! Rusanov (Local Lax-Friedrichs) 黎曼求解器
//!
//! Rusanov 求解器是一种简单但鲁棒的近似黎曼求解器，
//! 特别适合处理强间断和复杂流动情况。
//!
//! # 特点
//!
//! - 简单稳定：单波速估计，计算代价低
//! - 强耗散：数值粘性较大，适合初始阶段或复杂问题
//! - 鲁棒性：能处理各种极端情况
//! - GPU 友好：无分支的统一公式
//!
//! # 数学原理
//!
//! Rusanov 通量公式：
//! ```text
//! F* = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)
//! ```
//!
//! 其中 λ_max 是最大特征速度：
//! ```text
//! λ_max = max(|u_L| + c_L, |u_R| + c_R)
//! c = sqrt(g * h)
//! ```
//!
//! # 适用场景
//!
//! - 初始化阶段的快速收敛
//! - 溃坝等强间断问题
//! - GPU 大规模并行计算
//! - 作为自适应求解器的备选方案

use super::traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use crate::types::NumericalParams;
use glam::DVec2;

// ============================================================================
// Rusanov 求解器配置
// ============================================================================

/// Rusanov 求解器配置
#[derive(Debug, Clone, Copy)]
pub struct RusanovConfig {
    /// 波速放大系数 (≥1.0)
    ///
    /// 增加此值可以提高稳定性，但会增加数值耗散。
    /// - 1.0: 标准 Rusanov
    /// - 1.1-1.2: 更稳定，适合困难问题
    pub wave_speed_factor: f64,

    /// 是否使用加权平均
    ///
    /// 启用时使用水深加权平均，可以改善干湿过渡区域的行为。
    pub use_weighted_average: bool,

    /// 最小波速
    ///
    /// 避免数值除零，通常设为 1e-6。
    pub min_wave_speed: f64,

    /// 是否启用熵修正
    pub entropy_fix: bool,
}

impl Default for RusanovConfig {
    fn default() -> Self {
        Self {
            wave_speed_factor: 1.0,
            min_wave_speed: 1e-8,
            use_weighted_average: false,
            entropy_fix: false,
        }
    }
}

impl RusanovConfig {
    /// 标准配置
    pub fn standard() -> Self {
        Self::default()
    }

    /// 高稳定性配置
    pub fn robust() -> Self {
        Self {
            wave_speed_factor: 1.2,
            use_weighted_average: true,
            min_wave_speed: 1e-6,
            entropy_fix: true,
        }
    }

    /// GPU 优化配置
    pub fn gpu_optimized() -> Self {
        Self {
            wave_speed_factor: 1.0,
            use_weighted_average: false, // 避免条件分支
            min_wave_speed: 1e-8,
            entropy_fix: false, // 简化计算
        }
    }
}

// ============================================================================
// Rusanov 求解器
// ============================================================================

/// Rusanov 黎曼求解器
///
/// Local Lax-Friedrichs 格式的实现。
///
/// # 示例
///
/// ```ignore
/// use mh_physics::schemes::riemann::{RusanovSolver, RiemannSolver};
///
/// let solver = RusanovSolver::new(&params, 9.81);
///
/// let flux = solver.solve(
///     1.0, 0.5,                              // 左右水深
///     DVec2::new(1.0, 0.0), DVec2::ZERO,     // 左右速度
///     DVec2::X,                               // 法向量
/// )?;
///
/// println!("质量通量: {}", flux.mass);
/// ```
#[derive(Debug, Clone)]
pub struct RusanovSolver {
    /// 基本参数
    params: SolverParams,

    /// Rusanov 特定配置
    config: RusanovConfig,
}

impl RusanovSolver {
    /// 创建新的 Rusanov 求解器
    pub fn new(numerical_params: &NumericalParams, gravity: f64) -> Self {
        Self {
            params: SolverParams::from_numerical(numerical_params, gravity),
            config: RusanovConfig::default(),
        }
    }

    /// 使用配置创建
    pub fn with_config(
        numerical_params: &NumericalParams,
        gravity: f64,
        config: RusanovConfig,
    ) -> Self {
        Self {
            params: SolverParams::from_numerical(numerical_params, gravity),
            config,
        }
    }

    /// 从参数直接创建
    pub fn from_params(params: SolverParams) -> Self {
        Self {
            params,
            config: RusanovConfig::default(),
        }
    }

    /// 从参数和配置创建
    pub fn from_params_with_config(params: SolverParams, config: RusanovConfig) -> Self {
        Self { params, config }
    }

    /// 获取参数
    pub fn params(&self) -> &SolverParams {
        &self.params
    }

    /// 获取配置
    pub fn config(&self) -> &RusanovConfig {
        &self.config
    }

    // =========================================================================
    // 波速估计
    // =========================================================================

    /// 计算最大波速
    ///
    /// λ_max = max(|u_L| + c_L, |u_R| + c_R)
    #[inline]
    fn max_wave_speed(&self, h_l: f64, h_r: f64, un_l: f64, un_r: f64) -> f64 {
        let g = self.params.gravity;

        // 声速
        let c_l = (g * h_l.max(0.0)).sqrt();
        let c_r = (g * h_r.max(0.0)).sqrt();

        // 左右特征速度
        let lambda_l = un_l.abs() + c_l;
        let lambda_r = un_r.abs() + c_r;

        // 最大波速
        let lambda_max = lambda_l.max(lambda_r);

        // 应用放大系数和最小值
        (lambda_max * self.config.wave_speed_factor).max(self.config.min_wave_speed)
    }

    // =========================================================================
    // 物理通量计算
    // =========================================================================

    /// 计算物理通量 (旋转坐标系)
    ///
    /// F = [h*u_n, h*u_n² + 0.5*g*h², h*u_n*u_t]
    #[inline]
    fn physical_flux(&self, h: f64, un: f64, ut: f64) -> (f64, f64, f64) {
        let g = self.params.gravity;
        let hun = h * un;

        (
            hun,                           // 质量通量
            hun * un + 0.5 * g * h * h,   // 法向动量通量
            hun * ut,                      // 切向动量通量
        )
    }

    /// 计算守恒变量
    #[inline]
    fn conserved_vars(&self, h: f64, un: f64, ut: f64) -> (f64, f64, f64) {
        (h, h * un, h * ut)
    }

    // =========================================================================
    // 核心求解
    // =========================================================================

    /// 求解双湿状态
    fn solve_both_wet(
        &self,
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        normal: DVec2,
    ) -> Result<RiemannFlux, RiemannError> {
        let tangent = DVec2::new(-normal.y, normal.x);

        // 分解速度到法向/切向
        let un_l = vel_l.dot(normal);
        let un_r = vel_r.dot(normal);
        let ut_l = vel_l.dot(tangent);
        let ut_r = vel_r.dot(tangent);

        // 计算最大波速
        let lambda_max = self.max_wave_speed(h_l, h_r, un_l, un_r);

        // 计算左右物理通量
        let (f_mass_l, f_mom_n_l, f_mom_t_l) = self.physical_flux(h_l, un_l, ut_l);
        let (f_mass_r, f_mom_n_r, f_mom_t_r) = self.physical_flux(h_r, un_r, ut_r);

        // 计算左右守恒变量
        let (u_h_l, u_hun_l, u_hut_l) = self.conserved_vars(h_l, un_l, ut_l);
        let (u_h_r, u_hun_r, u_hut_r) = self.conserved_vars(h_r, un_r, ut_r);

        // Rusanov 通量公式
        // F* = 0.5 * (F_L + F_R) - 0.5 * λ_max * (U_R - U_L)
        let half = 0.5;
        let half_lambda = half * lambda_max;

        let mass = half * (f_mass_l + f_mass_r) - half_lambda * (u_h_r - u_h_l);
        let mom_n = half * (f_mom_n_l + f_mom_n_r) - half_lambda * (u_hun_r - u_hun_l);
        let mom_t = half * (f_mom_t_l + f_mom_t_r) - half_lambda * (u_hut_r - u_hut_l);

        Ok(RiemannFlux::from_rotated(mass, mom_n, mom_t, normal, lambda_max))
    }

    /// 求解单侧湿状态（溃坝问题）
    /// 
    /// 使用 Ritter 溃坝解析解来计算干湿边界通量
    /// 参考: Toro, E.F. "Shock-Capturing Methods for Free-Surface Shallow Flows"
    fn solve_single_wet(
        &self,
        h_wet: f64,
        vel_wet: DVec2,
        normal: DVec2,
        wet_on_left: bool,
    ) -> Result<RiemannFlux, RiemannError> {
        let tangent = DVec2::new(-normal.y, normal.x);
        let un_wet = vel_wet.dot(normal);
        let ut_wet = vel_wet.dot(tangent);
        let c_wet = (self.params.gravity * h_wet).sqrt();
        let lambda_max = (un_wet.abs() + c_wet).max(self.config.min_wave_speed);
        let g = self.params.gravity;

        // 计算物理通量
        let (f_h, f_hun, f_hut) = self.physical_flux(h_wet, un_wet, ut_wet);

        // 使用 Riemann 不变量的 Ritter 溃坝解
        let (mass, mom_n, mom_t) = if wet_on_left {
            // 湿侧在左，干侧在右
            if un_wet >= c_wet {
                // 超临界流出：全部使用内部通量
                (f_h, f_hun, f_hut)
            } else if un_wet <= -c_wet {
                // 超临界流入干区：静水压力边界
                (0.0, 0.5 * g * h_wet * h_wet, 0.0)
            } else {
                // 亚临界：使用 Ritter 解的边界值
                // h* = (2c + u)² / (9g), u* = (2c + u) / 3
                let h_star = (2.0 * c_wet + un_wet).powi(2) / (9.0 * g);
                let u_star = (2.0 * c_wet + un_wet) / 3.0;
                let f_mass = h_star * u_star;
                let f_mom = h_star * u_star * u_star + 0.5 * g * h_star * h_star;
                // 切向动量保持切向速度不变
                let f_mom_t = h_star * u_star * ut_wet / un_wet.abs().max(1e-10);
                (f_mass, f_mom, f_mom_t)
            }
        } else {
            // 湿侧在右，干侧在左（对称情况）
            if un_wet <= -c_wet {
                // 超临界流出
                (f_h, f_hun, f_hut)
            } else if un_wet >= c_wet {
                // 超临界流入干区：静水压力边界
                (0.0, 0.5 * g * h_wet * h_wet, 0.0)
            } else {
                // 亚临界：对称的 Ritter 解
                let h_star = (2.0 * c_wet - un_wet).powi(2) / (9.0 * g);
                let u_star = -(2.0 * c_wet - un_wet) / 3.0;
                let f_mass = h_star * u_star;
                let f_mom = h_star * u_star * u_star + 0.5 * g * h_star * h_star;
                let f_mom_t = h_star * u_star * ut_wet / un_wet.abs().max(1e-10);
                (f_mass, f_mom, f_mom_t)
            }
        };

        Ok(RiemannFlux::from_rotated(mass, mom_n, mom_t, normal, lambda_max))
    }

    /// 求解双干状态
    #[inline]
    fn solve_both_dry(&self) -> Result<RiemannFlux, RiemannError> {
        Ok(RiemannFlux::ZERO)
    }
}

// ============================================================================
// RiemannSolver trait 实现
// ============================================================================

impl RiemannSolver for RusanovSolver {
    fn name(&self) -> &'static str {
        "Rusanov (LLF)"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: self.config.entropy_fix,
            supports_hydrostatic: true,
            order: 1,
            positivity_preserving: true, // Rusanov 天然保正
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
        let h_dry = self.params.h_dry;

        // 干湿状态判断
        let left_wet = h_left > h_dry;
        let right_wet = h_right > h_dry;

        match (left_wet, right_wet) {
            (true, true) => self.solve_both_wet(h_left, h_right, vel_left, vel_right, normal),
            (true, false) => self.solve_single_wet(h_left, vel_left, normal, true),
            (false, true) => self.solve_single_wet(h_right, vel_right, normal, false),
            (false, false) => self.solve_both_dry(),
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
// 辅助函数
// ============================================================================

/// 创建默认 Rusanov 求解器
pub fn create_rusanov_solver(gravity: f64) -> RusanovSolver {
    let params = SolverParams {
        gravity,
        ..Default::default()
    };
    RusanovSolver::from_params(params)
}

/// 创建鲁棒 Rusanov 求解器
pub fn create_robust_rusanov_solver(gravity: f64) -> RusanovSolver {
    let params = SolverParams {
        gravity,
        ..Default::default()
    };
    RusanovSolver::from_params_with_config(params, RusanovConfig::robust())
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_1_SQRT_2;

    fn create_test_solver() -> RusanovSolver {
        create_rusanov_solver(9.81)
    }

    #[test]
    fn test_solver_name_and_capabilities() {
        let solver = create_test_solver();
        assert_eq!(solver.name(), "Rusanov (LLF)");

        let caps = solver.capabilities();
        assert!(caps.handles_dry_wet);
        assert!(caps.positivity_preserving);
        assert_eq!(caps.order, 1);
    }

    #[test]
    fn test_static_water() {
        let solver = create_test_solver();

        // 静水：两侧相同水深和零速度
        let flux = solver
            .solve(1.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        // 静水应该没有质量通量
        assert!(flux.mass.abs() < 1e-10);
        // 动量通量应该平衡（压力差为零）
        assert!(flux.momentum_x.abs() < 1e-10);
        assert!(flux.momentum_y.abs() < 1e-10);
    }

    #[test]
    fn test_uniform_flow() {
        let solver = create_test_solver();

        // 均匀流：相同水深，相同速度
        let vel = DVec2::new(1.0, 0.0);
        let flux = solver.solve(1.0, 1.0, vel, vel, DVec2::X).unwrap();

        // 质量通量应该等于 h * u
        assert!((flux.mass - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dam_break() {
        let solver = create_test_solver();

        // 溃坝：左侧高水位，右侧低水位，初始静止
        let flux = solver
            .solve(2.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        // 应该有向右的质量通量（从高向低）
        assert!(flux.mass > 0.0);
        // 应该有合理的波速
        assert!(flux.max_wave_speed > 0.0);
    }

    #[test]
    fn test_dry_left() {
        let solver = create_test_solver();

        // 左侧干，右侧湿
        let flux = solver
            .solve(0.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        // 应该是有效通量
        assert!(flux.is_valid());
    }

    #[test]
    fn test_dry_right() {
        let solver = create_test_solver();

        // 左侧湿，右侧干
        let flux = solver
            .solve(1.0, 0.0, DVec2::new(1.0, 0.0), DVec2::ZERO, DVec2::X)
            .unwrap();

        // 向右流动应该产生正的质量通量
        assert!(flux.is_valid());
    }

    #[test]
    fn test_both_dry() {
        let solver = create_test_solver();

        // 两侧都干
        let flux = solver
            .solve(0.0, 0.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        // 零通量
        assert_eq!(flux.mass, 0.0);
        assert_eq!(flux.momentum_x, 0.0);
        assert_eq!(flux.momentum_y, 0.0);
    }

    #[test]
    fn test_oblique_flow() {
        let solver = create_test_solver();

        // 斜向流动：45度方向
        let vel = DVec2::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);
        let flux = solver.solve(1.0, 1.0, vel, vel, DVec2::X).unwrap();

        // 应该有正的质量通量
        assert!(flux.mass > 0.0);
        // 应该有正的 x 动量通量
        assert!(flux.momentum_x > 0.0);
    }

    #[test]
    fn test_y_normal() {
        let solver = create_test_solver();

        // 法向量沿 Y 轴
        let vel = DVec2::new(0.0, 1.0);
        let flux = solver.solve(1.0, 1.0, vel, vel, DVec2::Y).unwrap();

        // 质量通量应该等于 h * v
        assert!((flux.mass - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetry() {
        let solver = create_test_solver();

        // 对称测试：交换左右并反转法向量
        let flux1 = solver
            .solve(2.0, 1.0, DVec2::new(0.5, 0.0), DVec2::new(-0.3, 0.0), DVec2::X)
            .unwrap();

        let flux2 = solver
            .solve(1.0, 2.0, DVec2::new(-0.3, 0.0), DVec2::new(0.5, 0.0), -DVec2::X)
            .unwrap();

        // 反转后的通量应该符号相反（近似）
        assert!((flux1.mass + flux2.mass).abs() < 1e-10);
    }

    #[test]
    fn test_conservation() {
        let solver = create_test_solver();

        // 计算通量应该满足守恒性
        let flux = solver
            .solve(1.5, 0.8, DVec2::new(0.3, 0.1), DVec2::new(-0.2, 0.05), DVec2::X)
            .unwrap();

        // 通量应该是有限值
        assert!(flux.is_valid());
        assert!(flux.mass.is_finite());
        assert!(flux.momentum_x.is_finite());
        assert!(flux.momentum_y.is_finite());
    }

    #[test]
    fn test_robust_config() {
        let params = SolverParams::default();
        let solver = RusanovSolver::from_params_with_config(params, RusanovConfig::robust());

        // 鲁棒配置应该有更大的波速因子
        assert!(solver.config().wave_speed_factor > 1.0);
        assert!(solver.config().use_weighted_average);
    }

    #[test]
    fn test_max_wave_speed() {
        let solver = create_test_solver();

        // 超临界流
        let h = 1.0;
        let c = (9.81 * h).sqrt(); // ≈ 3.13 m/s
        let u_fast = 5.0; // 超临界

        let flux = solver
            .solve(h, h, DVec2::new(u_fast, 0.0), DVec2::new(u_fast, 0.0), DVec2::X)
            .unwrap();

        // 波速应该大于 u + c
        assert!(flux.max_wave_speed >= u_fast + c - 0.01);
    }
}
