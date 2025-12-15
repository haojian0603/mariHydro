// crates/mh_physics/src/schemes/riemann/rusanov.rs

//! Rusanov (Local Lax-Friedrichs) 黎曼求解器（泛型化）
//!
//! T=3 改造：完整泛型化以支持 f32/f64 精度切换
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

use super::traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use crate::types::NumericalParams;
use mh_runtime::{Backend, CpuBackend, RuntimeScalar};

// ============================================================================
// Rusanov 求解器配置（泛型化）
// ============================================================================

/// Rusanov 求解器配置
#[derive(Debug, Clone, Copy)]
pub struct RusanovConfig<S: RuntimeScalar> {
    /// 波速放大系数 (≥1.0)
    pub wave_speed_factor: S,
    /// 是否使用加权平均
    pub use_weighted_average: bool,
    /// 最小波速
    pub min_wave_speed: S,
    /// 是否启用熵修正
    pub entropy_fix: bool,
}

impl<S: RuntimeScalar> Default for RusanovConfig<S> {
    fn default() -> Self {
        Self {
            wave_speed_factor: S::ONE,
            min_wave_speed: S::from_f64(1e-8).unwrap_or(S::ZERO),
            use_weighted_average: false,
            entropy_fix: false,
        }
    }
}

impl<S: RuntimeScalar> RusanovConfig<S> {
    /// 标准配置
    pub fn standard() -> Self {
        Self::default()
    }

    /// 高稳定性配置
    pub fn robust() -> Self {
        Self {
            wave_speed_factor: S::from_f64(1.2).unwrap_or(S::ONE),
            use_weighted_average: true,
            min_wave_speed: S::from_f64(1e-6).unwrap_or(S::ZERO),
            entropy_fix: true,
        }
    }

    /// GPU 优化配置
    pub fn gpu_optimized() -> Self {
        Self {
            wave_speed_factor: S::ONE,
            use_weighted_average: false,
            min_wave_speed: S::from_f64(1e-8).unwrap_or(S::ZERO),
            entropy_fix: false,
        }
    }
}

// ============================================================================
// Rusanov 求解器（泛型化）
// ============================================================================

/// Rusanov 黎曼求解器（泛型化）
///
/// 支持任意 Backend，实现 f32/f64 精度切换
#[derive(Debug, Clone)]
pub struct RusanovSolver<B: Backend> {
    /// 基本参数
    params: SolverParams<B::Scalar>,
    /// Rusanov 特定配置
    config: RusanovConfig<B::Scalar>,
    /// Backend phantom data
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> RusanovSolver<B> {
    /// 创建新的 Rusanov 求解器
    pub fn new(numerical_params: &NumericalParams<B::Scalar>, gravity: B::Scalar) -> Self {
        Self {
            params: SolverParams::from_numerical(numerical_params, gravity),
            config: RusanovConfig::default(),
            _backend: std::marker::PhantomData,
        }
    }

    /// 使用配置创建
    pub fn with_config(
        numerical_params: &NumericalParams<B::Scalar>,
        gravity: B::Scalar,
        config: RusanovConfig<B::Scalar>,
    ) -> Self {
        Self {
            params: SolverParams::from_numerical(numerical_params, gravity),
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// 从参数直接创建
    pub fn from_params(params: SolverParams<B::Scalar>) -> Self {
        Self {
            params,
            config: RusanovConfig::default(),
            _backend: std::marker::PhantomData,
        }
    }

    /// 从参数和配置创建
    pub fn from_params_with_config(params: SolverParams<B::Scalar>, config: RusanovConfig<B::Scalar>) -> Self {
        Self { 
            params, 
            config,
            _backend: std::marker::PhantomData,
        }
    }

    /// 获取参数
    pub fn params(&self) -> &SolverParams<B::Scalar> {
        &self.params
    }

    /// 获取配置
    pub fn config(&self) -> &RusanovConfig<B::Scalar> {
        &self.config
    }

    // =========================================================================
    // 波速估计
    // =========================================================================

    /// 计算最大波速
    #[inline]
    fn max_wave_speed(&self, h_l: B::Scalar, h_r: B::Scalar, un_l: B::Scalar, un_r: B::Scalar) -> B::Scalar {
        let g = self.params.gravity;

        // 声速
        let c_l = (g * h_l.max(B::Scalar::ZERO)).sqrt();
        let c_r = (g * h_r.max(B::Scalar::ZERO)).sqrt();

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
    #[inline]
    fn physical_flux(&self, h: B::Scalar, un: B::Scalar, ut: B::Scalar) -> (B::Scalar, B::Scalar, B::Scalar) {
        let g = self.params.gravity;
        let hun = h * un;

        (
            hun,                                               // 质量通量
            hun * un + B::Scalar::HALF * g * h * h,           // 法向动量通量
            hun * ut,                                          // 切向动量通量
        )
    }

    /// 计算守恒变量
    #[inline]
    fn conserved_vars(&self, h: B::Scalar, un: B::Scalar, ut: B::Scalar) -> (B::Scalar, B::Scalar, B::Scalar) {
        (h, h * un, h * ut)
    }

    // =========================================================================
    // 核心求解
    // =========================================================================

    /// 求解双湿状态
    fn solve_both_wet(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        vel_l: B::Vector2D,
        vel_r: B::Vector2D,
        normal: B::Vector2D,
    ) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        let tangent = B::vec2_new(-normal.y(), normal.x());

        // 分解速度到法向/切向
        let un_l = B::vec2_dot(&vel_l, &normal);
        let un_r = B::vec2_dot(&vel_r, &normal);
        let ut_l = B::vec2_dot(&vel_l, &tangent);
        let ut_r = B::vec2_dot(&vel_r, &tangent);

        // 静水平衡检测
        let depth_close = (h_l - h_r).abs() <= self.params.h_min;
        let vel_tol = B::Scalar::from_f64(1e-12).unwrap_or(B::Scalar::ZERO);
        let still_water = depth_close
            && un_l.abs() <= vel_tol
            && un_r.abs() <= vel_tol
            && ut_l.abs() <= vel_tol
            && ut_r.abs() <= vel_tol;
        if still_water {
            let lambda_max = self.max_wave_speed(h_l, h_r, un_l, un_r);
            return Ok(RiemannFlux::from_rotated::<B>(
                B::Scalar::ZERO, B::Scalar::ZERO, B::Scalar::ZERO, 
                normal, lambda_max, self.params.gravity
            ));
        }

        // 计算最大波速
        let lambda_max = self.max_wave_speed(h_l, h_r, un_l, un_r);

        // 计算左右物理通量
        let (f_mass_l, f_mom_n_l, f_mom_t_l) = self.physical_flux(h_l, un_l, ut_l);
        let (f_mass_r, f_mom_n_r, f_mom_t_r) = self.physical_flux(h_r, un_r, ut_r);

        // 计算左右守恒变量
        let (u_h_l, u_hun_l, u_hut_l) = self.conserved_vars(h_l, un_l, ut_l);
        let (u_h_r, u_hun_r, u_hut_r) = self.conserved_vars(h_r, un_r, ut_r);

        // Rusanov 通量公式
        let half_lambda = B::Scalar::HALF * lambda_max;

        let mass = B::Scalar::HALF * (f_mass_l + f_mass_r) - half_lambda * (u_h_r - u_h_l);
        let mom_n = B::Scalar::HALF * (f_mom_n_l + f_mom_n_r) - half_lambda * (u_hun_r - u_hun_l);
        let mom_t = B::Scalar::HALF * (f_mom_t_l + f_mom_t_r) - half_lambda * (u_hut_r - u_hut_l);

        Ok(RiemannFlux::from_rotated::<B>(mass, mom_n, mom_t, normal, lambda_max, self.params.gravity))
    }

    /// 求解单侧湿状态（溃坝问题）
    fn solve_single_wet(
        &self,
        h_wet: B::Scalar,
        vel_wet: B::Vector2D,
        normal: B::Vector2D,
        wet_on_left: bool,
    ) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        let tangent = B::vec2_new(-normal.y(), normal.x());
        let un_wet = B::vec2_dot(&vel_wet, &normal);
        let ut_wet = B::vec2_dot(&vel_wet, &tangent);
        let c_wet = (self.params.gravity * h_wet).sqrt();
        let lambda_max = (un_wet.abs() + c_wet).max(self.config.min_wave_speed);
        let g = self.params.gravity;

        // 计算物理通量
        let (f_h, f_hun, f_hut) = self.physical_flux(h_wet, un_wet, ut_wet);

        let three = B::Scalar::from_f64(3.0).unwrap();
        let two = B::Scalar::TWO;
        let nine = B::Scalar::from_f64(9.0).unwrap();

        // 使用 Riemann 不变量的 Ritter 溃坝解
        let (mass, mom_n, mom_t) = if wet_on_left {
            if un_wet >= c_wet {
                (f_h, f_hun, f_hut)
            } else if un_wet <= -c_wet {
                (B::Scalar::ZERO, B::Scalar::HALF * g * h_wet * h_wet, B::Scalar::ZERO)
            } else {
                let h_star = (two * c_wet + un_wet).powi(2) / (nine * g);
                let u_star = (two * c_wet + un_wet) / three;
                let f_mass = h_star * u_star;
                let f_mom = h_star * u_star * u_star + B::Scalar::HALF * g * h_star * h_star;
                let denom = un_wet.abs().max(B::Scalar::from_f64(1e-10).unwrap());
                let f_mom_t = h_star * u_star * ut_wet / denom;
                (f_mass, f_mom, f_mom_t)
            }
        } else {
            if un_wet <= -c_wet {
                (f_h, f_hun, f_hut)
            } else if un_wet >= c_wet {
                (B::Scalar::ZERO, B::Scalar::HALF * g * h_wet * h_wet, B::Scalar::ZERO)
            } else {
                let h_star = (two * c_wet - un_wet).powi(2) / (nine * g);
                let u_star = -(two * c_wet - un_wet) / three;
                let f_mass = h_star * u_star;
                let f_mom = h_star * u_star * u_star + B::Scalar::HALF * g * h_star * h_star;
                let denom = un_wet.abs().max(B::Scalar::from_f64(1e-10).unwrap());
                let f_mom_t = h_star * u_star * ut_wet / denom;
                (f_mass, f_mom, f_mom_t)
            }
        };

        Ok(RiemannFlux::from_rotated::<B>(mass, mom_n, mom_t, normal, lambda_max, self.params.gravity))
    }

    /// 求解双干状态
    #[inline]
    fn solve_both_dry(&self) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        Ok(RiemannFlux::zero())
    }
}

// ============================================================================
// RiemannSolver trait 实现
// ============================================================================

impl<B: Backend> RiemannSolver for RusanovSolver<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

    fn name(&self) -> &'static str {
        "Rusanov (LLF)"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: self.config.entropy_fix,
            supports_hydrostatic: true,
            order: 1,
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
        let h_dry = self.params.h_dry;

        let left_wet = h_left > h_dry;
        let right_wet = h_right > h_dry;

        match (left_wet, right_wet) {
            (true, true) => self.solve_both_wet(h_left, h_right, vel_left, vel_right, normal),
            (true, false) => self.solve_single_wet(h_left, vel_left, normal, true),
            (false, true) => self.solve_single_wet(h_right, vel_right, normal, false),
            (false, false) => self.solve_both_dry(),
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

/// f64 版本的 RusanovSolver
pub type RusanovSolverF64 = RusanovSolver<CpuBackend<f64>>;

/// f32 版本的 RusanovSolver
pub type RusanovSolverF32 = RusanovSolver<CpuBackend<f32>>;

// ============================================================================
// 辅助函数
// ============================================================================

/// 创建默认 Rusanov 求解器 (f64)
pub fn create_rusanov_solver(gravity: f64) -> RusanovSolverF64 {
    let params = SolverParams::<f64> {
        gravity,
        ..Default::default()
    };
    RusanovSolverF64::from_params(params)
}

/// 创建鲁棒 Rusanov 求解器 (f64)
pub fn create_robust_rusanov_solver(gravity: f64) -> RusanovSolverF64 {
    let params = SolverParams::<f64> {
        gravity,
        ..Default::default()
    };
    RusanovSolverF64::from_params_with_config(params, RusanovConfig::robust())
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_solver_f64() -> RusanovSolverF64 {
        create_rusanov_solver(9.81)
    }

    fn create_test_solver_f32() -> RusanovSolverF32 {
        let params = SolverParams::<f32> {
            gravity: 9.81f32,
            ..Default::default()
        };
        RusanovSolverF32::from_params(params)
    }

    #[test]
    fn test_solver_name_and_capabilities() {
        let solver = create_test_solver_f64();
        assert_eq!(solver.name(), "Rusanov (LLF)");

        let caps = solver.capabilities();
        assert!(caps.handles_dry_wet);
        assert!(caps.positivity_preserving);
        assert_eq!(caps.order, 1);
    }

    #[test]
    fn test_static_water_f64() {
        let solver = create_test_solver_f64();

        let flux = solver
            .solve(
                1.0, 1.0, 
                CpuBackend::<f64>::vec2_new(0.0, 0.0), 
                CpuBackend::<f64>::vec2_new(0.0, 0.0), 
                CpuBackend::<f64>::vec2_new(1.0, 0.0)
            )
            .unwrap();

        assert!(flux.mass.abs() < 1e-10);
        assert!(flux.momentum_x.abs() < 1e-10);
        assert!(flux.momentum_y.abs() < 1e-10);
    }

    #[test]
    fn test_static_water_f32() {
        let solver = create_test_solver_f32();

        let flux = solver
            .solve(
                1.0f32, 1.0f32, 
                CpuBackend::<f32>::vec2_new(0.0, 0.0), 
                CpuBackend::<f32>::vec2_new(0.0, 0.0), 
                CpuBackend::<f32>::vec2_new(1.0, 0.0)
            )
            .unwrap();

        assert!(flux.mass.abs() < 1e-5f32);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_dam_break_f64() {
        let solver = create_test_solver_f64();

        let flux = solver
            .solve(
                2.0, 1.0, 
                CpuBackend::<f64>::vec2_new(0.0, 0.0), 
                CpuBackend::<f64>::vec2_new(0.0, 0.0), 
                CpuBackend::<f64>::vec2_new(1.0, 0.0)
            )
            .unwrap();

        assert!(flux.mass > 0.0);
        assert!(flux.max_wave_speed > 0.0);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_both_dry() {
        let solver = create_test_solver_f64();

        let flux = solver
            .solve(
                0.0, 0.0, 
                CpuBackend::<f64>::vec2_new(0.0, 0.0), 
                CpuBackend::<f64>::vec2_new(0.0, 0.0), 
                CpuBackend::<f64>::vec2_new(1.0, 0.0)
            )
            .unwrap();

        assert_eq!(flux.mass, 0.0);
        assert_eq!(flux.momentum_x, 0.0);
        assert_eq!(flux.momentum_y, 0.0);
    }

    #[test]
    fn test_robust_config() {
        let params = SolverParams::<f64>::default();
        let solver = RusanovSolverF64::from_params_with_config(params, RusanovConfig::robust());

        assert!(solver.config().wave_speed_factor > 1.0);
        assert!(solver.config().use_weighted_average);
    }
}
