// crates/mh_physics/src/schemes/riemann/central.rs

//! 中心差分黎曼求解器
//!
//! 提供无耗散的中心通量，用于稳定性/对比测试。

use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use num_traits::Float;

use crate::types::NumericalParams;
use super::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};

/// 中心差分求解器（Backend 泛型化）
#[derive(Clone)]
pub struct CentralSolver<B: Backend> {
    params: SolverParams<B::Scalar>,
}

impl<B: Backend> CentralSolver<B> {
    /// 创建新的中心差分求解器
    pub fn new(numerical_params: &NumericalParams<B::Scalar>, gravity: B::Scalar) -> Self {
        Self {
            params: SolverParams::from_numerical(numerical_params, gravity),
        }
    }

    /// 从参数直接创建
    pub fn from_params(params: SolverParams<B::Scalar>) -> Self {
        Self { params }
    }

    /// 物理通量（局部坐标系）
    #[inline]
    fn physical_flux(
        &self,
        h: B::Scalar,
        un: B::Scalar,
        ut: B::Scalar,
    ) -> (B::Scalar, B::Scalar, B::Scalar) {
        let g = self.params.gravity;
        let mass = h * un;
        let mom_n = h * un * un + B::Scalar::HALF * g * h * h;
        let mom_t = h * un * ut;
        (mass, mom_n, mom_t)
    }

    /// 计算最大波速
    #[inline]
    fn max_wave_speed(&self, h_l: B::Scalar, h_r: B::Scalar, un_l: B::Scalar, un_r: B::Scalar) -> B::Scalar {
        let g = self.params.gravity;
        let c_l = (g * h_l.max(self.params.h_min)).sqrt();
        let c_r = (g * h_r.max(self.params.h_min)).sqrt();
        (un_l.abs() + c_l).max(un_r.abs() + c_r)
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
        let lambda_max = un_wet.abs() + c_wet;
        let g = self.params.gravity;

        let (f_h, f_hun, f_hut) = self.physical_flux(h_wet, un_wet, ut_wet);

        let three = B::Scalar::ONE + B::Scalar::ONE + B::Scalar::ONE;
        let two = B::Scalar::TWO;
        let nine = three * three;

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
                let denom = un_wet.abs().max(self.params.flux_eps);
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
                let denom = un_wet.abs().max(self.params.flux_eps);
                let f_mom_t = h_star * u_star * ut_wet / denom;
                (f_mass, f_mom, f_mom_t)
            }
        };

        Ok(RiemannFlux::from_rotated::<B>(mass, mom_n, mom_t, normal, lambda_max, self.params.gravity))
    }
}

impl<B: Backend> RiemannSolver for CentralSolver<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

    fn name(&self) -> &'static str {
        "Central"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: false,
            supports_hydrostatic: true,
            order: 1,
            positivity_preserving: false,
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
        let h_l = h_left.max(B::Scalar::ZERO);
        let h_r = h_right.max(B::Scalar::ZERO);

        let left_dry = h_l <= self.params.h_dry;
        let right_dry = h_r <= self.params.h_dry;
        if left_dry && right_dry {
            return Ok(RiemannFlux::zero());
        }
        if left_dry {
            return self.solve_single_wet(h_r, vel_right, normal, false);
        }
        if right_dry {
            return self.solve_single_wet(h_l, vel_left, normal, true);
        }

        let tangent = B::vec2_new(-normal.y(), normal.x());
        let un_l = B::vec2_dot(&vel_left, &normal);
        let un_r = B::vec2_dot(&vel_right, &normal);
        let ut_l = B::vec2_dot(&vel_left, &tangent);
        let ut_r = B::vec2_dot(&vel_right, &tangent);

        // 计算最大波速（用于 CFL 统计）
        let lambda_max = self.max_wave_speed(h_l, h_r, un_l, un_r);

        let (f_mass_l, f_mom_n_l, f_mom_t_l) = self.physical_flux(h_l, un_l, ut_l);
        let (f_mass_r, f_mom_n_r, f_mom_t_r) = self.physical_flux(h_r, un_r, ut_r);

        // 中心通量：F = 0.5 * (F_L + F_R)
        let mass = B::Scalar::HALF * (f_mass_l + f_mass_r);
        let mom_n = B::Scalar::HALF * (f_mom_n_l + f_mom_n_r);
        let mom_t = B::Scalar::HALF * (f_mom_t_l + f_mom_t_r);

        // 返回旋转回全局坐标的通量
        Ok(RiemannFlux::from_rotated::<B>(mass, mom_n, mom_t, normal, lambda_max, self.params.gravity))
    }

    fn gravity(&self) -> Self::Scalar {
        self.params.gravity
    }

    fn dry_threshold(&self) -> Self::Scalar {
        self.params.h_dry
    }
}
