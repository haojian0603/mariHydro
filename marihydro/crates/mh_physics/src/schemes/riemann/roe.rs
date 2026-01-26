// crates/mh_physics/src/schemes/riemann/roe.rs

//! Roe 近似黎曼求解器
//!
//! 使用 Roe 线性化与熵修正，提供对接触间断更高分辨率的通量。

use num_traits::{Float, FromPrimitive};
use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use super::traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};

/// Roe 求解器（Backend 泛型化）
#[derive(Debug, Clone)]
pub struct RoeSolver<B: Backend> {
    params: SolverParams<B::Scalar>,
    gravity: B::Scalar,
}

impl<B: Backend> RoeSolver<B> {
    /// 创建新的 Roe 求解器
    pub fn new(params: &SolverParams<B::Scalar>, gravity: B::Scalar) -> Self {
        Self {
            params: *params,
            gravity,
        }
    }

    /// 熵修正（Harten）
    #[inline]
    fn entropy_fix(&self, lambda: B::Scalar, eps: B::Scalar) -> B::Scalar {
        let abs_lambda = lambda.abs();
        if abs_lambda < eps {
            (lambda * lambda + eps * eps) / (eps + eps)
        } else {
            abs_lambda
        }
    }
}

impl<B: Backend> RiemannSolver for RoeSolver<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

    fn name(&self) -> &'static str {
        "Roe"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: false,
            order: 2,
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
        let h_left = h_left.max(B::Scalar::ZERO);
        let h_right = h_right.max(B::Scalar::ZERO);

        if h_left <= self.params.h_dry && h_right <= self.params.h_dry {
            return Ok(RiemannFlux::zero());
        }

        let vel_left = if h_left <= self.params.h_dry {
            B::vec2_new(B::Scalar::ZERO, B::Scalar::ZERO)
        } else {
            vel_left
        };
        let vel_right = if h_right <= self.params.h_dry {
            B::vec2_new(B::Scalar::ZERO, B::Scalar::ZERO)
        } else {
            vel_right
        };

        let tangent = B::vec2_new(-normal.y(), normal.x());
        let un_l = B::vec2_dot(&vel_left, &normal);
        let un_r = B::vec2_dot(&vel_right, &normal);
        let ut_l = B::vec2_dot(&vel_left, &tangent);
        let ut_r = B::vec2_dot(&vel_right, &tangent);

        let c_l = (self.gravity * h_left).sqrt();
        let c_r = (self.gravity * h_right).sqrt();

        let sqrt_hl = h_left.sqrt();
        let sqrt_hr = h_right.sqrt();
        let denom = sqrt_hl + sqrt_hr + self.params.flux_eps;
        let h_roe = (h_left + h_right) * B::Scalar::HALF;
        let u_roe = (sqrt_hl * un_l + sqrt_hr * un_r) / denom;
        let c_roe = (self.gravity * h_roe).sqrt();

        let dh = h_right - h_left;
        let du = un_r - un_l;

        let alpha1 = (dh - h_roe * du / c_roe) * B::Scalar::HALF;
        let alpha2 = (dh + h_roe * du / c_roe) * B::Scalar::HALF;

        let lambda1 = u_roe - c_roe;
        let lambda2 = u_roe + c_roe;
        let eps = self.params.entropy_threshold((lambda2 - lambda1).abs());

        let a1 = self.entropy_fix(lambda1, eps);
        let a2 = self.entropy_fix(lambda2, eps);

        let diss_mass = a1 * alpha1 + a2 * alpha2;
        let diss_mom = a1 * alpha1 * (u_roe - c_roe) + a2 * alpha2 * (u_roe + c_roe);

        let f_mass_l = h_left * un_l;
        let f_mass_r = h_right * un_r;
        let f_mom_n_l = h_left * un_l * un_l + B::Scalar::HALF * self.gravity * h_left * h_left;
        let f_mom_n_r = h_right * un_r * un_r + B::Scalar::HALF * self.gravity * h_right * h_right;
        let f_mom_t_l = h_left * un_l * ut_l;
        let f_mom_t_r = h_right * un_r * ut_r;

        let flux_mass = (f_mass_l + f_mass_r) * B::Scalar::HALF - diss_mass * B::Scalar::HALF;
        let flux_mom_n = (f_mom_n_l + f_mom_n_r) * B::Scalar::HALF - diss_mom * B::Scalar::HALF;

        let lambda_t = u_roe.abs();
        let diss_t = lambda_t * (h_right * ut_r - h_left * ut_l);
        let flux_mom_t = (f_mom_t_l + f_mom_t_r) * B::Scalar::HALF - diss_t * B::Scalar::HALF;

        let max_speed = Float::max(lambda1.abs(), lambda2.abs());

        Ok(RiemannFlux::from_rotated::<B>(
            flux_mass,
            flux_mom_n,
            flux_mom_t,
            normal,
            max_speed,
            self.gravity,
        ))
    }

    fn gravity(&self) -> B::Scalar {
        self.gravity
    }

    fn dry_threshold(&self) -> B::Scalar {
        self.params.h_dry
    }
}

/// f64 便捷构造
pub fn create_roe_solver(gravity: f64) -> RoeSolver<mh_runtime::CpuBackend<f64>> {
    let params = SolverParams::<f64>::default();
    RoeSolver::new(&params, gravity)
}
