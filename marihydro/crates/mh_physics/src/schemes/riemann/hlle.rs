// crates/mh_physics/src/schemes/riemann/hlle.rs

//! HLLE (Harten-Lax-van Leer-Einfeldt) 黎曼求解器
//!
//! 在 HLL 基础上使用更稳健的 Einfeldt 波速估计，
//! 对强间断和干湿过渡具有更好的稳定性。

use num_traits::{Float, FromPrimitive};
use mh_runtime::{Backend, RuntimeScalar, Vector2D};
use super::traits::{RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};

/// HLLE 求解器（Backend 泛型化）
#[derive(Debug, Clone)]
pub struct HlleSolver<B: Backend> {
    params: SolverParams<B::Scalar>,
    gravity: B::Scalar,
}

impl<B: Backend> HlleSolver<B> {
    /// 创建新的 HLLE 求解器
    pub fn new(params: &SolverParams<B::Scalar>, gravity: B::Scalar) -> Self {
        Self {
            params: *params,
            gravity,
        }
    }

    /// 波速估计（Einfeldt）
    #[inline]
    fn estimate_wave_speeds(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        un_l: B::Scalar,
        un_r: B::Scalar,
        c_l: B::Scalar,
        c_r: B::Scalar,
    ) -> (B::Scalar, B::Scalar) {
        let sqrt_hl = h_l.sqrt();
        let sqrt_hr = h_r.sqrt();
        let denom = sqrt_hl + sqrt_hr + self.params.flux_eps;

        let u_roe = if denom > self.params.h_min {
            (sqrt_hl * un_l + sqrt_hr * un_r) / denom
        } else {
            (un_l + un_r) * B::Scalar::HALF
        };

        let h_roe = (h_l + h_r) * B::Scalar::HALF;
        let c_roe = (self.gravity * h_roe).sqrt();

        let s_l = Float::min(un_l - c_l, u_roe - c_roe);
        let s_r = Float::max(un_r + c_r, u_roe + c_roe);
        (s_l, s_r)
    }

    /// 物理通量（旋转坐标系）
    #[inline]
    fn physical_flux(&self, h: B::Scalar, un: B::Scalar, ut: B::Scalar) -> (B::Scalar, B::Scalar, B::Scalar) {
        (
            h * un,
            h * un * un + B::Scalar::HALF * self.gravity * h * h,
            h * un * ut,
        )
    }
}

impl<B: Backend> RiemannSolver for HlleSolver<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

    fn name(&self) -> &'static str {
        "HLLE"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: false,
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
        let (s_l, s_r) = self.estimate_wave_speeds(h_left, h_right, un_l, un_r, c_l, c_r);

        let (f_h_l, f_hn_l, f_ht_l) = self.physical_flux(h_left, un_l, ut_l);
        let (f_h_r, f_hn_r, f_ht_r) = self.physical_flux(h_right, un_r, ut_r);

        let (u_h_l, u_hn_l, u_ht_l) = (h_left, h_left * un_l, h_left * ut_l);
        let (u_h_r, u_hn_r, u_ht_r) = (h_right, h_right * un_r, h_right * ut_r);

        let (f_h, f_hn, f_ht) = if s_l >= B::Scalar::ZERO {
            (f_h_l, f_hn_l, f_ht_l)
        } else if s_r <= B::Scalar::ZERO {
            (f_h_r, f_hn_r, f_ht_r)
        } else {
            let denom = s_r - s_l;
            if denom.abs() < self.params.flux_eps {
                (
                    (f_h_l + f_h_r) * B::Scalar::HALF,
                    (f_hn_l + f_hn_r) * B::Scalar::HALF,
                    (f_ht_l + f_ht_r) * B::Scalar::HALF,
                )
            } else {
                (
                    (s_r * f_h_l - s_l * f_h_r + s_l * s_r * (u_h_r - u_h_l)) / denom,
                    (s_r * f_hn_l - s_l * f_hn_r + s_l * s_r * (u_hn_r - u_hn_l)) / denom,
                    (s_r * f_ht_l - s_l * f_ht_r + s_l * s_r * (u_ht_r - u_ht_l)) / denom,
                )
            }
        };

        let max_speed = Float::max(s_l.abs(), s_r.abs());
        Ok(RiemannFlux::from_rotated::<B>(f_h, f_hn, f_ht, normal, max_speed, self.gravity))
    }

    fn gravity(&self) -> B::Scalar {
        self.gravity
    }

    fn dry_threshold(&self) -> B::Scalar {
        self.params.h_dry
    }
}

/// f64 便捷构造
pub fn create_hlle_solver(gravity: f64) -> HlleSolver<mh_runtime::CpuBackend<f64>> {
    let params = SolverParams::<f64>::default();
    HlleSolver::new(&params, gravity)
}
