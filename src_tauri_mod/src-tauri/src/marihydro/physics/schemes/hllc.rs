//! HLLC 近似 Riemann 求解器

use super::dry_wet::{DryWetHandler, WetDryState};
use super::flux_utils::{physical_flux_1d, InterfaceFlux, RotatedFlux};
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

pub struct HllcSolver {
    params: NumericalParams,
    g: f64,
    dry_wet: DryWetHandler,
}

impl HllcSolver {
    pub fn new(params: NumericalParams, g: f64) -> Self {
        let dry_wet = DryWetHandler::new(&params);
        Self { params, g, dry_wet }
    }

    pub fn solve(
        &self,
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        normal: DVec2,
    ) -> MhResult<InterfaceFlux> {
        match self.dry_wet.classify(h_l, h_r) {
            WetDryState::BothDry => Ok(InterfaceFlux::ZERO),
            WetDryState::LeftDry => self.solve_left_dry(h_r, vel_r, normal),
            WetDryState::RightDry => self.solve_right_dry(h_l, vel_l, normal),
            WetDryState::BothWet => self.solve_both_wet(h_l, h_r, vel_l, vel_r, normal),
        }
    }

    fn solve_both_wet(
        &self,
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        normal: DVec2,
    ) -> MhResult<InterfaceFlux> {
        let tangent = DVec2::new(-normal.y, normal.x);
        let (un_l, un_r) = (vel_l.dot(normal), vel_r.dot(normal));
        let (ut_l, ut_r) = (vel_l.dot(tangent), vel_r.dot(tangent));
        let (c_l, c_r) = ((self.g * h_l).sqrt(), (self.g * h_r).sqrt());

        let (s_l, s_r) = self.einfeldt_speeds(h_l, h_r, un_l, un_r, c_l, c_r);
        let max_speed = s_l.abs().max(s_r.abs());

        let rf = if s_l >= 0.0 {
            physical_flux_1d(h_l, un_l, ut_l, self.g)
        } else if s_r <= 0.0 {
            physical_flux_1d(h_r, un_r, ut_r, self.g)
        } else {
            self.hllc_star_flux(h_l, h_r, un_l, un_r, ut_l, ut_r, s_l, s_r)?
        };

        let mut flux = rf.to_global(normal);
        flux.max_wave_speed = max_speed;
        Ok(flux)
    }

    #[inline]
    fn einfeldt_speeds(
        &self,
        h_l: f64,
        h_r: f64,
        un_l: f64,
        un_r: f64,
        c_l: f64,
        c_r: f64,
    ) -> (f64, f64) {
        let (sh_l, sh_r) = (h_l.sqrt(), h_r.sqrt());
        let sum = sh_l + sh_r + self.params.flux_eps;
        let h_roe = 0.5 * (h_l + h_r);
        let u_roe = (sh_l * un_l + sh_r * un_r) / sum;
        let c_roe = (self.g * h_roe).sqrt();
        (
            (un_l - c_l).min(u_roe - c_roe),
            (un_r + c_r).max(u_roe + c_roe),
        )
    }

    fn hllc_star_flux(
        &self,
        h_l: f64,
        h_r: f64,
        un_l: f64,
        un_r: f64,
        ut_l: f64,
        ut_r: f64,
        s_l: f64,
        s_r: f64,
    ) -> MhResult<RotatedFlux> {
        let (q_l, q_r) = (h_l * (un_l - s_l), h_r * (un_r - s_r));
        let denom = q_l - q_r;
        let threshold = self.params.entropy_threshold((s_r - s_l).abs());

        let s_star = if denom.abs() < threshold {
            0.5 * (un_l + un_r)
        } else {
            let numer = q_l * un_l - q_r * un_r + 0.5 * self.g * (h_r * h_r - h_l * h_l);
            let s = numer / denom;
            if !s.is_finite() {
                return Err(MhError::Numerical {
                    message: format!("HLLC: s_star NaN: {}", s),
                });
            }
            self.entropy_fix(s.clamp(s_l, s_r), s_l, s_r)
        };

        let (h_star, ut) = if s_star >= 0.0 {
            (h_l * (s_l - un_l) / (s_l - s_star), ut_l)
        } else {
            (h_r * (s_r - un_r) / (s_r - s_star), ut_r)
        };
        let h_star = h_star.max(0.0);

        Ok(RotatedFlux {
            mass: h_star * s_star,
            momentum_n: h_star * s_star * s_star + 0.5 * self.g * h_star * h_star,
            momentum_t: h_star * s_star * ut,
        })
    }

    #[inline]
    fn entropy_fix(&self, s_star: f64, s_l: f64, s_r: f64) -> f64 {
        let eps = self.params.entropy_ratio * (s_r - s_l).abs();
        if s_star.abs() < eps {
            s_star.signum() * eps
        } else {
            s_star
        }
    }

    fn solve_left_dry(&self, h_r: f64, vel_r: DVec2, normal: DVec2) -> MhResult<InterfaceFlux> {
        let c_r = (self.g * h_r).sqrt();
        let un_r = vel_r.dot(normal);
        let s_front = un_r - 2.0 * c_r;
        if s_front >= 0.0 {
            return Ok(InterfaceFlux::ZERO);
        }

        let h_star = ((2.0 * c_r + un_r) / 3.0).powi(2) / self.g;
        if h_star < self.params.h_dry {
            return Ok(InterfaceFlux::ZERO);
        }

        let u_star = (2.0 * c_r + un_r) / 3.0;
        let t = DVec2::new(-normal.y, normal.x);
        let ut = vel_r.dot(t);
        let rf = physical_flux_1d(h_star, u_star, ut, self.g);
        let mut f = rf.to_global(normal);
        f.max_wave_speed = (un_r + c_r).abs().max(s_front.abs());
        Ok(f)
    }

    fn solve_right_dry(&self, h_l: f64, vel_l: DVec2, normal: DVec2) -> MhResult<InterfaceFlux> {
        let c_l = (self.g * h_l).sqrt();
        let un_l = vel_l.dot(normal);
        let s_front = un_l + 2.0 * c_l;
        if s_front <= 0.0 {
            return Ok(InterfaceFlux::ZERO);
        }

        let h_star = ((un_l + 2.0 * c_l) / 3.0).powi(2) / self.g;
        if h_star < self.params.h_dry {
            return Ok(InterfaceFlux::ZERO);
        }

        let u_star = (un_l + 2.0 * c_l) / 3.0;
        let t = DVec2::new(-normal.y, normal.x);
        let ut = vel_l.dot(t);
        let rf = physical_flux_1d(h_star, u_star, ut, self.g);
        let mut f = rf.to_global(normal);
        f.max_wave_speed = (un_l - c_l).abs().max(s_front.abs());
        Ok(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solver() -> HllcSolver {
        HllcSolver::new(NumericalParams::default(), 9.81)
    }

    #[test]
    fn test_both_dry() {
        let f = solver()
            .solve(0.0, 0.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert_eq!(f.mass, 0.0);
    }

    #[test]
    fn test_still_water() {
        let f = solver()
            .solve(10.0, 10.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert!(f.mass.abs() < 1e-10);
    }

    #[test]
    fn test_dam_break() {
        let f = solver()
            .solve(10.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert!(f.mass > 0.0);
        assert!(f.max_wave_speed > 0.0);
    }
}
