// crates/mh_physics/src/riemann/hlle.rs

//! HLLE Riemann 求解器（f64 版本）

use super::hllc::RiemannSolverTrait;

/// HLLE 求解器
pub struct HlleSolver {
    g: f64,
    dry_tolerance: f64,
}

impl HlleSolver {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            dry_tolerance: 1e-6,
        }
    }

    fn estimate_wave_speeds(&self, h_l: f64, h_r: f64, un_l: f64, un_r: f64) -> (f64, f64) {
        let c_l = (self.g * h_l.max(0.0)).sqrt();
        let c_r = (self.g * h_r.max(0.0)).sqrt();

        let sqrt_hl = h_l.sqrt();
        let sqrt_hr = h_r.sqrt();
        let denom = sqrt_hl + sqrt_hr;

        let u_roe = if denom > self.dry_tolerance {
            (sqrt_hl * un_l + sqrt_hr * un_r) / denom
        } else {
            0.5 * (un_l + un_r)
        };
        let h_roe = 0.5 * (h_l + h_r);
        let c_roe = (self.g * h_roe).sqrt();

        let s_l = (un_l - c_l).min(u_roe - c_roe);
        let s_r = (un_r + c_r).max(u_roe + c_roe);
        (s_l, s_r)
    }
}

impl RiemannSolverTrait for HlleSolver {
    fn compute_flux(
        &self,
        h_l: f64,
        hu_l: f64,
        hv_l: f64,
        h_r: f64,
        hu_r: f64,
        hv_r: f64,
        nx: f64,
        ny: f64,
    ) -> (f64, f64, f64) {
        let h_l = h_l.max(0.0);
        let h_r = h_r.max(0.0);

        if h_l < self.dry_tolerance && h_r < self.dry_tolerance {
            return (0.0, 0.0, 0.0);
        }

        let (u_l, v_l) = if h_l > self.dry_tolerance { (hu_l / h_l, hv_l / h_l) } else { (0.0, 0.0) };
        let (u_r, v_r) = if h_r > self.dry_tolerance { (hu_r / h_r, hv_r / h_r) } else { (0.0, 0.0) };

        let un_l = u_l * nx + v_l * ny;
        let un_r = u_r * nx + v_r * ny;
        let ut_l = -u_l * ny + v_l * nx;
        let ut_r = -u_r * ny + v_r * nx;

        let (s_l, s_r) = self.estimate_wave_speeds(h_l, h_r, un_l, un_r);

        let f_l = (
            h_l * un_l,
            h_l * un_l * un_l + 0.5 * self.g * h_l * h_l,
            h_l * un_l * ut_l,
        );
        let f_r = (
            h_r * un_r,
            h_r * un_r * un_r + 0.5 * self.g * h_r * h_r,
            h_r * un_r * ut_r,
        );

        let u_l_cons = (h_l, h_l * un_l, h_l * ut_l);
        let u_r_cons = (h_r, h_r * un_r, h_r * ut_r);

        let (f_h, f_hn, f_ht) = if s_l >= 0.0 {
            f_l
        } else if s_r <= 0.0 {
            f_r
        } else {
            let denom = s_r - s_l;
            if denom.abs() < 1e-12 {
                (0.5 * (f_l.0 + f_r.0), 0.5 * (f_l.1 + f_r.1), 0.5 * (f_l.2 + f_r.2))
            } else {
                (
                    (s_r * f_l.0 - s_l * f_r.0 + s_l * s_r * (u_r_cons.0 - u_l_cons.0)) / denom,
                    (s_r * f_l.1 - s_l * f_r.1 + s_l * s_r * (u_r_cons.1 - u_l_cons.1)) / denom,
                    (s_r * f_l.2 - s_l * f_r.2 + s_l * s_r * (u_r_cons.2 - u_l_cons.2)) / denom,
                )
            }
        };

        let f_hu = f_hn * nx - f_ht * ny;
        let f_hv = f_hn * ny + f_ht * nx;
        (f_h, f_hu, f_hv)
    }

    fn max_wave_speed(&self, h: f64, u: f64, v: f64) -> f64 {
        if h > self.dry_tolerance {
            let c = (self.g * h).sqrt();
            let speed = (u * u + v * v).sqrt();
            speed + c
        } else {
            0.0
        }
    }
}
