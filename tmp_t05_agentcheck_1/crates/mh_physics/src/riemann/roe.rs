// crates/mh_physics/src/riemann/roe.rs

//! Roe 近似 Riemann 求解器（f64 版本）

use super::hllc::RiemannSolverTrait;

/// Roe 求解器
pub struct RoeSolver {
    g: f64,
    dry_tolerance: f64,
    entropy_delta: f64,
}

impl RoeSolver {
    pub fn new(g: f64) -> Self {
        Self {
            g,
            dry_tolerance: 1e-6,
            entropy_delta: 0.1,
        }
    }

    fn entropy_fix(&self, lambda: f64, c: f64) -> f64 {
        let abs_lambda = lambda.abs();
        let eps = self.entropy_delta * c;
        if abs_lambda < eps {
            (lambda * lambda + eps * eps) / (2.0 * eps)
        } else {
            abs_lambda
        }
    }
}

impl RiemannSolverTrait for RoeSolver {
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

        let sqrt_hl = h_l.sqrt();
        let sqrt_hr = h_r.sqrt();
        let denom = sqrt_hl + sqrt_hr;
        let h_roe = 0.5 * (h_l + h_r);
        let u_roe = if denom > self.dry_tolerance {
            (sqrt_hl * un_l + sqrt_hr * un_r) / denom
        } else {
            0.5 * (un_l + un_r)
        };
        let c_roe = (self.g * h_roe).sqrt();

        let dh = h_r - h_l;
        let du = un_r - un_l;

        let alpha1 = 0.5 * (dh - h_roe * du / c_roe);
        let alpha2 = 0.5 * (dh + h_roe * du / c_roe);

        let lambda1 = u_roe - c_roe;
        let lambda2 = u_roe + c_roe;

        let a1 = self.entropy_fix(lambda1, c_roe);
        let a2 = self.entropy_fix(lambda2, c_roe);

        let diss_mass = a1 * alpha1 + a2 * alpha2;
        let diss_mom = a1 * alpha1 * (u_roe - c_roe) + a2 * alpha2 * (u_roe + c_roe);

        let f_mass_l = h_l * un_l;
        let f_mass_r = h_r * un_r;
        let f_mom_n_l = h_l * un_l * un_l + 0.5 * self.g * h_l * h_l;
        let f_mom_n_r = h_r * un_r * un_r + 0.5 * self.g * h_r * h_r;
        let f_mom_t_l = h_l * un_l * ut_l;
        let f_mom_t_r = h_r * un_r * ut_r;

        let flux_mass = 0.5 * (f_mass_l + f_mass_r) - 0.5 * diss_mass;
        let flux_mom_n = 0.5 * (f_mom_n_l + f_mom_n_r) - 0.5 * diss_mom;

        let lambda_t = u_roe.abs();
        let diss_t = lambda_t * (h_r * ut_r - h_l * ut_l);
        let flux_mom_t = 0.5 * (f_mom_t_l + f_mom_t_r) - 0.5 * diss_t;

        let f_hu = flux_mom_n * nx - flux_mom_t * ny;
        let f_hv = flux_mom_n * ny + flux_mom_t * nx;

        (flux_mass, f_hu, f_hv)
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
