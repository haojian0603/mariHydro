// src-tauri\src\marihydro\domain\state.rs
use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::marihydro::infra::error::{MhError, MhResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservedState {
    pub n_cells: usize,

    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub hc: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct GradientState {
    pub grad_h: Vec<DVec2>,
    pub grad_hu: Vec<DVec2>,
    pub grad_hv: Vec<DVec2>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Flux {
    pub mass: f64,
    pub mom_x: f64,
    pub mom_y: f64,
}

impl Flux {
    #[inline(always)]
    pub const fn new(mass: f64, mom_x: f64, mom_y: f64) -> Self {
        Self { mass, mom_x, mom_y }
    }

    #[inline(always)]
    pub fn scale(self, factor: f64) -> Self {
        Self {
            mass: self.mass * factor,
            mom_x: self.mom_x * factor,
            mom_y: self.mom_y * factor,
        }
    }

    #[inline(always)]
    pub fn magnitude(&self) -> f64 {
        (self.mass * self.mass + self.mom_x * self.mom_x + self.mom_y * self.mom_y).sqrt()
    }
}

impl std::ops::AddAssign for Flux {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.mass += rhs.mass;
        self.mom_x += rhs.mom_x;
        self.mom_y += rhs.mom_y;
    }
}

impl std::ops::SubAssign for Flux {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        self.mass -= rhs.mass;
        self.mom_x -= rhs.mom_x;
        self.mom_y -= rhs.mom_y;
    }
}

impl std::ops::Add for Flux {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self {
            mass: self.mass + rhs.mass,
            mom_x: self.mom_x + rhs.mom_x,
            mom_y: self.mom_y + rhs.mom_y,
        }
    }
}

impl std::ops::Sub for Flux {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self {
            mass: self.mass - rhs.mass,
            mom_x: self.mom_x - rhs.mom_x,
            mom_y: self.mom_y - rhs.mom_y,
        }
    }
}

impl std::ops::Neg for Flux {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            mass: -self.mass,
            mom_x: -self.mom_x,
            mom_y: -self.mom_y,
        }
    }
}

impl std::ops::Mul<f64> for Flux {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl std::ops::Mul<Flux> for f64 {
    type Output = Flux;
    #[inline(always)]
    fn mul(self, rhs: Flux) -> Flux {
        rhs.scale(self)
    }
}

impl ConservedState {
    pub fn new(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            hc: None,
        }
    }

    pub fn with_scalar(n_cells: usize) -> Self {
        Self {
            n_cells,
            h: vec![0.0; n_cells],
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            hc: Some(vec![0.0; n_cells]),
        }
    }

    pub fn cold_start(n_cells: usize, initial_eta: f64, z_bed: &[f64]) -> MhResult<Self> {
        if z_bed.len() != n_cells {
            return Err(MhError::InvalidMesh {
                message: format!("底高程数组长度 {} 与单元数 {} 不匹配", z_bed.len(), n_cells),
            });
        }

        let h: Vec<f64> = z_bed.iter().map(|&z| (initial_eta - z).max(0.0)).collect();

        Ok(Self {
            n_cells,
            h,
            hu: vec![0.0; n_cells],
            hv: vec![0.0; n_cells],
            hc: None,
        })
    }

    pub fn clone_structure(&self) -> Self {
        Self {
            n_cells: self.n_cells,
            h: vec![0.0; self.n_cells],
            hu: vec![0.0; self.n_cells],
            hv: vec![0.0; self.n_cells],
            hc: self.hc.as_ref().map(|_| vec![0.0; self.n_cells]),
        }
    }

    #[inline(always)]
    pub fn primitive(&self, idx: usize, eps: f64) -> (f64, f64, f64) {
        let h = self.h[idx];
        if h > eps {
            (h, self.hu[idx] / h, self.hv[idx] / h)
        } else {
            (h, 0.0, 0.0)
        }
    }

    #[inline(always)]
    pub fn velocity(&self, idx: usize, eps: f64) -> DVec2 {
        let h = self.h[idx];
        if h > eps {
            DVec2::new(self.hu[idx] / h, self.hv[idx] / h)
        } else {
            DVec2::ZERO
        }
    }

    #[inline(always)]
    pub fn water_level(&self, idx: usize, z_bed: f64) -> f64 {
        self.h[idx] + z_bed
    }

    #[inline(always)]
    pub fn set(&mut self, idx: usize, h: f64, hu: f64, hv: f64) {
        self.h[idx] = h;
        self.hu[idx] = hu;
        self.hv[idx] = hv;
    }

    #[inline(always)]
    pub fn set_from_primitive(&mut self, idx: usize, h: f64, u: f64, v: f64) {
        self.h[idx] = h;
        self.hu[idx] = h * u;
        self.hv[idx] = h * v;
    }

    pub fn reset(&mut self) {
        self.h.iter_mut().for_each(|x| *x = 0.0);
        self.hu.iter_mut().for_each(|x| *x = 0.0);
        self.hv.iter_mut().for_each(|x| *x = 0.0);
        if let Some(ref mut hc) = self.hc {
            hc.iter_mut().for_each(|x| *x = 0.0);
        }
    }

    pub fn total_mass(&self, cell_areas: &[f64]) -> f64 {
        self.h.iter().zip(cell_areas).map(|(h, a)| h * a).sum()
    }

    pub fn total_momentum(&self, cell_areas: &[f64]) -> DVec2 {
        let hux: f64 = self.hu.iter().zip(cell_areas).map(|(hu, a)| hu * a).sum();
        let hvx: f64 = self.hv.iter().zip(cell_areas).map(|(hv, a)| hv * a).sum();
        DVec2::new(hux, hvx)
    }

    pub fn validate(&self, time: f64) -> MhResult<()> {
        self.validate_with_limits(time, 15000.0, 100.0)
    }

    pub fn validate_with_limits(
        &self,
        time: f64,
        max_depth: f64,
        max_velocity: f64,
    ) -> MhResult<()> {
        for (idx, &h) in self.h.iter().enumerate() {
            if h.is_nan() || h.is_infinite() {
                return Err(MhError::NumericalInstability {
                    message: format!("水深异常 (NaN/Inf) 在单元 {}", idx),
                    time,
                    location: None,
                });
            }

            if h < 0.0 {
                return Err(MhError::NumericalInstability {
                    message: format!("水深为负 {:.6} m 在单元 {}", h, idx),
                    time,
                    location: None,
                });
            }

            if h > max_depth {
                return Err(MhError::NumericalInstability {
                    message: format!("水深过大: {:.2} m 在单元 {}", h, idx),
                    time,
                    location: None,
                });
            }

            let hu = self.hu[idx];
            let hv = self.hv[idx];

            if hu.is_nan() || hu.is_infinite() {
                return Err(MhError::NumericalInstability {
                    message: format!("x 方向动量异常 (NaN/Inf) 在单元 {}", idx),
                    time,
                    location: None,
                });
            }

            if hv.is_nan() || hv.is_infinite() {
                return Err(MhError::NumericalInstability {
                    message: format!("y 方向动量异常 (NaN/Inf) 在单元 {}", idx),
                    time,
                    location: None,
                });
            }

            if h > 1e-6 {
                let u = hu / h;
                let v = hv / h;

                if u.abs() > max_velocity {
                    return Err(MhError::NumericalInstability {
                        message: format!("x 方向流速过大: {:.2} m/s 在单元 {}", u, idx),
                        time,
                        location: None,
                    });
                }

                if v.abs() > max_velocity {
                    return Err(MhError::NumericalInstability {
                        message: format!("y 方向流速过大: {:.2} m/s 在单元 {}", v, idx),
                        time,
                        location: None,
                    });
                }
            }
        }

        Ok(())
    }
}

impl GradientState {
    pub fn new(n_cells: usize) -> Self {
        Self {
            grad_h: vec![DVec2::ZERO; n_cells],
            grad_hu: vec![DVec2::ZERO; n_cells],
            grad_hv: vec![DVec2::ZERO; n_cells],
        }
    }

    pub fn reset(&mut self) {
        self.grad_h.iter_mut().for_each(|v| *v = DVec2::ZERO);
        self.grad_hu.iter_mut().for_each(|v| *v = DVec2::ZERO);
        self.grad_hv.iter_mut().for_each(|v| *v = DVec2::ZERO);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conserved_state_creation() {
        let state = ConservedState::new(100);
        assert_eq!(state.n_cells, 100);
        assert_eq!(state.h.len(), 100);
        assert!(state.hc.is_none());
    }

    #[test]
    fn test_cold_start() {
        let z_bed = vec![-10.0, -5.0, 0.0, 5.0];
        let state = ConservedState::cold_start(4, 0.0, &z_bed).unwrap();

        assert_eq!(state.h[0], 10.0);
        assert_eq!(state.h[1], 5.0);
        assert_eq!(state.h[2], 0.0);
        assert_eq!(state.h[3], 0.0);
    }

    #[test]
    fn test_primitive_extraction() {
        let mut state = ConservedState::new(1);
        state.h[0] = 2.0;
        state.hu[0] = 4.0;
        state.hv[0] = 6.0;

        let (h, u, v) = state.primitive(0, 1e-6);
        assert_eq!(h, 2.0);
        assert_eq!(u, 2.0);
        assert_eq!(v, 3.0);
    }

    #[test]
    fn test_flux_operations() {
        let f1 = Flux::new(1.0, 2.0, 3.0);
        let f2 = Flux::new(0.5, 1.0, 1.5);

        let sum = f1 + f2;
        assert_eq!(sum.mass, 1.5);

        let scaled = f1 * 2.0;
        assert_eq!(scaled.mass, 2.0);

        let scaled2 = 2.0 * f1;
        assert_eq!(scaled2.mass, 2.0);
    }

    #[test]
    fn test_total_momentum() {
        let mut state = ConservedState::new(2);
        state.hu[0] = 1.0;
        state.hv[0] = 2.0;
        state.hu[1] = 3.0;
        state.hv[1] = 4.0;

        let areas = vec![1.0, 1.0];
        let mom = state.total_momentum(&areas);

        assert_eq!(mom.x, 4.0);
        assert_eq!(mom.y, 6.0);
    }

    #[test]
    fn test_validation_nan() {
        let mut state = ConservedState::new(1);
        state.h[0] = f64::NAN;

        assert!(state.validate(0.0).is_err());
    }

    #[test]
    fn test_validation_negative_depth() {
        let mut state = ConservedState::new(1);
        state.h[0] = -1.0;

        assert!(state.validate(0.0).is_err());
    }
}
