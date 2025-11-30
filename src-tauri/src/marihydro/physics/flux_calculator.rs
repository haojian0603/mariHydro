// src-tauri\src\marihydro\physics\flux_calculator.rs
use glam::DVec2;

pub struct FluxCalculator {
    gravity: f64,
    h_min: f64,
}

impl FluxCalculator {
    pub fn new(gravity: f64, h_min: f64) -> Self {
        Self { gravity, h_min }
    }

    #[inline]
    pub fn compute_euler_flux(&self, h: f64, u: f64, v: f64) -> EulerFlux {
        if h < self.h_min {
            return EulerFlux::zero();
        }

        let hu = h * u;
        let hv = h * v;
        let p = 0.5 * self.gravity * h * h;

        EulerFlux {
            mass: hu,
            momentum_x: hu * u + p,
            momentum_y: hu * v,
        }
    }

    #[inline]
    pub fn compute_rotated_flux(&self, h: f64, vel: DVec2, normal: DVec2) -> RotatedFlux {
        let un = vel.dot(normal);
        let ut = vel.dot(DVec2::new(-normal.y, normal.x));

        let q = h * un;
        let p = 0.5 * self.gravity * h * h;

        RotatedFlux {
            mass: q,
            momentum_n: q * un + p,
            momentum_t: q * ut,
        }
    }

    #[inline]
    pub fn compute_wave_speed(&self, h: f64, vel: DVec2) -> f64 {
        if h < self.h_min {
            return 0.0;
        }

        let c = (self.gravity * h).sqrt();
        let speed = vel.length();
        speed + c
    }

    #[inline]
    pub fn is_dry(&self, h: f64) -> bool {
        h < self.h_min
    }

    #[inline]
    pub fn gravity(&self) -> f64 {
        self.gravity
    }

    #[inline]
    pub fn h_min(&self) -> f64 {
        self.h_min
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct EulerFlux {
    pub mass: f64,
    pub momentum_x: f64,
    pub momentum_y: f64,
}

impl EulerFlux {
    pub fn zero() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RotatedFlux {
    pub mass: f64,
    pub momentum_n: f64,
    pub momentum_t: f64,
}

impl RotatedFlux {
    pub fn rotate_back(&self, normal: DVec2) -> EulerFlux {
        let tangent = DVec2::new(-normal.y, normal.x);

        EulerFlux {
            mass: self.mass,
            momentum_x: self.momentum_n * normal.x + self.momentum_t * tangent.x,
            momentum_y: self.momentum_n * normal.y + self.momentum_t * tangent.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_flux() {
        let calc = FluxCalculator::new(9.81, 1e-6);
        let flux = calc.compute_euler_flux(2.0, 1.0, 0.5);

        assert_eq!(flux.mass, 2.0);
        assert!((flux.momentum_x - (2.0 + 0.5 * 9.81 * 4.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dry_bed() {
        let calc = FluxCalculator::new(9.81, 0.05);
        assert!(calc.is_dry(0.01));
        assert!(!calc.is_dry(0.1));
    }

    #[test]
    fn test_wave_speed() {
        let calc = FluxCalculator::new(9.81, 1e-6);
        let speed = calc.compute_wave_speed(1.0, DVec2::new(1.0, 0.0));

        let expected = 1.0 + (9.81_f64).sqrt();
        assert!((speed - expected).abs() < 1e-10);
    }
}
