// src-tauri/src/marihydro/physics/schemes/flux_utils.rs

//! 通量计算公共工具

use glam::DVec2;

#[derive(Debug, Clone, Copy, Default)]
pub struct InterfaceFlux {
    pub mass: f64,
    pub momentum_x: f64,
    pub momentum_y: f64,
    pub max_wave_speed: f64,
}

impl InterfaceFlux {
    pub const ZERO: Self = Self {
        mass: 0.0,
        momentum_x: 0.0,
        momentum_y: 0.0,
        max_wave_speed: 0.0,
    };

    pub fn new(mass: f64, mx: f64, my: f64) -> Self {
        Self {
            mass,
            momentum_x: mx,
            momentum_y: my,
            max_wave_speed: 0.0,
        }
    }
    pub fn with_wave_speed(mut self, s: f64) -> Self {
        self.max_wave_speed = s;
        self
    }
    #[inline]
    pub fn momentum(&self) -> DVec2 {
        DVec2::new(self.momentum_x, self.momentum_y)
    }
    #[inline]
    pub fn scale(self, f: f64) -> Self {
        Self {
            mass: self.mass * f,
            momentum_x: self.momentum_x * f,
            momentum_y: self.momentum_y * f,
            max_wave_speed: self.max_wave_speed,
        }
    }
}

impl std::ops::Add for InterfaceFlux {
    type Output = Self;
    fn add(self, r: Self) -> Self {
        Self {
            mass: self.mass + r.mass,
            momentum_x: self.momentum_x + r.momentum_x,
            momentum_y: self.momentum_y + r.momentum_y,
            max_wave_speed: self.max_wave_speed.max(r.max_wave_speed),
        }
    }
}

impl std::ops::Neg for InterfaceFlux {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            mass: -self.mass,
            momentum_x: -self.momentum_x,
            momentum_y: -self.momentum_y,
            max_wave_speed: self.max_wave_speed,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RotatedState {
    pub h: f64,
    pub un: f64,
    pub ut: f64,
}

impl RotatedState {
    #[inline]
    pub fn from_global(h: f64, vel: DVec2, normal: DVec2) -> Self {
        let t = DVec2::new(-normal.y, normal.x);
        Self {
            h,
            un: vel.dot(normal),
            ut: vel.dot(t),
        }
    }
    #[inline]
    pub fn velocity_to_global(&self, normal: DVec2) -> DVec2 {
        let t = DVec2::new(-normal.y, normal.x);
        normal * self.un + t * self.ut
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RotatedFlux {
    pub mass: f64,
    pub momentum_n: f64,
    pub momentum_t: f64,
}

impl RotatedFlux {
    pub const ZERO: Self = Self {
        mass: 0.0,
        momentum_n: 0.0,
        momentum_t: 0.0,
    };

    #[inline]
    pub fn to_global(&self, normal: DVec2) -> InterfaceFlux {
        let t = DVec2::new(-normal.y, normal.x);
        let mom = normal * self.momentum_n + t * self.momentum_t;
        InterfaceFlux {
            mass: self.mass,
            momentum_x: mom.x,
            momentum_y: mom.y,
            max_wave_speed: 0.0,
        }
    }
}

#[inline]
pub fn physical_flux_1d(h: f64, un: f64, ut: f64, g: f64) -> RotatedFlux {
    let q = h * un;
    RotatedFlux {
        mass: q,
        momentum_n: q * un + 0.5 * g * h * h,
        momentum_t: q * ut,
    }
}

#[inline]
pub fn wave_speed(h: f64, g: f64, h_min: f64) -> f64 {
    if h < h_min {
        0.0
    } else {
        (g * h).sqrt()
    }
}
