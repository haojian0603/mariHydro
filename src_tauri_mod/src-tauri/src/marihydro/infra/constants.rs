// src-tauri/src/marihydro/infra/constants.rs
use std::f64::consts::PI;

pub mod physics {
    use super::PI;
    pub const EARTH_ROTATION_RATE_RAD: f64 = 7.292115e-5;
    pub const STANDARD_GRAVITY: f64 = 9.80665;
    pub const EARTH_RADIUS: f64 = 6_371_000.0;
    pub const SECONDS_PER_DAY: f64 = 86400.0;
    pub const DEG_TO_RAD: f64 = PI / 180.0;
    pub const RAD_TO_DEG: f64 = 180.0 / PI;
    pub const STD_SEAWATER_DENSITY: f64 = 1025.0;
    pub const STD_FRESHWATER_DENSITY: f64 = 1000.0;
    pub const STD_AIR_DENSITY: f64 = 1.225;
    pub const STD_ATM_PRESSURE: f64 = 101325.0;
}

pub mod validation {
    pub const MAX_REASONABLE_WIND_SPEED: f64 = 130.0;
    pub const MAX_REASONABLE_DEPTH: f64 = 15_000.0;
    pub const MAX_REASONABLE_VELOCITY: f64 = 100.0;
    pub const SUSPICIOUS_ELEVATION_HIGH: f64 = 8900.0;
    pub const MAX_SCALE_FACTOR: f64 = 1e6;
    pub const MIN_SCALE_FACTOR: f64 = 1e-6;
    pub const MIN_ACTIVE_RATIO: f64 = 0.001;
}

pub mod defaults {
    pub const GHOST_WIDTH: usize = 2;
    pub const H_MIN: f64 = 0.05;
    pub const ELEVATION: f64 = -10.0;
    pub const MANNING_N: f64 = 0.025;
    pub const EDDY_VISCOSITY: f64 = 1.0;
    pub const SEDIMENT_W_S: f64 = 0.001;
    pub const SEDIMENT_TAU_CR: f64 = 0.1;
    pub const CFL: f64 = 0.9;
    pub const MAX_DT: f64 = 60.0;
    pub const MIN_DT: f64 = 1e-6;
}
