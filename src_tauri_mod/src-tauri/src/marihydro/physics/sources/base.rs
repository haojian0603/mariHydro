// src-tauri/src/marihydro/physics/sources/base.rs
use crate::marihydro::core::types::NumericalParams;

pub struct FrictionDecayCalculator {
    g: f64,
    h_min: f64,
    h_friction: f64,
}

impl FrictionDecayCalculator {
    pub fn new(g: f64, params: &NumericalParams) -> Self {
        Self { g, h_min: params.h_min, h_friction: params.h_friction }
    }

    #[inline]
    pub fn compute_manning_cf(&self, h: f64, n: f64) -> f64 {
        let h_safe = h.max(self.h_friction);
        self.g * n * n / h_safe.cbrt()
    }

    #[inline]
    pub fn compute_chezy_cf(&self, chezy_c: f64) -> f64 {
        self.g / (chezy_c * chezy_c)
    }

    #[inline]
    pub fn compute_decay(&self, cf: f64, speed: f64, dt: f64) -> f64 {
        1.0 / (1.0 + dt * cf * speed)
    }

    #[inline]
    pub fn apply_implicit(&self, hu: f64, hv: f64, h: f64, cf: f64, dt: f64) -> (f64, f64) {
        if h < self.h_min {
            return (0.0, 0.0);
        }
        let speed_sq = (hu * hu + hv * hv) / (h * h);
        if speed_sq < 1e-20 {
            return (hu, hv);
        }
        let speed = speed_sq.sqrt();
        let decay = self.compute_decay(cf, speed, dt);
        (hu * decay, hv * decay)
    }

    pub fn compute_decay_field(&self, h: &[f64], hu: &[f64], hv: &[f64], n: &[f64], dt: f64, output: &mut [f64]) {
        for i in 0..h.len() {
            if h[i] < self.h_min {
                output[i] = 0.0;
                continue;
            }
            let cf = self.compute_manning_cf(h[i], n[i]);
            let speed_sq = (hu[i] * hu[i] + hv[i] * hv[i]) / (h[i] * h[i]);
            if speed_sq < 1e-20 {
                output[i] = 1.0;
            } else {
                output[i] = self.compute_decay(cf, speed_sq.sqrt(), dt);
            }
        }
    }
}

pub struct SourceHelpers;

impl SourceHelpers {
    #[inline]
    pub fn safe_accumulate(acc: &mut f64, val: f64) {
        if val.is_finite() {
            *acc += val;
        }
    }

    #[inline]
    pub fn validate_contribution(val: f64, max_abs: f64) -> f64 {
        if !val.is_finite() { return 0.0; }
        val.clamp(-max_abs, max_abs)
    }

    #[inline]
    pub fn smooth_transition(h: f64, h_dry: f64, h_wet: f64) -> f64 {
        if h <= h_dry { 0.0 }
        else if h >= h_wet { 1.0 }
        else { (h - h_dry) / (h_wet - h_dry) }
    }
}
