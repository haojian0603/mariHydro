// src-tauri/src/marihydro/physics/schemes/mod.rs

pub mod hllc;
pub mod hydrostatic;
pub mod muscl;

use std::ops::{Add, Mul, Sub};

// 导出非结构化网格核心类型
pub use hllc::{solve_hllc, solve_hllc_simple, HllcResult};
pub use hydrostatic::{
    compute_bed_slope_source, hydrostatic_reconstruction, muscl_reconstruct, BedSlopeSource,
    HydrostaticFaceState, ReconstructedState, SlopeLimiter,
};

// 导出结构化网格类型（可选使用）
pub use muscl::{reconstruct_interface, SlopeLimiterType};

/// 守恒变量微元
#[derive(Debug, Clone, Copy, Default)]
pub struct ConservedVars {
    pub h: f64,
    pub hu: f64,
    pub hv: f64,
    pub hc: f64,
}

/// 原始变量微元
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimitiveVars {
    pub h: f64,
    pub u: f64,
    pub v: f64,
    pub c: f64,
    pub z: f64,
    pub eta: f64,
}

/// 通量微元
#[derive(Debug, Clone, Copy, Default)]
pub struct FluxVars {
    pub mass: f64,
    pub x_mom: f64,
    pub y_mom: f64,
    pub sed: f64,
}

impl Add for ConservedVars {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            h: self.h + rhs.h,
            hu: self.hu + rhs.hu,
            hv: self.hv + rhs.hv,
            hc: self.hc + rhs.hc,
        }
    }
}

impl Sub for ConservedVars {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            h: self.h - rhs.h,
            hu: self.hu - rhs.hu,
            hv: self.hv - rhs.hv,
            hc: self.hc - rhs.hc,
        }
    }
}

impl Mul<f64> for ConservedVars {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            h: self.h * rhs,
            hu: self.hu * rhs,
            hv: self.hv * rhs,
            hc: self.hc * rhs,
        }
    }
}

impl ConservedVars {
    #[inline(always)]
    pub fn to_primitive(&self, z: f64, h_min: f64) -> PrimitiveVars {
        if self.h < h_min {
            PrimitiveVars {
                h: self.h,
                u: 0.0,
                v: 0.0,
                c: 0.0,
                z,
                eta: self.h + z,
            }
        } else {
            let inv_h = 1.0 / self.h;
            PrimitiveVars {
                h: self.h,
                u: self.hu * inv_h,
                v: self.hv * inv_h,
                c: self.hc * inv_h,
                z,
                eta: self.h + z,
            }
        }
    }

    #[inline(always)]
    pub fn from_primitive(p: &PrimitiveVars) -> Self {
        Self {
            h: p.h,
            hu: p.h * p.u,
            hv: p.h * p.v,
            hc: p.h * p.c,
        }
    }
}
