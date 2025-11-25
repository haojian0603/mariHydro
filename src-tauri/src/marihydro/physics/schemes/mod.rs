// src-tauri/src/marihydro/physics/schemes/mod.rs

// pub mod friction;
pub mod hllc;
pub mod muscl;

use std::ops::{Add, Mul, Sub};

/// 物理状态微元 (Conservative Variables)
/// 用于内核计算，不涉及内存分配
#[derive(Debug, Clone, Copy, Default)]
pub struct ConservedVars {
    pub h: f64,  // 水深
    pub hu: f64, // 单宽流量 X
    pub hv: f64, // 单宽流量 Y
    pub hc: f64, // 泥沙质量/浓度积
}

/// 原始变量微元 (Primitive Variables)
#[derive(Debug, Clone, Copy, Default)]
pub struct PrimitiveVars {
    pub h: f64,
    pub u: f64,
    pub v: f64,
    pub c: f64,
    pub z: f64,   // 地形高程 (Bed Elevation)
    pub eta: f64, // 水位 (h + z)
}

/// 通量微元 (Flux)
#[derive(Debug, Clone, Copy, Default)]
pub struct FluxVars {
    pub mass: f64,  // 质量通量
    pub x_mom: f64, // X动量通量
    pub y_mom: f64, // Y动量通量
    pub sed: f64,   // 泥沙通量
}

// --- 运算符重载 (方便公式编写) ---

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
    /// 转换为原始变量 (去奇异化)
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
            // 这里的 1.0/h 是热点，编译器通常会自动优化为乘法逆元
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

    /// 从原始变量构建
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
