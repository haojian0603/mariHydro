// crates/mh_physics/src/sediment/shear_stress.rs

//! 床面剪切应力计算模块
//!
//! 提供统一的床面剪切应力计算，消除 manager.rs 和 bed_load.rs 中的重复代码。
//!
//! # 物理背景
//!
//! ## Manning 公式
//!
//! ```text
//! τ_b = ρ × g × n² × |V|² / h^(1/3)
//! ```
//!
//! 其中：
//! - τ_b: 床面剪切应力 [Pa]
//! - ρ: 水密度 [kg/m³]
//! - g: 重力加速度 [m/s²]
//! - n: Manning 糙率系数 [s/m^(1/3)]
//! - V: 流速 [m/s]
//! - h: 水深 [m]
//!
//! ## Chezy 公式
//!
// crates/mh_physics/src/sediment/shear_stress.rs

//! 床面剪切应力计算模块（Backend 无关标量版本）

use crate::core::Backend;
use crate::types::PhysicalConstants;
use mh_runtime::RuntimeScalar;
use serde::{Deserialize, Serialize};

/// 剪切应力计算结果
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShearStress<S: RuntimeScalar = f64> {
    /// 剪切应力大小 [Pa]
    pub magnitude: S,
    /// x 方向分量 [Pa]
    pub tau_x: S,
    /// y 方向分量 [Pa]
    pub tau_y: S,
    /// 剪切流速 [m/s]
    pub u_star: S,
}

impl<S: RuntimeScalar> Default for ShearStress<S> {
    fn default() -> Self {
        Self {
            magnitude: S::ZERO,
            tau_x: S::ZERO,
            tau_y: S::ZERO,
            u_star: S::ZERO,
        }
    }
}

impl<S: RuntimeScalar> ShearStress<S> {
    pub fn zero() -> Self {
        Self::default()
    }
}

/// 床面剪切应力计算器
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShearStressCalculator<S: RuntimeScalar = f64> {
    /// 最小水深 [m]
    pub h_min: S,
    /// 水密度 [kg/m³]
    pub rho_water: S,
    /// 重力加速度 [m/s²]
    pub g: S,
}

impl<S: RuntimeScalar> Default for ShearStressCalculator<S> {
    fn default() -> Self {
        let physics = PhysicalConstants::seawater();
        Self {
            h_min: S::from_f64(0.01).unwrap_or(S::ZERO),
            rho_water: S::from_f64(physics.rho_water).unwrap_or(S::ONE),
            g: S::from_f64(physics.g).unwrap_or(S::ONE),
        }
    }
}

impl<S: RuntimeScalar> ShearStressCalculator<S> {
    /// 创建新的计算器
    pub fn new(h_min: S, rho_water: S, g: S) -> Self {
        Self { h_min, rho_water, g }
    }

    /// 从物理常数创建
    pub fn from_physics(physics: &PhysicalConstants, h_min: S) -> Self {
        Self {
            h_min,
            rho_water: S::from_f64(physics.rho_water).unwrap_or(S::ONE),
            g: S::from_f64(physics.g).unwrap_or(S::ONE),
        }
    }

    /// 使用 Manning 公式计算剪切应力（单个点）
    pub fn manning(&self, h: S, u: S, v: S, manning_n: S) -> ShearStress<S> {
        if h < self.h_min {
            return ShearStress::default();
        }

        let speed_sq = u * u + v * v;
        let eps = S::from_f64(1e-12).unwrap_or(S::ZERO);
        let speed = (speed_sq + eps).sqrt();
        let h_pow = h.powf(S::from_f64(1.0 / 3.0).unwrap_or(S::ONE));

        let magnitude = self.rho_water * self.g * manning_n * manning_n * speed_sq / h_pow;
        let tau_x = magnitude * u / speed;
        let tau_y = magnitude * v / speed;

        let u_star = (magnitude / self.rho_water).sqrt();

        ShearStress {
            magnitude,
            tau_x,
            tau_y,
            u_star,
        }
    }

    /// 使用 Chezy 公式计算剪切应力（单个点）
    pub fn chezy(&self, h: S, u: S, v: S, chezy_c: S) -> ShearStress<S> {
        if h < self.h_min || chezy_c < S::from_f64(1e-6).unwrap_or(S::ZERO) {
            return ShearStress::default();
        }

        let speed_sq = u * u + v * v;
        let eps = S::from_f64(1e-12).unwrap_or(S::ZERO);
        let speed = (speed_sq + eps).sqrt();
        let magnitude = self.rho_water * self.g * speed_sq / (chezy_c * chezy_c);
        let tau_x = magnitude * u / speed;
        let tau_y = magnitude * v / speed;

        let u_star = (magnitude / self.rho_water).sqrt();

        ShearStress {
            magnitude,
            tau_x,
            tau_y,
            u_star,
        }
    }

    /// 从水力半径和能量坡度计算剪切应力
    pub fn from_energy_slope(&self, hydraulic_radius: S, energy_slope: S) -> S {
        self.rho_water * self.g * hydraulic_radius * energy_slope.abs()
    }

    /// 批量计算 Manning 剪切应力
    pub fn manning_batch(
        &self,
        h: &[S],
        u: &[S],
        v: &[S],
        manning_n: ManningCoeff<'_, S>,
    ) -> (Vec<S>, Vec<S>, Vec<S>, Vec<S>) {
        debug_assert_eq!(h.len(), u.len());
        debug_assert_eq!(h.len(), v.len());

        let mut tau_out = vec![S::ZERO; h.len()];
        let mut tau_x_out = vec![S::ZERO; h.len()];
        let mut tau_y_out = vec![S::ZERO; h.len()];
        let mut u_star_out = vec![S::ZERO; h.len()];

        for i in 0..h.len() {
            let n = manning_n.get(i);
            let shear = self.manning(h[i], u[i], v[i], n);
            tau_out[i] = shear.magnitude;
            tau_x_out[i] = shear.tau_x;
            tau_y_out[i] = shear.tau_y;
            u_star_out[i] = shear.u_star;
        }

        (tau_out, tau_x_out, tau_y_out, u_star_out)
    }

    /// 批量计算 Chezy 剪切应力
    pub fn chezy_batch(
        &self,
        h: &[S],
        u: &[S],
        v: &[S],
        chezy_c: ChezyCoeff<'_, S>,
    ) -> (Vec<S>, Vec<S>, Vec<S>, Vec<S>) {
        debug_assert_eq!(h.len(), u.len());
        debug_assert_eq!(h.len(), v.len());

        let mut tau_out = vec![S::ZERO; h.len()];
        let mut tau_x_out = vec![S::ZERO; h.len()];
        let mut tau_y_out = vec![S::ZERO; h.len()];
        let mut u_star_out = vec![S::ZERO; h.len()];

        for i in 0..h.len() {
            let c = chezy_c.get(i);
            let shear = self.chezy(h[i], u[i], v[i], c);
            tau_out[i] = shear.magnitude;
            tau_x_out[i] = shear.tau_x;
            tau_y_out[i] = shear.tau_y;
            u_star_out[i] = shear.u_star;
        }

        (tau_out, tau_x_out, tau_y_out, u_star_out)
    }
}

/// Manning 糙率，可为常数或数组
#[derive(Debug, Clone, Copy, Serialize)]
pub enum ManningCoeff<'a, S: RuntimeScalar = f64> {
    /// 常数糙率
    Uniform(S),
    /// 按单元提供的糙率数组
    Array(&'a [S]),
}

impl<'a, S: RuntimeScalar> ManningCoeff<'a, S> {
    pub fn get(&self, idx: usize) -> S {
        match self {
            ManningCoeff::Uniform(v) => *v,
            ManningCoeff::Array(arr) => arr[idx],
        }
    }
}

impl<'a, S: RuntimeScalar> From<S> for ManningCoeff<'a, S> {
    fn from(val: S) -> Self {
        ManningCoeff::Uniform(val)
    }
}

impl<'a, S: RuntimeScalar> From<&'a [S]> for ManningCoeff<'a, S> {
    fn from(arr: &'a [S]) -> Self {
        ManningCoeff::Array(arr)
    }
}

// Removed From implementation for ManningCoeff

// Removed From implementation for ChezyCoeff

/// Chezy 系数，可为常数或数组
#[derive(Debug, Clone, Copy, Serialize)]
pub enum ChezyCoeff<'a, S: RuntimeScalar = f64> {
    /// 常数 Chezy 系数
    Uniform(S),
    /// 按单元提供的 Chezy 系数数组
    Array(&'a [S]),
}

impl<'a, S: RuntimeScalar> ChezyCoeff<'a, S> {
    pub fn get(&self, idx: usize) -> S {
        match self {
            ChezyCoeff::Uniform(v) => *v,
            ChezyCoeff::Array(arr) => arr[idx],
        }
    }
}

impl<'a, S: RuntimeScalar> From<S> for ChezyCoeff<'a, S> {
    fn from(val: S) -> Self {
        ChezyCoeff::Uniform(val)
    }
}

impl<'a, S: RuntimeScalar> From<&'a [S]> for ChezyCoeff<'a, S> {
    fn from(arr: &'a [S]) -> Self {
        ChezyCoeff::Array(arr)
    }
}

// Removed From implementation for ChezyCoeff

impl<S: RuntimeScalar> ShearStressCalculator<S> {
    /// 使用 Backend 缓冲区批量计算 Manning 剪切应力
    pub fn manning_batch_backend<B>(
        &self,
        backend: &B,
        h: &B::Buffer<S>,
        u: &B::Buffer<S>,
        v: &B::Buffer<S>,
        manning_n: ManningCoeff<'_, S>,
        tau_out: &mut B::Buffer<S>,
        tau_x_out: &mut B::Buffer<S>,
        tau_y_out: &mut B::Buffer<S>,
        u_star_out: &mut B::Buffer<S>,
    )
    where
        B: Backend<Scalar = S>,
    {
        let n = h.len().min(u.len()).min(v.len());
        for i in 0..n {
            let n_coeff = manning_n.get(i);
            let shear = self.manning(h[i], u[i], v[i], n_coeff);
            tau_out[i] = shear.magnitude;
            tau_x_out[i] = shear.tau_x;
            tau_y_out[i] = shear.tau_y;
            u_star_out[i] = shear.u_star;
        }
        let _ = backend;
    }

    /// 使用 Backend 缓冲区批量计算 Chezy 剪切应力
    pub fn chezy_batch_backend<B>(
        &self,
        backend: &B,
        h: &B::Buffer<S>,
        u: &B::Buffer<S>,
        v: &B::Buffer<S>,
        chezy_c: ChezyCoeff<'_, S>,
        tau_out: &mut B::Buffer<S>,
        tau_x_out: &mut B::Buffer<S>,
        tau_y_out: &mut B::Buffer<S>,
        u_star_out: &mut B::Buffer<S>,
    )
    where
        B: Backend<Scalar = S>,
    {
        let n = h.len().min(u.len()).min(v.len());
        for i in 0..n {
            let c_coeff = chezy_c.get(i);
            let shear = self.chezy(h[i], u[i], v[i], c_coeff);
            tau_out[i] = shear.magnitude;
            tau_x_out[i] = shear.tau_x;
            tau_y_out[i] = shear.tau_y;
            u_star_out[i] = shear.u_star;
        }
        let _ = backend;
    }
}

/// Shields 参数，用于判断是否发生起动
pub fn shields_parameter(
    tau_b: f64,
    rho_water: f64,
    rho_sediment: f64,
    d50: f64,
) -> f64 {
    let physics = PhysicalConstants::default();
    let submerged_weight = (rho_sediment - rho_water) * physics.g * d50;
    if submerged_weight.abs() < 1e-12 {
        return 0.0;
    }

    tau_b / submerged_weight
}

/// 计算临界 Shields 参数（Soulsby-Whitehouse 公式）
pub fn shields_critical_soulsby(d50: f64, rho_sediment: f64, rho_water: f64) -> f64 {
    let relative_density = (rho_sediment / rho_water) - 1.0;
    let nu = 1e-6; // 运动粘度 [m²/s]

    let physics = PhysicalConstants::default();
    let d_star = d50 * (relative_density * physics.g / (nu * nu)).powf(1.0 / 3.0);
    let term1 = 0.30 / (1.0 + 1.2 * d_star);
    let term2 = 0.055 * (1.0 - (-0.02 * d_star).exp());
    term1 + term2
}

/// 计算床面切应力到 Shields 参数
pub fn shields_from_tau(tau_b: f64, rho_water: f64, rho_sediment: f64, d50: f64) -> f64 {
    shields_parameter(tau_b, rho_water, rho_sediment, d50)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manning_single_point() {
        let calculator: ShearStressCalculator<f64> = ShearStressCalculator::default();

        let h = 1.0;
        let u = 2.0;
        let v = 1.0;
        let n = 0.03;

        let shear = calculator.manning(h, u, v, n);

        assert!(shear.magnitude > 0.0);
        assert!(shear.tau_x.abs() > 0.0);
        assert!(shear.tau_y.abs() > 0.0);
        assert!(shear.u_star > 0.0);
    }

    #[test]
    fn test_chezy_single_point() {
        let calculator: ShearStressCalculator<f64> = ShearStressCalculator::default();

        let h = 1.0;
        let u = 1.5;
        let v = 0.5;
        let c = 50.0;

        let shear = calculator.chezy(h, u, v, c);

        assert!(shear.magnitude > 0.0);
        assert!(shear.u_star > 0.0);
    }

    #[test]
    fn test_shields_parameter() {
        let tau_b = 2.0;
        let rho_w = 1000.0;
        let rho_s = 2650.0;
        let d50 = 0.002;

        let theta = shields_parameter(tau_b, rho_w, rho_s, d50);

        assert!(theta > 0.0);
    }
    #[test]
    fn test_batch_calculation() {
        let calc = ShearStressCalculator::default();
        
        let h = vec![2.0, 1.5, 1.0, 0.5];
        let u = vec![1.0, 1.5, 2.0, 0.5];
        let v = vec![0.0, 0.5, 0.0, 0.0];
        let (tau, tau_x, tau_y, u_star): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) =
            calc.manning_batch(&h, &u, &v, 0.03.into());
        
        // 所有值应为正
        for (&t, (&tx, (&ty, &us))) in tau.iter().zip(tau_x.iter().zip(tau_y.iter().zip(u_star.iter()))) {
            assert!(t > 0.0);
            assert!(tx.is_finite());
            assert!(ty.is_finite());
            assert!(us > 0.0);
        }
    }
}
