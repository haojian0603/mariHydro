// crates/mh_physics/src/sediment/shear_stress.rs

//! 床面剪切应力计算模块（Backend 无关标量版本）
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

use crate::prelude::*;
use crate::types::PhysicalConstants;
use serde::{Deserialize, Serialize};

/// 剪切应力计算结果
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ShearStress<S: RuntimeScalar> {
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
#[derive(Debug, Clone)]
pub struct ShearStressCalculator<B: Backend> {
    /// 后端
    backend: B,
    /// 最小水深 [m]
    pub h_min: B::Scalar,
    /// 水密度 [kg/m³]
    pub rho_water: B::Scalar,
    /// 重力加速度 [m/s²]
    pub g: B::Scalar,
}

impl<B: Backend> ShearStressCalculator<B> {
    /// 创建新的计算器
    pub fn new(backend: B, h_min: B::Scalar, rho_water: B::Scalar, g: B::Scalar) -> Self {
        Self { backend, h_min, rho_water, g }
    }

    /// 使用默认物理常数创建
    pub fn with_backend(backend: B) -> Self {
        let physics = PhysicalConstants::seawater();
        Self {
            h_min: backend.config_scalar(0.01, "ShearStressCalculator.with_backend.h_min"),
            rho_water: backend.config_scalar(physics.rho_water, "ShearStressCalculator.with_backend.rho_water"),
            g: backend.config_scalar(physics.g, "ShearStressCalculator.with_backend.g"),
            backend,
        }
    }

    /// 从物理常数创建
    pub fn from_physics(backend: B, physics: &PhysicalConstants, h_min: B::Scalar) -> Self {
        let rho_water =
            backend.config_scalar(physics.rho_water, "ShearStressCalculator.from_physics.rho_water");
        let g = backend.config_scalar(physics.g, "ShearStressCalculator.from_physics.g");
        Self {
            h_min,
            rho_water,
            g,
            backend,
        }
    }

    /// 使用 Manning 公式计算剪切应力（单个点）
    pub fn manning(
        &self,
        h: B::Scalar,
        u: B::Scalar,
        v: B::Scalar,
        manning_n: B::Scalar,
    ) -> ShearStress<B::Scalar> {
        if h < self.h_min {
            return ShearStress::default();
        }

        let cfg = |v| self.backend.config_scalar(v, "ShearStressCalculator.manning");
        let speed_sq = u * u + v * v;
        let eps = cfg(1e-12);
        let speed = (speed_sq + eps).sqrt();
        let h_pow = h.powf(cfg(1.0 / 3.0));

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
    pub fn chezy(
        &self,
        h: B::Scalar,
        u: B::Scalar,
        v: B::Scalar,
        chezy_c: B::Scalar,
    ) -> ShearStress<B::Scalar> {
        let cfg = |v| self.backend.config_scalar(v, "ShearStressCalculator.chezy");
        if h < self.h_min || chezy_c < cfg(1e-6) {
            return ShearStress::default();
        }

        let speed_sq = u * u + v * v;
        let eps = cfg(1e-12);
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
    pub fn from_energy_slope(&self, hydraulic_radius: B::Scalar, energy_slope: B::Scalar) -> B::Scalar {
        self.rho_water * self.g * hydraulic_radius * energy_slope.abs()
    }
}

/// Manning 糙率（Backend 缓冲区版本，Layer 3 推荐）
#[derive(Debug, Clone, Copy)]
pub enum ManningCoeffBuf<'a, B: Backend> {
    Uniform(B::Scalar),
    Buffer(&'a B::Buffer<B::Scalar>),
}

impl<'a, B: Backend> ManningCoeffBuf<'a, B> {
    pub fn get(&self, idx: usize) -> B::Scalar {
        match self {
            ManningCoeffBuf::Uniform(v) => *v,
            ManningCoeffBuf::Buffer(buf) => buf[idx],
        }
    }
}

impl<'a, B: Backend> ManningCoeffBuf<'a, B> {
    /// 从 Backend 缓冲区创建 ManningCoeffBuf
    pub fn from_buffer(buf: &'a B::Buffer<B::Scalar>) -> Self {
        ManningCoeffBuf::Buffer(buf)
    }
}

/// Chezy 系数（Backend 缓冲区版本，Layer 3 推荐）
#[derive(Debug, Clone, Copy)]
pub enum ChezyCoeffBuf<'a, B: Backend> {
    Uniform(B::Scalar),
    Buffer(&'a B::Buffer<B::Scalar>),
}

impl<'a, B: Backend> ChezyCoeffBuf<'a, B> {
    pub fn get(&self, idx: usize) -> B::Scalar {
        match self {
            ChezyCoeffBuf::Uniform(v) => *v,
            ChezyCoeffBuf::Buffer(buf) => buf[idx],
        }
    }
}

impl<'a, B: Backend> ChezyCoeffBuf<'a, B> {
    /// 从 Backend 缓冲区创建 ChezyCoeffBuf
    pub fn from_buffer(buf: &'a B::Buffer<B::Scalar>) -> Self {
        ChezyCoeffBuf::Buffer(buf)
    }
}

/// 剪切应力批量计算错误
#[derive(Debug, Clone)]
pub enum ShearStressBatchError {
    SizeMismatch { expected: usize, actual: usize },
}

impl<B: Backend> ShearStressCalculator<B> {
    /// 使用 Backend 缓冲区批量计算 Manning 剪切应力
    pub fn manning_batch(
        &self,
        h: &B::Buffer<B::Scalar>,
        u: &B::Buffer<B::Scalar>,
        v: &B::Buffer<B::Scalar>,
        manning_n: ManningCoeffBuf<'_, B>,
        tau_out: &mut B::Buffer<B::Scalar>,
        tau_x_out: &mut B::Buffer<B::Scalar>,
        tau_y_out: &mut B::Buffer<B::Scalar>,
        u_star_out: &mut B::Buffer<B::Scalar>,
    ) -> Result<(), ShearStressBatchError> {
        let n = h.len().min(u.len()).min(v.len());
        if tau_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: tau_out.len() });
        }
        if tau_x_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: tau_x_out.len() });
        }
        if tau_y_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: tau_y_out.len() });
        }
        if u_star_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: u_star_out.len() });
        }

        for i in 0..n {
            let n_coeff = manning_n.get(i);
            let shear = self.manning(h[i], u[i], v[i], n_coeff);
            tau_out[i] = shear.magnitude;
            tau_x_out[i] = shear.tau_x;
            tau_y_out[i] = shear.tau_y;
            u_star_out[i] = shear.u_star;
        }
        Ok(())
    }

    /// 使用 Backend 缓冲区批量计算 Chezy 剪切应力
    pub fn chezy_batch(
        &self,
        h: &B::Buffer<B::Scalar>,
        u: &B::Buffer<B::Scalar>,
        v: &B::Buffer<B::Scalar>,
        chezy_c: ChezyCoeffBuf<'_, B>,
        tau_out: &mut B::Buffer<B::Scalar>,
        tau_x_out: &mut B::Buffer<B::Scalar>,
        tau_y_out: &mut B::Buffer<B::Scalar>,
        u_star_out: &mut B::Buffer<B::Scalar>,
    ) -> Result<(), ShearStressBatchError> {
        let n = h.len().min(u.len()).min(v.len());
        if tau_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: tau_out.len() });
        }
        if tau_x_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: tau_x_out.len() });
        }
        if tau_y_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: tau_y_out.len() });
        }
        if u_star_out.len() < n {
            return Err(ShearStressBatchError::SizeMismatch { expected: n, actual: u_star_out.len() });
        }

        for i in 0..n {
            let c_coeff = chezy_c.get(i);
            let shear = self.chezy(h[i], u[i], v[i], c_coeff);
            tau_out[i] = shear.magnitude;
            tau_x_out[i] = shear.tau_x;
            tau_y_out[i] = shear.tau_y;
            u_star_out[i] = shear.u_star;
        }
        Ok(())
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
    use crate::core::CpuBackend;

    #[test]
    fn test_manning_single_point() {
        let backend = CpuBackend::<f64>::new();
        let calculator = ShearStressCalculator::with_backend(backend);

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
        let backend = CpuBackend::<f64>::new();
        let calculator = ShearStressCalculator::with_backend(backend);

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
        let backend = CpuBackend::<f64>::new();
        let calc = ShearStressCalculator::with_backend(backend);

        let h = vec![2.0, 1.5, 1.0, 0.5];
        let u = vec![1.0, 1.5, 2.0, 0.5];
        let v = vec![0.0, 0.5, 0.0, 0.0];
        let mut h_buf = backend.alloc(h.len());
        let mut u_buf = backend.alloc(u.len());
        let mut v_buf = backend.alloc(v.len());
        h_buf.copy_from_slice(&h);
        u_buf.copy_from_slice(&u);
        v_buf.copy_from_slice(&v);

        let mut tau = backend.alloc(h.len());
        let mut tau_x = backend.alloc(h.len());
        let mut tau_y = backend.alloc(h.len());
        let mut u_star = backend.alloc(h.len());

        calc.manning_batch(
            &h_buf,
            &u_buf,
            &v_buf,
            ManningCoeffBuf::Uniform(0.03),
            &mut tau,
            &mut tau_x,
            &mut tau_y,
            &mut u_star,
        )
        .unwrap();

        let tau_slice = tau.as_slice();
        let tau_x_slice = tau_x.as_slice();
        let tau_y_slice = tau_y.as_slice();
        let u_star_slice = u_star.as_slice();

        // 所有值应为正
        for (&t, (&tx, (&ty, &us))) in tau_slice.iter().zip(tau_x_slice.iter().zip(tau_y_slice.iter().zip(u_star_slice.iter()))) {
            assert!(t > 0.0);
            assert!(tx.is_finite());
            assert!(ty.is_finite());
            assert!(us > 0.0);
        }
    }
}
