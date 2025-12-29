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
//! ```text
//! τ_b = ρ × g × |V|² / C²
//! ```
//!
//! 其中 C 是 Chezy 系数。
//!
//! ## 剪切流速
//!
//! ```text
//! u_* = √(τ_b / ρ)
//! ```

use crate::types::PhysicalConstants;
use mh_runtime::RuntimeScalar;

/// 剪切应力计算结果
#[derive(Debug, Clone, Copy, Default)]
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

impl<S: RuntimeScalar> ShearStress<S> {
    /// 零剪切应力
    pub fn zero() -> Self {
        Self {
            magnitude: S::ZERO,
            tau_x: S::ZERO,
            tau_y: S::ZERO,
            u_star: S::ZERO,
        }
    }
}

/// 床面剪切应力计算器
#[derive(Debug, Clone, Copy)]
pub struct ShearStressCalculator {
    /// 最小水深 [m]
    pub h_min: f64,
    /// 水密度 [kg/m³]
    pub rho_water: f64,
    /// 重力加速度 [m/s²]
    pub g: f64,
}

impl Default for ShearStressCalculator {
    fn default() -> Self {
        Self {
            h_min: 0.01,
            rho_water: 1000.0,
            g: 9.81,
        }
    }
}

impl ShearStressCalculator {
    /// 创建新的计算器
    pub fn new(h_min: f64, rho_water: f64, g: f64) -> Self {
        Self { h_min, rho_water, g }
    }

    /// 从物理常数创建
    pub fn from_physics(physics: &PhysicalConstants, h_min: f64) -> Self {
        Self {
            h_min,
            rho_water: physics.rho_water,
            g: physics.g,
        }
    }

    /// 使用 Manning 公式计算剪切应力（单个点）
    ///
    /// τ_b = ρ × g × n² × |V|² / h^(1/3)
    ///
    /// # 参数
    ///
    /// - `h`: 水深 [m]
    /// - `u`: x 方向流速 [m/s]
    /// - `v`: y 方向流速 [m/s]
    /// - `manning_n`: Manning 糙率系数 [s/m^(1/3)]
    pub fn manning(&self, h: f64, u: f64, v: f64, manning_n: f64) -> ShearStress<f64> {
        if h < self.h_min {
            return ShearStress::default();
        }

        let speed_sq = u * u + v * v;
        let speed = speed_sq.sqrt();
        let h_pow = h.powf(1.0 / 3.0);

        // τ_b = ρ × g × n² × |V|² / h^(1/3)
        let magnitude = self.rho_water * self.g * manning_n * manning_n * speed_sq / h_pow;

        // 分量
        let (tau_x, tau_y) = if speed > 1e-10 {
            (magnitude * u / speed, magnitude * v / speed)
        } else {
            (0.0, 0.0)
        };

        // 剪切流速
        let u_star = (magnitude / self.rho_water).sqrt();

        ShearStress {
            magnitude,
            tau_x,
            tau_y,
            u_star,
        }
    }

    /// 使用 Chezy 公式计算剪切应力（单个点）
    ///
    /// τ_b = ρ × g × |V|² / C²
    ///
    /// # 参数
    ///
    /// - `h`: 水深 [m]
    /// - `u`: x 方向流速 [m/s]
    /// - `v`: y 方向流速 [m/s]
    /// - `chezy_c`: Chezy 系数 [m^(1/2)/s]
    pub fn chezy(&self, h: f64, u: f64, v: f64, chezy_c: f64) -> ShearStress<f64> {
        if h < self.h_min || chezy_c < 1e-6 {
            return ShearStress::default();
        }

        let speed_sq = u * u + v * v;
        let speed = speed_sq.sqrt();

        // τ_b = ρ × g × |V|² / C²
        let magnitude = self.rho_water * self.g * speed_sq / (chezy_c * chezy_c);

        // 分量
        let (tau_x, tau_y) = if speed > 1e-10 {
            (magnitude * u / speed, magnitude * v / speed)
        } else {
            (0.0, 0.0)
        };

        // 剪切流速
        let u_star = (magnitude / self.rho_water).sqrt();

        ShearStress {
            magnitude,
            tau_x,
            tau_y,
            u_star,
        }
    }

    /// 从水力半径和能量坡度计算剪切应力
    ///
    /// τ_b = ρ × g × R × S_f
    ///
    /// # 参数
    ///
    /// - `hydraulic_radius`: 水力半径 [m]
    /// - `energy_slope`: 能量坡度 [-]
    pub fn from_energy_slope(&self, hydraulic_radius: f64, energy_slope: f64) -> f64 {
        self.rho_water * self.g * hydraulic_radius * energy_slope.abs()
    }

    /// 批量计算 Manning 剪切应力
    ///
    /// # 参数
    ///
    /// - `h`: 水深数组 [m]
    /// - `u`: x 方向流速数组 [m/s]
    /// - `v`: y 方向流速数组 [m/s]
    /// - `manning_n`: Manning 系数（可以是单一值或数组）
    /// - `tau_out`: 输出剪切应力大小数组 [Pa]
    /// - `tau_x_out`: 输出 x 分量数组（可选）
    /// - `tau_y_out`: 输出 y 分量数组（可选）
    pub fn compute_manning_batch(
        &self,
        h: &[f64],
        u: &[f64],
        v: &[f64],
        manning_n: ManningCoeff<'_>,
        tau_out: &mut [f64],
        mut tau_x_out: Option<&mut [f64]>,
        mut tau_y_out: Option<&mut [f64]>,
    ) {
        let n_cells = h.len().min(u.len()).min(v.len()).min(tau_out.len());
        let rho_g = self.rho_water * self.g;

        for i in 0..n_cells {
            let hi = h[i];
            if hi < self.h_min {
                tau_out[i] = 0.0;
                if let Some(ref mut tx_arr) = tau_x_out {
                    if i < tx_arr.len() {
                        tx_arr[i] = 0.0;
                    }
                }
                if let Some(ref mut ty_arr) = tau_y_out {
                    if i < ty_arr.len() {
                        ty_arr[i] = 0.0;
                    }
                }
                continue;
            }

            let ui = u[i];
            let vi = v[i];
            let speed_sq = ui * ui + vi * vi;
            let speed = speed_sq.sqrt();
            let h_pow = hi.powf(1.0 / 3.0);

            let n = manning_n.get(i);
            let tau_mag = rho_g * n * n * speed_sq / h_pow;
            tau_out[i] = tau_mag;

            // 分量
            if speed > 1e-10 {
                if let Some(ref mut tx_arr) = tau_x_out {
                    if i < tx_arr.len() {
                        tx_arr[i] = tau_mag * ui / speed;
                    }
                }
                if let Some(ref mut ty_arr) = tau_y_out {
                    if i < ty_arr.len() {
                        ty_arr[i] = tau_mag * vi / speed;
                    }
                }
            } else {
                if let Some(ref mut tx_arr) = tau_x_out {
                    if i < tx_arr.len() {
                        tx_arr[i] = 0.0;
                    }
                }
                if let Some(ref mut ty_arr) = tau_y_out {
                    if i < ty_arr.len() {
                        ty_arr[i] = 0.0;
                    }
                }
            }
        }
    }

    /// 批量计算剪切流速
    ///
    /// u_* = √(τ_b / ρ)
    pub fn compute_shear_velocity(&self, tau: &[f64], u_star_out: &mut [f64]) {
        let n = tau.len().min(u_star_out.len());
        let inv_rho = 1.0 / self.rho_water;
        for i in 0..n {
            u_star_out[i] = (tau[i] * inv_rho).sqrt();
        }
    }
}

/// Manning 系数输入类型
#[derive(Debug, Clone, Copy)]
pub enum ManningCoeff<'a> {
    /// 均匀值
    Uniform(f64),
    /// 按单元变化的数组
    Array(&'a [f64]),
}

impl<'a> ManningCoeff<'a> {
    /// 获取指定索引处的 Manning 系数
    #[inline]
    pub fn get(&self, index: usize) -> f64 {
        match self {
            ManningCoeff::Uniform(n) => *n,
            ManningCoeff::Array(arr) => arr.get(index).copied().unwrap_or(0.03),
        }
    }
}

impl From<f64> for ManningCoeff<'_> {
    fn from(n: f64) -> Self {
        ManningCoeff::Uniform(n)
    }
}

impl<'a> From<&'a [f64]> for ManningCoeff<'a> {
    fn from(arr: &'a [f64]) -> Self {
        ManningCoeff::Array(arr)
    }
}

/// Shields 参数计算
///
/// θ = τ_b / ((ρ_s - ρ_w) × g × d)
///
/// # 参数
///
/// - `tau_b`: 床面剪切应力 [Pa]
/// - `rho_s`: 泥沙密度 [kg/m³]
/// - `rho_w`: 水密度 [kg/m³]
/// - `g`: 重力加速度 [m/s²]
/// - `d50`: 中值粒径 [m]
#[inline]
pub fn shields_parameter(tau_b: f64, rho_s: f64, rho_w: f64, g: f64, d50: f64) -> f64 {
    let denominator = (rho_s - rho_w) * g * d50;
    if denominator > 1e-12 {
        tau_b / denominator
    } else {
        0.0
    }
}

/// 临界 Shields 参数（经验公式）
///
/// 使用 Soulsby-Whitehouse (1997) 公式
///
/// θ_cr = 0.30 / (1 + 1.2 D_*) + 0.055 (1 - exp(-0.020 D_*))
///
/// 其中 D_* = d × [(s-1)g/ν²]^(1/3) 是无量纲粒径
pub fn critical_shields(d50: f64, relative_density: f64, g: f64, nu: f64) -> f64 {
    // 无量纲粒径
    let d_star = d50 * ((relative_density - 1.0) * g / (nu * nu)).powf(1.0 / 3.0);

    // Soulsby-Whitehouse 公式
    0.30 / (1.0 + 1.2 * d_star) + 0.055 * (1.0 - (-0.020 * d_star).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manning_shear_stress() {
        let calc = ShearStressCalculator::default();
        
        // 典型河流条件：h=2m, u=1m/s, v=0, n=0.03
        let stress = calc.manning(2.0, 1.0, 0.0, 0.03);
        
        // τ = ρ g n² |V|² / h^(1/3)
        // τ = 1000 × 9.81 × 0.0009 × 1 / 2^(1/3)
        // τ ≈ 7.0 Pa
        assert!(stress.magnitude > 5.0 && stress.magnitude < 10.0);
        assert!(stress.tau_x > 0.0);
        assert!(stress.tau_y.abs() < 1e-10);
        assert!(stress.u_star > 0.0);
    }

    #[test]
    fn test_chezy_shear_stress() {
        let calc = ShearStressCalculator::default();
        
        // C = 50
        let stress = calc.chezy(2.0, 1.0, 0.0, 50.0);
        
        // τ = ρ g |V|² / C²
        // τ = 1000 × 9.81 × 1 / 2500
        // τ ≈ 3.9 Pa
        assert!(stress.magnitude > 3.0 && stress.magnitude < 5.0);
    }

    #[test]
    fn test_dry_cell() {
        let calc = ShearStressCalculator::default();
        
        let stress = calc.manning(0.001, 1.0, 0.0, 0.03);
        assert!(stress.magnitude < 1e-10);
    }

    #[test]
    fn test_shields_parameter() {
        // 砂粒：d50=0.5mm, τ=5Pa
        let theta = shields_parameter(5.0, 2650.0, 1000.0, 9.81, 0.0005);
        
        // θ = 5 / (1650 × 9.81 × 0.0005) ≈ 0.62
        assert!(theta > 0.5 && theta < 0.7);
    }

    #[test]
    fn test_critical_shields() {
        // 中等砂粒
        let theta_cr = critical_shields(0.0005, 2.65, 9.81, 1e-6);
        
        // 典型值应在 0.03-0.06 范围
        assert!(theta_cr > 0.02 && theta_cr < 0.10);
    }

    #[test]
    fn test_batch_calculation() {
        let calc = ShearStressCalculator::default();
        
        let h = vec![2.0, 1.5, 1.0, 0.5];
        let u = vec![1.0, 1.5, 2.0, 0.5];
        let v = vec![0.0, 0.5, 0.0, 0.0];
        let mut tau = vec![0.0; 4];
        
        calc.compute_manning_batch(&h, &u, &v, 0.03.into(), &mut tau, None, None);
        
        // 所有值应为正
        for t in &tau {
            assert!(*t > 0.0);
        }
    }
}
