// src-tauri/src/marihydro/physics/waves/radiation_stress.rs
//! 辐射应力计算
//!
//! 实现波浪辐射应力及其对水流的驱动力计算。
//! 参考：Longuet-Higgins & Stewart (1964), Dean & Dalrymple (1991)

use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use glam::DVec2;
use std::f64::consts::PI;

/// 波浪参数
#[derive(Debug, Clone, Copy)]
pub struct WaveParameters {
    /// 有效波高 Hs [m]
    pub height: f64,
    /// 波周期 T [s]
    pub period: f64,
    /// 波向 θ [rad]，从x正向逆时针
    pub direction: f64,
}

impl WaveParameters {
    /// 创建新的波浪参数
    pub fn new(height: f64, period: f64, direction_deg: f64) -> Self {
        Self {
            height,
            period,
            direction: direction_deg.to_radians(),
        }
    }

    /// 计算角频率 ω = 2π/T
    pub fn angular_frequency(&self) -> f64 {
        2.0 * PI / self.period
    }

    /// 计算深水波长 L0 = g*T²/(2π)
    pub fn deep_water_wavelength(&self, g: f64) -> f64 {
        g * self.period.powi(2) / (2.0 * PI)
    }

    /// 计算波浪能量 E = ρgH²/8
    pub fn energy(&self, rho: f64, g: f64) -> f64 {
        rho * g * self.height.powi(2) / 8.0
    }

    /// 波向单位向量
    pub fn direction_vector(&self) -> DVec2 {
        DVec2::new(self.direction.cos(), self.direction.sin())
    }
}

/// 波浪场状态
#[derive(Debug, Clone)]
pub struct WaveField {
    /// 各单元波高 [m]
    pub height: Vec<f64>,
    /// 各单元波周期 [s]
    pub period: Vec<f64>,
    /// 各单元波向 [rad]
    pub direction: Vec<f64>,
    /// 各单元波长 [m]
    pub wavelength: Vec<f64>,
    /// 各单元波数 k [1/m]
    pub wavenumber: Vec<f64>,
    /// 群速度系数 n = Cg/C
    pub group_factor: Vec<f64>,
    /// 波浪能量 [J/m²]
    pub energy: Vec<f64>,
}

impl WaveField {
    /// 创建新的波浪场
    pub fn new(n_cells: usize) -> Self {
        Self {
            height: vec![0.0; n_cells],
            period: vec![0.0; n_cells],
            direction: vec![0.0; n_cells],
            wavelength: vec![0.0; n_cells],
            wavenumber: vec![0.0; n_cells],
            group_factor: vec![0.5; n_cells],
            energy: vec![0.0; n_cells],
        }
    }

    /// 从均匀波浪参数初始化
    pub fn from_uniform(n_cells: usize, params: WaveParameters, depths: &[f64], g: f64, rho: f64) -> Self {
        let mut field = Self::new(n_cells);
        field.set_uniform(params, depths, g, rho);
        field
    }

    /// 设置均匀波浪场
    pub fn set_uniform(&mut self, params: WaveParameters, depths: &[f64], g: f64, rho: f64) {
        for i in 0..self.height.len().min(depths.len()) {
            self.height[i] = params.height;
            self.period[i] = params.period;
            self.direction[i] = params.direction;

            let h = depths[i].max(0.01);
            let (k, n) = compute_wavenumber_and_n(params.period, h, g);

            self.wavenumber[i] = k;
            self.wavelength[i] = 2.0 * PI / k;
            self.group_factor[i] = n;
            self.energy[i] = params.energy(rho, g);
        }
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.height.resize(n_cells, 0.0);
        self.period.resize(n_cells, 0.0);
        self.direction.resize(n_cells, 0.0);
        self.wavelength.resize(n_cells, 0.0);
        self.wavenumber.resize(n_cells, 0.0);
        self.group_factor.resize(n_cells, 0.5);
        self.energy.resize(n_cells, 0.0);
    }
}

/// 使用迭代法计算波数 k 和群速度系数 n
/// 
/// 色散关系：ω² = gk*tanh(kh)
fn compute_wavenumber_and_n(period: f64, depth: f64, g: f64) -> (f64, f64) {
    let omega = 2.0 * PI / period;
    let omega2 = omega * omega;

    // 深水波数初始估计
    let k0 = omega2 / g;

    // Newton-Raphson 迭代求解
    let mut k = k0;
    for _ in 0..20 {
        let kh = k * depth;
        let tanh_kh = kh.tanh();
        let f = g * k * tanh_kh - omega2;
        let df = g * (tanh_kh + k * depth * (1.0 - tanh_kh * tanh_kh));

        let dk = f / df;
        k -= dk;

        if dk.abs() < 1e-10 * k {
            break;
        }
    }

    // 群速度系数 n = 0.5 * (1 + 2kh/sinh(2kh))
    let kh = k * depth;
    let n = if kh > 10.0 {
        0.5 // 深水极限
    } else if kh < 0.01 {
        1.0 // 浅水极限
    } else {
        0.5 * (1.0 + 2.0 * kh / (2.0 * kh).sinh())
    };

    (k, n)
}

/// 辐射应力张量
#[derive(Debug, Clone, Copy, Default)]
pub struct RadiationStressTensor {
    /// Sxx 分量
    pub sxx: f64,
    /// Syy 分量
    pub syy: f64,
    /// Sxy 分量
    pub sxy: f64,
}

impl RadiationStressTensor {
    /// 计算辐射应力张量
    /// 
    /// Sxx = E * (n*cos²θ + n - 0.5)
    /// Syy = E * (n*sin²θ + n - 0.5)
    /// Sxy = E * n * sinθ * cosθ
    pub fn compute(energy: f64, n: f64, direction: f64) -> Self {
        let cos_theta = direction.cos();
        let sin_theta = direction.sin();
        let cos2 = cos_theta * cos_theta;
        let sin2 = sin_theta * sin_theta;

        Self {
            sxx: energy * (n * cos2 + n - 0.5),
            syy: energy * (n * sin2 + n - 0.5),
            sxy: energy * n * sin_theta * cos_theta,
        }
    }
}

/// 辐射应力计算器
pub struct RadiationStress {
    /// 水密度 [kg/m³]
    rho: f64,
    /// 重力加速度 [m/s²]
    g: f64,
    /// 辐射应力场
    stress: Vec<RadiationStressTensor>,
    /// 辐射应力梯度（驱动力）
    force_x: Vec<f64>,
    force_y: Vec<f64>,
}

impl RadiationStress {
    /// 创建新的辐射应力计算器
    pub fn new(n_cells: usize, rho: f64, g: f64) -> Self {
        Self {
            rho,
            g,
            stress: vec![RadiationStressTensor::default(); n_cells],
            force_x: vec![0.0; n_cells],
            force_y: vec![0.0; n_cells],
        }
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.stress.resize(n_cells, RadiationStressTensor::default());
        self.force_x.resize(n_cells, 0.0);
        self.force_y.resize(n_cells, 0.0);
    }

    /// 计算辐射应力场
    pub fn compute_stress(&mut self, wave_field: &WaveField) {
        for i in 0..self.stress.len().min(wave_field.energy.len()) {
            self.stress[i] = RadiationStressTensor::compute(
                wave_field.energy[i],
                wave_field.group_factor[i],
                wave_field.direction[i],
            );
        }
    }

    /// 计算辐射应力梯度（波浪驱动力）
    /// 
    /// Fx = -1/(ρh) * (∂Sxx/∂x + ∂Sxy/∂y)
    /// Fy = -1/(ρh) * (∂Sxy/∂x + ∂Syy/∂y)
    pub fn compute_gradient<M: MeshAccess>(
        &mut self,
        mesh: &M,
        depths: &[f64],
        params: &NumericalParams,
    ) {
        self.force_x.fill(0.0);
        self.force_y.fill(0.0);

        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            let h = depths[i];

            if area < 1e-14 || params.is_dry(h) {
                continue;
            }

            // Green-Gauss 梯度计算
            let mut grad_sxx = DVec2::ZERO;
            let mut grad_syy = DVec2::ZERO;
            let mut grad_sxy = DVec2::ZERO;

            for &face in mesh.cell_faces(cell) {
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);

                let sign = if i == owner.0 { 1.0 } else { -1.0 };
                let ds = normal * length * sign;

                let stress_face = if !neighbor.is_valid() {
                    self.stress[i]
                } else {
                    let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                    RadiationStressTensor {
                        sxx: (self.stress[i].sxx + self.stress[o].sxx) * 0.5,
                        syy: (self.stress[i].syy + self.stress[o].syy) * 0.5,
                        sxy: (self.stress[i].sxy + self.stress[o].sxy) * 0.5,
                    }
                };

                grad_sxx += ds * stress_face.sxx;
                grad_syy += ds * stress_face.syy;
                grad_sxy += ds * stress_face.sxy;
            }

            let dsxx_dx = grad_sxx.x / area;
            let dsyy_dy = grad_syy.y / area;
            let dsxy_dx = grad_sxy.x / area;
            let dsxy_dy = grad_sxy.y / area;

            // 波浪驱动力
            let inv_rho_h = 1.0 / (self.rho * h);
            self.force_x[i] = -inv_rho_h * (dsxx_dx + dsxy_dy);
            self.force_y[i] = -inv_rho_h * (dsxy_dx + dsyy_dy);
        }
    }

    /// 应用辐射应力到动量方程
    pub fn apply_to_momentum(
        &self,
        depths: &[f64],
        dt: f64,
        acc_hu: &mut [f64],
        acc_hv: &mut [f64],
    ) {
        for i in 0..self.force_x.len().min(acc_hu.len()) {
            let h = depths[i];
            if h < 1e-6 {
                continue;
            }

            // Δ(hu) = h * Fx * dt
            acc_hu[i] += h * self.force_x[i] * dt;
            acc_hv[i] += h * self.force_y[i] * dt;
        }
    }

    /// 获取辐射应力场
    pub fn stress(&self) -> &[RadiationStressTensor] {
        &self.stress
    }

    /// 获取 x 方向波浪驱动力
    pub fn force_x(&self) -> &[f64] {
        &self.force_x
    }

    /// 获取 y 方向波浪驱动力
    pub fn force_y(&self) -> &[f64] {
        &self.force_y
    }
}

/// 波浪源项（用于源项耦合）
pub struct WaveSource {
    /// 辐射应力计算器
    radiation_stress: RadiationStress,
    /// 波浪场
    wave_field: WaveField,
}

impl WaveSource {
    /// 创建新的波浪源项
    pub fn new(n_cells: usize, rho: f64, g: f64) -> Self {
        Self {
            radiation_stress: RadiationStress::new(n_cells, rho, g),
            wave_field: WaveField::new(n_cells),
        }
    }

    /// 设置波浪场
    pub fn set_wave_field(&mut self, wave_field: WaveField) {
        self.wave_field = wave_field;
        if self.wave_field.height.len() != self.radiation_stress.stress.len() {
            self.radiation_stress.resize(self.wave_field.height.len());
        }
    }

    /// 更新均匀波浪场
    pub fn update_uniform(
        &mut self,
        params: WaveParameters,
        depths: &[f64],
        g: f64,
        rho: f64,
    ) {
        self.wave_field.set_uniform(params, depths, g, rho);
    }

    /// 计算波浪驱动力
    pub fn compute<M: MeshAccess>(
        &mut self,
        mesh: &M,
        depths: &[f64],
        params: &NumericalParams,
    ) {
        self.radiation_stress.compute_stress(&self.wave_field);
        self.radiation_stress.compute_gradient(mesh, depths, params);
    }

    /// 应用到动量方程
    pub fn apply(
        &self,
        depths: &[f64],
        dt: f64,
        acc_hu: &mut [f64],
        acc_hv: &mut [f64],
    ) {
        self.radiation_stress.apply_to_momentum(depths, dt, acc_hu, acc_hv);
    }

    /// 获取波浪场引用
    pub fn wave_field(&self) -> &WaveField {
        &self.wave_field
    }

    /// 获取辐射应力引用
    pub fn radiation_stress(&self) -> &RadiationStress {
        &self.radiation_stress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wave_parameters() {
        let params = WaveParameters::new(1.0, 8.0, 45.0);
        assert!((params.direction - 0.7854).abs() < 0.01);
        assert!((params.angular_frequency() - 0.785).abs() < 0.01);
    }

    #[test]
    fn test_wavenumber_deep_water() {
        // 深水条件：T=10s, h=100m
        let (k, n) = compute_wavenumber_and_n(10.0, 100.0, 9.81);
        // 深水波数 k = ω²/g = (2π/10)²/9.81 ≈ 0.0403
        assert!((k - 0.0403).abs() < 0.01);
        // 深水群速度系数 n ≈ 0.5
        assert!((n - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_wavenumber_shallow_water() {
        // 浅水条件：T=10s, h=1m
        let (k, n) = compute_wavenumber_and_n(10.0, 1.0, 9.81);
        // 浅水群速度系数 n ≈ 1.0
        assert!(n > 0.9);
    }

    #[test]
    fn test_radiation_stress_tensor() {
        // 能量 = 1000 J/m², n = 0.5（深水），θ = 0（顺x方向）
        let s = RadiationStressTensor::compute(1000.0, 0.5, 0.0);
        // Sxx = 1000 * (0.5*1 + 0.5 - 0.5) = 500
        assert!((s.sxx - 500.0).abs() < 1.0);
        // Syy = 1000 * (0.5*0 + 0.5 - 0.5) = 0
        assert!(s.syy.abs() < 1.0);
        // Sxy = 0
        assert!(s.sxy.abs() < 1.0);
    }

    #[test]
    fn test_wave_field_creation() {
        let field = WaveField::new(100);
        assert_eq!(field.height.len(), 100);
        assert_eq!(field.group_factor[0], 0.5);
    }
}
