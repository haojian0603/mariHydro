// src-tauri/src/marihydro/physics/sources/baroclinic.rs
//! 斜压梯度力（密度驱动流）
//!
//! 实现盐度/温度分层引起的密度变化和斜压梯度力。
//! 适用于河口、近岸和分层水体模拟。

use crate::marihydro::core::traits::mesh::MeshAccess;
use crate::marihydro::core::types::{CellIndex, NumericalParams};
use glam::DVec2;

/// 状态方程类型
#[derive(Debug, Clone, Copy)]
pub enum EquationOfState {
    /// 线性状态方程
    /// ρ = ρ₀ * (1 + βs*S + βt*(T-T₀))
    Linear {
        /// 参考密度 [kg/m³]
        rho_0: f64,
        /// 盐度膨胀系数 [1/psu]
        beta_s: f64,
        /// 热膨胀系数 [1/°C]
        beta_t: f64,
        /// 参考温度 [°C]
        t_0: f64,
    },
    /// UNESCO (1981) 海水状态方程
    Unesco1981,
    /// TEOS-10 标准海水状态方程（简化版）
    Teos10Simplified,
}

impl Default for EquationOfState {
    fn default() -> Self {
        Self::Linear {
            rho_0: 1000.0,
            beta_s: 7.8e-4,  // 典型海水值
            beta_t: -2.0e-4, // 典型海水值（负值表示温度升高密度降低）
            t_0: 15.0,
        }
    }
}

impl EquationOfState {
    /// 创建线性状态方程（仅盐度）
    pub fn linear_salinity(rho_0: f64, beta_s: f64) -> Self {
        Self::Linear {
            rho_0,
            beta_s,
            beta_t: 0.0,
            t_0: 0.0,
        }
    }

    /// 创建线性状态方程（仅温度）
    pub fn linear_temperature(rho_0: f64, beta_t: f64, t_0: f64) -> Self {
        Self::Linear {
            rho_0,
            beta_s: 0.0,
            beta_t,
            t_0,
        }
    }

    /// 计算密度
    pub fn density(&self, salinity: f64, temperature: f64) -> f64 {
        match *self {
            Self::Linear { rho_0, beta_s, beta_t, t_0 } => {
                rho_0 * (1.0 + beta_s * salinity + beta_t * (temperature - t_0))
            }
            Self::Unesco1981 => {
                self.unesco_density(salinity, temperature, 0.0)
            }
            Self::Teos10Simplified => {
                self.teos10_simplified_density(salinity, temperature)
            }
        }
    }

    /// 计算密度对盐度的偏导数 ∂ρ/∂S
    pub fn drho_ds(&self, salinity: f64, temperature: f64) -> f64 {
        match *self {
            Self::Linear { rho_0, beta_s, .. } => {
                rho_0 * beta_s
            }
            Self::Unesco1981 | Self::Teos10Simplified => {
                // 数值微分
                let ds = 0.01;
                let rho_p = self.density(salinity + ds, temperature);
                let rho_m = self.density(salinity - ds, temperature);
                (rho_p - rho_m) / (2.0 * ds)
            }
        }
    }

    /// 计算密度对温度的偏导数 ∂ρ/∂T
    pub fn drho_dt(&self, salinity: f64, temperature: f64) -> f64 {
        match *self {
            Self::Linear { rho_0, beta_t, .. } => {
                rho_0 * beta_t
            }
            Self::Unesco1981 | Self::Teos10Simplified => {
                // 数值微分
                let dt = 0.01;
                let rho_p = self.density(salinity, temperature + dt);
                let rho_m = self.density(salinity, temperature - dt);
                (rho_p - rho_m) / (2.0 * dt)
            }
        }
    }

    /// UNESCO (1981) 海水状态方程
    /// 参考：Millero & Poisson (1981)
    fn unesco_density(&self, s: f64, t: f64, _p: f64) -> f64 {
        // 纯水密度
        let rho_w = 999.842594
            + 6.793952e-2 * t
            - 9.095290e-3 * t.powi(2)
            + 1.001685e-4 * t.powi(3)
            - 1.120083e-6 * t.powi(4)
            + 6.536336e-9 * t.powi(5);

        // 盐度项
        let a = 8.24493e-1
            - 4.0899e-3 * t
            + 7.6438e-5 * t.powi(2)
            - 8.2467e-7 * t.powi(3)
            + 5.3875e-9 * t.powi(4);

        let b = -5.72466e-3
            + 1.0227e-4 * t
            - 1.6546e-6 * t.powi(2);

        let c = 4.8314e-4;

        rho_w + a * s + b * s.powf(1.5) + c * s.powi(2)
    }

    /// TEOS-10 简化版密度公式
    fn teos10_simplified_density(&self, sa: f64, ct: f64) -> f64 {
        // 简化的多项式拟合
        // SA: 绝对盐度 [g/kg], CT: 保守温度 [°C]
        let rho_0 = 1028.0;

        // 温度效应
        let t_term = -0.20 * (ct - 10.0) - 0.003 * (ct - 10.0).powi(2);

        // 盐度效应
        let s_term = 0.78 * (sa - 35.0);

        rho_0 + t_term + s_term
    }
}

/// 密度场状态
#[derive(Debug, Clone)]
pub struct DensityField {
    /// 密度 [kg/m³]
    pub rho: Vec<f64>,
    /// 盐度 [psu 或 g/kg]
    pub salinity: Vec<f64>,
    /// 温度 [°C]
    pub temperature: Vec<f64>,
}

impl DensityField {
    /// 创建新的密度场
    pub fn new(n_cells: usize) -> Self {
        Self {
            rho: vec![1000.0; n_cells],
            salinity: vec![0.0; n_cells],
            temperature: vec![15.0; n_cells],
        }
    }

    /// 从盐度场创建
    pub fn from_salinity(salinity: Vec<f64>, eos: &EquationOfState) -> Self {
        let n = salinity.len();
        let mut field = Self {
            rho: vec![0.0; n],
            salinity,
            temperature: vec![15.0; n],
        };
        field.update_density(eos);
        field
    }

    /// 从温度场创建
    pub fn from_temperature(temperature: Vec<f64>, eos: &EquationOfState) -> Self {
        let n = temperature.len();
        let mut field = Self {
            rho: vec![0.0; n],
            salinity: vec![35.0; n],
            temperature,
        };
        field.update_density(eos);
        field
    }

    /// 更新密度场
    pub fn update_density(&mut self, eos: &EquationOfState) {
        for i in 0..self.rho.len() {
            self.rho[i] = eos.density(self.salinity[i], self.temperature[i]);
        }
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.rho.resize(n_cells, 1000.0);
        self.salinity.resize(n_cells, 0.0);
        self.temperature.resize(n_cells, 15.0);
    }
}

/// 斜压梯度力计算配置
#[derive(Debug, Clone)]
pub struct BaroclinicConfig {
    /// 状态方程
    pub eos: EquationOfState,
    /// 参考密度 [kg/m³]
    pub rho_ref: f64,
    /// 重力加速度 [m/s²]
    pub g: f64,
    /// 是否使用深度平均斜压梯度
    pub depth_averaged: bool,
}

impl Default for BaroclinicConfig {
    fn default() -> Self {
        Self {
            eos: EquationOfState::default(),
            rho_ref: 1000.0,
            g: 9.81,
            depth_averaged: true,
        }
    }
}

/// 斜压梯度力求解器
pub struct BaroclinicSolver {
    /// 配置
    config: BaroclinicConfig,
    /// 密度梯度
    drho_dx: Vec<f64>,
    drho_dy: Vec<f64>,
    /// 斜压加速度
    acc_x: Vec<f64>,
    acc_y: Vec<f64>,
}

impl BaroclinicSolver {
    /// 创建新的求解器
    pub fn new(config: BaroclinicConfig, n_cells: usize) -> Self {
        Self {
            config,
            drho_dx: vec![0.0; n_cells],
            drho_dy: vec![0.0; n_cells],
            acc_x: vec![0.0; n_cells],
            acc_y: vec![0.0; n_cells],
        }
    }

    /// 使用默认配置创建
    pub fn with_default(n_cells: usize) -> Self {
        Self::new(BaroclinicConfig::default(), n_cells)
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.drho_dx.resize(n_cells, 0.0);
        self.drho_dy.resize(n_cells, 0.0);
        self.acc_x.resize(n_cells, 0.0);
        self.acc_y.resize(n_cells, 0.0);
    }

    /// 计算密度梯度
    fn compute_density_gradient<M: MeshAccess>(
        &mut self,
        mesh: &M,
        density_field: &DensityField,
    ) {
        self.drho_dx.fill(0.0);
        self.drho_dy.fill(0.0);

        for i in 0..mesh.n_cells() {
            let cell = CellIndex(i);
            let area = mesh.cell_area(cell);
            if area < 1e-14 {
                continue;
            }

            let mut grad = DVec2::ZERO;

            for &face in mesh.cell_faces(cell) {
                let owner = mesh.face_owner(face);
                let neighbor = mesh.face_neighbor(face);
                let normal = mesh.face_normal(face);
                let length = mesh.face_length(face);

                let sign = if i == owner.0 { 1.0 } else { -1.0 };
                let ds = normal * length * sign;

                let rho_face = if !neighbor.is_valid() {
                    density_field.rho[i]
                } else {
                    let o = if i == owner.0 { neighbor.0 } else { owner.0 };
                    (density_field.rho[i] + density_field.rho[o]) * 0.5
                };

                grad += ds * rho_face;
            }

            self.drho_dx[i] = grad.x / area;
            self.drho_dy[i] = grad.y / area;
        }
    }

    /// 计算斜压梯度力
    /// 
    /// 深度平均形式：
    /// Fx = -g*h/(2*ρ₀) * ∂ρ/∂x
    /// Fy = -g*h/(2*ρ₀) * ∂ρ/∂y
    pub fn compute<M: MeshAccess>(
        &mut self,
        mesh: &M,
        depths: &[f64],
        density_field: &DensityField,
        params: &NumericalParams,
    ) {
        let n = mesh.n_cells();
        if n != self.drho_dx.len() {
            self.resize(n);
        }

        // 计算密度梯度
        self.compute_density_gradient(mesh, density_field);

        // 计算斜压加速度
        let g = self.config.g;
        let rho_ref = self.config.rho_ref;

        for i in 0..n {
            let h = depths[i];
            if params.is_dry(h) {
                self.acc_x[i] = 0.0;
                self.acc_y[i] = 0.0;
                continue;
            }

            if self.config.depth_averaged {
                // 深度平均斜压梯度（2D浅水）
                // 假设密度线性分布，积分后得到 h/2 因子
                let coeff = -g * h / (2.0 * rho_ref);
                self.acc_x[i] = coeff * self.drho_dx[i];
                self.acc_y[i] = coeff * self.drho_dy[i];
            } else {
                // 简单斜压梯度（假设均匀分布）
                let coeff = -g * h / rho_ref;
                self.acc_x[i] = coeff * self.drho_dx[i];
                self.acc_y[i] = coeff * self.drho_dy[i];
            }
        }
    }

    /// 应用斜压梯度力到动量方程
    pub fn apply_to_momentum(
        &self,
        depths: &[f64],
        dt: f64,
        acc_hu: &mut [f64],
        acc_hv: &mut [f64],
    ) {
        for i in 0..self.acc_x.len().min(acc_hu.len()) {
            let h = depths[i];
            if h < 1e-6 {
                continue;
            }

            // Δ(hu) = h * ax * dt
            acc_hu[i] += h * self.acc_x[i] * dt;
            acc_hv[i] += h * self.acc_y[i] * dt;
        }
    }

    /// 获取 x 方向加速度
    pub fn acceleration_x(&self) -> &[f64] {
        &self.acc_x
    }

    /// 获取 y 方向加速度
    pub fn acceleration_y(&self) -> &[f64] {
        &self.acc_y
    }

    /// 获取密度梯度
    pub fn density_gradient(&self) -> (&[f64], &[f64]) {
        (&self.drho_dx, &self.drho_dy)
    }

    /// 获取配置
    pub fn config(&self) -> &BaroclinicConfig {
        &self.config
    }

    /// 设置状态方程
    pub fn set_eos(&mut self, eos: EquationOfState) {
        self.config.eos = eos;
    }
}

/// 盐水入侵模拟辅助函数
pub mod estuary {
    use super::*;

    /// 计算盐度扩散系数（随盐度变化）
    /// 
    /// 参考：Fischer et al. (1979)
    pub fn salinity_diffusivity(
        velocity_mag: f64,
        depth: f64,
        width: f64,
    ) -> f64 {
        // Elder 横向扩散公式：Dy = 0.6 * h * u*
        // 纵向扩散：Dx = 5.9 * h * u*
        // u* = sqrt(g * h * S) 约 = 0.1 * U

        let u_star = 0.1 * velocity_mag;
        let k_longitudinal = 5.9 * depth * u_star;
        let k_lateral = 0.6 * depth * u_star;

        // 加入河宽效应
        let aspect_ratio = width / depth;
        let mixing_coeff = if aspect_ratio > 20.0 {
            k_longitudinal * (1.0 + 0.01 * aspect_ratio)
        } else {
            k_longitudinal
        };

        mixing_coeff.max(1e-6)
    }

    /// 估算盐度锋面位置（一维稳态解）
    /// 
    /// 返回盐度从海洋到淡水的距离估计
    pub fn estimate_salt_intrusion_length(
        river_discharge: f64,
        channel_width: f64,
        channel_depth: f64,
        tidal_velocity: f64,
        ocean_salinity: f64,
    ) -> f64 {
        // 简化的 Savenije 公式
        // L = K * H * U_tidal / (Q/A)
        // K 经验常数约 20-50

        let k = 30.0;
        let area = channel_width * channel_depth;
        let u_river = river_discharge / area;

        if u_river < 1e-6 {
            return f64::MAX;
        }

        k * channel_depth * tidal_velocity / u_river
    }

    /// 计算理查森数（判断分层稳定性）
    /// 
    /// Ri = g * (∂ρ/∂z) / (ρ * (∂u/∂z)²)
    /// Ri > 0.25 表示稳定分层
    pub fn richardson_number(
        drho_dz: f64,
        du_dz: f64,
        rho: f64,
        g: f64,
    ) -> f64 {
        if du_dz.abs() < 1e-10 {
            return f64::MAX;
        }

        -g * drho_dz / (rho * du_dz.powi(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_eos() {
        let eos = EquationOfState::default();
        
        // 淡水
        let rho_fresh = eos.density(0.0, 15.0);
        assert!((rho_fresh - 1000.0).abs() < 1.0);

        // 海水
        let rho_sea = eos.density(35.0, 15.0);
        assert!(rho_sea > rho_fresh);
        assert!((rho_sea - 1027.0).abs() < 3.0);
    }

    #[test]
    fn test_unesco_eos() {
        let eos = EquationOfState::Unesco1981;

        // 标准海水 S=35, T=15°C
        let rho = eos.density(35.0, 15.0);
        assert!((rho - 1026.0).abs() < 2.0);

        // 淡水 T=4°C（最大密度点附近）
        let rho_4c = eos.density(0.0, 4.0);
        assert!((rho_4c - 1000.0).abs() < 0.5);
    }

    #[test]
    fn test_density_derivatives() {
        let eos = EquationOfState::Linear {
            rho_0: 1000.0,
            beta_s: 7.8e-4,
            beta_t: -2.0e-4,
            t_0: 15.0,
        };

        let drho_ds = eos.drho_ds(35.0, 15.0);
        assert!((drho_ds - 0.78).abs() < 0.01);

        let drho_dt = eos.drho_dt(35.0, 15.0);
        assert!((drho_dt - (-0.20)).abs() < 0.01);
    }

    #[test]
    fn test_density_field() {
        let eos = EquationOfState::default();
        let field = DensityField::from_salinity(vec![0.0, 17.5, 35.0], &eos);

        assert!(field.rho[0] < field.rho[1]);
        assert!(field.rho[1] < field.rho[2]);
    }

    #[test]
    fn test_salt_intrusion_estimate() {
        let length = estuary::estimate_salt_intrusion_length(
            100.0,  // 100 m³/s
            200.0,  // 200 m width
            5.0,    // 5 m depth
            0.5,    // 0.5 m/s tidal velocity
            35.0,   // ocean salinity
        );

        // 应该返回合理的入侵长度
        assert!(length > 0.0);
        assert!(length < 1e6);
    }

    #[test]
    fn test_richardson_number() {
        // 稳定分层
        let ri = estuary::richardson_number(-0.5, 0.1, 1025.0, 9.81);
        assert!(ri > 0.25);

        // 不稳定（密度倒置）
        let ri_unstable = estuary::richardson_number(0.5, 0.1, 1025.0, 9.81);
        assert!(ri_unstable < 0.0);
    }
}
