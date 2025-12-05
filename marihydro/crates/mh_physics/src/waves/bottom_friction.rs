// crates/mh_physics/src/waves/bottom_friction.rs

//! 波浪底摩擦计算
//! 
//! 实现多种波浪底摩擦模型，包括 Jonswap, Madsen, Nielsen 等。

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// 重力加速度
const G: f64 = 9.81;

/// 波浪底摩擦模型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveBottomFrictionModel {
    /// JONSWAP 经验公式
    Jonswap,
    /// Madsen (1988) 公式
    Madsen,
    /// Nielsen (1992) 公式
    Nielsen,
    /// 常数摩擦系数
    Constant,
}

impl Default for WaveBottomFrictionModel {
    fn default() -> Self {
        Self::Jonswap
    }
}

/// 波浪轨道速度计算器
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WaveOrbitalVelocity {
    /// 最大水平轨道速度 [m/s]
    pub u_max: f64,
    /// 轨道位移幅值 [m]
    pub amplitude: f64,
    /// 轨道周期 [s]
    pub period: f64,
}

impl WaveOrbitalVelocity {
    /// 从波浪参数计算底部轨道速度
    /// 
    /// u_max = πH / (T sinh(kh))
    /// a = H / (2 sinh(kh))
    pub fn compute(
        height: f64,
        period: f64,
        wavenumber: f64,
        depth: f64,
    ) -> Self {
        let h = depth.max(0.1);
        let kh = wavenumber * h;
        let sinh_kh = kh.sinh();
        
        if sinh_kh < 1e-10 {
            // 浅水极限
            let c = (G * h).sqrt();
            return Self {
                u_max: height / (2.0 * h) * c,
                amplitude: height / 2.0,
                period,
            };
        }
        
        Self {
            u_max: PI * height / (period * sinh_kh),
            amplitude: height / (2.0 * sinh_kh),
            period,
        }
    }

    /// 获取轨道速度随时间的变化
    pub fn velocity_at_phase(&self, phase: f64) -> f64 {
        self.u_max * phase.cos()
    }

    /// 获取轨道位移随时间的变化
    pub fn displacement_at_phase(&self, phase: f64) -> f64 {
        self.amplitude * phase.sin()
    }
}

/// 波浪底摩擦计算配置
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WaveBottomFrictionConfig {
    /// 摩擦模型
    pub model: WaveBottomFrictionModel,
    /// 床面糙率高度 [m]
    pub roughness_height: f64,
    /// 常数摩擦系数（用于 Constant 模型）
    pub friction_coefficient: f64,
}

impl Default for WaveBottomFrictionConfig {
    fn default() -> Self {
        Self {
            model: WaveBottomFrictionModel::Jonswap,
            roughness_height: 0.05,
            friction_coefficient: 0.01,
        }
    }
}

impl WaveBottomFrictionConfig {
    /// 创建 JONSWAP 模型配置
    pub fn jonswap() -> Self {
        Self {
            model: WaveBottomFrictionModel::Jonswap,
            ..Default::default()
        }
    }

    /// 创建 Madsen 模型配置
    pub fn madsen(roughness_height: f64) -> Self {
        Self {
            model: WaveBottomFrictionModel::Madsen,
            roughness_height,
            ..Default::default()
        }
    }

    /// 创建 Nielsen 模型配置
    pub fn nielsen(roughness_height: f64) -> Self {
        Self {
            model: WaveBottomFrictionModel::Nielsen,
            roughness_height,
            ..Default::default()
        }
    }

    /// 创建常数摩擦系数配置
    pub fn constant(friction_coefficient: f64) -> Self {
        Self {
            model: WaveBottomFrictionModel::Constant,
            friction_coefficient,
            ..Default::default()
        }
    }
}

/// 波浪底摩擦计算器
pub struct WaveBottomFriction {
    /// 配置
    config: WaveBottomFrictionConfig,
    /// 摩擦系数
    fw: Vec<f64>,
    /// 床面剪切应力振幅 [Pa]
    tau_wave: Vec<f64>,
    /// 能量耗散率 [W/m²]
    dissipation: Vec<f64>,
}

impl WaveBottomFriction {
    /// 创建新的计算器
    pub fn new(n_cells: usize, config: WaveBottomFrictionConfig) -> Self {
        Self {
            config,
            fw: vec![0.01; n_cells],
            tau_wave: vec![0.0; n_cells],
            dissipation: vec![0.0; n_cells],
        }
    }

    /// 计算波浪底摩擦
    pub fn compute_friction(
        &mut self,
        height: &[f64],
        period: &[f64],
        wavenumber: &[f64],
        depth: &[f64],
    ) {
        let n = self.fw.len()
            .min(height.len())
            .min(period.len())
            .min(wavenumber.len())
            .min(depth.len());
        
        let rho = 1025.0;

        for i in 0..n {
            let orbital = WaveOrbitalVelocity::compute(
                height[i],
                period[i],
                wavenumber[i],
                depth[i],
            );

            // 计算摩擦系数
            self.fw[i] = self.compute_friction_coefficient(orbital.amplitude, orbital.period);

            // 床面剪切应力振幅
            // τ_wave = 0.5 × ρ × fw × u_max²
            self.tau_wave[i] = 0.5 * rho * self.fw[i] * orbital.u_max * orbital.u_max;

            // 能量耗散率
            // D = (2/3π) × ρ × fw × u_max³
            self.dissipation[i] = 2.0 / (3.0 * PI) * rho * self.fw[i] 
                * orbital.u_max * orbital.u_max * orbital.u_max;
        }
    }

    /// 根据模型计算摩擦系数
    fn compute_friction_coefficient(&self, amplitude: f64, period: f64) -> f64 {
        match self.config.model {
            WaveBottomFrictionModel::Jonswap => {
                // JONSWAP 经验公式
                // fw = 0.067 for typical conditions
                0.067
            }
            WaveBottomFrictionModel::Madsen => {
                // Madsen (1988)
                // fw = exp(-5.977 + 5.213(a/ks)^(-0.194))
                let a = amplitude.max(1e-6);
                let ks = self.config.roughness_height.max(1e-6);
                let ratio = a / ks;
                
                if ratio < 1.57 {
                    0.3  // 最大值
                } else {
                    (-5.977 + 5.213 * ratio.powf(-0.194)).exp()
                }
            }
            WaveBottomFrictionModel::Nielsen => {
                // Nielsen (1992)
                // fw = exp(5.5(a/ks)^(-0.2) - 6.3)
                let a = amplitude.max(1e-6);
                let ks = self.config.roughness_height.max(1e-6);
                let ratio = a / ks;
                
                if ratio < 1.0 {
                    0.3
                } else {
                    (5.5 * ratio.powf(-0.2) - 6.3).exp()
                }
            }
            WaveBottomFrictionModel::Constant => {
                self.config.friction_coefficient
            }
        }
    }

    /// 获取摩擦系数
    pub fn friction_coefficients(&self) -> &[f64] {
        &self.fw
    }

    /// 获取波浪床面剪切应力
    pub fn wave_shear_stress(&self) -> &[f64] {
        &self.tau_wave
    }

    /// 获取能量耗散率
    pub fn energy_dissipation(&self) -> &[f64] {
        &self.dissipation
    }

    /// 获取配置
    pub fn config(&self) -> &WaveBottomFrictionConfig {
        &self.config
    }

    /// 计算波流联合剪切应力
    /// 
    /// 使用 Soulsby (1997) 的波流联合公式
    pub fn compute_combined_stress(
        tau_current: f64,
        tau_wave: f64,
        angle_between: f64,
    ) -> f64 {
        let cos_phi = angle_between.cos();
        
        // Soulsby 非线性公式
        let tau_mean = tau_current * (1.0 + 1.2 * (tau_wave / (tau_current + tau_wave + 1e-14)).powf(3.2));
        let tau_max = ((tau_mean + tau_wave * cos_phi).powi(2) 
            + (tau_wave * angle_between.sin()).powi(2)).sqrt();
        
        tau_max
    }
}

/// 波流联合底摩擦计算器
pub struct WaveCurrentInteraction {
    /// 波浪摩擦计算器
    wave_friction: WaveBottomFriction,
    /// 联合剪切应力 [Pa]
    tau_combined: Vec<f64>,
    /// 联合摩擦系数
    fc_combined: Vec<f64>,
}

impl WaveCurrentInteraction {
    /// 创建新的计算器
    pub fn new(n_cells: usize, config: WaveBottomFrictionConfig) -> Self {
        Self {
            wave_friction: WaveBottomFriction::new(n_cells, config),
            tau_combined: vec![0.0; n_cells],
            fc_combined: vec![0.0; n_cells],
        }
    }

    /// 计算波流联合剪切应力
    pub fn compute(
        &mut self,
        tau_current: &[f64],
        current_direction: &[f64],
        height: &[f64],
        period: &[f64],
        wavenumber: &[f64],
        wave_direction: &[f64],
        depth: &[f64],
    ) {
        // 先计算波浪底摩擦
        self.wave_friction.compute_friction(height, period, wavenumber, depth);
        let tau_wave = self.wave_friction.wave_shear_stress();

        let n = self.tau_combined.len()
            .min(tau_current.len())
            .min(current_direction.len())
            .min(wave_direction.len())
            .min(tau_wave.len());

        for i in 0..n {
            let angle_diff = wave_direction[i] - current_direction[i];
            self.tau_combined[i] = WaveBottomFriction::compute_combined_stress(
                tau_current[i],
                tau_wave[i],
                angle_diff,
            );
        }
    }

    /// 获取联合剪切应力
    pub fn combined_shear_stress(&self) -> &[f64] {
        &self.tau_combined
    }

    /// 获取波浪摩擦计算器
    pub fn wave_friction(&self) -> &WaveBottomFriction {
        &self.wave_friction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbital_velocity() {
        let orbital = WaveOrbitalVelocity::compute(2.0, 8.0, 0.1, 10.0);
        
        assert!(orbital.u_max > 0.0);
        assert!(orbital.amplitude > 0.0);
        assert_eq!(orbital.period, 8.0);
    }

    #[test]
    fn test_orbital_velocity_shallow() {
        let orbital = WaveOrbitalVelocity::compute(1.0, 8.0, 1.0, 0.5);
        
        assert!(orbital.u_max > 0.0);
    }

    #[test]
    fn test_wave_bottom_friction_config() {
        let config = WaveBottomFrictionConfig::jonswap();
        assert_eq!(config.model, WaveBottomFrictionModel::Jonswap);
        
        let config = WaveBottomFrictionConfig::madsen(0.03);
        assert_eq!(config.model, WaveBottomFrictionModel::Madsen);
        assert!((config.roughness_height - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_wave_bottom_friction_jonswap() {
        let config = WaveBottomFrictionConfig::jonswap();
        let mut friction = WaveBottomFriction::new(10, config);
        
        let height = vec![1.0; 10];
        let period = vec![8.0; 10];
        let wavenumber = vec![0.1; 10];
        let depth = vec![10.0; 10];
        
        friction.compute_friction(&height, &period, &wavenumber, &depth);
        
        let fw = friction.friction_coefficients();
        assert!(fw.iter().all(|&f| f > 0.0));
        
        let tau = friction.wave_shear_stress();
        assert!(tau.iter().all(|&t| t >= 0.0));
    }

    #[test]
    fn test_wave_bottom_friction_madsen() {
        let config = WaveBottomFrictionConfig::madsen(0.05);
        let mut friction = WaveBottomFriction::new(10, config);
        
        let height = vec![2.0; 10];
        let period = vec![10.0; 10];
        let wavenumber = vec![0.08; 10];
        let depth = vec![15.0; 10];
        
        friction.compute_friction(&height, &period, &wavenumber, &depth);
        
        let dissipation = friction.energy_dissipation();
        assert!(dissipation.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_wave_bottom_friction_nielsen() {
        let config = WaveBottomFrictionConfig::nielsen(0.03);
        let mut friction = WaveBottomFriction::new(5, config);
        
        let height = vec![1.5; 5];
        let period = vec![8.0; 5];
        let wavenumber = vec![0.1; 5];
        let depth = vec![8.0; 5];
        
        friction.compute_friction(&height, &period, &wavenumber, &depth);
        
        let fw = friction.friction_coefficients();
        assert!(fw.iter().all(|&f| f > 0.0 && f < 1.0));
    }

    #[test]
    fn test_combined_stress() {
        let tau_current = 1.0;
        let tau_wave = 2.0;
        let angle = 0.0;  // 同向
        
        let tau_combined = WaveBottomFriction::compute_combined_stress(
            tau_current, tau_wave, angle
        );
        
        // 同向时联合应力应该接近叠加
        assert!(tau_combined > tau_current);
        assert!(tau_combined > tau_wave);
    }

    #[test]
    fn test_combined_stress_perpendicular() {
        let tau_current = 1.0;
        let tau_wave = 1.0;
        let angle = PI / 2.0;  // 垂直
        
        let tau_combined = WaveBottomFriction::compute_combined_stress(
            tau_current, tau_wave, angle
        );
        
        // 垂直时应该是矢量合成
        assert!(tau_combined > 0.0);
    }

    #[test]
    fn test_wave_current_interaction() {
        let config = WaveBottomFrictionConfig::jonswap();
        let mut interaction = WaveCurrentInteraction::new(10, config);
        
        let tau_current = vec![0.5; 10];
        let current_direction = vec![0.0; 10];
        let height = vec![1.0; 10];
        let period = vec![8.0; 10];
        let wavenumber = vec![0.1; 10];
        let wave_direction = vec![PI / 4.0; 10];
        let depth = vec![10.0; 10];
        
        interaction.compute(
            &tau_current,
            &current_direction,
            &height,
            &period,
            &wavenumber,
            &wave_direction,
            &depth,
        );
        
        let tau_combined = interaction.combined_shear_stress();
        assert!(tau_combined.iter().all(|&t| t > 0.0));
    }
}
