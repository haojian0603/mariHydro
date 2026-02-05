// crates/mh_physics/src/waves/bottom_friction.rs

//! 波浪底摩擦计算
//! 
//! 实现多种波浪底摩擦模型，包括 Jonswap, Madsen, Nielsen 等。

use serde::{Deserialize, Serialize};
use crate::prelude::*;

fn scalar_const<B: Backend>(backend: &B, v: f64) -> B::Scalar {
    backend.scalar_from_f64(v)
}

fn scalar_pi<B: Backend>(backend: &B) -> B::Scalar {
    backend.scalar_from_f64(std::f64::consts::PI)
}

/// 波浪底摩擦模型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum WaveBottomFrictionModel {
    /// JONSWAP 经验公式
    #[default]
    Jonswap,
    /// Madsen (1988) 公式
    Madsen,
    /// Nielsen (1992) 公式
    Nielsen,
    /// 常数摩擦系数
    Constant,
}

/// 波浪底摩擦错误
#[derive(Debug, Clone)]
pub enum WaveBottomFrictionError {
    /// 后端缓冲区不可直接访问
    BackendAccess(String),
}

impl std::fmt::Display for WaveBottomFrictionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BackendAccess(msg) => write!(f, "后端访问错误: {}", msg),
        }
    }
}

impl std::error::Error for WaveBottomFrictionError {}


/// 波浪轨道速度计算器
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WaveOrbitalVelocity<S: RuntimeScalar> {
    /// 最大水平轨道速度 [m/s]
    pub u_max: S,
    /// 轨道位移幅值 [m]
    pub amplitude: S,
    /// 轨道周期 [s]
    pub period: S,
}

impl<S: RuntimeScalar> WaveOrbitalVelocity<S> {
    /// 从波浪参数计算底部轨道速度
    /// 
    /// u_max = πH / (T sinh(kh))
    /// a = H / (2 sinh(kh))
    pub fn compute<B: Backend<Scalar = S>>(
        backend: &B,
        height: S,
        period: S,
        wavenumber: S,
        depth: S,
    ) -> Self {
        let h_min = scalar_const(backend, 0.1);
        let h = if depth > h_min { depth } else { h_min };
        let kh = wavenumber * h;
        let sinh_kh = kh.sinh();
        let eps = scalar_const(backend, 1e-10);
        
        if sinh_kh < eps {
            // 浅水极限
            let c = (scalar_const(backend, 9.81) * h).sqrt();
            return Self {
                u_max: height / (scalar_const(backend, 2.0) * h) * c,
                amplitude: height / scalar_const(backend, 2.0),
                period,
            };
        }
        
        let pi = scalar_pi(backend);
        Self {
            u_max: pi * height / (period * sinh_kh),
            amplitude: height / (scalar_const(backend, 2.0) * sinh_kh),
            period,
        }
    }

    /// 获取轨道速度随时间的变化
    pub fn velocity_at_phase(&self, phase: S) -> S {
        self.u_max * phase.cos()
    }

    /// 获取轨道位移随时间的变化
    pub fn displacement_at_phase(&self, phase: S) -> S {
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
pub struct WaveBottomFriction<B: Backend> {
    /// 配置
    config: WaveBottomFrictionConfig,
    /// 后端实例
    backend: B,
    /// 摩擦系数
    fw: B::Buffer<B::Scalar>,
    /// 床面剪切应力振幅 [Pa]
    tau_wave: B::Buffer<B::Scalar>,
    /// 能量耗散率 [W/m²]
    dissipation: B::Buffer<B::Scalar>,
}

impl<B: Backend> WaveBottomFriction<B> {
    /// 创建新的计算器
    pub fn new(backend: B, n_cells: usize, config: WaveBottomFrictionConfig) -> Self {
        let mut fw = backend.alloc(n_cells);
        let mut tau_wave = backend.alloc(n_cells);
        let mut dissipation = backend.alloc(n_cells);
        fw.fill(scalar_const(&backend, 0.01));
        tau_wave.fill(B::Scalar::ZERO);
        dissipation.fill(B::Scalar::ZERO);
        Self {
            config,
            backend,
            fw,
            tau_wave,
            dissipation,
        }
    }

    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// 计算波浪底摩擦
    pub fn compute_friction(
        &mut self,
        height: &B::Buffer<B::Scalar>,
        period: &B::Buffer<B::Scalar>,
        wavenumber: &B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
    ) -> Result<(), WaveBottomFrictionError> {
        let height = height.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("height buffer not accessible".to_string())
        })?;
        let period = period.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("period buffer not accessible".to_string())
        })?;
        let wavenumber = wavenumber.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("wavenumber buffer not accessible".to_string())
        })?;
        let depth = depth.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("depth buffer not accessible".to_string())
        })?;
        let fw = self.fw.try_as_slice_mut().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("fw buffer not accessible".to_string())
        })?;
        let tau_wave = self.tau_wave.try_as_slice_mut().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("tau_wave buffer not accessible".to_string())
        })?;
        let dissipation = self.dissipation.try_as_slice_mut().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("dissipation buffer not accessible".to_string())
        })?;

        let n = fw.len()
            .min(height.len())
            .min(period.len())
            .min(wavenumber.len())
            .min(depth.len());
        
        let rho = scalar_const(&self.backend, 1025.0);
        let half = scalar_const(&self.backend, 0.5);
        let two = scalar_const(&self.backend, 2.0);
        let three = scalar_const(&self.backend, 3.0);
        let pi = scalar_pi(&self.backend);

        for i in 0..n {
            let orbital = WaveOrbitalVelocity::compute(
                &self.backend,
                height[i],
                period[i],
                wavenumber[i],
                depth[i],
            );

            // 计算摩擦系数
            fw[i] = Self::compute_friction_coefficient(
                &self.config,
                &self.backend,
                orbital.amplitude,
                orbital.period,
            );

            // 床面剪切应力振幅
            // τ_wave = 0.5 × ρ × fw × u_max²
            tau_wave[i] = half * rho * fw[i] * orbital.u_max * orbital.u_max;

            // 能量耗散率
            // D = (2/3π) × ρ × fw × u_max³
            dissipation[i] = two / (three * pi) * rho * fw[i]
                * orbital.u_max * orbital.u_max * orbital.u_max;
        }
        Ok(())
    }

    /// 根据模型计算摩擦系数
    fn compute_friction_coefficient(
        config: &WaveBottomFrictionConfig,
        backend: &B,
        amplitude: B::Scalar,
        _period: B::Scalar,
    ) -> B::Scalar {
        let amplitude = amplitude.to_f64_lossy();
        match config.model {
            WaveBottomFrictionModel::Jonswap => {
                // JONSWAP 经验公式
                // fw = 0.067 for typical conditions
                scalar_const(backend, 0.067)
            }
            WaveBottomFrictionModel::Madsen => {
                // Madsen (1988)
                // fw = exp(-5.977 + 5.213(a/ks)^(-0.194))
                let a = amplitude.max(1e-6);
                let ks = config.roughness_height.max(1e-6);
                let ratio = a / ks;
                
                if ratio < 1.57 {
                    scalar_const(backend, 0.3)  // 最大值
                } else {
                    scalar_const(backend, (-5.977 + 5.213 * ratio.powf(-0.194)).exp())
                }
            }
            WaveBottomFrictionModel::Nielsen => {
                // Nielsen (1992)
                // fw = exp(5.5(a/ks)^(-0.2) - 6.3)
                let a = amplitude.max(1e-6);
                let ks = config.roughness_height.max(1e-6);
                let ratio = a / ks;
                
                if ratio < 1.0 {
                    scalar_const(backend, 0.3)
                } else {
                    scalar_const(backend, (5.5 * ratio.powf(-0.2) - 6.3).exp())
                }
            }
            WaveBottomFrictionModel::Constant => {
                scalar_const(backend, config.friction_coefficient)
            }
        }
    }

    /// 获取摩擦系数
    pub fn friction_coefficients(&self) -> &B::Buffer<B::Scalar> {
        &self.fw
    }

    /// 获取波浪床面剪切应力
    pub fn wave_shear_stress(&self) -> &B::Buffer<B::Scalar> {
        &self.tau_wave
    }

    /// 获取能量耗散率
    pub fn energy_dissipation(&self) -> &B::Buffer<B::Scalar> {
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
        backend: &B,
        tau_current: B::Scalar,
        tau_wave: B::Scalar,
        angle_between: B::Scalar,
    ) -> B::Scalar {
        let cos_phi = angle_between.cos();
        let one = scalar_const(backend, 1.0);
        let one_point_two = scalar_const(backend, 1.2);
        let three_point_two = scalar_const(backend, 3.2);
        let eps = scalar_const(backend, 1e-14);
        
        // Soulsby 非线性公式
        let ratio = tau_wave / (tau_current + tau_wave + eps);
        let tau_mean = tau_current * (one + one_point_two * ratio.powf(three_point_two));
        
        ((tau_mean + tau_wave * cos_phi).powi(2)
            + (tau_wave * angle_between.sin()).powi(2)).sqrt()
    }
}

/// 波流联合底摩擦计算器
pub struct WaveCurrentInteraction<B: Backend> {
    /// 波浪摩擦计算器
    wave_friction: WaveBottomFriction<B>,
    /// 联合剪切应力 [Pa]
    tau_combined: B::Buffer<B::Scalar>,
    /// 联合摩擦系数（暂未使用）
    _fc_combined: B::Buffer<B::Scalar>,
}

impl<B: Backend> WaveCurrentInteraction<B> {
    /// 创建新的计算器
    pub fn new(backend: B, n_cells: usize, config: WaveBottomFrictionConfig) -> Self {
        let mut tau_combined = backend.alloc(n_cells);
        let mut fc_combined = backend.alloc(n_cells);
        tau_combined.fill(B::Scalar::ZERO);
        fc_combined.fill(B::Scalar::ZERO);
        Self {
            wave_friction: WaveBottomFriction::new(backend, n_cells, config),
            tau_combined,
            _fc_combined: fc_combined,
        }
    }

    /// 计算波流联合剪切应力
    pub fn compute(
        &mut self,
        tau_current: &B::Buffer<B::Scalar>,
        current_direction: &B::Buffer<B::Scalar>,
        height: &B::Buffer<B::Scalar>,
        period: &B::Buffer<B::Scalar>,
        wavenumber: &B::Buffer<B::Scalar>,
        wave_direction: &B::Buffer<B::Scalar>,
        depth: &B::Buffer<B::Scalar>,
    ) -> Result<(), WaveBottomFrictionError> {
        // 先计算波浪底摩擦
        self.wave_friction.compute_friction(height, period, wavenumber, depth)?;
        let tau_wave = self.wave_friction.wave_shear_stress();

        let tau_current = tau_current.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("tau_current buffer not accessible".to_string())
        })?;
        let current_direction = current_direction.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("current_direction buffer not accessible".to_string())
        })?;
        let wave_direction = wave_direction.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("wave_direction buffer not accessible".to_string())
        })?;
        let tau_wave = tau_wave.try_as_slice().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("tau_wave buffer not accessible".to_string())
        })?;
        let tau_combined = self.tau_combined.try_as_slice_mut().ok_or_else(|| {
            WaveBottomFrictionError::BackendAccess("tau_combined buffer not accessible".to_string())
        })?;

        let n = tau_combined.len()
            .min(tau_current.len())
            .min(current_direction.len())
            .min(wave_direction.len())
            .min(tau_wave.len());

        for i in 0..n {
            let angle_diff = wave_direction[i] - current_direction[i];
            tau_combined[i] = WaveBottomFriction::compute_combined_stress(
                self.wave_friction.backend(),
                tau_current[i],
                tau_wave[i],
                angle_diff,
            );
        }
        Ok(())
    }

    /// 获取联合剪切应力
    pub fn combined_shear_stress(&self) -> &B::Buffer<B::Scalar> {
        &self.tau_combined
    }
    pub fn wave_friction(&self) -> &WaveBottomFriction<B> {
        &self.wave_friction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_orbital_velocity() {
        let backend = CpuBackend::<f64>::new();
        let orbital = WaveOrbitalVelocity::compute(&backend, 2.0, 8.0, 0.1, 10.0);
        assert!(orbital.u_max > 0.0);
        assert!(orbital.amplitude > 0.0);
        assert_eq!(orbital.period, 8.0);
    }

    #[test]
    fn test_orbital_velocity_shallow() {
        let backend = CpuBackend::<f64>::new();
        let orbital = WaveOrbitalVelocity::compute(&backend, 1.0, 8.0, 1.0, 0.5);
        assert!(orbital.u_max > 0.0);
    }

    #[test]
    fn test_wave_bottom_friction_config() {
        let config = WaveBottomFrictionConfig::jonswap();
        assert_eq!(config.model, WaveBottomFrictionModel::Jonswap);
        
        let config = WaveBottomFrictionConfig::madsen(0.03);
        assert!((config.roughness_height - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_wave_bottom_friction_jonswap() {
        let config = WaveBottomFrictionConfig::jonswap();
        let backend = CpuBackend::<f64>::new();
        let mut friction = WaveBottomFriction::new(backend.clone(), 10, config);
        
        let mut height = backend.alloc(10);
        let mut period = backend.alloc(10);
        let mut wavenumber = backend.alloc(10);
        let mut depth = backend.alloc(10);
        height.copy_from_slice(&[1.0; 10]);
        period.copy_from_slice(&[8.0; 10]);
        wavenumber.copy_from_slice(&[0.1; 10]);
        depth.copy_from_slice(&[10.0; 10]);
        
        friction.compute_friction(&height, &period, &wavenumber, &depth).unwrap();
        
        let fw = friction.friction_coefficients().as_slice();
        assert!(fw.iter().all(|&f| f > 0.0));
        
        let tau = friction.wave_shear_stress().as_slice();
        assert!(tau.iter().all(|&t| t >= 0.0));
    }

    #[test]
    fn test_wave_bottom_friction_madsen() {
        let config = WaveBottomFrictionConfig::madsen(0.05);
        let backend = CpuBackend::<f64>::new();
        let mut friction = WaveBottomFriction::new(backend.clone(), 10, config);
        
        let mut height = backend.alloc(10);
        let mut period = backend.alloc(10);
        let mut wavenumber = backend.alloc(10);
        let mut depth = backend.alloc(10);
        height.copy_from_slice(&[2.0; 10]);
        period.copy_from_slice(&[10.0; 10]);
        wavenumber.copy_from_slice(&[0.08; 10]);
        depth.copy_from_slice(&[15.0; 10]);
        
        friction.compute_friction(&height, &period, &wavenumber, &depth).unwrap();
        
        let dissipation = friction.energy_dissipation().as_slice();
        assert!(dissipation.iter().all(|&d| d >= 0.0));
    }

    #[test]
    fn test_wave_bottom_friction_nielsen() {
        let config = WaveBottomFrictionConfig::nielsen(0.03);
        let backend = CpuBackend::<f64>::new();
        let mut friction = WaveBottomFriction::new(backend.clone(), 5, config);
        
        let mut height = backend.alloc(5);
        let mut period = backend.alloc(5);
        let mut wavenumber = backend.alloc(5);
        let mut depth = backend.alloc(5);
        height.copy_from_slice(&[1.5; 5]);
        period.copy_from_slice(&[8.0; 5]);
        wavenumber.copy_from_slice(&[0.1; 5]);
        depth.copy_from_slice(&[8.0; 5]);
        
        friction.compute_friction(&height, &period, &wavenumber, &depth).unwrap();
        
        let fw = friction.friction_coefficients().as_slice();
        assert!(fw.iter().all(|&f| f > 0.0 && f < 1.0));
    }

    #[test]
    fn test_combined_stress() {
        let backend = CpuBackend::<f64>::new();
        let tau_current = 1.0;
        let tau_wave = 2.0;
        let angle = 0.0;  // 同向
        
        let tau_combined = WaveBottomFriction::compute_combined_stress(
            &backend, tau_current, tau_wave, angle
        );
        
        // 同向时联合应力应该接近叠加
        assert!(tau_combined > tau_current);
        assert!(tau_combined > tau_wave);
    }

    #[test]
    fn test_combined_stress_perpendicular() {
        let backend = CpuBackend::<f64>::new();
        let tau_current = 1.0;
        let tau_wave = 1.0;
        let angle = std::f64::consts::PI / 2.0;  // 垂直
        
        let tau_combined = WaveBottomFriction::compute_combined_stress(
            &backend, tau_current, tau_wave, angle
        );
        
        // 垂直时应该是矢量合成
        assert!(tau_combined > 0.0);
    }

    #[test]
    fn test_wave_current_interaction() {
        let config = WaveBottomFrictionConfig::jonswap();
        let backend = CpuBackend::<f64>::new();
        let mut interaction = WaveCurrentInteraction::new(backend.clone(), 10, config);
        
        let mut tau_current = backend.alloc(10);
        let mut current_direction = backend.alloc(10);
        let mut height = backend.alloc(10);
        let mut period = backend.alloc(10);
        let mut wavenumber = backend.alloc(10);
        let mut wave_direction = backend.alloc(10);
        let mut depth = backend.alloc(10);
        tau_current.copy_from_slice(&[0.5; 10]);
        current_direction.copy_from_slice(&[0.0; 10]);
        height.copy_from_slice(&[1.0; 10]);
        period.copy_from_slice(&[8.0; 10]);
        wavenumber.copy_from_slice(&[0.1; 10]);
        wave_direction.copy_from_slice(&[std::f64::consts::PI / 4.0; 10]);
        depth.copy_from_slice(&[10.0; 10]);
        
        interaction.compute(
            &tau_current,
            &current_direction,
            &height,
            &period,
            &wavenumber,
            &wave_direction,
            &depth,
        ).unwrap();
        
        let tau_combined = interaction.combined_shear_stress().as_slice();
        assert!(tau_combined.iter().all(|&t| t > 0.0));
    }
}
