// src-tauri/src/marihydro/physics/waves/bottom_friction.rs
//! 波浪底部摩擦
//!
//! 实现波浪引起的底部剪应力和能量耗散。
//! 参考：Jonsson (1966), Soulsby (1997)

use std::f64::consts::PI;

/// 波浪轨道速度计算
#[derive(Debug, Clone, Copy)]
pub struct WaveOrbitalVelocity {
    /// 有效波高 [m]
    pub height: f64,
    /// 波周期 [s]
    pub period: f64,
    /// 波数 [1/m]
    pub wavenumber: f64,
}

impl WaveOrbitalVelocity {
    /// 创建新实例
    pub fn new(height: f64, period: f64, wavenumber: f64) -> Self {
        Self {
            height,
            period,
            wavenumber,
        }
    }

    /// 计算底部轨道速度振幅
    /// 
    /// Ub = πH / (T * sinh(kh))
    pub fn bottom_velocity(&self, depth: f64) -> f64 {
        let kh = self.wavenumber * depth;
        if kh < 0.01 {
            // 浅水极限
            PI * self.height / (self.period * kh)
        } else if kh > 10.0 {
            // 深水极限（趋近于零）
            0.0
        } else {
            PI * self.height / (self.period * kh.sinh())
        }
    }

    /// 计算底部轨道运动幅值
    /// 
    /// Ab = Ub * T / (2π)
    pub fn bottom_excursion(&self, depth: f64) -> f64 {
        let ub = self.bottom_velocity(depth);
        ub * self.period / (2.0 * PI)
    }
}

/// 波浪底部摩擦模型
#[derive(Debug, Clone, Copy)]
pub enum WaveBottomFrictionModel {
    /// Jonsson (1966) 摩擦因子
    Jonsson {
        /// Nikuradse 粗糙度 ks [m]
        roughness: f64,
    },
    /// Soulsby (1997) 简化公式
    Soulsby {
        /// 粗糙度 z0 [m]
        z0: f64,
    },
    /// 常数摩擦因子
    Constant {
        /// 摩擦因子 fw
        fw: f64,
    },
}

impl Default for WaveBottomFrictionModel {
    fn default() -> Self {
        Self::Jonsson { roughness: 0.01 }
    }
}

impl WaveBottomFrictionModel {
    /// 计算波浪摩擦因子 fw
    pub fn friction_factor(&self, orbital: &WaveOrbitalVelocity, depth: f64) -> f64 {
        match *self {
            Self::Jonsson { roughness } => {
                let ab = orbital.bottom_excursion(depth);
                if ab < 1e-10 {
                    return 0.0;
                }
                let r = ab / roughness;
                if r < 1.57 {
                    // 光滑床面
                    0.3
                } else {
                    // Jonsson (1966) 经验公式
                    // fw = exp(-5.977 + 5.213 * (Ab/ks)^(-0.194))
                    (-5.977 + 5.213 * r.powf(-0.194)).exp()
                }
            }
            Self::Soulsby { z0 } => {
                let ab = orbital.bottom_excursion(depth);
                if ab < 1e-10 {
                    return 0.0;
                }
                // Soulsby (1997): fw = 1.39 * (Ab/z0)^(-0.52)
                1.39 * (ab / z0).powf(-0.52)
            }
            Self::Constant { fw } => fw,
        }
    }
}

/// 波浪底部摩擦源项计算器
pub struct WaveBottomFriction {
    /// 摩擦模型
    model: WaveBottomFrictionModel,
    /// 水密度 [kg/m³]
    rho: f64,
    /// 底部剪应力振幅 [N/m²]
    tau_w: Vec<f64>,
    /// 波浪能量耗散率 [W/m²]
    dissipation: Vec<f64>,
}

impl WaveBottomFriction {
    /// 创建新的波浪底部摩擦计算器
    pub fn new(n_cells: usize, model: WaveBottomFrictionModel, rho: f64) -> Self {
        Self {
            model,
            rho,
            tau_w: vec![0.0; n_cells],
            dissipation: vec![0.0; n_cells],
        }
    }

    /// 使用默认模型创建
    pub fn with_default(n_cells: usize, rho: f64) -> Self {
        Self::new(n_cells, WaveBottomFrictionModel::default(), rho)
    }

    /// 调整大小
    pub fn resize(&mut self, n_cells: usize) {
        self.tau_w.resize(n_cells, 0.0);
        self.dissipation.resize(n_cells, 0.0);
    }

    /// 计算底部剪应力和能量耗散
    /// 
    /// τw = 0.5 * ρ * fw * Ub²
    /// D = 2/(3π) * ρ * fw * Ub³
    pub fn compute(
        &mut self,
        heights: &[f64],
        periods: &[f64],
        wavenumbers: &[f64],
        depths: &[f64],
    ) {
        let n = self.tau_w.len()
            .min(heights.len())
            .min(periods.len())
            .min(wavenumbers.len())
            .min(depths.len());

        for i in 0..n {
            if heights[i] < 1e-6 || depths[i] < 0.01 {
                self.tau_w[i] = 0.0;
                self.dissipation[i] = 0.0;
                continue;
            }

            let orbital = WaveOrbitalVelocity::new(heights[i], periods[i], wavenumbers[i]);
            let ub = orbital.bottom_velocity(depths[i]);

            if ub < 1e-10 {
                self.tau_w[i] = 0.0;
                self.dissipation[i] = 0.0;
                continue;
            }

            let fw = self.model.friction_factor(&orbital, depths[i]);

            // 底部剪应力振幅
            self.tau_w[i] = 0.5 * self.rho * fw * ub * ub;

            // 能量耗散率 (Jonsson 1966)
            self.dissipation[i] = 2.0 / (3.0 * PI) * self.rho * fw * ub.powi(3);
        }
    }

    /// 计算波流联合底部剪应力
    /// 
    /// 使用 Soulsby (1997) 的波流联合公式
    pub fn compute_combined(
        &mut self,
        heights: &[f64],
        periods: &[f64],
        wavenumbers: &[f64],
        depths: &[f64],
        current_velocities: &[(f64, f64)],
        current_tau: &[f64],
    ) {
        let n = self.tau_w.len()
            .min(heights.len())
            .min(current_velocities.len())
            .min(current_tau.len());

        for i in 0..n {
            if heights[i] < 1e-6 || depths[i] < 0.01 {
                self.tau_w[i] = current_tau[i];
                continue;
            }

            let orbital = WaveOrbitalVelocity::new(heights[i], periods[i], wavenumbers[i]);
            let ub = orbital.bottom_velocity(depths[i]);

            if ub < 1e-10 {
                self.tau_w[i] = current_tau[i];
                continue;
            }

            let fw = self.model.friction_factor(&orbital, depths[i]);
            let tau_wave = 0.5 * self.rho * fw * ub * ub;
            let tau_current = current_tau[i];

            // Soulsby (1997) 非线性增强公式
            // τm = τc * [1 + 1.2 * (τw/(τc + τw))^3.2]
            if tau_current < 1e-10 {
                self.tau_w[i] = tau_wave;
            } else {
                let ratio = tau_wave / (tau_current + tau_wave);
                let enhancement = 1.0 + 1.2 * ratio.powf(3.2);
                self.tau_w[i] = tau_current * enhancement;
            }
        }
    }

    /// 获取底部剪应力
    pub fn tau_w(&self) -> &[f64] {
        &self.tau_w
    }

    /// 获取能量耗散率
    pub fn dissipation(&self) -> &[f64] {
        &self.dissipation
    }

    /// 获取摩擦模型
    pub fn model(&self) -> &WaveBottomFrictionModel {
        &self.model
    }

    /// 设置摩擦模型
    pub fn set_model(&mut self, model: WaveBottomFrictionModel) {
        self.model = model;
    }
}

/// 波浪破碎能量耗散（用于近岸）
#[derive(Debug, Clone, Copy)]
pub struct WaveBreaking {
    /// 破碎系数 γ
    pub gamma: f64,
    /// 最大波陡 (H/L)max
    pub max_steepness: f64,
    /// 破碎类型指数
    pub breaking_index: f64,
}

impl Default for WaveBreaking {
    fn default() -> Self {
        Self {
            gamma: 0.78,
            max_steepness: 0.142,
            breaking_index: 0.8,
        }
    }
}

impl WaveBreaking {
    /// 判断是否破碎
    pub fn is_breaking(&self, height: f64, depth: f64, wavelength: f64) -> bool {
        // 深度限制破碎：H > γ * h
        let depth_limited = height > self.gamma * depth;
        // 波陡限制破碎：H/L > 0.142
        let steepness_limited = height / wavelength > self.max_steepness;

        depth_limited || steepness_limited
    }

    /// 计算破碎后波高
    pub fn limit_height(&self, height: f64, depth: f64, wavelength: f64) -> f64 {
        let h_depth = self.gamma * depth;
        let h_steep = self.max_steepness * wavelength;
        height.min(h_depth).min(h_steep)
    }

    /// 计算破碎能量耗散（Battjes-Janssen 模型）
    /// 
    /// D = 0.25 * α * Qb * f * ρg * Hmax²
    pub fn dissipation_rate(
        &self,
        height: f64,
        depth: f64,
        period: f64,
        rho: f64,
        g: f64,
    ) -> f64 {
        let h_max = self.gamma * depth;
        if height <= h_max {
            return 0.0;
        }

        // 破碎概率 Qb
        let qb = 1.0 - (h_max / height).powi(2);

        // 耗散率
        let alpha = 1.0; // 耗散系数
        let f = 1.0 / period;
        0.25 * alpha * qb * f * rho * g * h_max.powi(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbital_velocity() {
        // 浅水条件测试
        let orbital = WaveOrbitalVelocity::new(1.0, 10.0, 0.1);
        let ub = orbital.bottom_velocity(5.0);
        assert!(ub > 0.0);
        assert!(ub < 10.0); // 合理范围
    }

    #[test]
    fn test_jonsson_friction_factor() {
        let model = WaveBottomFrictionModel::Jonsson { roughness: 0.01 };
        let orbital = WaveOrbitalVelocity::new(1.0, 8.0, 0.1);
        let fw = model.friction_factor(&orbital, 5.0);
        assert!(fw > 0.0);
        assert!(fw < 1.0);
    }

    #[test]
    fn test_constant_friction() {
        let model = WaveBottomFrictionModel::Constant { fw: 0.05 };
        let orbital = WaveOrbitalVelocity::new(1.0, 8.0, 0.1);
        let fw = model.friction_factor(&orbital, 5.0);
        assert!((fw - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_wave_breaking() {
        let breaking = WaveBreaking::default();

        // 未破碎
        assert!(!breaking.is_breaking(0.5, 2.0, 50.0));

        // 深度限制破碎
        assert!(breaking.is_breaking(2.0, 2.0, 50.0));

        // 限制波高
        let h_limited = breaking.limit_height(2.0, 2.0, 50.0);
        assert!(h_limited < 2.0);
    }

    #[test]
    fn test_breaking_dissipation() {
        let breaking = WaveBreaking::default();
        
        // 未破碎时耗散为零
        let d1 = breaking.dissipation_rate(0.5, 2.0, 8.0, 1025.0, 9.81);
        assert!(d1.abs() < 1e-10);

        // 破碎时有耗散
        let d2 = breaking.dissipation_rate(2.0, 2.0, 8.0, 1025.0, 9.81);
        assert!(d2 > 0.0);
    }
}
