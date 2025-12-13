// marihydro\crates\mh_physics\src\sources\wave_forcing.rs
//! 波浪驱动源项
//!
//! 实现辐射应力梯度对流场的驱动作用：
//! - 辐射应力梯度 → 动量源
//! - 波浪增强底摩擦
//! - 波浪增强泥沙起动
//!
//! # 辐射应力梯度
//!
//! 在近岸区域，波浪破碎产生的辐射应力梯度是重要的动力因素：
//! ```text
//! S_x = -1/ρh × (∂S_xx/∂x + ∂S_xy/∂y)
//! S_y = -1/ρh × (∂S_xy/∂x + ∂S_yy/∂y)
//! ```

use crate::sources::traits::{SourceContribution, SourceContext, SourceTerm};
use crate::state::ShallowWaterState;
use mh_foundation::AlignedVec;
use serde::{Deserialize, Serialize};

/// 波浪驱动源项配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveForcingConfig {
    /// 是否启用
    pub enabled: bool,
    /// 水密度 [kg/m³]
    pub rho_water: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最小水深 [m]
    pub h_min: f64, // ALLOW_F64: Layer 4 配置参数
}

impl Default for WaveForcingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rho_water: 1025.0,
            h_min: 0.1,
        }
    }
}

/// 波浪驱动源项
///
/// 将辐射应力梯度转换为动量源项
pub struct WaveForcing {
    /// 配置
    config: WaveForcingConfig,
    /// 单元数
    n_cells: usize,
    /// x 方向辐射应力梯度 ∂S_xx/∂x + ∂S_xy/∂y [N/m²]
    grad_sxx_sxy: AlignedVec<f64>, // ALLOW_F64: 源项计算
    /// y 方向辐射应力梯度 ∂S_xy/∂x + ∂S_yy/∂y [N/m²]
    grad_sxy_syy: AlignedVec<f64>, // ALLOW_F64: 源项计算
    /// 波浪轨道速度 [m/s]
    orbital_velocity: AlignedVec<f64>, // ALLOW_F64: 源项计算
    /// 有效（波流联合）剪切应力 [Pa]
    effective_shear: AlignedVec<f64>, // ALLOW_F64: 源项计算
}

impl WaveForcing {
    /// 创建新的波浪源项
    pub fn new(n_cells: usize, config: WaveForcingConfig) -> Self {
        Self {
            config,
            n_cells,
            grad_sxx_sxy: AlignedVec::zeros(n_cells),
            grad_sxy_syy: AlignedVec::zeros(n_cells),
            orbital_velocity: AlignedVec::zeros(n_cells),
            effective_shear: AlignedVec::zeros(n_cells),
        }
    }

    /// 使用默认配置创建
    pub fn with_defaults(n_cells: usize) -> Self {
        Self::new(n_cells, WaveForcingConfig::default())
    }

    /// 从辐射应力张量更新梯度
    ///
    /// # 参数
    /// - `stress`: 辐射应力张量计算器
    /// - `dx`, `dy`: 网格间距（简化版，假设均匀网格）
    pub fn update_from_radiation_stress(
        &mut self,
        s_xx: &[f64],
        s_xy: &[f64],
        s_yy: &[f64],
        grad_x: &[f64], // ∂/∂x 算子结果
        grad_y: &[f64], // ∂/∂y 算子结果
    ) {
        let n = self.n_cells.min(s_xx.len()).min(s_xy.len()).min(s_yy.len());

        for i in 0..n {
            // 简化：直接使用预计算的梯度
            // 实际应该用单独的梯度计算
            self.grad_sxx_sxy[i] = grad_x.get(i).copied().unwrap_or(0.0);
            self.grad_sxy_syy[i] = grad_y.get(i).copied().unwrap_or(0.0);
        }
    }

    /// 设置辐射应力梯度
    pub fn set_stress_gradients(
        &mut self,
        grad_sxx_sxy: &[f64],
        grad_sxy_syy: &[f64],
    ) {
        let n = self.n_cells.min(grad_sxx_sxy.len()).min(grad_sxy_syy.len());
        self.grad_sxx_sxy[..n].copy_from_slice(&grad_sxx_sxy[..n]);
        self.grad_sxy_syy[..n].copy_from_slice(&grad_sxy_syy[..n]);
    }

    /// 设置波浪轨道速度
    pub fn set_orbital_velocity(&mut self, u_orb: &[f64]) {
        let n = self.n_cells.min(u_orb.len());
        self.orbital_velocity[..n].copy_from_slice(&u_orb[..n]);
    }

    /// 计算有效剪切应力（波流联合）
    ///
    /// # 参数
    /// - `tau_current`: 流动引起的床面剪切应力 [Pa]
    pub fn compute_effective_shear(&mut self, tau_current: &[f64]) {
        for i in 0..self.n_cells.min(tau_current.len()) {
            let u_orb = self.orbital_velocity[i];
            
            // 波浪引起的剪切应力：τ_wave = 0.5 × ρ × fw × u_orb²
            let fw = 0.05; // 简化摩擦系数
            let tau_wave = 0.5 * self.config.rho_water * fw * u_orb * u_orb;
            
            // 简单叠加（Soulsby 公式的简化版）
            self.effective_shear[i] = tau_current[i] + tau_wave;
        }
    }

    /// 获取有效剪切应力（用于泥沙计算）
    pub fn effective_shear(&self) -> &[f64] {
        &self.effective_shear
    }

    /// 获取波浪轨道速度
    pub fn orbital_velocity(&self) -> &[f64] {
        &self.orbital_velocity
    }

    /// 计算辐射应力梯度（有限体积法）
    ///
    /// 使用 Green-Gauss 方法计算应力散度:
    /// - grad_x = ∂S_xx/∂x + ∂S_xy/∂y
    /// - grad_y = ∂S_xy/∂x + ∂S_yy/∂y
    ///
    /// # 参数
    /// - `s_xx`: S_xx 应力场 [N/m]
    /// - `s_xy`: S_xy 应力场 [N/m]
    /// - `s_yy`: S_yy 应力场 [N/m]
    /// - `cell_areas`: 单元面积 [m²]
    /// - `neighbors`: 邻域信息 (邻居索引, 面长度, 法向量x, 法向量y)
    pub fn compute_stress_gradients(
        &mut self,
        s_xx: &[f64],
        s_xy: &[f64],
        s_yy: &[f64],
        cell_areas: &[f64],
        neighbors: &[Vec<(usize, f64, f64, f64)>],
    ) {
        let n = self.n_cells.min(s_xx.len()).min(s_xy.len()).min(s_yy.len());

        for i in 0..n {
            let area = cell_areas.get(i).copied().unwrap_or(1.0).max(1e-10);

            let mut grad_sxx_x = 0.0;  // ∂S_xx/∂x
            let mut grad_sxy_x = 0.0;  // ∂S_xy/∂x
            let mut grad_sxy_y = 0.0;  // ∂S_xy/∂y
            let mut grad_syy_y = 0.0;  // ∂S_yy/∂y

            // Green-Gauss: ∂φ/∂x ≈ (1/A) × ∑_f φ_f × n_x × L_f
            if let Some(neigh_list) = neighbors.get(i) {
                for &(j, face_len, nx, ny) in neigh_list {
                    if j >= n {
                        continue;
                    }

                    // 面上插值（算术平均）
                    let sxx_face = 0.5 * (s_xx[i] + s_xx[j]);
                    let sxy_face = 0.5 * (s_xy[i] + s_xy[j]);
                    let syy_face = 0.5 * (s_yy[i] + s_yy[j]);

                    grad_sxx_x += sxx_face * nx * face_len;
                    grad_sxy_x += sxy_face * nx * face_len;
                    grad_sxy_y += sxy_face * ny * face_len;
                    grad_syy_y += syy_face * ny * face_len;
                }
            }

            // 除以面积
            grad_sxx_x /= area;
            grad_sxy_x /= area;
            grad_sxy_y /= area;
            grad_syy_y /= area;

            // 组合为动量方程源项的梯度
            self.grad_sxx_sxy[i] = grad_sxx_x + grad_sxy_y;
            self.grad_sxy_syy[i] = grad_sxy_x + grad_syy_y;
        }
    }

    /// 获取 x 方向应力梯度
    pub fn grad_x(&self) -> &[f64] {
        &self.grad_sxx_sxy
    }

    /// 获取 y 方向应力梯度
    pub fn grad_y(&self) -> &[f64] {
        &self.grad_sxy_syy
    }
}

impl SourceTerm for WaveForcing {
    fn name(&self) -> &'static str {
        "WaveForcing"
    }

    fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    fn compute_cell(
        &self,
        state: &ShallowWaterState,
        cell: usize,
        _ctx: &SourceContext,
    ) -> SourceContribution {
        let h = state.h[cell];
        if h < self.config.h_min {
            return SourceContribution::ZERO;
        }

        let rho = self.config.rho_water;

        // S_x = -1/(ρh) × (∂S_xx/∂x + ∂S_xy/∂y)
        let s_hu = -self.grad_sxx_sxy[cell] / (rho * h);
        let s_hv = -self.grad_sxy_syy[cell] / (rho * h);

        SourceContribution::momentum(s_hu, s_hv)
    }

    fn is_explicit(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NumericalParams;

    fn create_test_state(n_cells: usize, h: f64) -> ShallowWaterState {
        let mut state = ShallowWaterState::new(n_cells);
        for i in 0..n_cells {
            state.h[i] = h;
        }
        state
    }

    #[test]
    fn test_wave_forcing_creation() {
        let wf = WaveForcing::with_defaults(10);
        assert_eq!(wf.n_cells, 10);
    }

    #[test]
    fn test_zero_gradient() {
        let wf = WaveForcing::with_defaults(10);
        let state = create_test_state(10, 2.0);
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);

        let contrib = wf.compute_cell(&state, 0, &ctx);
        
        // 零梯度 → 零源项
        assert!((contrib.s_hu).abs() < 1e-10);
        assert!((contrib.s_hv).abs() < 1e-10);
    }

    #[test]
    fn test_nonzero_gradient() {
        let mut wf = WaveForcing::with_defaults(10);
        let state = create_test_state(10, 2.0);
        
        // 设置非零梯度
        wf.grad_sxx_sxy.fill(100.0); // N/m²
        
        let params = NumericalParams::default();
        let ctx = SourceContext::new(0.0, 1.0, &params);
        let contrib = wf.compute_cell(&state, 0, &ctx);
        
        // 应该有负的 x 动量源（辐射应力驱动）
        assert!(contrib.s_hu < 0.0);
    }

    #[test]
    fn test_effective_shear() {
        let mut wf = WaveForcing::with_defaults(10);
        
        // 设置轨道速度
        wf.orbital_velocity.fill(0.5);
        
        let tau_current = vec![1.0; 10];
        wf.compute_effective_shear(&tau_current);
        
        // 有效应力应该大于流动应力
        assert!(wf.effective_shear[0] > 1.0);
    }
}

