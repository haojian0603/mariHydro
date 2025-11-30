// src_tauri_mod\src-tauri\src\marihydro\physics\schemes\hydrostatic.rs
//! 静水重构与底坡源项处理
//!
//! 实现 Audusse 等人提出的静水重构方法，保证浅水方程在静止状态下的
//! well-balanced 性质。
//!
//! ## 理论背景
//!
//! 浅水方程的动量方程包含压力梯度项和底坡源项：
//!
//! $$ \frac{\partial (hu)}{\partial t} + \cdots = -gh\frac{\partial h}{\partial x} - gh\frac{\partial z_b}{\partial x} $$
//!
//! 在静止状态下（$u = v = 0$, $\eta = h + z_b = \text{const}$），这两项应精确抵消。
//! 然而，直接离散化会产生数值不平衡，导致虚假流动。
//!
//! ## Audusse 方法
//!
//! Audusse 等人 (2004) 提出的方法关键思想：
//!
//! 1. **界面高程取最大值**: $z_f = \max(z_L, z_R)$
//! 2. **修正界面水深**: $h^*_L = \max(0, \eta_L - z_f)$, $h^*_R = \max(0, \eta_R - z_f)$
//! 3. **底坡源项与通量修正配对**
//!
//! ### Well-balanced 性质
//!
//! 在静止水面条件下 ($\eta = const$, $u = v = 0$):
//! - 界面通量精确为零（因为 $h^*_L = h^*_R$, $u = 0$）
//! - 底坡源项精确抵消压力梯度
//!
//! ### 正性保持
//!
//! 修正后的水深保证非负，这对于干湿边界处理至关重要。
//!
//! ## 数值优势
//!
//! - 避免 $\eta - z$ 大数相减带来的精度损失
//! - 自然处理干湿边界
//! - 与任意黎曼求解器兼容
//!
//! ## 参考文献
//!
//! 1. Audusse, E., Bouchut, F., Bristeau, M.-O., Klein, R., & Perthame, B. (2004).
//!    A fast and stable well-balanced scheme with hydrostatic reconstruction for
//!    shallow water flows. SIAM Journal on Scientific Computing, 25(6), 2050-2065.
//!
//! 2. Kurganov, A., & Petrova, G. (2007). A second-order well-balanced positivity
//!    preserving central-upwind scheme for the Saint-Venant system.
//!    Communications in Mathematical Sciences, 5(1), 133-160.

use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

#[derive(Debug, Clone, Copy)]
pub struct HydrostaticFaceState {
    pub h_left: f64,
    pub h_right: f64,
    pub vel_left: DVec2,
    pub vel_right: DVec2,
    pub z_face: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BedSlopeSource {
    pub source_x: f64,
    pub source_y: f64,
}

impl BedSlopeSource {
    #[inline]
    pub fn as_vec(&self) -> DVec2 {
        DVec2::new(self.source_x, self.source_y)
    }
}

/// Audusse 静水重构器
pub struct HydrostaticReconstruction {
    h_dry: f64,
    h_min: f64,
    g: f64,
}

impl HydrostaticReconstruction {
    pub fn new(params: &NumericalParams, g: f64) -> Self {
        Self {
            h_dry: params.h_dry,
            h_min: params.h_min,
            g,
        }
    }

    /// 对单个界面进行 Audusse 重构（避免 eta-z 大数相减）
    pub fn reconstruct_face(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        grad_h_l: DVec2,
        grad_h_r: DVec2,
        r_l: DVec2,
        r_r: DVec2,
    ) -> HydrostaticFaceState {
        let z_face = z_l.max(z_r);

        // MUSCL 重构水深
        let h_l_recon = (h_l + grad_h_l.dot(r_l)).max(0.0);
        let h_r_recon = (h_r + grad_h_r.dot(r_r)).max(0.0);

        // 关键：使用 (h + z_cell - z_face) 而非 (eta - z_face)
        let delta_z_l = z_face - z_l;
        let delta_z_r = z_face - z_r;

        let h_l_star = (h_l_recon - delta_z_l).max(0.0);
        let h_r_star = (h_r_recon - delta_z_r).max(0.0);

        // 干湿处理
        let (h_l_final, vel_l_final) = if h_l_star < self.h_dry {
            (0.0, DVec2::ZERO)
        } else {
            (h_l_star, vel_l)
        };

        let (h_r_final, vel_r_final) = if h_r_star < self.h_dry {
            (0.0, DVec2::ZERO)
        } else {
            (h_r_star, vel_r)
        };

        HydrostaticFaceState {
            h_left: h_l_final,
            h_right: h_r_final,
            vel_left: vel_l_final,
            vel_right: vel_r_final,
            z_face,
        }
    }

    /// 简化版本（无梯度重构）
    pub fn reconstruct_face_simple(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
    ) -> HydrostaticFaceState {
        self.reconstruct_face(
            h_l,
            h_r,
            z_l,
            z_r,
            vel_l,
            vel_r,
            DVec2::ZERO,
            DVec2::ZERO,
            DVec2::ZERO,
            DVec2::ZERO,
        )
    }

    /// 计算底坡源项修正（保持 well-balanced）
    pub fn bed_slope_correction(
        &self,
        h_l: f64,
        h_r: f64,
        z_l: f64,
        z_r: f64,
        normal: DVec2,
        length: f64,
    ) -> BedSlopeSource {
        let z_face = z_l.max(z_r);
        let h_l_star = (h_l + z_l - z_face).max(0.0);
        let h_r_star = (h_r + z_r - z_face).max(0.0);

        let delta_z_l = z_face - z_l;
        let delta_z_r = z_face - z_r;

        // Well-balanced 修正通量
        let correction_l = 0.5 * self.g * (h_l * h_l - h_l_star * h_l_star + 2.0 * h_l * delta_z_l);
        let correction_r = 0.5 * self.g * (h_r * h_r - h_r_star * h_r_star + 2.0 * h_r * delta_z_r);

        let source = normal * (correction_r - correction_l) * length;

        BedSlopeSource {
            source_x: source.x,
            source_y: source.y,
        }
    }

    /// 验证静水平衡
    pub fn check_well_balanced(&self, h: &[f64], z: &[f64], tolerance: f64) -> bool {
        // 找参考水位
        let mut eta_ref = None;
        for i in 0..h.len() {
            if h[i] > self.h_dry * 10.0 {
                eta_ref = Some(h[i] + z[i]);
                break;
            }
        }

        let Some(eta_ref) = eta_ref else {
            return true;
        };

        // 检查所有湿单元
        for i in 0..h.len() {
            if h[i] > self.h_dry * 10.0 {
                let eta = h[i] + z[i];
                if (eta - eta_ref).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

/// 高精度水位计算（当底高程绝对值很大时使用）
pub struct HighPrecisionWaterLevel {
    reference_elevation: f64,
}

impl HighPrecisionWaterLevel {
    pub fn new(z_bed: &[f64]) -> Self {
        let sum: f64 = z_bed.iter().sum();
        let reference = if z_bed.is_empty() {
            0.0
        } else {
            sum / z_bed.len() as f64
        };
        Self {
            reference_elevation: reference,
        }
    }

    #[inline]
    pub fn relative_eta(&self, h: f64, z: f64) -> f64 {
        h + (z - self.reference_elevation)
    }

    #[inline]
    pub fn depth_from_relative_eta(&self, eta_rel: f64, z: f64) -> f64 {
        (eta_rel - (z - self.reference_elevation)).max(0.0)
    }

    pub fn reference(&self) -> f64 {
        self.reference_elevation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydrostatic_balance() {
        let params = NumericalParams::default();
        let recon = HydrostaticReconstruction::new(&params, 9.81);

        let eta = 10.0;
        let z_l = 5.0;
        let z_r = 3.0;
        let h_l = eta - z_l;
        let h_r = eta - z_r;

        let result = recon.reconstruct_face_simple(h_l, h_r, z_l, z_r, DVec2::ZERO, DVec2::ZERO);

        let eta_left = result.h_left + result.z_face;
        let eta_right = result.h_right + result.z_face;
        assert!((eta_left - eta_right).abs() < 1e-10);
    }

    #[test]
    fn test_high_precision() {
        let z = vec![1000.0, 1001.0, 1002.0];
        let hp = HighPrecisionWaterLevel::new(&z);

        let eta_rel = hp.relative_eta(5.0, 1001.0);
        let h_back = hp.depth_from_relative_eta(eta_rel, 1001.0);
        assert!((h_back - 5.0).abs() < 1e-10);
    }
}
