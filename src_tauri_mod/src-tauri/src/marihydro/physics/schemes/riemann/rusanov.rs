// src-tauri/src/marihydro/physics/schemes/riemann/rusanov.rs

//! Rusanov (局部 Lax-Friedrichs) 求解器
//!
//! 一种简单、稳健的近似黎曼求解器，特别适合处理强激波。
//! 数值耗散较大，但稳定性好。

use super::traits::{RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use crate::marihydro::core::error::MhResult;
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

/// Rusanov 求解器
#[derive(Debug, Clone)]
pub struct RusanovSolver {
    params: SolverParams,
}

impl RusanovSolver {
    /// 创建新的 Rusanov 求解器
    pub fn new(numerical_params: NumericalParams, gravity: f64) -> Self {
        Self {
            params: SolverParams::from_numerical(&numerical_params, gravity),
        }
    }

    /// 从参数直接创建
    pub fn from_params(params: SolverParams) -> Self {
        Self { params }
    }

    /// 获取参数
    pub fn params(&self) -> &SolverParams {
        &self.params
    }

    /// 计算最大波速
    #[inline]
    fn max_wave_speed(&self, h_l: f64, h_r: f64, un_l: f64, un_r: f64) -> f64 {
        let g = self.params.gravity;
        let c_l = (g * h_l).sqrt();
        let c_r = (g * h_r).sqrt();

        // 使用特征速度的最大值
        let s1 = (un_l - c_l).abs();
        let s2 = (un_l + c_l).abs();
        let s3 = (un_r - c_r).abs();
        let s4 = (un_r + c_r).abs();

        s1.max(s2).max(s3).max(s4)
    }

    /// 计算物理通量向量
    #[inline]
    fn physical_flux(&self, h: f64, vel: DVec2, normal: DVec2) -> (f64, f64, f64) {
        let g = self.params.gravity;
        let un = vel.dot(normal);
        let tangent = DVec2::new(-normal.y, normal.x);
        let ut = vel.dot(tangent);

        (
            h * un,
            h * un * un + 0.5 * g * h * h,
            h * un * ut,
        )
    }

    /// 计算守恒变量
    #[inline]
    fn conservative(&self, h: f64, vel: DVec2, normal: DVec2) -> (f64, f64, f64) {
        let tangent = DVec2::new(-normal.y, normal.x);
        let un = vel.dot(normal);
        let ut = vel.dot(tangent);
        (h, h * un, h * ut)
    }
}

impl RiemannSolver for RusanovSolver {
    fn name(&self) -> &'static str {
        "Rusanov"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: false, // Rusanov 不需要熵修正
            supports_hydrostatic: false,
            order: 1,
            positivity_preserving: true,
        }
    }

    fn solve(
        &self,
        h_left: f64,
        h_right: f64,
        vel_left: DVec2,
        vel_right: DVec2,
        normal: DVec2,
    ) -> MhResult<RiemannFlux> {
        // 干湿检测
        let is_dry_l = h_left <= self.params.h_dry;
        let is_dry_r = h_right <= self.params.h_dry;

        if is_dry_l && is_dry_r {
            return Ok(RiemannFlux::ZERO);
        }

        // 使用安全水深
        let h_l = if is_dry_l { self.params.h_min } else { h_left };
        let h_r = if is_dry_r { self.params.h_min } else { h_right };
        let v_l = if is_dry_l { DVec2::ZERO } else { vel_left };
        let v_r = if is_dry_r { DVec2::ZERO } else { vel_right };

        // 法向速度
        let un_l = v_l.dot(normal);
        let un_r = v_r.dot(normal);

        // 最大波速
        let s_max = self.max_wave_speed(h_l, h_r, un_l, un_r);

        // 左右物理通量
        let (f_mass_l, f_mom_n_l, f_mom_t_l) = self.physical_flux(h_l, v_l, normal);
        let (f_mass_r, f_mom_n_r, f_mom_t_r) = self.physical_flux(h_r, v_r, normal);

        // 左右守恒变量
        let (u_mass_l, u_mom_n_l, u_mom_t_l) = self.conservative(h_l, v_l, normal);
        let (u_mass_r, u_mom_n_r, u_mom_t_r) = self.conservative(h_r, v_r, normal);

        // Rusanov 通量: F = 0.5 * (F_L + F_R) - 0.5 * s_max * (U_R - U_L)
        let mass = 0.5 * (f_mass_l + f_mass_r) - 0.5 * s_max * (u_mass_r - u_mass_l);
        let mom_n = 0.5 * (f_mom_n_l + f_mom_n_r) - 0.5 * s_max * (u_mom_n_r - u_mom_n_l);
        let mom_t = 0.5 * (f_mom_t_l + f_mom_t_r) - 0.5 * s_max * (u_mom_t_r - u_mom_t_l);

        // 转换回全局坐标
        let flux = RiemannFlux::from_rotated(mass, mom_n, mom_t, normal, s_max);

        // 干区通量限制
        if is_dry_l || is_dry_r {
            let factor = if is_dry_l && is_dry_r {
                0.0
            } else if is_dry_l {
                // 左干：只允许流入
                if flux.mass < 0.0 { 0.0 } else { 1.0 }
            } else {
                // 右干：只允许流出
                if flux.mass > 0.0 { 0.0 } else { 1.0 }
            };
            return Ok(flux.scaled(factor));
        }

        Ok(flux)
    }

    fn gravity(&self) -> f64 {
        self.params.gravity
    }

    fn dry_threshold(&self) -> f64 {
        self.params.h_dry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_solver() -> RusanovSolver {
        RusanovSolver::from_params(SolverParams::default())
    }

    #[test]
    fn test_both_dry() {
        let solver = create_solver();
        let flux = solver
            .solve(0.0, 0.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert_eq!(flux.mass, 0.0);
    }

    #[test]
    fn test_still_water() {
        let solver = create_solver();
        let flux = solver
            .solve(10.0, 10.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        // Rusanov 在静水情况下应该没有通量
        assert!(flux.mass.abs() < 1e-10, "静水应无通量");
    }

    #[test]
    fn test_dam_break() {
        let solver = create_solver();
        let flux = solver
            .solve(10.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        // 溃坝应产生从高水位流向低水位的通量
        assert!(flux.mass > 0.0, "溃坝应产生正通量");
        assert!(flux.max_wave_speed > 0.0);
    }

    #[test]
    fn test_symmetry() {
        let solver = create_solver();

        // 交换左右应该得到相反的通量
        let flux1 = solver
            .solve(2.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        let flux2 = solver
            .solve(1.0, 2.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        assert!((flux1.mass + flux2.mass).abs() < 1e-10, "通量应该反对称");
    }

    #[test]
    fn test_is_more_diffusive_than_hllc() {
        use super::super::hllc::HllcSolver;

        let rusanov = create_solver();
        let hllc = HllcSolver::from_params(SolverParams::default());

        // 对于同样的溃坝问题
        let flux_r = rusanov
            .solve(5.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        let flux_h = hllc
            .solve(5.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();

        // Rusanov 的波速应该更大（更保守）
        assert!(
            flux_r.max_wave_speed >= flux_h.max_wave_speed * 0.99,
            "Rusanov 应该至少与 HLLC 一样保守"
        );
    }
}
