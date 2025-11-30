// src-tauri/src/marihydro/physics/schemes/riemann/hllc.rs

//! HLLC 近似黎曼求解器
//!
//! HLLC (Harten-Lax-van Leer-Contact) 求解器是一种高精度的近似黎曼求解器，
//! 能够正确处理接触间断。

use super::traits::{RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams};
use super::InterfaceState;
use crate::marihydro::core::error::{MhError, MhResult};
use crate::marihydro::core::types::NumericalParams;
use glam::DVec2;

/// HLLC 求解器
#[derive(Debug, Clone)]
pub struct HllcSolver {
    params: SolverParams,
}

impl HllcSolver {
    /// 创建新的 HLLC 求解器
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

    // ================= 波速估计 =================

    /// Einfeldt 波速估计
    ///
    /// 使用 Roe 平均计算特征波速
    #[inline]
    fn einfeldt_speeds(
        &self,
        h_l: f64,
        h_r: f64,
        un_l: f64,
        un_r: f64,
        c_l: f64,
        c_r: f64,
    ) -> (f64, f64) {
        let sh_l = h_l.sqrt();
        let sh_r = h_r.sqrt();
        let sum = sh_l + sh_r + self.params.flux_eps;

        // Roe 平均
        let h_roe = 0.5 * (h_l + h_r);
        let u_roe = (sh_l * un_l + sh_r * un_r) / sum;
        let c_roe = (self.params.gravity * h_roe).sqrt();

        (
            (un_l - c_l).min(u_roe - c_roe),
            (un_r + c_r).max(u_roe + c_roe),
        )
    }

    /// 熵修正
    #[inline]
    fn entropy_fix(&self, s_star: f64, s_l: f64, s_r: f64) -> f64 {
        let eps = self.params.entropy_threshold((s_r - s_l).abs());
        if s_star.abs() < eps {
            s_star.signum() * eps
        } else {
            s_star
        }
    }

    // ================= 核心求解 =================

    /// 求解双湿状态
    fn solve_both_wet(
        &self,
        h_l: f64,
        h_r: f64,
        vel_l: DVec2,
        vel_r: DVec2,
        normal: DVec2,
    ) -> MhResult<RiemannFlux> {
        let tangent = DVec2::new(-normal.y, normal.x);

        // 分解速度到法向/切向
        let un_l = vel_l.dot(normal);
        let un_r = vel_r.dot(normal);
        let ut_l = vel_l.dot(tangent);
        let ut_r = vel_r.dot(tangent);

        // 波速
        let c_l = (self.params.gravity * h_l).sqrt();
        let c_r = (self.params.gravity * h_r).sqrt();
        let (s_l, s_r) = self.einfeldt_speeds(h_l, h_r, un_l, un_r, c_l, c_r);
        let max_speed = s_l.abs().max(s_r.abs());

        // 选择通量区域
        if s_l >= 0.0 {
            // 全在左侧区域
            let flux = self.physical_flux(h_l, un_l, ut_l);
            return Ok(RiemannFlux::from_rotated(
                flux.0,
                flux.1,
                flux.2,
                normal,
                max_speed,
            ));
        }

        if s_r <= 0.0 {
            // 全在右侧区域
            let flux = self.physical_flux(h_r, un_r, ut_r);
            return Ok(RiemannFlux::from_rotated(
                flux.0,
                flux.1,
                flux.2,
                normal,
                max_speed,
            ));
        }

        // 星区域通量
        let (mass, mom_n, mom_t) =
            self.hllc_star_flux(h_l, h_r, un_l, un_r, ut_l, ut_r, s_l, s_r)?;

        Ok(RiemannFlux::from_rotated(mass, mom_n, mom_t, normal, max_speed))
    }

    /// 计算物理通量 (1D)
    #[inline]
    fn physical_flux(&self, h: f64, un: f64, ut: f64) -> (f64, f64, f64) {
        let g = self.params.gravity;
        (
            h * un,
            h * un * un + 0.5 * g * h * h,
            h * un * ut,
        )
    }

    /// 计算 HLLC 星区域通量
    fn hllc_star_flux(
        &self,
        h_l: f64,
        h_r: f64,
        un_l: f64,
        un_r: f64,
        ut_l: f64,
        ut_r: f64,
        s_l: f64,
        s_r: f64,
    ) -> MhResult<(f64, f64, f64)> {
        let g = self.params.gravity;

        // 计算星区域速度
        let q_l = h_l * (un_l - s_l);
        let q_r = h_r * (un_r - s_r);
        let denom = q_l - q_r;
        let threshold = self.params.entropy_threshold((s_r - s_l).abs());

        let s_star = if denom.abs() < threshold {
            0.5 * (un_l + un_r)
        } else {
            let numer = q_l * un_l - q_r * un_r + 0.5 * g * (h_r * h_r - h_l * h_l);
            let s = numer / denom;
            if !s.is_finite() {
                return Err(MhError::Numerical {
                    message: format!("HLLC: s_star 无效: {}", s),
                });
            }
            self.entropy_fix(s.clamp(s_l, s_r), s_l, s_r)
        };

        // 确定使用左侧还是右侧状态
        // P0-001 修复: 添加分母保护，避免 s_l ≈ s_star 或 s_r ≈ s_star 时除零
        let (h_star, ut) = if s_star >= 0.0 {
            let denom_l = s_l - s_star;
            if denom_l.abs() < threshold {
                // 分母过小，回退到左状态
                (h_l, ut_l)
            } else {
                let h_s = h_l * (s_l - un_l) / denom_l;
                (h_s.max(0.0), ut_l)
            }
        } else {
            let denom_r = s_r - s_star;
            if denom_r.abs() < threshold {
                // 分母过小，回退到右状态
                (h_r, ut_r)
            } else {
                let h_s = h_r * (s_r - un_r) / denom_r;
                (h_s.max(0.0), ut_r)
            }
        };

        Ok((
            h_star * s_star,
            h_star * s_star * s_star + 0.5 * g * h_star * h_star,
            h_star * s_star * ut,
        ))
    }

    // ================= 干湿处理 =================

    /// 求解左干右湿
    fn solve_left_dry(
        &self,
        h_r: f64,
        vel_r: DVec2,
        normal: DVec2,
    ) -> MhResult<RiemannFlux> {
        let g = self.params.gravity;
        let c_r = (g * h_r).sqrt();
        let un_r = vel_r.dot(normal);

        // 稀疏波前沿
        let s_front = un_r - 2.0 * c_r;
        if s_front >= 0.0 {
            return Ok(RiemannFlux::ZERO);
        }

        // 干床状态
        let h_star = ((2.0 * c_r + un_r) / 3.0).powi(2) / g;
        if h_star < self.params.h_dry {
            return Ok(RiemannFlux::ZERO);
        }

        let u_star = (2.0 * c_r + un_r) / 3.0;
        let tangent = DVec2::new(-normal.y, normal.x);
        let ut = vel_r.dot(tangent);

        let (mass, mom_n, mom_t) = self.physical_flux(h_star, u_star, ut);
        let max_speed = (un_r + c_r).abs().max(s_front.abs());

        Ok(RiemannFlux::from_rotated(mass, mom_n, mom_t, normal, max_speed))
    }

    /// 求解左湿右干
    fn solve_right_dry(
        &self,
        h_l: f64,
        vel_l: DVec2,
        normal: DVec2,
    ) -> MhResult<RiemannFlux> {
        let g = self.params.gravity;
        let c_l = (g * h_l).sqrt();
        let un_l = vel_l.dot(normal);

        // 稀疏波前沿
        let s_front = un_l + 2.0 * c_l;
        if s_front <= 0.0 {
            return Ok(RiemannFlux::ZERO);
        }

        // 干床状态
        let h_star = ((un_l + 2.0 * c_l) / 3.0).powi(2) / g;
        if h_star < self.params.h_dry {
            return Ok(RiemannFlux::ZERO);
        }

        let u_star = (un_l + 2.0 * c_l) / 3.0;
        let tangent = DVec2::new(-normal.y, normal.x);
        let ut = vel_l.dot(tangent);

        let (mass, mom_n, mom_t) = self.physical_flux(h_star, u_star, ut);
        let max_speed = (un_l - c_l).abs().max(s_front.abs());

        Ok(RiemannFlux::from_rotated(mass, mom_n, mom_t, normal, max_speed))
    }
}

impl RiemannSolver for HllcSolver {
    fn name(&self) -> &'static str {
        "HLLC"
    }

    fn capabilities(&self) -> SolverCapabilities {
        SolverCapabilities {
            handles_dry_wet: true,
            has_entropy_fix: true,
            supports_hydrostatic: true,
            order: 2,
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
        let is_dry_l = h_left <= self.params.h_dry;
        let is_dry_r = h_right <= self.params.h_dry;

        match (is_dry_l, is_dry_r) {
            (true, true) => Ok(RiemannFlux::ZERO),
            (true, false) => self.solve_left_dry(h_right, vel_right, normal),
            (false, true) => self.solve_right_dry(h_left, vel_left, normal),
            (false, false) => self.solve_both_wet(h_left, h_right, vel_left, vel_right, normal),
        }
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

    fn create_solver() -> HllcSolver {
        HllcSolver::from_params(SolverParams::default())
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
        assert!(flux.mass.abs() < 1e-10, "静水应无通量");
    }

    #[test]
    fn test_dam_break() {
        let solver = create_solver();
        let flux = solver
            .solve(10.0, 1.0, DVec2::ZERO, DVec2::ZERO, DVec2::X)
            .unwrap();
        assert!(flux.mass > 0.0, "溃坝应产生正通量");
        assert!(flux.max_wave_speed > 0.0);
    }

    #[test]
    fn test_uniform_flow() {
        let solver = create_solver();
        let vel = DVec2::new(1.0, 0.0);
        let flux = solver.solve(1.0, 1.0, vel, vel, DVec2::X).unwrap();
        assert!(flux.mass > 0.0, "顺流应产生正通量");
    }

    #[test]
    fn test_flux_validity() {
        let solver = create_solver();
        let flux = solver
            .solve(5.0, 3.0, DVec2::new(0.5, 0.2), DVec2::new(-0.3, 0.1), DVec2::X)
            .unwrap();
        assert!(flux.is_valid());
    }
}
