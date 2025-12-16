// marihydro/crates/mh_physics/src/schemes/riemann/hllc.rs
//! HLLC 近似黎曼求解器
//!
//! HLLC (Harten-Lax-van Leer-Contact) 求解器能够正确处理接触间断和干湿界面，
//! 为浅水方程提供高精度通量计算。本实现支持 f32/f64 精度切换和 GPU 后端扩展。
//!
//! # 精度支持
//!
//! - `CpuBackend<f32>`: GPU 加速模式，内存减半
//! - `CpuBackend<f64>`: 默认高精度模式
//!
//! # 核心算法
//!
//! 通过波速估计、星区域通量计算和熵修正，在接触间断处保持高分辨率。

use crate::schemes::riemann::traits::{
    RiemannError, RiemannFlux, RiemannSolver, SolverCapabilities, SolverParams,
};
use mh_runtime::{Backend, CpuBackend, RuntimeScalar, Vector2D};
use num_traits::{Float, FromPrimitive};

/// HLLC 求解器（Backend 泛型化）
///
/// 支持 f32/f64 精度和 GPU 后端，消除所有硬编码类型。
#[derive(Debug, Clone)]
pub struct HllcSolver<B: Backend> {
    params: SolverParams<B::Scalar>,
    gravity: B::Scalar, // 缓存重力加速度，避免重复访问
}

impl<B: Backend> HllcSolver<B> {
    /// 创建新的 HLLC 求解器
    pub fn new(params: &SolverParams<B::Scalar>, gravity: B::Scalar) -> Self {
        Self {
            params: params.clone(),
            gravity,
        }
    }

    /// 获取参数引用
    pub fn params(&self) -> &SolverParams<B::Scalar> {
        &self.params
    }

    /// Einfeldt 波速估计
    ///
    /// 使用 Roe 平均计算左右波速，确保鲁棒性。
    #[inline]
    fn einfeldt_speeds(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        un_l: B::Scalar,
        un_r: B::Scalar,
        c_l: B::Scalar,
        c_r: B::Scalar,
    ) -> (B::Scalar, B::Scalar) {
        let sqrt_h_l = Float::sqrt(h_l);
        let sqrt_h_r = Float::sqrt(h_r);
        let sum = sqrt_h_l + sqrt_h_r + self.params.flux_eps;

        // Roe 平均
        let h_roe = (h_l + h_r) * B::Scalar::HALF;
        let u_roe = (sqrt_h_l * un_l + sqrt_h_r * un_r) / sum;
        let c_roe = Float::sqrt(self.gravity * h_roe);

        (
            Float::min(un_l - c_l, u_roe - c_roe),
            Float::max(un_r + c_r, u_roe + c_roe),
        )
    }

    /// 熵修正
    ///
    /// 修正接近零的特征速度，避免音速跨越导致的数值振荡。
    #[inline]
    fn entropy_fix(&self, s_star: B::Scalar, s_l: B::Scalar, s_r: B::Scalar) -> B::Scalar {
        let threshold = FromPrimitive::from_f64(1e-14).unwrap_or(B::Scalar::MIN_POSITIVE);
        if Float::abs(s_star) < threshold {
            return B::Scalar::ZERO;
        }

        let eps = self.params.entropy_threshold(Float::abs(s_r - s_l));
        if Float::abs(s_star) < eps {
            Float::signum(s_star) * eps
        } else {
            s_star
        }
    }

    /// 物理通量（1D 形式）
    #[inline]
    fn physical_flux(
        &self,
        h: B::Scalar,
        un: B::Scalar,
        ut: B::Scalar,
    ) -> (B::Scalar, B::Scalar, B::Scalar) {
        (
            h * un,
            h * un * un + B::Scalar::HALF * self.gravity * h * h,
            h * un * ut,
        )
    }

    /// HLLC 星区域通量计算
    #[inline]
    fn hllc_star_flux(
        &self,
        h_l: B::Scalar,
        h_r: B::Scalar,
        un_l: B::Scalar,
        un_r: B::Scalar,
        ut_l: B::Scalar,
        ut_r: B::Scalar,
        s_l: B::Scalar,
        s_r: B::Scalar,
    ) -> Result<(B::Scalar, B::Scalar, B::Scalar), RiemannError> {
        let q_l = h_l * (s_l - un_l);
        let q_r = h_r * (s_r - un_r);
        let denom = q_l - q_r;
        let threshold = self.params.entropy_threshold(Float::abs(s_r - s_l));

        let s_star = if Float::abs(denom) < threshold {
            (un_l + un_r) * B::Scalar::HALF
        } else {
            let numer = h_l * un_l * (s_l - un_l)
                - h_r * un_r * (s_r - un_r)
                + B::Scalar::HALF * self.gravity * (h_r * h_r - h_l * h_l);
            let s = numer / denom;
            if !Float::is_finite(s) {
                return Err(RiemannError::Numerical {
                    message: "Invalid s_star calculation".to_string(),
                });
            }
            self.entropy_fix(s.clamp_value(s_l, s_r), s_l, s_r)
        };

        // 根据星区域速度选择左右状态
        let (h_star, ut_star) = if s_star >= B::Scalar::ZERO {
            let denom_l = s_l - s_star;
            if Float::abs(denom_l) < threshold {
                (h_l, ut_l)
            } else {
                let h_s = h_l * (s_l - un_l) / denom_l;
                (Float::max(h_s, B::Scalar::ZERO), ut_l)
            }
        } else {
            let denom_r = s_r - s_star;
            if Float::abs(denom_r) < threshold {
                (h_r, ut_r)
            } else {
                let h_s = h_r * (s_r - un_r) / denom_r;
                (Float::max(h_s, B::Scalar::ZERO), ut_r)
            }
        };

        Ok((
            h_star * s_star,
            h_star * s_star * s_star + B::Scalar::HALF * self.gravity * h_star * h_star,
            h_star * s_star * ut_star,
        ))
    }

    /// 左干右湿情况求解
    #[inline]
    fn solve_left_dry(
        &self,
        h_r: B::Scalar,
        vel_r: B::Vector2D,
        normal: B::Vector2D,
    ) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        // 使用 Backend 几何方法求切向量
        let tangent = B::vec2_new(-normal.y(), normal.x());
        let un_r = B::vec2_dot(&vel_r, &normal);
        let ut_r = B::vec2_dot(&vel_r, &tangent);
        let c_r = Float::sqrt(self.gravity * h_r);

        let s_front = un_r - B::Scalar::TWO * c_r;
        if s_front >= B::Scalar::ZERO {
            return Ok(RiemannFlux::zero());
        }

        // 干床状态计算
        let three: B::Scalar = FromPrimitive::from_f64(3.0).unwrap();
        let factor = (B::Scalar::TWO * c_r + un_r) / three;
        let h_star = Float::powi(factor, 2) / self.gravity;

        if h_star < self.params.h_dry {
            return Ok(RiemannFlux::zero());
        }

        let u_star = factor;

        let (mass, mom_n, mom_t) = self.physical_flux(h_star, u_star, ut_r);
        let max_speed = Float::max(Float::abs(un_r + c_r), Float::abs(s_front));

        Ok(RiemannFlux::from_rotated::<B>(
            mass,
            mom_n,
            mom_t,
            normal,
            max_speed,
            self.gravity,
        ))
    }

    /// 右干左湿情况求解
    #[inline]
    fn solve_right_dry(
        &self,
        h_l: B::Scalar,
        vel_l: B::Vector2D,
        normal: B::Vector2D,
    ) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        let tangent = B::vec2_new(-normal.y(), normal.x());
        let un_l = B::vec2_dot(&vel_l, &normal);
        let ut_l = B::vec2_dot(&vel_l, &tangent);
        let c_l = Float::sqrt(self.gravity * h_l);

        let s_front = un_l + B::Scalar::TWO * c_l;
        if s_front <= B::Scalar::ZERO {
            return Ok(RiemannFlux::zero());
        }

        let three: B::Scalar = FromPrimitive::from_f64(3.0).unwrap();
        let factor = (un_l + B::Scalar::TWO * c_l) / three;
        let h_star = Float::powi(factor, 2) / self.gravity;

        if h_star < self.params.h_dry {
            return Ok(RiemannFlux::zero());
        }

        let u_star = factor;

        let (mass, mom_n, mom_t) = self.physical_flux(h_star, u_star, ut_l);
        let max_speed = Float::max(Float::abs(un_l - c_l), Float::abs(s_front));

        Ok(RiemannFlux::from_rotated::<B>(
            mass,
            mom_n,
            mom_t,
            normal,
            max_speed,
            self.gravity,
        ))
    }
}

impl<B: Backend> RiemannSolver for HllcSolver<B> {
    type Scalar = B::Scalar;
    type Vector2D = B::Vector2D;

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
        h_left: B::Scalar,
        h_right: B::Scalar,
        vel_left: B::Vector2D,
        vel_right: B::Vector2D,
        normal: B::Vector2D,
    ) -> Result<RiemannFlux<B::Scalar>, RiemannError> {
        let is_dry_l = h_left <= self.params.h_dry;
        let is_dry_r = h_right <= self.params.h_dry;

        match (is_dry_l, is_dry_r) {
            (true, true) => Ok(RiemannFlux::zero()),
            (true, false) => self.solve_left_dry(h_right, vel_right, normal),
            (false, true) => self.solve_right_dry(h_left, vel_left, normal),
            (false, false) => {
                let tangent = B::vec2_new(-normal.y(), normal.x());
                let un_l = B::vec2_dot(&vel_left, &normal);
                let un_r = B::vec2_dot(&vel_right, &normal);
                let ut_l = B::vec2_dot(&vel_left, &tangent);
                let ut_r = B::vec2_dot(&vel_right, &tangent);
                let c_l = Float::sqrt(self.gravity * h_left);
                let c_r = Float::sqrt(self.gravity * h_right);

                let (s_l, s_r) = self.einfeldt_speeds(h_left, h_right, un_l, un_r, c_l, c_r);
                let (mass, mom_n, mom_t) = self.hllc_star_flux(
                    h_left, h_right, un_l, un_r, ut_l, ut_r, s_l, s_r,
                )?;
                let max_speed = Float::max(Float::abs(s_l), Float::abs(s_r));

                Ok(RiemannFlux::from_rotated::<B>(
                    mass,
                    mom_n,
                    mom_t,
                    normal,
                    max_speed,
                    self.gravity,
                ))
            }
        }
    }

    fn gravity(&self) -> B::Scalar {
        self.gravity
    }

    fn dry_threshold(&self) -> B::Scalar {
        self.params.h_dry
    }
}

/// f64 类型别名（向后兼容）
pub type HllcSolverF64 = HllcSolver<CpuBackend<f64>>;

/// f32 类型别名（高性能模式）
pub type HllcSolverF32 = HllcSolver<CpuBackend<f32>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemes::riemann::SolverParams;
    use mh_runtime::CpuBackend;

    fn create_solver<B: Backend>(gravity: B::Scalar) -> HllcSolver<B> {
        let mut params = SolverParams::default();
        params.h_dry = B::Scalar::from_f64(1e-6).unwrap();
        HllcSolver::new(&params, gravity)
    }

    #[test]
    fn test_both_dry_f64() {
        let solver = create_solver::<CpuBackend<f64>>(9.81f64);
        let flux = solver
            .solve(
                0.0f64,
                0.0f64,
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
            )
            .unwrap();
        assert_eq!(flux.mass, 0.0);
    }

    #[test]
    fn test_still_water_f64() {
        let solver = create_solver::<CpuBackend<f64>>(9.81f64);
        let flux = solver
            .solve(
                10.0f64,
                10.0f64,
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
            )
            .unwrap();
        assert!(flux.mass.abs() < 1e-10);
    }

    #[test]
    fn test_dam_break_f64() {
        let solver = create_solver::<CpuBackend<f64>>(9.81f64);
        let flux = solver
            .solve(
                10.0f64,
                1.0f64,
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
            )
            .unwrap();
        assert!(flux.mass.abs() > 0.1);
        assert!(flux.max_wave_speed > 0.0);
        assert!(flux.is_valid());
    }

    #[test]
    fn test_f32_precision() {
        let solver = create_solver::<CpuBackend<f32>>(9.81f32);
        let flux = solver
            .solve(
                5.0f32,
                3.0f32,
                CpuBackend::<f32>::vec2_new(0.5, 0.2),
                CpuBackend::<f32>::vec2_new(-0.3, 0.1),
                CpuBackend::<f32>::vec2_new(1.0, 0.0),
            )
            .unwrap();
        assert!(flux.is_valid());
        assert_eq!(std::mem::size_of_val(&flux.mass), 4); // f32验证
    }

    #[test]
    fn test_left_dry_f64() {
        let solver = create_solver::<CpuBackend<f64>>(9.81f64);
        let flux = solver
            .solve(
                0.0f64,
                5.0f64,
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(-1.0, 0.0),
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
            )
            .unwrap();
        assert!(flux.is_valid());
    }

    #[test]
    fn test_right_dry_f64() {
        let solver = create_solver::<CpuBackend<f64>>(9.81f64);
        let flux = solver
            .solve(
                5.0f64,
                0.0f64,
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
                CpuBackend::<f64>::vec2_new(0.0, 0.0),
                CpuBackend::<f64>::vec2_new(1.0, 0.0),
            )
            .unwrap();
        assert!(flux.is_valid());
    }
}