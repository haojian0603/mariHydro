// src-tauri/src/marihydro/physics/schemes/muscl.rs

//! MUSCL 斜率重构
//!
//! 本模块提供两种实现：
//! - 结构化网格版本：使用 4 点模板的 `reconstruct_interface`
//! - 非结构化网格版本：使用 `hydrostatic::muscl_reconstruct`

use super::{ConservedVars, PrimitiveVars};

/// 斜坡限制器类型（结构化网格版本）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SlopeLimiterType {
    FirstOrder,
    Minmod,
    #[default]
    VanLeer,
}

impl SlopeLimiterType {
    /// 转换为 hydrostatic 模块的 SlopeLimiter 类型
    pub fn to_hydrostatic(&self) -> super::hydrostatic::SlopeLimiter {
        match self {
            Self::FirstOrder => super::hydrostatic::SlopeLimiter::None,
            Self::Minmod => super::hydrostatic::SlopeLimiter::MinMod,
            Self::VanLeer => super::hydrostatic::SlopeLimiter::VanLeer,
        }
    }
}

/// 计算斜率限制器 phi(r)
#[inline(always)]
fn slope_limiter(r: f64, kind: SlopeLimiterType) -> f64 {
    match kind {
        SlopeLimiterType::FirstOrder => 0.0,
        SlopeLimiterType::Minmod => r.max(0.0).min(1.0),
        SlopeLimiterType::VanLeer => {
            let abs_r = r.abs();
            (r + abs_r) / (1.0 + abs_r)
        }
    }
}

/// 计算梯度 Delta
#[inline(always)]
fn compute_slope(left: f64, center: f64, right: f64, kind: SlopeLimiterType) -> f64 {
    let d_l = center - left;
    let d_r = right - center;
    if d_r.abs() < 1e-10 {
        return 0.0;
    }
    let r = d_l / d_r;
    slope_limiter(r, kind) * d_r
}

/// MUSCL 重构 + 静水修正（结构化网格版本）
///
/// # 注意
/// 此函数仅用于**结构化网格**（规则网格）。
/// 对于非结构化网格，请使用 `hydrostatic::muscl_reconstruct`。
///
/// # 参数
/// - `p_im1`, `p_i`, `p_ip1`, `p_ip2`: 4 点模板 (i-1, i, i+1, i+2)
/// - `limiter`: 斜坡限制器类型
/// - `h_min`: 最小水深阈值
///
/// # 返回
/// 界面 i+1/2 处的左右状态 (State_L, State_R)
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_interface(
    p_im1: &PrimitiveVars,
    p_i: &PrimitiveVars,
    p_ip1: &PrimitiveVars,
    p_ip2: &PrimitiveVars,
    limiter: SlopeLimiterType,
    h_min: f64,
) -> (ConservedVars, ConservedVars) {
    let d_eta_l = compute_slope(p_im1.eta, p_i.eta, p_ip1.eta, limiter);
    let d_u_l = compute_slope(p_im1.u, p_i.u, p_ip1.u, limiter);
    let d_v_l = compute_slope(p_im1.v, p_i.v, p_ip1.v, limiter);
    let d_c_l = compute_slope(p_im1.c, p_i.c, p_ip1.c, limiter);

    let eta_l = p_i.eta + 0.5 * d_eta_l;
    let u_l = p_i.u + 0.5 * d_u_l;
    let v_l = p_i.v + 0.5 * d_v_l;
    let c_l = p_i.c + 0.5 * d_c_l;

    let d_eta_r = compute_slope(p_i.eta, p_ip1.eta, p_ip2.eta, limiter);
    let d_u_r = compute_slope(p_i.u, p_ip1.u, p_ip2.u, limiter);
    let d_v_r = compute_slope(p_i.v, p_ip1.v, p_ip2.v, limiter);
    let d_c_r = compute_slope(p_i.c, p_ip1.c, p_ip2.c, limiter);

    let eta_r = p_ip1.eta - 0.5 * d_eta_r;
    let u_r = p_ip1.u - 0.5 * d_u_r;
    let v_r = p_ip1.v - 0.5 * d_v_r;
    let c_r = p_ip1.c - 0.5 * d_c_r;

    let z_face = p_i.z.max(p_ip1.z);

    let h_l_star = (eta_l - z_face).max(0.0);
    let h_r_star = (eta_r - z_face).max(0.0);

    let make_conserved = |h: f64, u: f64, v: f64, c: f64| -> ConservedVars {
        if h < h_min {
            ConservedVars {
                h,
                hu: 0.0,
                hv: 0.0,
                hc: 0.0,
            }
        } else {
            ConservedVars {
                h,
                hu: h * u,
                hv: h * v,
                hc: h * c,
            }
        }
    };

    (
        make_conserved(h_l_star, u_l, v_l, c_l),
        make_conserved(h_r_star, u_r, v_r, c_r),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slope_limiters() {
        assert_eq!(slope_limiter(0.5, SlopeLimiterType::FirstOrder), 0.0);
        assert_eq!(slope_limiter(0.5, SlopeLimiterType::Minmod), 0.5);

        let vl = slope_limiter(1.0, SlopeLimiterType::VanLeer);
        assert!((vl - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_slope_limiter_type_conversion() {
        let vl = SlopeLimiterType::VanLeer;
        let hl = vl.to_hydrostatic();
        assert_eq!(hl, super::super::hydrostatic::SlopeLimiter::VanLeer);
    }

    #[test]
    fn test_reconstruct_flat() {
        let p = PrimitiveVars {
            h: 1.0,
            u: 0.0,
            v: 0.0,
            c: 0.0,
            z: 0.0,
            eta: 1.0,
        };

        let (left, right) = reconstruct_interface(&p, &p, &p, &p, SlopeLimiterType::VanLeer, 1e-6);

        assert_eq!(left.h, 1.0);
        assert_eq!(right.h, 1.0);
    }
}
