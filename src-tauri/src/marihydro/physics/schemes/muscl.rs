// src-tauri/src/marihydro/physics/schemes/muscl.rs

use super::{ConservedVars, PrimitiveVars};
use crate::marihydro::infra::config::SlopeLimiterType;

/// 计算斜率限制器 phi(r)
#[inline(always)]
fn slope_limiter(r: f64, kind: SlopeLimiterType) -> f64 {
    match kind {
        SlopeLimiterType::FirstOrder => 0.0, // 降级为一阶
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

/// MUSCL 重构 + 静水修正
/// 输入: 4点模版 (i-1, i, i+1, i+2)
/// 输出: 界面 i+1/2 处的 (State_L, State_R)
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_interface(
    p_im1: &PrimitiveVars, // i-1
    p_i: &PrimitiveVars,   // i
    p_ip1: &PrimitiveVars, // i+1
    p_ip2: &PrimitiveVars, // i+2
    limiter: SlopeLimiterType,
    h_min: f64,
) -> (ConservedVars, ConservedVars) {
    // --- 1. 重构左状态 (Left State at i+1/2) ---
    // 基于 Cell i，利用 i-1, i, i+1 计算梯度
    let d_eta_l = compute_slope(p_im1.eta, p_i.eta, p_ip1.eta, limiter);
    let d_u_l = compute_slope(p_im1.u, p_i.u, p_ip1.u, limiter);
    let d_v_l = compute_slope(p_im1.v, p_i.v, p_ip1.v, limiter);
    let d_c_l = compute_slope(p_im1.c, p_i.c, p_ip1.c, limiter);

    let eta_l = p_i.eta + 0.5 * d_eta_l;
    let u_l = p_i.u + 0.5 * d_u_l;
    let v_l = p_i.v + 0.5 * d_v_l;
    let c_l = p_i.c + 0.5 * d_c_l;

    // --- 2. 重构右状态 (Right State at i+1/2) ---
    // 基于 Cell i+1，利用 i, i+1, i+2 计算梯度
    let d_eta_r = compute_slope(p_i.eta, p_ip1.eta, p_ip2.eta, limiter);
    let d_u_r = compute_slope(p_i.u, p_ip1.u, p_ip2.u, limiter);
    let d_v_r = compute_slope(p_i.v, p_ip1.v, p_ip2.v, limiter);
    let d_c_r = compute_slope(p_i.c, p_ip1.c, p_ip2.c, limiter);

    // 注意方向：这是 i+1 单元的左界面，相当于中心减去半个梯度
    let eta_r = p_ip1.eta - 0.5 * d_eta_r;
    let u_r = p_ip1.u - 0.5 * d_u_r;
    let v_r = p_ip1.v - 0.5 * d_v_r;
    let c_r = p_ip1.c - 0.5 * d_c_r;

    // --- 3. Audusse 地形修正 (Hydrostatic Fix) ---
    // 界面地形取两侧地形的最大值 (构建一个台阶)
    // 注意：这里的 z 使用的是原单元中心高程。
    // 更高阶的方法应对 z 也进行重构，但标准 Audusse 方法使用中心 z 即可保证 C-property
    let z_face = p_i.z.max(p_ip1.z);

    // 修正水深 h* = max(0, eta_reconstructed - z_face)
    let h_l_star = (eta_l - z_face).max(0.0);
    let h_r_star = (eta_r - z_face).max(0.0);

    // 构建守恒变量
    // 如果水深过小，强制速度为0 (Desingularization)
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
