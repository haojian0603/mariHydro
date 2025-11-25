use super::{ConservedVars, FluxVars};
use crate::marihydro::physics::schemes::hllc::ComputeAxis::{X, Y};

#[derive(Clone, Copy)]
pub enum ComputeAxis {
    X,
    Y,
}

/// 1D 计算辅助结构
struct State1D {
    h: f64,
    u_n: f64, // 法向速度
    u_t: f64, // 切向速度
    c: f64,
}

/// HLLC 求解器（包含Toro干湿边界处理）
///
/// # 算法
/// 1. 检查干湿状态：双干/左干/右干/双湿
/// 2. 双湿情况：标准HLLC三波模型
/// 3. 干湿界面：Toro单侧稀疏波解析解
pub fn solve_hllc(
    u_l: &ConservedVars,
    u_r: &ConservedVars,
    axis: ComputeAxis,
    g: f64,
    h_min: f64,
) -> FluxVars {
    // === 干湿边界处理（Toro） ===

    // 1. 双干：无通量
    if u_l.h < h_min && u_r.h < h_min {
        return FluxVars::default();
    }

    // 2. 左干右湿
    if u_l.h < h_min && u_r.h >= h_min {
        return solve_dry_wet_interface(u_r, axis, g, h_min, true);
    }

    // 3. 左湿右干
    if u_l.h >= h_min && u_r.h < h_min {
        return solve_dry_wet_interface(u_l, axis, g, h_min, false);
    }

    // === 双湿：标准HLLC ===

    // 4. 旋转坐标系 -> 1D
    let to_1d = |u: &ConservedVars| -> State1D {
        let inv_h = 1.0 / u.h;
        match axis {
            X => State1D {
                h: u.h,
                u_n: u.hu * inv_h,
                u_t: u.hv * inv_h,
                c: u.hc * inv_h,
            },
            Y => State1D {
                h: u.h,
                u_n: u.hv * inv_h,
                u_t: u.hu * inv_h,
                c: u.hc * inv_h,
            },
        }
    };

    let s_l = to_1d(u_l);
    let s_r = to_1d(u_r);

    // 5. 波速估计（Einfeldt）
    let a_l = (g * s_l.h).sqrt();
    let a_r = (g * s_r.h).sqrt();
    let u_l_n = s_l.u_n;
    let u_r_n = s_r.u_n;

    let sl = (u_l_n - a_l).min(u_r_n - a_r);
    let sr = (u_l_n + a_l).max(u_r_n + a_r);

    // 6. 计算通量（1D HLLC）
    let flux_1d = if sl >= 0.0 {
        physical_flux(&s_l, g)
    } else if sr <= 0.0 {
        physical_flux(&s_r, g)
    } else {
        // 星区（接触间断）
        let h_l = s_l.h;
        let h_r = s_r.h;

        // 接触波速度 S*
        let num = u_r_n * h_r * (sr - u_r_n) - u_l_n * h_l * (sl - u_l_n)
            + 0.5 * g * (h_l * h_l - h_r * h_r);
        let den = h_r * (sr - u_r_n) - h_l * (sl - u_l_n);
        let s_star = if den.abs() < 1e-10 { 0.0 } else { num / den };

        // 选择K侧（左或右）
        let (sk, state_k, flux_k) = if s_star >= 0.0 {
            (sl, &s_l, physical_flux(&s_l, g))
        } else {
            (sr, &s_r, physical_flux(&s_r, g))
        };

        // HLLC通量公式: F* = F_K + S_K * (U*_K - U_K)
        let factor = state_k.h * (sk - state_k.u_n) / (sk - s_star);

        let ds_h = factor - state_k.h;
        let ds_mom_n = factor * s_star - state_k.h * state_k.u_n;
        let ds_mom_t = factor * state_k.u_t - state_k.h * state_k.u_t;
        let ds_sed = factor * state_k.c - state_k.h * state_k.c;

        FluxVars {
            mass: flux_k.mass + sk * ds_h,
            x_mom: flux_k.x_mom + sk * ds_mom_n,
            y_mom: flux_k.y_mom + sk * ds_mom_t,
            sed: flux_k.sed + sk * ds_sed,
        }
    };

    // 7. 反向旋转（1D -> 2D）
    match axis {
        X => flux_1d,
        Y => FluxVars {
            mass: flux_1d.mass,
            x_mom: flux_1d.y_mom,
            y_mom: flux_1d.x_mom,
            sed: flux_1d.sed,
        },
    }
}

/// Toro干湿界面求解器
///
/// # 参数
/// - `u_wet`: 湿侧守恒变量
/// - `axis`: 计算方向
/// - `g`: 重力加速度
/// - `h_min`: 最小水深
/// - `wet_on_right`: true=湿在右，false=湿在左
fn solve_dry_wet_interface(
    u_wet: &ConservedVars,
    axis: ComputeAxis,
    g: f64,
    h_min: f64,
    wet_on_right: bool,
) -> FluxVars {
    // 提取1D状态
    let (h, u_n, u_t, c) = match axis {
        X => {
            let inv_h = 1.0 / u_wet.h.max(h_min);
            (
                u_wet.h,
                u_wet.hu * inv_h,
                u_wet.hv * inv_h,
                u_wet.hc * inv_h,
            )
        }
        Y => {
            let inv_h = 1.0 / u_wet.h.max(h_min);
            (
                u_wet.h,
                u_wet.hv * inv_h,
                u_wet.hu * inv_h,
                u_wet.hc * inv_h,
            )
        }
    };

    let c_wave = (g * h).sqrt();

    // Toro稀疏波解析解
    let flux_1d = if wet_on_right {
        // 右侧湿，左侧干，稀疏波向左传播
        let s_head = u_n - 2.0 * c_wave; // 稀疏波波头
        let s_tail = u_n + c_wave; // 稀疏波波尾

        if s_head >= 0.0 {
            // 稀疏波完全在界面右侧，界面处为干底
            FluxVars::default()
        } else if s_tail <= 0.0 {
            // 稀疏波完全在界面左侧，使用湿侧通量
            physical_flux_components(h, u_n, u_t, c, g)
        } else {
            // 界面在稀疏波内部，计算星区状态
            // 星区速度: u* = (1/3)(u_R - 2c_R)（向干侧传播）
            let u_star = (u_n - 2.0 * c_wave) / 3.0;

            // 星区水深: h* = (1/g) * ((u* + c_R)/2)²
            let h_star = ((u_star + c_wave) / 2.0).powi(2) / g;

            // 星区通量（标量保持湿侧值）
            physical_flux_components(h_star, u_star, u_t, c, g)
        }
    } else {
        // 左侧湿，右侧干，稀疏波向右传播
        let s_head = u_n + 2.0 * c_wave;
        let s_tail = u_n - c_wave;

        if s_head <= 0.0 {
            FluxVars::default()
        } else if s_tail >= 0.0 {
            physical_flux_components(h, u_n, u_t, c, g)
        } else {
            let u_star = (u_n + 2.0 * c_wave) / 3.0;
            let h_star = ((c_wave - u_star) / 2.0).powi(2) / g;

            physical_flux_components(h_star, u_star, u_t, c, g)
        }
    };

    // 旋转回2D
    match axis {
        X => flux_1d,
        Y => FluxVars {
            mass: flux_1d.mass,
            x_mom: flux_1d.y_mom,
            y_mom: flux_1d.x_mom,
            sed: flux_1d.sed,
        },
    }
}

#[inline(always)]
fn physical_flux(s: &State1D, g: f64) -> FluxVars {
    let qn = s.h * s.u_n;
    FluxVars {
        mass: qn,
        x_mom: qn * s.u_n + 0.5 * g * s.h * s.h,
        y_mom: qn * s.u_t,
        sed: qn * s.c,
    }
}

#[inline(always)]
fn physical_flux_components(h: f64, u_n: f64, u_t: f64, c: f64, g: f64) -> FluxVars {
    let qn = h * u_n;
    FluxVars {
        mass: qn,
        x_mom: qn * u_n + 0.5 * g * h * h,
        y_mom: qn * u_t,
        sed: qn * c,
    }
}
