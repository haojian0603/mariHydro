// crates/mh_runtime/src/simd/kernels.rs

//! SIMD 计算核心
//!
//! 提供浅水方程求解的关键 SIMD 核心：
//! - 通量计算
//! - 状态更新
//! - 斜率限制器
//!
//! # 多版本支持
//!
//! 每个核心有三个版本：
//! - AVX-512 (`_avx512` 后缀)
//! - AVX2 (`_avx2` 后缀)
//! - 标量 (`_scalar` 后缀)
//!
//! 分发函数自动选择最优版本。

#![allow(unsafe_code)]

use super::{simd_capability, SimdCapability};

// ============================================================================
// 通量计算核心
// ============================================================================

/// 批量计算浅水方程通量 (f64)
///
/// # 参数
/// - `h`: 水深数组
/// - `hu`: x动量数组
/// - `hv`: y动量数组
/// - `nx`: 法向量x分量
/// - `ny`: 法向量y分量
/// - `g`: 重力加速度
/// - `out_mass`: 质量通量输出
/// - `out_mom_x`: x动量通量输出
/// - `out_mom_y`: y动量通量输出
pub fn compute_flux_batch_f64(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    nx: &[f64],
    ny: &[f64],
    g: f64,
    out_mass: &mut [f64],
    out_mom_x: &mut [f64],
    out_mom_y: &mut [f64],
) {
    let n = h.len();
    debug_assert_eq!(hu.len(), n);
    debug_assert_eq!(hv.len(), n);
    debug_assert_eq!(nx.len(), n);
    debug_assert_eq!(ny.len(), n);
    debug_assert_eq!(out_mass.len(), n);
    debug_assert_eq!(out_mom_x.len(), n);
    debug_assert_eq!(out_mom_y.len(), n);

    match simd_capability() {
        SimdCapability::Avx512 => {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: 已检测 AVX-512 支持
                unsafe {
                    compute_flux_avx512_f64(h, hu, hv, nx, ny, g, out_mass, out_mom_x, out_mom_y);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                compute_flux_scalar_f64(h, hu, hv, nx, ny, g, out_mass, out_mom_x, out_mom_y);
            }
        }
        SimdCapability::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: 已检测 AVX2 支持
                unsafe {
                    compute_flux_avx2_f64(h, hu, hv, nx, ny, g, out_mass, out_mom_x, out_mom_y);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                compute_flux_scalar_f64(h, hu, hv, nx, ny, g, out_mass, out_mom_x, out_mom_y);
            }
        }
        _ => {
            compute_flux_scalar_f64(h, hu, hv, nx, ny, g, out_mass, out_mom_x, out_mom_y);
        }
    }
}

/// 标量版本通量计算
fn compute_flux_scalar_f64(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    nx: &[f64],
    ny: &[f64],
    g: f64,
    out_mass: &mut [f64],
    out_mom_x: &mut [f64],
    out_mom_y: &mut [f64],
) {
    let n = h.len();
    for i in 0..n {
        let hi = h[i];
        if hi <= 0.0 {
            out_mass[i] = 0.0;
            out_mom_x[i] = 0.0;
            out_mom_y[i] = 0.0;
            continue;
        }

        let u = hu[i] / hi;
        let v = hv[i] / hi;
        let nxi = nx[i];
        let nyi = ny[i];

        // 法向速度
        let un = u * nxi + v * nyi;

        // 质量通量: h * un
        let mass_flux = hi * un;

        // 压力项
        let pressure = 0.5 * g * hi * hi;

        // 动量通量
        out_mass[i] = mass_flux;
        out_mom_x[i] = hu[i] * un + pressure * nxi;
        out_mom_y[i] = hv[i] * un + pressure * nyi;
    }
}

/// AVX2 版本通量计算
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_flux_avx2_f64(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    nx: &[f64],
    ny: &[f64],
    g: f64,
    out_mass: &mut [f64],
    out_mom_x: &mut [f64],
    out_mom_y: &mut [f64],
) {
    use std::arch::x86_64::*;

    let n = h.len();
    let lanes = 4; // AVX2 处理 4 个 f64
    let aligned_n = n - (n % lanes);

    let g_vec = _mm256_set1_pd(g);
    let half = _mm256_set1_pd(0.5);
    let epsilon = _mm256_set1_pd(1e-12);

    let mut i = 0;
    while i < aligned_n {
        // 加载数据
        let h_vec = _mm256_loadu_pd(h.as_ptr().add(i));
        let hu_vec = _mm256_loadu_pd(hu.as_ptr().add(i));
        let hv_vec = _mm256_loadu_pd(hv.as_ptr().add(i));
        let nx_vec = _mm256_loadu_pd(nx.as_ptr().add(i));
        let ny_vec = _mm256_loadu_pd(ny.as_ptr().add(i));

        // 安全除法: h + epsilon
        let h_safe = _mm256_max_pd(h_vec, epsilon);
        let inv_h = _mm256_div_pd(_mm256_set1_pd(1.0), h_safe);

        // 速度
        let u_vec = _mm256_mul_pd(hu_vec, inv_h);
        let v_vec = _mm256_mul_pd(hv_vec, inv_h);

        // 法向速度: un = u*nx + v*ny
        let un_vec = _mm256_fmadd_pd(u_vec, nx_vec, _mm256_mul_pd(v_vec, ny_vec));

        // 质量通量: h * un
        let mass_flux = _mm256_mul_pd(h_vec, un_vec);

        // 压力: 0.5 * g * h^2
        let h_sq = _mm256_mul_pd(h_vec, h_vec);
        let pressure = _mm256_mul_pd(half, _mm256_mul_pd(g_vec, h_sq));

        // 动量通量
        let mom_x = _mm256_fmadd_pd(hu_vec, un_vec, _mm256_mul_pd(pressure, nx_vec));
        let mom_y = _mm256_fmadd_pd(hv_vec, un_vec, _mm256_mul_pd(pressure, ny_vec));

        // 掩码：h > 0
        let mask = _mm256_cmp_pd(h_vec, _mm256_setzero_pd(), _CMP_GT_OQ);
        let mass_final = _mm256_and_pd(mass_flux, mask);
        let mom_x_final = _mm256_and_pd(mom_x, mask);
        let mom_y_final = _mm256_and_pd(mom_y, mask);

        // 存储
        _mm256_storeu_pd(out_mass.as_mut_ptr().add(i), mass_final);
        _mm256_storeu_pd(out_mom_x.as_mut_ptr().add(i), mom_x_final);
        _mm256_storeu_pd(out_mom_y.as_mut_ptr().add(i), mom_y_final);

        i += lanes;
    }

    // 处理尾部
    for i in aligned_n..n {
        let hi = h[i];
        if hi <= 0.0 {
            out_mass[i] = 0.0;
            out_mom_x[i] = 0.0;
            out_mom_y[i] = 0.0;
            continue;
        }
        let u = hu[i] / hi;
        let v = hv[i] / hi;
        let un = u * nx[i] + v * ny[i];
        let pressure = 0.5 * g * hi * hi;
        out_mass[i] = hi * un;
        out_mom_x[i] = hu[i] * un + pressure * nx[i];
        out_mom_y[i] = hv[i] * un + pressure * ny[i];
    }
}

/// AVX-512 版本通量计算
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn compute_flux_avx512_f64(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    nx: &[f64],
    ny: &[f64],
    g: f64,
    out_mass: &mut [f64],
    out_mom_x: &mut [f64],
    out_mom_y: &mut [f64],
) {
    use std::arch::x86_64::*;

    let n = h.len();
    let lanes = 8; // AVX-512 处理 8 个 f64
    let aligned_n = n - (n % lanes);

    let g_vec = _mm512_set1_pd(g);
    let half = _mm512_set1_pd(0.5);
    let zero = _mm512_setzero_pd();
    let epsilon = _mm512_set1_pd(1e-12);

    let mut i = 0;
    while i < aligned_n {
        // 加载数据
        let h_vec = _mm512_loadu_pd(h.as_ptr().add(i));
        let hu_vec = _mm512_loadu_pd(hu.as_ptr().add(i));
        let hv_vec = _mm512_loadu_pd(hv.as_ptr().add(i));
        let nx_vec = _mm512_loadu_pd(nx.as_ptr().add(i));
        let ny_vec = _mm512_loadu_pd(ny.as_ptr().add(i));

        // 安全除法
        let h_safe = _mm512_max_pd(h_vec, epsilon);
        let inv_h = _mm512_div_pd(_mm512_set1_pd(1.0), h_safe);

        // 速度
        let u_vec = _mm512_mul_pd(hu_vec, inv_h);
        let v_vec = _mm512_mul_pd(hv_vec, inv_h);

        // 法向速度
        let un_vec = _mm512_fmadd_pd(u_vec, nx_vec, _mm512_mul_pd(v_vec, ny_vec));

        // 质量通量
        let mass_flux = _mm512_mul_pd(h_vec, un_vec);

        // 压力
        let h_sq = _mm512_mul_pd(h_vec, h_vec);
        let pressure = _mm512_mul_pd(half, _mm512_mul_pd(g_vec, h_sq));

        // 动量通量
        let mom_x = _mm512_fmadd_pd(hu_vec, un_vec, _mm512_mul_pd(pressure, nx_vec));
        let mom_y = _mm512_fmadd_pd(hv_vec, un_vec, _mm512_mul_pd(pressure, ny_vec));

        // 掩码
        let mask = _mm512_cmp_pd_mask(h_vec, zero, _CMP_GT_OQ);
        let mass_final = _mm512_maskz_mov_pd(mask, mass_flux);
        let mom_x_final = _mm512_maskz_mov_pd(mask, mom_x);
        let mom_y_final = _mm512_maskz_mov_pd(mask, mom_y);

        // 存储
        _mm512_storeu_pd(out_mass.as_mut_ptr().add(i), mass_final);
        _mm512_storeu_pd(out_mom_x.as_mut_ptr().add(i), mom_x_final);
        _mm512_storeu_pd(out_mom_y.as_mut_ptr().add(i), mom_y_final);

        i += lanes;
    }

    // 处理尾部
    if aligned_n < n {
        let remaining = n - aligned_n;
        let mask = (1u8 << remaining) - 1;
        
        let h_vec = _mm512_maskz_loadu_pd(mask, h.as_ptr().add(aligned_n));
        let hu_vec = _mm512_maskz_loadu_pd(mask, hu.as_ptr().add(aligned_n));
        let hv_vec = _mm512_maskz_loadu_pd(mask, hv.as_ptr().add(aligned_n));
        let nx_vec = _mm512_maskz_loadu_pd(mask, nx.as_ptr().add(aligned_n));
        let ny_vec = _mm512_maskz_loadu_pd(mask, ny.as_ptr().add(aligned_n));

        let h_safe = _mm512_max_pd(h_vec, epsilon);
        let inv_h = _mm512_div_pd(_mm512_set1_pd(1.0), h_safe);
        let u_vec = _mm512_mul_pd(hu_vec, inv_h);
        let v_vec = _mm512_mul_pd(hv_vec, inv_h);
        let un_vec = _mm512_fmadd_pd(u_vec, nx_vec, _mm512_mul_pd(v_vec, ny_vec));
        let mass_flux = _mm512_mul_pd(h_vec, un_vec);
        let h_sq = _mm512_mul_pd(h_vec, h_vec);
        let pressure = _mm512_mul_pd(half, _mm512_mul_pd(g_vec, h_sq));
        let mom_x = _mm512_fmadd_pd(hu_vec, un_vec, _mm512_mul_pd(pressure, nx_vec));
        let mom_y = _mm512_fmadd_pd(hv_vec, un_vec, _mm512_mul_pd(pressure, ny_vec));

        let h_mask = _mm512_cmp_pd_mask(h_vec, zero, _CMP_GT_OQ);
        let final_mask = mask & h_mask;

        _mm512_mask_storeu_pd(out_mass.as_mut_ptr().add(aligned_n), final_mask, mass_flux);
        _mm512_mask_storeu_pd(out_mom_x.as_mut_ptr().add(aligned_n), final_mask, mom_x);
        _mm512_mask_storeu_pd(out_mom_y.as_mut_ptr().add(aligned_n), final_mask, mom_y);
    }
}

// ============================================================================
// 状态更新核心
// ============================================================================

/// 批量 Euler 前向更新
pub fn update_state_euler_f64(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    flux_h: &[f64],
    flux_hu: &[f64],
    flux_hv: &[f64],
    areas: &[f64],
    dt: f64,
    h_min: f64,
) {
    let n = h.len();
    debug_assert_eq!(hu.len(), n);
    debug_assert_eq!(hv.len(), n);
    debug_assert_eq!(flux_h.len(), n);
    debug_assert_eq!(flux_hu.len(), n);
    debug_assert_eq!(flux_hv.len(), n);
    debug_assert_eq!(areas.len(), n);

    match simd_capability() {
        SimdCapability::Avx512 | SimdCapability::Avx2 => {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: 已检测 SIMD 支持
                unsafe {
                    update_state_euler_avx2_f64(h, hu, hv, flux_h, flux_hu, flux_hv, areas, dt, h_min);
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                update_state_euler_scalar_f64(h, hu, hv, flux_h, flux_hu, flux_hv, areas, dt, h_min);
            }
        }
        _ => {
            update_state_euler_scalar_f64(h, hu, hv, flux_h, flux_hu, flux_hv, areas, dt, h_min);
        }
    }
}

/// 标量 Euler 更新
fn update_state_euler_scalar_f64(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    flux_h: &[f64],
    flux_hu: &[f64],
    flux_hv: &[f64],
    areas: &[f64],
    dt: f64,
    h_min: f64,
) {
    for i in 0..h.len() {
        let inv_area = 1.0 / areas[i];
        h[i] = (h[i] - dt * flux_h[i] * inv_area).max(h_min);
        hu[i] -= dt * flux_hu[i] * inv_area;
        hv[i] -= dt * flux_hv[i] * inv_area;

        // 干单元动量置零
        if h[i] <= h_min {
            hu[i] = 0.0;
            hv[i] = 0.0;
        }
    }
}

/// AVX2 Euler 更新
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn update_state_euler_avx2_f64(
    h: &mut [f64],
    hu: &mut [f64],
    hv: &mut [f64],
    flux_h: &[f64],
    flux_hu: &[f64],
    flux_hv: &[f64],
    areas: &[f64],
    dt: f64,
    h_min: f64,
) {
    use std::arch::x86_64::*;

    let n = h.len();
    let lanes = 4;
    let aligned_n = n - (n % lanes);

    let dt_vec = _mm256_set1_pd(dt);
    let h_min_vec = _mm256_set1_pd(h_min);
    let mut i = 0;
    while i < aligned_n {
        let h_vec = _mm256_loadu_pd(h.as_ptr().add(i));
        let hu_vec = _mm256_loadu_pd(hu.as_ptr().add(i));
        let hv_vec = _mm256_loadu_pd(hv.as_ptr().add(i));
        let fh = _mm256_loadu_pd(flux_h.as_ptr().add(i));
        let fhu = _mm256_loadu_pd(flux_hu.as_ptr().add(i));
        let fhv = _mm256_loadu_pd(flux_hv.as_ptr().add(i));
        let area = _mm256_loadu_pd(areas.as_ptr().add(i));

        let inv_area = _mm256_div_pd(_mm256_set1_pd(1.0), area);
        let dt_inv_area = _mm256_mul_pd(dt_vec, inv_area);

        // h = max(h - dt * flux_h / area, h_min)
        let h_new = _mm256_max_pd(
            _mm256_fnmadd_pd(dt_inv_area, fh, h_vec),
            h_min_vec,
        );

        // hu -= dt * flux_hu / area
        let hu_new = _mm256_fnmadd_pd(dt_inv_area, fhu, hu_vec);
        let hv_new = _mm256_fnmadd_pd(dt_inv_area, fhv, hv_vec);

        // 干单元掩码
        let wet_mask = _mm256_cmp_pd(h_new, h_min_vec, _CMP_GT_OQ);
        let hu_final = _mm256_and_pd(hu_new, wet_mask);
        let hv_final = _mm256_and_pd(hv_new, wet_mask);

        _mm256_storeu_pd(h.as_mut_ptr().add(i), h_new);
        _mm256_storeu_pd(hu.as_mut_ptr().add(i), hu_final);
        _mm256_storeu_pd(hv.as_mut_ptr().add(i), hv_final);

        i += lanes;
    }

    // 尾部处理
    for i in aligned_n..n {
        let inv_area = 1.0 / areas[i];
        h[i] = (h[i] - dt * flux_h[i] * inv_area).max(h_min);
        hu[i] -= dt * flux_hu[i] * inv_area;
        hv[i] -= dt * flux_hv[i] * inv_area;
        if h[i] <= h_min {
            hu[i] = 0.0;
            hv[i] = 0.0;
        }
    }
}

// ============================================================================
// 波速计算
// ============================================================================

/// 批量计算波速 (用于 CFL)
pub fn compute_wave_speed_batch_f64(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    g: f64,
    h_min: f64,
    out: &mut [f64],
) {
    let n = h.len();
    for i in 0..n {
        if h[i] <= h_min {
            out[i] = 0.0;
        } else {
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let vel = (u * u + v * v).sqrt();
            let c = (g * h[i]).sqrt();
            out[i] = vel + c;
        }
    }
}

/// 获取最大波速
pub fn max_wave_speed_f64(
    h: &[f64],
    hu: &[f64],
    hv: &[f64],
    g: f64,
    h_min: f64,
) -> f64 {
    let mut max_speed = 0.0;
    for i in 0..h.len() {
        if h[i] > h_min {
            let u = hu[i] / h[i];
            let v = hv[i] / h[i];
            let vel = (u * u + v * v).sqrt();
            let c = (g * h[i]).sqrt();
            let speed = vel + c;
            if speed > max_speed {
                max_speed = speed;
            }
        }
    }
    max_speed
}

// ============================================================================
// 斜率限制器
// ============================================================================

/// Venkatakrishnan 限制器参数
pub struct VenkatakrishnanParams {
    /// 限制器强度参数 K
    pub k: f64,
    /// 特征长度（单元尺寸）
    pub dx: f64,
}

impl Default for VenkatakrishnanParams {
    fn default() -> Self {
        Self { k: 1.0, dx: 1.0 }
    }
}

/// 批量 Venkatakrishnan 限制器
pub fn venkatakrishnan_limiter_batch_f64(
    phi_max: &[f64],  // 邻居最大值
    phi_min: &[f64],  // 邻居最小值
    phi_i: &[f64],    // 单元中心值
    delta: &[f64],    // 重构增量
    params: &VenkatakrishnanParams,
    out: &mut [f64],  // 限制因子
) {
    let n = phi_i.len();
    let eps_sq = (params.k * params.dx).powi(3);

    for i in 0..n {
        let d = delta[i];
        if d.abs() < 1e-14 {
            out[i] = 1.0;
            continue;
        }

        let (delta_plus, delta_minus) = if d > 0.0 {
            (phi_max[i] - phi_i[i], phi_min[i] - phi_i[i])
        } else {
            (phi_min[i] - phi_i[i], phi_max[i] - phi_i[i])
        };

        let delta_use = if d > 0.0 { delta_plus } else { delta_minus };

        let r = delta_use / d;
        let phi = (r * r + 2.0 * r + eps_sq) / (r * r + r + 2.0 + eps_sq);
        out[i] = phi.min(1.0).max(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_flux_scalar() {
        let h = vec![1.0, 2.0, 0.0, 1.5];
        let hu = vec![1.0, 2.0, 0.0, 0.0];
        let hv = vec![0.0, 0.0, 0.0, 1.5];
        let nx = vec![1.0, 1.0, 1.0, 0.0];
        let ny = vec![0.0, 0.0, 0.0, 1.0];
        let g = 9.81;

        let mut out_mass = vec![0.0; 4];
        let mut out_mom_x = vec![0.0; 4];
        let mut out_mom_y = vec![0.0; 4];

        compute_flux_batch_f64(&h, &hu, &hv, &nx, &ny, g, &mut out_mass, &mut out_mom_x, &mut out_mom_y);

        // 第一个单元: h=1, u=1, v=0, nx=1
        // mass_flux = h * un = 1 * 1 = 1
        assert!((out_mass[0] - 1.0).abs() < 1e-10);

        // 第三个单元: h=0, 应该都是 0
        assert_eq!(out_mass[2], 0.0);
        assert_eq!(out_mom_x[2], 0.0);
        assert_eq!(out_mom_y[2], 0.0);
    }

    #[test]
    fn test_update_state_euler() {
        let mut h = vec![1.0, 2.0, 0.5, 1.5];
        let mut hu = vec![0.5, 1.0, 0.0, 0.75];
        let mut hv = vec![0.0, 0.0, 0.0, 0.0];
        let flux_h = vec![0.1, 0.2, 0.0, 0.15];
        let flux_hu = vec![0.05, 0.1, 0.0, 0.075];
        let flux_hv = vec![0.0; 4];
        let areas = vec![1.0; 4];
        let dt = 0.01;
        let h_min = 0.001;

        update_state_euler_f64(
            &mut h, &mut hu, &mut hv,
            &flux_h, &flux_hu, &flux_hv,
            &areas, dt, h_min,
        );

        // h 应该减少
        assert!(h[0] < 1.0);
        assert!(h[1] < 2.0);
    }

    #[test]
    fn test_max_wave_speed() {
        let h = vec![1.0, 2.0, 0.0, 4.0];
        let hu = vec![1.0, 0.0, 0.0, 0.0];
        let hv = vec![0.0, 2.0, 0.0, 0.0];
        let g = 10.0; // 简化
        let h_min = 0.001;

        let max_speed = max_wave_speed_f64(&h, &hu, &hv, g, h_min);

        // 第4个单元: c = sqrt(10*4) = sqrt(40) ≈ 6.32
        // 应该是最大的（无流速）
        assert!(max_speed > 6.0);
    }

    #[test]
    fn test_simd_capability() {
        let cap = simd_capability();
        println!("Detected SIMD capability: {:?}", cap.name());
        assert!(cap.f64_lanes() >= 1);
    }
}
