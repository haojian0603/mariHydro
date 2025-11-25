//! 大气-海洋界面物理过程
//!
//! 功能模块：
//! 1. 气压梯度力计算（海洋模型驱动力）
//! 2. 风阻系数参数化（Large & Pond 1981, Wu 1982）
//! 3. 风应力计算（动量通量）

use ndarray::{s, Array2, ArrayView2, Zip};

use crate::marihydro::domain::mesh::Mesh;
use crate::marihydro::infra::constants::validation;
use crate::marihydro::types::Gradient2D;

// ============================================================================
// 气压梯度计算（Pressure Gradient Force）
// ============================================================================

/// 场级别向量化气压梯度计算（推荐）
///
/// 利用 ndarray 切片操作自动实现 SIMD 加速。
///
/// # 参数
///
/// - `pressure`: 全域气压场（含 Ghost）
/// - `mesh`: 网格信息
/// - `rho`: 水体密度 [kg/m³]
/// - `u_acc`, `v_acc`: 输出加速度场（物理区域尺寸 ny×nx）
pub fn compute_pressure_gradient_field(
    pressure: &ArrayView2<f64>,
    mesh: &Mesh,
    rho: f64,
    u_acc: &mut Array2<f64>,
    v_acc: &mut Array2<f64>,
) {
    let dx = mesh.transform.dx.abs();
    let dy = mesh.transform.dy.abs();

    let ng = mesh.ng;
    let ny = mesh.ny;
    let nx = mesh.nx;

    // 验证输出尺寸
    debug_assert_eq!(u_acc.dim(), (ny, nx));
    debug_assert_eq!(v_acc.dim(), (ny, nx));

    // 定义切片窗口（物理区域）
    // p(i+1): 东邻
    let p_east = pressure.slice(s![ng..ng + ny, ng + 1..ng + nx + 1]);
    // p(i-1): 西邻
    let p_west = pressure.slice(s![ng..ng + ny, ng - 1..ng + nx - 1]);

    // p(j+1): 北邻
    let p_north = pressure.slice(s![ng + 1..ng + ny + 1, ng..ng + nx]);
    // p(j-1): 南邻
    let p_south = pressure.slice(s![ng - 1..ng + ny - 1, ng..ng + nx]);

    let factor = -1.0 / rho;
    let idx = 1.0 / (2.0 * dx);
    let idy = 1.0 / (2.0 * dy);

    // 并行计算
    Zip::from(u_acc)
        .and(v_acc)
        .and(&p_east)
        .and(&p_west)
        .and(&p_north)
        .and(&p_south)
        .par_for_each(|u, v, &e, &w, &n, &s| {
            *u = factor * (e - w) * idx;
            *v = factor * (n - s) * idy;
        });
}

/// 单点气压梯度计算（安全版本）
pub fn compute_pressure_gradient_acc(
    pressure_field: &[f64],
    idx: usize,
    stride: usize,
    dx: f64,
    dy: f64,
    rho: f64,
) -> Option<Gradient2D> {
    if idx < stride || idx >= pressure_field.len() - stride {
        return None;
    }
    if idx % stride == 0 || idx % stride == stride - 1 {
        return None;
    }

    Some(unsafe {
        compute_pressure_gradient_acc_unchecked(pressure_field, idx, stride, dx, dy, rho)
    })
}

/// 单点气压梯度计算（无检查版本）
///
/// # Safety
///
/// 调用者必须保证：
/// 1. `stride <= idx < pressure_field.len() - stride`
/// 2. `idx % stride > 0 && idx % stride < stride - 1`
/// 3. `pressure_field` 内存连续且为 Row-Major 布局
#[inline(always)]
pub unsafe fn compute_pressure_gradient_acc_unchecked(
    pressure_field: &[f64],
    idx: usize,
    stride: usize,
    dx: f64,
    dy: f64,
    rho: f64,
) -> Gradient2D {
    let p_east = *pressure_field.get_unchecked(idx + 1);
    let p_west = *pressure_field.get_unchecked(idx - 1);
    let p_north = *pressure_field.get_unchecked(idx + stride);
    let p_south = *pressure_field.get_unchecked(idx - stride);

    let dp_dx = (p_east - p_west) / (2.0 * dx);
    let dp_dy = (p_north - p_south) / (2.0 * dy);

    let factor = -1.0 / rho;

    (factor * dp_dx, factor * dp_dy)
}

// ============================================================================
// 风阻系数参数化（Wind Drag Coefficient）
// ============================================================================

/// Large & Pond (1981) 风阻系数参数化
///
/// 分段线性模型，适用于10m高度风速
///
/// # 公式
/// - W < 11 m/s:  C_d = 1.2e-3
/// - 11 ≤ W < 25: C_d = (0.49 + 0.065*W) * 1e-3
/// - W ≥ 25:      C_d = 2.11e-3 (饱和)
///
/// # 参数
/// - `wind_speed_10m`: 10m高度风速 [m/s]
///
/// # 返回
/// 风阻系数 C_d (无量纲)
#[inline]
pub fn wind_drag_coefficient_lp81(wind_speed_10m: f64) -> f64 {
    let w = wind_speed_10m.abs();

    // 防御性检查
    if w > validation::MAX_REASONABLE_WIND_SPEED {
        log::warn!("风速异常: {:.1} m/s，限制到最大值", w);
        return 2.11e-3;
    }

    if w < 11.0 {
        1.2e-3
    } else if w < 25.0 {
        (0.49 + 0.065 * w) * 1e-3
    } else {
        2.11e-3
    }
}

/// Wu (1982) 风阻系数参数化
///
/// 连续光滑模型，适用于全风速范围
///
/// # 公式
/// C_d = (0.8 + 0.065*W) * 1e-3
#[inline]
pub fn wind_drag_coefficient_wu82(wind_speed_10m: f64) -> f64 {
    let w = wind_speed_10m
        .abs()
        .min(validation::MAX_REASONABLE_WIND_SPEED);
    (0.8 + 0.065 * w) * 1e-3
}

/// 计算风应力 (τ_x, τ_y)
///
/// # 公式
/// τ = ρ_air * C_d * |W| * W
///
/// # 参数
/// - `wind_u`, `wind_v`: 风速分量 [m/s]
/// - `air_density`: 空气密度 [kg/m³]
///
/// # 返回
/// 风应力分量 (τ_x, τ_y) [N/m²]
pub fn compute_wind_stress(wind_u: f64, wind_v: f64, air_density: f64) -> (f64, f64) {
    let wind_mag = (wind_u * wind_u + wind_v * wind_v).sqrt();

    if wind_mag < 1e-8 {
        return (0.0, 0.0);
    }

    let cd = wind_drag_coefficient_lp81(wind_mag);
    let tau_mag = air_density * cd * wind_mag;

    (tau_mag * wind_u, tau_mag * wind_v)
}

/// 计算风应力加速度（直接施加到水体）
///
/// # 公式
/// a = τ / (ρ_water * h)
///
/// # 参数
/// - `wind_u`, `wind_v`: 风速分量 [m/s]
/// - `h`: 水深 [m]
/// - `air_density`: 空气密度 [kg/m³]
/// - `water_density`: 水体密度 [kg/m³]
///
/// # 返回
/// 加速度分量 (a_x, a_y) [m/s²]
pub fn compute_wind_acceleration(
    wind_u: f64,
    wind_v: f64,
    h: f64,
    air_density: f64,
    water_density: f64,
) -> (f64, f64) {
    if h < 1e-6 {
        return (0.0, 0.0);
    }

    let (tau_x, tau_y) = compute_wind_stress(wind_u, wind_v, air_density);
    let factor = 1.0 / (water_density * h);

    (tau_x * factor, tau_y * factor)
}

// ============================================================================
// 单元测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marihydro::infra::manifest::{PhysicsParameters, ProjectManifest};
    use ndarray::Array2;

    // ------------------------------------------------------------------------
    // 气压梯度测试（现有测试保留）
    // ------------------------------------------------------------------------

    /// 辅助函数：创建测试用的 Manifest
    fn create_test_manifest(nx: usize, ny: usize, dx: f64, dy: f64) -> ProjectManifest {
        let mut manifest = ProjectManifest::new("Test Atmosphere");

        manifest.grid_nx = nx;
        manifest.grid_ny = ny;
        manifest.grid_dx = dx;
        manifest.grid_dy = dy;

        manifest.physics = PhysicsParameters::default();
        manifest.crs_wkt = "EPSG:32651".into();

        manifest
    }

    #[test]
    fn test_field_computation() {
        let nx = 3;
        let ny = 3;

        let manifest = create_test_manifest(nx, ny, 1.0, 1.0);
        let mesh = Mesh::init(&manifest).expect("Mesh初始化失败");
        let ng = mesh.ng;

        let (total_ny, total_nx) = mesh.total_size();
        let mut pressure = Array2::zeros((total_ny, total_nx));

        for j in 0..total_ny {
            for i in 0..total_nx {
                let x = (i as f64) - ng as f64;
                let y = (j as f64) - ng as f64;
                pressure[[j, i]] = 10.0 * x + 20.0 * y;
            }
        }

        let mut u_acc = Array2::zeros((ny, nx));
        let mut v_acc = Array2::zeros((ny, nx));

        compute_pressure_gradient_field(&pressure.view(), &mesh, 1000.0, &mut u_acc, &mut v_acc);

        let center = (1, 1);
        assert!((u_acc[center] - (-0.01)).abs() < 1e-10);
        assert!((v_acc[center] - (-0.02)).abs() < 1e-10);
    }

    #[test]
    fn test_single_point() {
        let stride = 3;
        let mut pressure = vec![0.0; 9];

        for j in 0..3 {
            for i in 0..3 {
                let x = i as f64;
                let y = j as f64;
                pressure[j * stride + i] = 10.0 * x + 20.0 * y;
            }
        }

        let (ax, ay) =
            compute_pressure_gradient_acc(&pressure, 4, stride, 1.0, 1.0, 1000.0).unwrap();

        assert!((ax - (-0.01)).abs() < 1e-10);
        assert!((ay - (-0.02)).abs() < 1e-10);
    }

    // ------------------------------------------------------------------------
    // 风阻系数测试（新增）
    // ------------------------------------------------------------------------

    #[test]
    fn test_lp81_piecewise() {
        // 低风速段
        let cd_low = wind_drag_coefficient_lp81(5.0);
        assert!((cd_low - 1.2e-3).abs() < 1e-6);

        // 中等风速段
        let cd_mid = wind_drag_coefficient_lp81(15.0);
        let expected = (0.49 + 0.065 * 15.0) * 1e-3;
        assert!((cd_mid - expected).abs() < 1e-9);

        // 高风速饱和
        let cd_high = wind_drag_coefficient_lp81(30.0);
        assert!((cd_high - 2.11e-3).abs() < 1e-6);
    }

    #[test]
    fn test_wu82_continuity() {
        // Wu82应该是连续函数
        let cd1 = wind_drag_coefficient_wu82(10.0);
        let cd2 = wind_drag_coefficient_wu82(10.1);
        assert!((cd2 - cd1).abs() < 1e-5);
    }

    #[test]
    fn test_wind_stress_direction() {
        // 标准空气密度
        let rho_air = 1.225;

        // X方向风
        let (tau_x, tau_y) = compute_wind_stress(10.0, 0.0, rho_air);

        assert!(tau_x > 0.0);
        assert!(tau_y.abs() < 1e-12);

        // 验证量级
        let cd = wind_drag_coefficient_lp81(10.0);
        let expected_tau = rho_air * cd * 10.0 * 10.0;
        assert!((tau_x - expected_tau).abs() < 1e-6);
    }

    #[test]
    fn test_wind_stress_zero_wind() {
        let (tau_x, tau_y) = compute_wind_stress(0.0, 0.0, 1.225);

        assert_eq!(tau_x, 0.0);
        assert_eq!(tau_y, 0.0);
    }

    #[test]
    fn test_wind_acceleration() {
        let rho_air = 1.225;
        let rho_water = 1025.0;

        // 10m/s风速，1m水深
        let (ax, ay) = compute_wind_acceleration(10.0, 0.0, 1.0, rho_air, rho_water);

        // 验证风应力加速度为正
        assert!(ax > 0.0);
        assert!(ay.abs() < 1e-12);

        // 验证量级合理性（应该在1e-4量级）
        assert!(ax < 1e-3);
        assert!(ax > 1e-5);
    }

    #[test]
    fn test_wind_acceleration_shallow_water() {
        let rho_air = 1.225;
        let rho_water = 1025.0;

        // 相同风速，浅水应该产生更大加速度
        let (ax_deep, _) = compute_wind_acceleration(10.0, 0.0, 10.0, rho_air, rho_water);
        let (ax_shallow, _) = compute_wind_acceleration(10.0, 0.0, 1.0, rho_air, rho_water);

        assert!(ax_shallow > ax_deep);
        assert!((ax_shallow / ax_deep - 10.0).abs() < 1e-6); // 应该正好是10倍
    }

    #[test]
    fn test_wind_acceleration_dry_cell() {
        // 干单元应该返回零加速度
        let (ax, ay) = compute_wind_acceleration(10.0, 10.0, 1e-10, 1.225, 1025.0);

        assert_eq!(ax, 0.0);
        assert_eq!(ay, 0.0);
    }
}
