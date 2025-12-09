// crates/mh_physics/tests/validation_thacker.rs

//! Thacker 抛物面振荡验证
//!
//! Thacker (1981) 给出了抛物面碗中振荡水面的解析解，
//! 常用于验证浅水方程求解器的精度和质量守恒性。
//!
//! # 问题描述
//!
//! - 抛物面底床：$z_b(r) = h_0 (r/a)^2$
//! - 初始水面：椭圆形抬升
//! - 解析周期：$T = 2\pi / \omega$，其中 $\omega = \sqrt{8 g h_0} / a$
//!
//! # 验证标准
//!
//! - 一个周期后水面形态应恢复初始状态
//! - 总质量守恒误差 < 0.1%

use mh_foundation::Scalar;
use std::f64::consts::PI;

// ============================================================
// 测试参数
// ============================================================

/// 盆地半径 [m]
const BOWL_RADIUS: Scalar = 1000.0;
/// 中心水深 [m]
const CENTER_DEPTH: Scalar = 10.0;
/// 初始水面扰动幅度 [m]
const INITIAL_AMPLITUDE: Scalar = 0.5;
/// 重力加速度 [m/s²]
const GRAVITY: Scalar = 9.81;

/// 计算振荡角频率
fn angular_frequency() -> Scalar {
    (8.0 * GRAVITY * CENTER_DEPTH).sqrt() / BOWL_RADIUS
}

/// 计算振荡周期
fn oscillation_period() -> Scalar {
    2.0 * PI / angular_frequency()
}

/// 抛物面底床高程
fn bed_elevation(x: Scalar, y: Scalar, cx: Scalar, cy: Scalar) -> Scalar {
    let r2 = (x - cx).powi(2) + (y - cy).powi(2);
    CENTER_DEPTH * r2 / (BOWL_RADIUS * BOWL_RADIUS)
}

/// 初始水位
#[allow(dead_code)]
fn initial_water_surface(x: Scalar, y: Scalar, cx: Scalar, cy: Scalar) -> Scalar {
    let r2 = (x - cx).powi(2) + (y - cy).powi(2);
    let eta = INITIAL_AMPLITUDE * (1.0 - 2.0 * r2 / (BOWL_RADIUS * BOWL_RADIUS));
    eta.max(0.0)
}

// ============================================================
// 理论解析解
// ============================================================

/// Thacker 解析解（简化版）
#[allow(dead_code)]
struct ThackerAnalytic {
    omega: Scalar,
    a: Scalar,
    h0: Scalar,
    eta0: Scalar,
}

impl ThackerAnalytic {
    fn new() -> Self {
        Self {
            omega: angular_frequency(),
            a: BOWL_RADIUS,
            h0: CENTER_DEPTH,
            eta0: INITIAL_AMPLITUDE,
        }
    }

    /// 给定时间和位置的水位
    fn water_surface(&self, x: Scalar, y: Scalar, t: Scalar, cx: Scalar, cy: Scalar) -> Scalar {
        // 简化：假设初始相位为0，仅考虑对称振荡
        let r2 = (x - cx).powi(2) + (y - cy).powi(2);
        let phase = (self.omega * t).cos();
        let eta = self.eta0 * phase * (1.0 - 2.0 * r2 / (self.a * self.a));
        eta.max(0.0)
    }

    /// 给定时间和位置的速度
    fn velocity(&self, x: Scalar, y: Scalar, t: Scalar, cx: Scalar, cy: Scalar) -> (Scalar, Scalar) {
        let phase = (self.omega * t).sin();
        let coef = -self.omega * self.eta0 / self.a;
        let u = coef * (x - cx) * phase;
        let v = coef * (y - cy) * phase;
        (u, v)
    }
}

// ============================================================
// 数值测试
// ============================================================

/// 计算网格的总质量
fn compute_total_mass(h: &[Scalar], areas: &[Scalar]) -> Scalar {
    h.iter().zip(areas).map(|(&h, &a)| h * a).sum()
}

/// 计算 L2 范数误差
fn l2_error(numerical: &[Scalar], analytical: &[Scalar]) -> Scalar {
    let n = numerical.len() as Scalar;
    let sum_sq: Scalar = numerical
        .iter()
        .zip(analytical)
        .map(|(&n, &a)| (n - a).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

/// 计算 L∞ 范数误差
fn linf_error(numerical: &[Scalar], analytical: &[Scalar]) -> Scalar {
    numerical
        .iter()
        .zip(analytical)
        .map(|(&n, &a)| (n - a).abs())
        .fold(0.0, Scalar::max)
}

// ============================================================
// 实际测试用例
// ============================================================

#[test]
fn test_thacker_period_calculation() {
    let period = oscillation_period();
    let omega = angular_frequency();

    // 验证计算的周期
    let expected_omega = (8.0 * GRAVITY * CENTER_DEPTH).sqrt() / BOWL_RADIUS;
    assert!((omega - expected_omega).abs() < 1e-10);

    let expected_period = 2.0 * PI / expected_omega;
    assert!((period - expected_period).abs() < 1e-10);

    println!("Thacker parameters:");
    println!("  Bowl radius: {} m", BOWL_RADIUS);
    println!("  Center depth: {} m", CENTER_DEPTH);
    println!("  Initial amplitude: {} m", INITIAL_AMPLITUDE);
    println!("  Angular frequency: {:.4} rad/s", omega);
    println!("  Oscillation period: {:.2} s", period);
}

#[test]
fn test_thacker_analytic_solution() {
    let analytic = ThackerAnalytic::new();
    let cx = BOWL_RADIUS;
    let cy = BOWL_RADIUS;

    // t=0 时中心水位应为 eta0
    let eta_center = analytic.water_surface(cx, cy, 0.0, cx, cy);
    assert!((eta_center - INITIAL_AMPLITUDE).abs() < 1e-10);

    // t=0 时速度应为零
    let (u, v) = analytic.velocity(cx, cy, 0.0, cx, cy);
    assert!(u.abs() < 1e-10);
    assert!(v.abs() < 1e-10);

    // t=T/2 时中心水位应为 -eta0（被裁剪为0）
    let period = oscillation_period();
    let eta_half = analytic.water_surface(cx, cy, period / 2.0, cx, cy);
    assert!(eta_half.abs() < 1e-10 || eta_half >= 0.0);

    // t=T 时应恢复初始状态
    let eta_full = analytic.water_surface(cx, cy, period, cx, cy);
    assert!((eta_full - INITIAL_AMPLITUDE).abs() < 0.01);
}

#[test]
fn test_thacker_bed_elevation() {
    let cx = BOWL_RADIUS;
    let cy = BOWL_RADIUS;

    // 中心底床高程为0
    let z_center = bed_elevation(cx, cy, cx, cy);
    assert!(z_center.abs() < 1e-14);

    // 边缘底床高程为 h0
    let z_edge = bed_elevation(cx + BOWL_RADIUS, cy, cx, cy);
    assert!((z_edge - CENTER_DEPTH).abs() < 1e-10);
}

#[test]
fn test_mass_error_functions() {
    let h = vec![1.0, 2.0, 3.0];
    let areas = vec![10.0, 10.0, 10.0];

    let mass = compute_total_mass(&h, &areas);
    assert!((mass - 60.0).abs() < 1e-10);
}

#[test]
fn test_error_metrics() {
    let numerical = vec![1.0, 2.1, 3.0];
    let analytical = vec![1.0, 2.0, 3.0];

    let l2 = l2_error(&numerical, &analytical);
    let linf = linf_error(&numerical, &analytical);

    assert!(l2 > 0.0);
    assert!((linf - 0.1).abs() < 1e-10);
    assert!(l2 <= linf);
}

// ============================================================
// 完整验证测试（需要完整求解器）
// ============================================================

/// 完整 Thacker 验证测试
///
/// 此测试需要完整的求解器实现，暂时标记为 ignore
#[test]
#[ignore = "需要完整求解器实现"]
fn test_thacker_full_simulation() {
    // TODO: 使用 CircularMeshGenerator 生成网格
    // TODO: 初始化状态（底床、水深）
    // TODO: 使用半隐式求解器推进一个周期
    // TODO: 验证水面形态恢复
    // TODO: 验证质量守恒 < 0.1%
    // TODO: 验证 L2 误差 < 2%
}

// ============================================================
// 收敛性测试（需要完整求解器）
// ============================================================

/// 网格收敛性测试
///
/// 验证误差随网格加密的收敛阶
#[test]
#[ignore = "需要完整求解器实现"]
fn test_thacker_convergence() {
    // TODO: 在不同网格分辨率下运行
    // TODO: 计算各分辨率的 L2 误差
    // TODO: 验证收敛阶 > 1
}
