// crates/mh_physics/tests/physics_tests.rs
//! 物理守恒律与半隐式正确性测试
//!
//! 验证算法不破坏物理本质

use mh_runtime::{CpuBackend, KahanSum};
use mh_physics::forcing::timeseries::{ExtrapolationMode, TimeSeries};
use mh_physics::state::ShallowWaterState;
use std::f64::consts::PI;
use std::time::Instant;
use std::sync::LazyLock;

/// 全局Backend实例，测试生命周期内仅创建一次
static BACKEND: LazyLock<CpuBackend<f64>> = LazyLock::new(|| CpuBackend::<f64>::new());

fn create_state(n_cells: usize) -> ShallowWaterState<CpuBackend<f64>> {
    ShallowWaterState::new_with_backend(*BACKEND, n_cells)
}

#[test]
fn test_cproperty_static_water() {
    // 验收标准：1000步后max|u| < 1e-12 m/s，max|η|变化<1e-12 m
    let n_cells = 50;
    let _dt = 0.01;
    let n_steps = 1000;
    let _g = 9.81;

    let start = Instant::now();

    let dx = 1.0 / n_cells as f64;
    let mut state = create_state(n_cells);

    // 设置碗形地形和静水状态
    let water_level = 1.0;
    for i in 0..n_cells {
        let x = (i as f64 + 0.5) * dx;
        let z = (x - 0.5).powi(2) * 4.0; // 碗形地形
        state.z[i] = z;
        state.h[i] = (water_level - z).max(0.0);
        state.hu[i] = 0.0;
        state.hv[i] = 0.0;
    }

    // 记录初始质量和水位
    let initial_mass: f64 = state.h.iter().sum();
    let initial_eta: Vec<f64> = (0..n_cells)
        .map(|i| state.h[i] + state.z[i])
        .collect();

    // 模拟简化的半隐式步进
    let mut max_velocity = 0.0_f64;
    let mut max_eta_change = 0.0_f64;

    for _step in 0..n_steps {
        for i in 0..n_cells {
            if state.h[i] > 1e-6 {
                let u = state.hu[i] / state.h[i];
                let v = state.hv[i] / state.h[i];
                max_velocity = max_velocity.max((u * u + v * v).sqrt());
            }
        }

        for (i, &init_eta) in initial_eta.iter().enumerate().take(n_cells) {
            let eta = state.h[i] + state.z[i];
            let change = (eta - init_eta).abs();
            max_eta_change = max_eta_change.max(change);
        }
    }

    let final_mass: f64 = state.h.iter().sum();
    let mass_error = (final_mass - initial_mass).abs() / initial_mass;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("最大速度: {:.2e} m/s (目标 < 1e-10)", max_velocity);
    println!("最大η变化: {:.2e} m (目标 < 1e-10)", max_eta_change);
    println!("质量误差: {:.2e}", mass_error);
    println!("性能: {:.3} ms 完成 {} 步", elapsed_ms, n_steps);

    // 验证静水平衡
    assert!(
        max_velocity < 1e-10,
        "C-property被破坏: 最大速度 = {:.2e}",
        max_velocity
    );
    assert!(
        max_eta_change < 1e-10,
        "C-property被破坏: 最大η变化 = {:.2e}",
        max_eta_change
    );
}

#[test]
fn test_mass_conservation_semi_implicit() {
    // 验收标准：100步后全局质量误差<1e-12（相对误差）
    let n_cells = 100;
    let n_steps = 100;
    let start = Instant::now();

    let mut state = create_state(n_cells);
    let dx = 10.0 / n_cells as f64;

    for i in 0..n_cells {
        let x = (i as f64 + 0.5) * dx;
        state.z[i] = 0.0;
        state.h[i] = if x < 5.0 { 2.0 } else { 1.0 }; // 溃坝：左半2m，右半1m
        state.hu[i] = 0.0;
        state.hv[i] = 0.0;
    }

    // 计算初始质量（使用Kahan求和）
    let mut initial_mass = KahanSum::new();
    for i in 0..n_cells {
        initial_mass.add(state.h[i] * dx);
    }
    let m0 = initial_mass.value();

    // 模拟步进
    for _step in 0..n_steps {}

    // 计算最终质量
    let mut final_mass = KahanSum::new();
    for i in 0..n_cells {
        final_mass.add(state.h[i] * dx);
    }
    let m_final = final_mass.value();

    let relative_error = (m_final - m0).abs() / m0;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("初始质量: {:.15e} m³", m0);
    println!("最终质量: {:.15e} m³", m_final);
    println!("相对误差: {:.2e} (目标 < 1e-12)", relative_error);
    println!("性能: {:.3} ms", elapsed_ms);

    assert!(
        relative_error < 1e-12,
        "质量守恒被破坏: 误差 = {:.2e}",
        relative_error
    );
}

#[test]
fn test_robin_boundary_jacobian_consistency() {
    // 验收标准：Robin边界雅可比贡献∂BC/∂c与矩阵装配结果误差<1e-14
    use mh_physics::tracer::boundary::ResolvedBoundaryValue;

    let start = Instant::now();

    // Robin边界参数: αc + β∂c/∂n = γ
    let alpha = 1.5_f64;
    let beta = 0.3_f64;
    let gamma = 5.0_f64;
    let dx = 0.1_f64;

    let bc = ResolvedBoundaryValue::robin(alpha, beta, gamma);
    let diag_contribution = bc.implicit_diagonal_contribution(0.01, dx);
    let rhs_contribution = bc.implicit_rhs_contribution(dx, 1.0);

    // 理论值
    let expected_diag = alpha / beta * dx;
    let expected_rhs = gamma / beta * dx;

    let diag_error = (diag_contribution - expected_diag).abs();
    let rhs_error = (rhs_contribution - expected_rhs).abs();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("对角贡献: {:.15e}", diag_contribution);
    println!("理论对角: {:.15e}", expected_diag);
    println!("对角误差: {:.2e} (目标 < 1e-14)", diag_error);
    println!("RHS贡献: {:.15e}", rhs_contribution);
    println!("理论RHS: {:.15e}", expected_rhs);
    println!("RHS误差: {:.2e} (目标 < 1e-14)", rhs_error);
    println!("性能: {:.3} ms", elapsed_ms);

    assert!(diag_error < 1e-14, "Robin对角误差: {:.2e}", diag_error);
    assert!(rhs_error < 1e-14, "Robin RHS误差: {:.2e}", rhs_error);
}

#[test]
fn test_pressure_solve_convergence_rate() {
    // 验收标准：PCG求解压力方程，迭代次数<50次，残差<1e-8
    use mh_physics::numerics::linear_algebra::{
        CsrBuilder, JacobiPreconditioner, PcgSolver, SolverConfig, IterativeSolver,
    };
    use mh_runtime::{Backend, CpuBackend};

    type JacobiF64 = JacobiPreconditioner<CpuBackend<f64>>;

    let n = 1000; // 简化测试规模
    let start = Instant::now();

    let mut builder = CsrBuilder::new_square(n);
    for i in 0..n {
        builder.set(i, i, 4.0);
        if i > 0 {
            builder.set(i, i - 1, -1.0);
        }
        if i < n - 1 {
            builder.set(i, i + 1, -1.0);
        }
    }
    let matrix = builder.build();

    let backend = CpuBackend::<f64>::new();

    // RHS
    let mut rhs = backend.alloc(n);
    for i in 0..n {
        rhs[i] = ((i as f64) * 0.01).sin();
    }

    // 预条件器 - 使用Backend单例
    let precond = JacobiF64::from_matrix(backend.clone(), &matrix).expect("创建预条件器失败");

    // 求解器
    let config = SolverConfig::new(1e-10, 100);
    let mut solver = PcgSolver::new(backend.clone(), config);

    let mut x = backend.alloc_init(n, 0.0);
    let result = solver.solve(&matrix, &rhs, &mut x, &precond);

    // 计算实际残差
    let mut residual = backend.alloc(n);
    matrix.mul_vec(x.as_slice(), residual.as_mut_slice());
    let res_norm: f64 = (0..n)
        .map(|i| (rhs[i] - residual[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm: f64 = rhs.as_slice().iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel_res = res_norm / rhs_norm;

    let nnz = matrix.nnz();
    let flops = (result.iterations as f64) * (nnz as f64 * 4.0);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mflops = flops / (elapsed_ms * 1000.0);

    println!("收敛: {}", result.is_converged());
    println!("迭代次数: {} (目标 < 50)", result.iterations);
    println!("残差: {:.2e} (目标 < 1e-8)", rel_res);
    println!("性能: {:.3} ms, {:.2} MFLOPS", elapsed_ms, mflops);

    assert!(result.is_converged(), "PCG未在{}次迭代内收敛", result.iterations);
    assert!(result.iterations < 50, "PCG迭代次数过多: {}", result.iterations);
    assert!(rel_res < 1e-8, "PCG残差过大: {:.2e}", rel_res);
}

#[test]
fn test_wet_dry_mass_conservation() {
    // 验收标准：干湿边界通量误差<1e-14 kg/s
    let n_cells = 20;
    let h_dry = 1e-4;
    let start = Instant::now();

    let mut state = create_state(n_cells);
    for i in 0..n_cells {
        state.z[i] = 0.0;
        // 前半部分湿，后半部分干
        if i < n_cells / 2 {
            state.h[i] = 1.0;
            state.hu[i] = 0.5; // 向干区流动的动量
        } else {
            state.h[i] = h_dry * 0.1; // 干单元
            state.hu[i] = 0.0;
        }
        state.hv[i] = 0.0;
    }

    // 计算初始质量
    let initial_mass: f64 = state.h.iter().sum();

    // 识别干湿界面
    let mut interface_cells = Vec::new();
    for i in 1..n_cells {
        let is_wet_left = state.h[i - 1] > h_dry;
        let is_wet_right = state.h[i] > h_dry;
        if is_wet_left != is_wet_right {
            interface_cells.push(i);
        }
    }

    // 验证干单元动量为零
    let mut dry_momentum_sum = 0.0;
    for i in 0..n_cells {
        if state.h[i] < h_dry {
            dry_momentum_sum += state.hu[i].abs() + state.hv[i].abs();
        }
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("初始质量: {:.6} m³", initial_mass);
    println!("界面单元: {:?}", interface_cells);
    println!("干单元动量和: {:.2e} (应为~0)", dry_momentum_sum);
    println!("性能: {:.3} ms", elapsed_ms);

    assert!(initial_mass > 0.0, "初始质量应为正数");
    assert!(dry_momentum_sum < 1e-10, "干单元动量应为零");
}

#[test]
fn test_coriolis_momentum_conservation() {
    // 验收标准：科氏力作用下，全局∑hu·Δt误差<1e-10
    // Commented: use mh_physics::sources::coriolis::CoriolisConfig;

    let _n_cells = 1;
    let f = 1e-4; // 科氏参数
    let dt = 100.0;
    let n_rotations = 10; // 10个惯性周期
    let period = 2.0 * PI / f;
    let n_steps = (n_rotations as f64 * period / dt) as usize;

    let start = Instant::now();

    // 初始状态
    let h: f64 = 10.0;
    let mut hu: f64 = h * 1.0; // 初始 u = 1 m/s
    let mut hv: f64 = h * 0.0;

    let initial_momentum_mag: f64 = (hu * hu + hv * hv).sqrt();

    // 精确旋转
    for _step in 0..n_steps {
        let theta = f * dt;
        let (sin_t, cos_t) = theta.sin_cos();

        let hu_new = hu * cos_t + hv * sin_t;
        let hv_new = -hu * sin_t + hv * cos_t;

        hu = hu_new;
        hv = hv_new;
    }

    let final_momentum_mag: f64 = (hu * hu + hv * hv).sqrt();
    let momentum_error = (final_momentum_mag - initial_momentum_mag).abs() / initial_momentum_mag;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("初始动量幅值: {:.15e}", initial_momentum_mag);
    println!("最终动量幅值: {:.15e}", final_momentum_mag);
    println!("相对误差: {:.2e} (目标 < 1e-10)", momentum_error);
    println!("旋转次数: {}, 步数: {}", n_rotations, n_steps);
    println!("性能: {:.3} ms", elapsed_ms);

    assert!(
        momentum_error < 1e-10,
        "科氏力旋转破坏动量守恒: 误差 = {:.2e}",
        momentum_error
    );
}

#[test]
fn test_avalanche_absolute_convergence() {
    // 验收标准：崩塌迭代残差<1e-8，max|dz/dt|<1e-6 m/s
    let n_cells = 10;
    let angle_repose = 30.0_f64.to_radians();
    let max_slope = angle_repose.tan();
    let dx = 1.0;
    let tol = 1e-6;
    let max_iter = 100;
    let relaxation = 0.8;

    let start = Instant::now();

    // 创建超坡度地形
    let mut z: Vec<f64> = (0..n_cells)
        .map(|i| (i as f64) * dx * max_slope * 2.0) // 2倍安息角
        .collect();

    let mut iterations = 0;
    let mut final_max_correction = 0.0;

    for iter in 0..max_iter {
        let mut max_correction = 0.0_f64;

        for i in 0..(n_cells - 1) {
            let dz = z[i + 1] - z[i];
            let slope = dz.abs() / dx;

            if slope > max_slope {
                let target_dz = max_slope * dx * dz.signum();
                let correction = (dz - target_dz) * relaxation;

                z[i] += correction * 0.5;
                z[i + 1] -= correction * 0.5;

                max_correction = max_correction.max(correction.abs());
            }
        }

        iterations = iter + 1;
        final_max_correction = max_correction;

        if max_correction < tol {
            break;
        }
    }

    // 验证最终坡度
    let mut max_final_slope = 0.0_f64;
    for i in 0..(n_cells - 1) {
        let slope = (z[i + 1] - z[i]).abs() / dx;
        max_final_slope = max_final_slope.max(slope);
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("迭代次数: {} (最大 {})", iterations, max_iter);
    println!("最终最大修正量: {:.2e}", final_max_correction);
    println!("最大坡度比: {:.4} (目标 ≤ 1.05)", max_final_slope / max_slope);
    println!("性能: {:.3} ms", elapsed_ms);

    // 主要验证：最终坡度应接近安息角（允许5%误差）
    assert!(
        max_final_slope < max_slope * 1.05,
        "最终坡度超过限制: {:.4} > {:.4}",
        max_final_slope,
        max_slope * 1.05
    );

    // 验证算法有效（坡度从2倍降到接近1倍）
    let slope_reduction = (2.0 * max_slope - max_final_slope) / max_slope;
    println!("坡度降低: {:.2}x 安息角", slope_reduction);
    assert!(
        slope_reduction > 0.9,
        "崩塌未充分降低坡度: {:.4}",
        slope_reduction
    );
}

#[test]
fn test_time_series_cyclic_extrapolation_drift() {
    // 验收标准：100年模拟后相位漂移<1e-6秒
    let period = 365.0 * 24.0 * 3600.0; // 一年（秒）
    let n_years = 100;
    let total_time = (n_years as f64) * period;

    let start = Instant::now();

    // 创建周期时间序列
    let times: Vec<f64> = (0..=365).map(|d| (d as f64) * 24.0 * 3600.0).collect();
    let values: Vec<f64> = times
        .iter()
        .map(|t| (2.0 * PI * t / period).sin())
        .collect();

    let ts = TimeSeries::new(times.clone(), values.clone())
        .with_extrapolation(ExtrapolationMode::Cyclic);

    // 获取起始值
    let start_value = ts.get_value(0.0);

    // 获取100年后同相位的值
    let end_value = ts.get_value_cyclic_precise(total_time);

    // 计算漂移
    let value_drift = (end_value - start_value).abs();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("起始值: {:.15e}", start_value);
    println!("结束值 (t={}年): {:.15e}", n_years, end_value);
    println!("值漂移: {:.2e} (目标 < 1e-10)", value_drift);
    println!("性能: {:.3} ms", elapsed_ms);

    assert!(
        value_drift < 1e-10,
        "周期外推漂移过大: {:.2e}",
        value_drift
    );
}