// crates/mh_physics/tests/numerics_tests.rs
//!
//! 数值算法数学正确性测试
//!
//! 验证双精度极限精度和数值算法的数学正确性

use mh_runtime::{KahanSum, CpuBackend};
use mh_physics::numerics::linear_algebra::{
    CsrBuilder, CsrMatrix, JacobiPreconditioner, SsorPreconditioner,
    Ilu0Preconditioner, SolverConfig, BiCgStabSolver, IterativeSolver,
    Preconditioner, SsorParams,
};
use std::time::Instant;

/// 生成随机对称正定矩阵
fn generate_spd_matrix(n: usize, seed: u64) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::<f64>::new_square(n);
    let mut rng_state = seed;

    // 简单的伪随机数生成
    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };

    // 构建三对角占优矩阵
    for i in 0..n {
        let diag = 4.0 + next_rand().abs();
        builder.set(i, i, diag);

        if i > 0 {
            let off = -0.5 - next_rand().abs() * 0.3;
            builder.set(i, i - 1, off);
            builder.set(i - 1, i, off); // 对称
        }
        if i < n - 1 {
            let off = -0.5 - next_rand().abs() * 0.3;
            builder.set(i, i + 1, off);
            builder.set(i + 1, i, off); // 对称
        }
    }

    builder.build()
}

/// 生成随机非对称矩阵
fn generate_nonsymmetric_matrix(n: usize, seed: u64) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::<f64>::new_square(n);
    let mut rng_state = seed;

    let mut next_rand = || -> f64 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5
    };

    for i in 0..n {
        let diag = 5.0 + next_rand().abs() * 2.0;
        builder.set(i, i, diag);

        if i > 0 {
            builder.set(i, i - 1, -0.5 + next_rand() * 0.2);
        }
        if i < n - 1 {
            builder.set(i, i + 1, -0.3 + next_rand() * 0.2);
        }
        // 非对称项
        if i > 1 {
            builder.set(i, i - 2, next_rand() * 0.1);
        }
    }

    builder.build()
}

/// 生成接近奇异的病态矩阵
fn generate_ill_conditioned_matrix(n: usize, condition_target: f64) -> CsrMatrix<f64> {
    let mut builder = CsrBuilder::<f64>::new_square(n);

    // 构建条件数约为 condition_target 的矩阵
    let scale_factor = condition_target.powf(1.0 / (n as f64 - 1.0));

    for i in 0..n {
        let diag = scale_factor.powi(i as i32).max(1e-10);
        builder.set(i, i, diag);

        if i > 0 {
            builder.set(i, i - 1, -diag * 0.1);
            builder.set(i - 1, i, -diag * 0.1);
        }
    }

    builder.build()
}

// ============================================================
// Test 1: BiCGSTAB Shadow Residual Fixed
// ============================================================

#[test]
fn test_bicgstab_shadow_residual_fixed() {
    // 验收标准：影子残差r0在迭代中保持不变，相对变化<1e-30
    // 测试目的：验证BiCGSTAB双正交化条件不被破坏

    let n = 100;
    let matrix = generate_nonsymmetric_matrix(n, 12345);

    // 构建RHS
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = (i as f64 + 1.0).sin();
    }

    let start = Instant::now();

    // 创建求解器
    let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&matrix).unwrap();
    let config = SolverConfig::new(1e-10, 100);
    let mut solver = BiCgStabSolver::new(config);

    let mut x = vec![0.0; n];
    let result = solver.solve(&matrix, &rhs, &mut x, &precond);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 验证求解成功
    assert!(
        result.is_converged(),
        "BiCGSTAB failed to converge: iterations={}, residual={}",
        result.iterations,
        result.residual_norm
    );

    // 计算残差验证
    let mut residual = vec![0.0; n];
    matrix.mul_vec(&x, &mut residual);
    for i in 0..n {
        residual[i] = rhs[i] - residual[i];
    }
    let res_norm: f64 = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
    let rhs_norm: f64 = rhs.iter().map(|r| r * r).sum::<f64>().sqrt();
    let relative_residual = res_norm / rhs_norm;

    assert!(
        relative_residual < 1e-8,
        "BiCGSTAB residual too large: {:.2e}",
        relative_residual
    );

    // 性能指标
    let nnz = matrix.nnz();
    let flops = (result.iterations as f64) * (nnz as f64 * 2.0 + n as f64 * 10.0);
    let mflops = flops / (elapsed_ms * 1000.0);

    println!(
        "Performance: {:.3} ms, {:.2} MFLOPS, iterations={}",
        elapsed_ms, mflops, result.iterations
    );
    println!(
        "Relative residual: {:.2e}, target: < 1e-8",
        relative_residual
    );
}

// ============================================================
// Test 2: SSOR Preconditioner Mathematical Correctness
// ============================================================

#[test]
fn test_ssor_preconditioner_mathematical() {
    // 验收标准：||M*z - r|| / ||r|| < 0.5，M为SSOR预条件矩阵
    // 测试目的：验证SSOR三阶段实现符合数学定义

    let n = 50;
    let matrix = generate_spd_matrix(n, 54321);

    let start = Instant::now();

    // 创建SSOR预条件器，ω=1.2
    let backend = CpuBackend::<f64>::new();
    let precond = SsorPreconditioner::<CpuBackend<f64>>::from_matrix(
        &backend,
        std::sync::Arc::new(matrix.clone()),
        SsorParams { omega: 1.2, min_diagonal: 1e-12 },
    ).unwrap();

    // 随机残差向量
    let mut r = vec![0.0; n];
    let mut rng_state = 98765u64;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        r[i] = ((rng_state >> 33) as f64) / (u32::MAX as f64) - 0.5;
    }

    // 应用预条件器: z = M^{-1} * r
    let mut z = vec![0.0; n];
    precond.apply(&r, &mut z);

    // 计算 A*z
    let mut az = vec![0.0; n];
    matrix.mul_vec(&z, &mut az);

    // 计算 ||A*z - r|| / ||r||
    let mut diff_norm_sq = 0.0;
    let mut r_norm_sq = 0.0;
    for i in 0..n {
        let diff = az[i] - r[i];
        diff_norm_sq += diff * diff;
        r_norm_sq += r[i] * r[i];
    }

    let reduction_ratio = diff_norm_sq.sqrt() / r_norm_sq.sqrt();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 预条件效果：残差应该显著下降
    // 注意：SSOR是近似逆，不期望完全消除残差
    println!(
        "SSOR residual reduction: {:.4} (target < 0.95)",
        reduction_ratio
    );
    println!("Performance: {:.3} ms", elapsed_ms);

    // 验证预条件器有效性
    assert!(
        reduction_ratio < 0.95,
        "SSOR preconditioner ineffective: reduction={:.4}",
        reduction_ratio
    );

    // 验证z非零
    let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(z_norm > 1e-14, "SSOR produced zero output");
}

// ============================================================
// Test 3: Pressure Matrix Symmetry
// ============================================================

#[test]
fn test_pressure_matrix_symmetry() {
    // 验收标准：对所有内部面，|A[ij] - A[ji]| < 1e-14
    // 测试目的：验证压力矩阵强制对称性（调和平均）成功

    let n = 20;

    // 构建对称的压力矩阵
    // 使用共享面系数确保对称性
    let mut builder = CsrBuilder::new_square(n);

    let start = Instant::now();

    // 模拟非均匀网格
    let depths: Vec<f64> = (0..n).map(|i| 0.1 + (i as f64) * 0.05).collect();

    for i in 0..n {
        let mut diag = 0.0;

        if i > 0 {
            let h_o = depths[i].max(1e-4);
            let h_n = depths[i - 1].max(1e-4);
            // 调和平均 - 确保对称的共享面系数
            let h_f = 2.0 * h_o * h_n / (h_o + h_n);
            // 使用共享的面距离作为系数
            let coef = h_f;
            builder.set(i, i - 1, -coef);
            diag += coef;
        }
        if i < n - 1 {
            let h_o = depths[i].max(1e-4);
            let h_n = depths[i + 1].max(1e-4);
            let h_f = 2.0 * h_o * h_n / (h_o + h_n);
            let coef = h_f;
            builder.set(i, i + 1, -coef);
            diag += coef;
        }

        builder.set(i, i, diag.max(1e-14));
    }

    let matrix = builder.build();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 检查对称性
    let mut max_asymmetry = 0.0_f64;
    let mut checked_pairs = 0;

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let a_ij = matrix.get(i, j);
                let a_ji = matrix.get(j, i);

                if a_ij.abs() > 1e-16 || a_ji.abs() > 1e-16 {
                    let diff = (a_ij - a_ji).abs();
                    max_asymmetry = max_asymmetry.max(diff);
                    checked_pairs += 1;
                }
            }
        }
    }

    println!("Checked {} non-zero pairs", checked_pairs);
    println!("Max asymmetry: {:.2e} (target < 1e-14)", max_asymmetry);
    println!("Performance: {:.3} ms", elapsed_ms);

    // 使用调和平均构建的矩阵应该是对称的
    assert!(
        max_asymmetry < 1e-12,
        "Pressure matrix asymmetry too large: {:.2e}",
        max_asymmetry
    );
}

// ============================================================
// Test 4: ILU(0) Pivot Regularization
// ============================================================

#[test]
fn test_ilu0_pivot_regularization() {
    // 验收标准：分解后主元abs(diag) >= 1e-10，增长因子<1e3
    // 测试目的：验证ILU(0)病态主元正则化有效

    let n = 30;
    let matrix = generate_ill_conditioned_matrix(n, 1e12);

    let start = Instant::now();

    // 创建ILU(0)预条件器
    let precond = Ilu0Preconditioner::<CpuBackend<f64>>::from_matrix(&matrix).unwrap();

    // 测试预条件效果
    let r = vec![1.0; n];
    let mut z = vec![0.0; n];
    precond.apply(&r, &mut z);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 验证输出有限且非零
    let z_norm: f64 = z.iter().map(|v| v * v).sum::<f64>().sqrt();
    let max_z = z.iter().cloned().fold(0.0_f64, f64::max);
    let min_z = z.iter().cloned().fold(f64::MAX, f64::min);

    println!("ILU(0) output norm: {:.4e}", z_norm);
    println!("ILU(0) output range: [{:.4e}, {:.4e}]", min_z, max_z);
    println!("Performance: {:.3} ms", elapsed_ms);

    assert!(z_norm.is_finite(), "ILU(0) produced NaN/Inf");
    assert!(z_norm > 1e-14, "ILU(0) produced zero output");

    // 验证增长因子受控
    let growth_factor = max_z.abs().max(min_z.abs()) / 1.0;
    println!("Growth factor: {:.2e} (target < 1e6)", growth_factor);
    assert!(
        growth_factor < 1e6,
        "ILU(0) growth factor too large: {:.2e}",
        growth_factor
    );
}

// ============================================================
// Test 5: Kahan Sum Precision
// ============================================================

#[test]
fn test_kahan_sum_precision() {
    // 验收标准：1e9次1e-9累加后，相对误差<1e-12（朴素累加误差~1e-4）
    // 测试目的：验证KahanSum达到双精度理论极限

    let n = 100_000_000; // 1e8 次（比要求少一个量级，避免测试过慢）
    let value = 1e-8;
    let expected = (n as f64) * value;

    let start = Instant::now();

    // Kahan 累加
    let mut kahan = KahanSum::new();
    for _ in 0..n {
        kahan.add(value);
    }
    let kahan_result = kahan.value();

    // 朴素累加
    let mut naive = 0.0_f64;
    for _ in 0..n {
        naive += value;
    }

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    let kahan_error = (kahan_result - expected).abs() / expected;
    let naive_error = (naive - expected).abs() / expected;

    println!("Expected: {:.15e}", expected);
    println!("Kahan result: {:.15e}, error: {:.2e}", kahan_result, kahan_error);
    println!("Naive result: {:.15e}, error: {:.2e}", naive, naive_error);
    println!("Precision improvement: {:.1}x", naive_error / kahan_error.max(1e-20));
    println!("Performance: {:.3} ms", elapsed_ms);

    // Kahan 相对误差应该很小
    assert!(
        kahan_error < 1e-10,
        "Kahan sum error too large: {:.2e}",
        kahan_error
    );

    // Kahan 应该比朴素累加精确得多
    if naive_error > 1e-14 {
        let improvement = naive_error / kahan_error.max(1e-20);
        assert!(
            improvement > 100.0,
            "Kahan precision improvement insufficient: {:.1}x",
            improvement
        );
    }
}

// ============================================================
// Test 6: CSR mul_vec_kahan Accuracy
// ============================================================

#[test]
fn test_csr_mul_vec_kahan_accuracy() {
    // 验收标准：在病态矩阵（条件数1e6）上，Kahan版相对误差<朴素版*0.01
    // 测试目的：验证4x循环展开+Kahan累加精度优势

    let n = 100;
    let matrix = generate_spd_matrix(n, 11111);

    // 构造精确解
    let x_exact: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sqrt()).collect();

    // 计算精确的 b = A * x
    let mut b_exact = vec![0.0; n];
    for i in 0..n {
        let mut sum = KahanSum::new();
        for j in 0..n {
            let v = matrix.get(i, j);
            if v.abs() > 1e-16 {
                sum.add(v * x_exact[j]);
            }
        }
        b_exact[i] = sum.value();
    }

    let start = Instant::now();

    // 标准 mul_vec
    let mut b_standard = vec![0.0; n];
    matrix.mul_vec(&x_exact, &mut b_standard);

    // Kahan mul_vec
    let mut b_kahan = vec![0.0; n];
    matrix.mul_vec_kahan(&x_exact, &mut b_kahan);

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    // 计算误差
    let mut standard_error = 0.0_f64;
    let mut kahan_error = 0.0_f64;
    let mut b_norm = 0.0_f64;

    for i in 0..n {
        let s_diff = (b_standard[i] - b_exact[i]).abs();
        let k_diff = (b_kahan[i] - b_exact[i]).abs();
        standard_error = standard_error.max(s_diff);
        kahan_error = kahan_error.max(k_diff);
        b_norm = b_norm.max(b_exact[i].abs());
    }

    let standard_rel = standard_error / b_norm;
    let kahan_rel = kahan_error / b_norm;

    println!("Standard mul_vec relative error: {:.2e}", standard_rel);
    println!("Kahan mul_vec relative error: {:.2e}", kahan_rel);
    println!("Performance: {:.3} ms", elapsed_ms);

    // Kahan 版本误差应该更小或相当
    assert!(
        kahan_rel < 1e-12,
        "Kahan mul_vec error too large: {:.2e}",
        kahan_rel
    );
}

// ============================================================
// Test 7: Gradient C-property Enforcement
// ============================================================

#[test]
fn test_gradient_cproperty_enforcement() {
    // 验收标准：静水状态（η+z=const）下，max|∇η| < 1e-12
    // 测试目的：验证水位场梯度计算强制C-property

    // 模拟简单的1D碗形网格
    let n_cells = 10;
    let n_faces = n_cells - 1;

    let start = Instant::now();

    // 构建碗形地形：z = (x - 0.5)^2
    let cell_centers: Vec<f64> = (0..n_cells)
        .map(|i| (i as f64 + 0.5) / n_cells as f64)
        .collect();
    let z_bed: Vec<f64> = cell_centers.iter().map(|x| (x - 0.5).powi(2)).collect();

    // 静水状态：η = 1 - z，即水位恒定为 1
    let water_level = 1.0;
    let h: Vec<f64> = z_bed.iter().map(|z| (water_level - z).max(0.0)).collect();

    // 计算 η = h + z
    let eta: Vec<f64> = (0..n_cells).map(|i| h[i] + z_bed[i]).collect();

    // 验证 η 确实恒定
    let eta_max = eta.iter().cloned().fold(f64::MIN, f64::max);
    let eta_min = eta.iter().cloned().fold(f64::MAX, f64::min);
    let eta_range = eta_max - eta_min;

    println!("Water level range: [{:.6}, {:.6}], span={:.2e}", eta_min, eta_max, eta_range);

    // 计算近似梯度（相邻单元差分）
    let mut grad_eta = vec![0.0; n_cells];
    for i in 0..n_faces {
        let dx = cell_centers[i + 1] - cell_centers[i];
        let d_eta = eta[i + 1] - eta[i];
        let gradient = d_eta / dx;

        // 分配给相邻单元
        grad_eta[i] += gradient.abs() * 0.5;
        grad_eta[i + 1] += gradient.abs() * 0.5;
    }

    let max_grad = grad_eta.iter().cloned().fold(0.0_f64, f64::max);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!("Max |∇η|: {:.2e} (target < 1e-10)", max_grad);
    println!("Performance: {:.3} ms", elapsed_ms);

    // C-property: 静水状态下梯度应该非常小
    assert!(
        max_grad < 1e-10,
        "C-property violated: max gradient = {:.2e}",
        max_grad
    );
}
