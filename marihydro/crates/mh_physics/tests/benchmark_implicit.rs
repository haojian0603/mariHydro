// crates/mh_physics/tests/benchmark_implicit.rs

//! 隐式求解器性能基准测试
//! 
//! # 运行方法
//! 
//! 所有基准测试都需要在 release 模式下运行：
//! 
//! ```bash
//! # 运行所有基准测试
//! cargo test --release benchmark_ -- --ignored --nocapture
//! 
//! # 运行单个测试
//! cargo test --release benchmark_scaling -- --ignored --nocapture
//! ```
//! 
//! # 性能预期
//! 
//! - **小规模问题** (N=100)：求解时间 < 10ms
//! - **中等规模** (N=10,000)：求解时间 < 100ms
//! - **大规模** (N=100,000)：求解时间 < 1s
//! - **SpMV性能**：>5 GFLOPS (AVX2)，>10 GFLOPS (AVX-512)
//! - **迭代次数**：应随问题规模缓慢增长（O(N^0.5)）
//! 
//! # 注意事项
//! 
//! 1. 确保关闭其他占用CPU的程序
//! 2. 多次运行取平均值
//! 3. 在CI环境中应设置宽松的性能阈值

use mh_foundation::AlignedVec;
use mh_physics::numerics::linear_algebra::{
    CsrBuilder, CsrMatrix, ConjugateGradient, IterativeSolver, JacobiPreconditioner,
    IdentityPreconditioner, Preconditioner, ScalarPreconditioner, SolverConfig, SolverResult,
};
use mh_runtime::CpuBackend;
use std::time::{Duration, Instant};

// ============================================================
// 基准测试配置
// ============================================================

/// 基准测试结果
#[derive(Debug, Clone)]
struct BenchmarkResult {
    /// 问题规模
    problem_size: usize,
    /// 非零元素数
    nnz: usize,
    /// 组装时间
    assembly_time: Duration,
    /// 求解时间
    solve_time: Duration,
    /// 迭代次数
    iterations: usize,
    /// 残差
    residual: f64,
    /// 是否收敛
    converged: bool,
}

impl BenchmarkResult {
    fn print(&self) {
        println!("=== Benchmark Result ===");
        println!("  Problem size: {} cells", self.problem_size);
        println!("  Non-zeros: {} ({:.2} avg per row)", 
            self.nnz, 
            self.nnz as f64 / self.problem_size as f64
        );
        println!("  Assembly time: {:?}", self.assembly_time);
        println!("  Solve time: {:?}", self.solve_time);
        println!("  Iterations: {}", self.iterations);
        println!("  Residual: {:.2e}", self.residual);
        println!("  Converged: {}", self.converged);
    }
}

// ============================================================
// 测试矩阵生成
// ============================================================

/// 生成 Laplacian 测试矩阵（五点差分格式）
fn generate_laplacian_5pt(n: usize) -> CsrMatrix<f64> {
    let size = n * n;
    let mut builder = CsrBuilder::<f64>::new(size, size);

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;

            // 对角线元素
            builder.set(idx, idx, 4.0);

            // 左邻居
            if j > 0 {
                builder.set(idx, idx - 1, -1.0);
            }

            // 右邻居
            if j < n - 1 {
                builder.set(idx, idx + 1, -1.0);
            }

            // 上邻居
            if i > 0 {
                builder.set(idx, idx - n, -1.0);
            }

            // 下邻居
            if i < n - 1 {
                builder.set(idx, idx + n, -1.0);
            }
        }
    }

    builder.build()
}

/// 生成带权重的测试矩阵（模拟非均匀网格）
#[allow(dead_code)]
fn generate_weighted_laplacian(n: usize, weights: &[f64]) -> CsrMatrix<f64> {
    let size = n * n;
    let mut builder = CsrBuilder::<f64>::new(size, size);

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let w = weights.get(idx).copied().unwrap_or(1.0);

            builder.set(idx, idx, 4.0 * w);

            if j > 0 {
                let w_nb = weights.get(idx - 1).copied().unwrap_or(1.0);
                builder.set(idx, idx - 1, -(w + w_nb) / 2.0);
            }

            if j < n - 1 {
                let w_nb = weights.get(idx + 1).copied().unwrap_or(1.0);
                builder.set(idx, idx + 1, -(w + w_nb) / 2.0);
            }

            if i > 0 {
                let w_nb = weights.get(idx - n).copied().unwrap_or(1.0);
                builder.set(idx, idx - n, -(w + w_nb) / 2.0);
            }

            if i < n - 1 {
                let w_nb = weights.get(idx + n).copied().unwrap_or(1.0);
                builder.set(idx, idx + n, -(w + w_nb) / 2.0);
            }
        }
    }

    builder.build()
}

/// 生成对称正定随机右端向量
fn generate_rhs(size: usize, seed: u64) -> AlignedVec<f64> {
    // 简单 LCG 伪随机
    let mut state = seed;
    let mut rhs = AlignedVec::zeros(size);
    
    for i in 0..size {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        rhs[i] = ((state >> 16) as i32 % 1000) as f64 / 1000.0;
    }
    
    rhs
}

// ============================================================
// 求解器基准测试
// ============================================================

/// 运行单个基准测试
fn run_benchmark<P: Preconditioner<CpuBackend<f64>> + ScalarPreconditioner<f64>>(
    matrix: &CsrMatrix<f64>,
    rhs: &[f64],
    preconditioner: &P,
    config: &SolverConfig,
) -> (Duration, SolverResult<f64>, AlignedVec<f64>) {
    let n = matrix.n_rows();
    let mut x = AlignedVec::zeros(n);
    
    let mut solver = ConjugateGradient::new(config.clone());
    
    let start = Instant::now();
    let result = solver.solve(matrix, rhs, &mut x, preconditioner);
    let elapsed = start.elapsed();
    
    (elapsed, result, x)
}

// ============================================================
// 测试用例
// ============================================================

#[test]
fn test_csr_matrix_generation() {
    // 验证矩阵生成的正确性
    let n = 10;
    let matrix = generate_laplacian_5pt(n);

    assert_eq!(matrix.n_rows(), n * n);
    assert_eq!(matrix.n_cols(), n * n);

    // 对角线元素应为 4
    for i in 0..n * n {
        let diag = matrix.get(i, i);
        assert!((diag - 4.0).abs() < 1e-10);
    }
}

#[test]
fn test_solver_convergence_small() {
    let n = 10;
    let matrix = generate_laplacian_5pt(n);
    let rhs = generate_rhs(n * n, 42);
    
    let config = SolverConfig::new(1e-10, 1000);
    
    let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&matrix).unwrap();
    let (elapsed, result, _x) = run_benchmark(&matrix, &rhs, &precond, &config);
    
    println!("Small problem ({}x{} = {} cells):", n, n, n * n);
    println!("  Solve time: {:?}", elapsed);
    println!("  Iterations: {}", result.iterations);
    println!("  Residual: {:.2e}", result.residual_norm);
    println!("  Converged: {}", result.is_converged());
    
    assert!(result.is_converged());
    assert!(result.residual_norm < 1e-8);
}

#[test]
fn test_preconditioner_comparison() {
    let n = 20;
    let matrix = generate_laplacian_5pt(n);
    let rhs = generate_rhs(n * n, 42);
    
    let config = SolverConfig::new(1e-10, 1000);
    
    // 无预条件器
    let backend = CpuBackend::<f64>::new();
    let no_precond = IdentityPreconditioner::<CpuBackend<f64>>::new(&backend);
    let (time_no, result_no, _) = run_benchmark(&matrix, &rhs, &no_precond, &config);
    
    // Jacobi 预条件器
    let jacobi = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&matrix).unwrap();
    let (time_jacobi, result_jacobi, _) = run_benchmark(&matrix, &rhs, &jacobi, &config);
    
    println!("Preconditioner comparison ({}x{}):", n, n);
    println!("  No preconditioner: {} iters, {:?}", result_no.iterations, time_no);
    println!("  Jacobi: {} iters, {:?}", result_jacobi.iterations, time_jacobi);
    
    // Jacobi 通常应减少迭代次数
    // 但这取决于具体问题，这里只验证都收敛
    assert!(result_no.is_converged());
    assert!(result_jacobi.is_converged());
}

// ============================================================
// 规模扩展性测试（ignored，需要 --release）
// ============================================================

/// 测试不同问题规模的求解性能
/// 
/// 运行命令：cargo test --release benchmark_scaling -- --ignored --nocapture
/// 预期：求解时间近似线性增长（O(N)），验证算法可扩展性
#[test]
#[ignore = "性能基准测试：需 --release -- --ignored --nocapture"]
fn benchmark_scaling() {
    let sizes = [16, 32, 64, 128, 256];
    let mut results = Vec::new();
    
    let config = SolverConfig::new(1e-10, 5000);
    
    println!("\n=== Scaling Benchmark ===\n");
    println!("{:>10} {:>12} {:>15} {:>15} {:>10}", 
        "N", "Cells", "NNZ", "Time(ms)", "Iterations");
    println!("{}", "-".repeat(65));
    
    for &n in &sizes {
        let size = n * n;
        
        // 组装
        let start_asm = Instant::now();
        let matrix = generate_laplacian_5pt(n);
        let assembly_time = start_asm.elapsed();
        
        let rhs = generate_rhs(size, 42);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&matrix).unwrap();
        
        // 求解
        let (solve_time, result, _) = run_benchmark(&matrix, &rhs, &precond, &config);
        
        let bench_result = BenchmarkResult {
            problem_size: size,
            nnz: matrix.nnz(),
            assembly_time,
            solve_time,
            iterations: result.iterations,
            residual: result.residual_norm,
            converged: result.is_converged(),
        };
        
        println!("{:>10} {:>12} {:>15} {:>15.2} {:>10}",
            n,
            size,
            matrix.nnz(),
            solve_time.as_secs_f64() * 1000.0,
            result.iterations
        );
        
        results.push(bench_result);
    }
    
    println!("\n=== Summary ===\n");
    for result in &results {
        result.print();
        println!();
    }
    
    // 验证可扩展性：检查时间增长是否近似线性
    if results.len() >= 3 {
        let time_16 = results[0].solve_time.as_secs_f64();
        let time_64 = results[2].solve_time.as_secs_f64();
        let scale_factor = (64.0 / 16.0) as f64;
        let expected_time = time_16 * scale_factor;
        
        // 允许50%的偏差（包含预条件器 setup 开销）
        assert!(time_64 < expected_time * 1.5, 
            "求解时间增长非线性：64x64 耗时 {:.2?}，预期 < {:.2?}", 
            time_64, expected_time * 1.5);
    }
}

/// 测试迭代次数与问题规模的关系
/// 
/// 运行命令：cargo test --release benchmark_iteration_count -- --ignored --nocapture
/// 预期：迭代次数随问题规模缓慢增长（O(N^0.5)）
#[test]
#[ignore = "性能基准测试：需 --release -- --ignored --nocapture"]
fn benchmark_iteration_count() {
    let sizes = [16, 32, 64, 128];
    
    let config = SolverConfig::new(1e-12, 10000);
    
    println!("\n=== Iteration Count Scaling ===\n");
    println!("{:>10} {:>12} {:>15} {:>15}", "N", "Cells", "Iterations", "Iter/√N");
    println!("{}", "-".repeat(55));
    
    for &n in &sizes {
        let matrix = generate_laplacian_5pt(n);
        let rhs = generate_rhs(n * n, 42);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(&matrix).unwrap();
        
        let (_time, result, _) = run_benchmark(&matrix, &rhs, &precond, &config);
        
        let iter_per_sqrt_n = result.iterations as f64 / (n as f64).sqrt();
        
        println!("{:>10} {:>12} {:>15} {:>15.2}", n, n * n, result.iterations, iter_per_sqrt_n);
    }
}

// ============================================================
// 矩阵向量乘法性能
// ============================================================

#[test]
fn test_spmv_performance_small() {
    let n = 50;
    let matrix = generate_laplacian_5pt(n);
    let x = generate_rhs(n * n, 42);
    let mut y = vec![0.0; n * n];
    
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        matrix.mul_vec(&x, &mut y);
    }
    
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    
    println!("SpMV performance ({}x{} matrix, {} nnz):", n * n, n * n, matrix.nnz());
    println!("  {} iterations in {:?}", iterations, elapsed);
    println!("  Per iteration: {:?}", per_iter);
}

/// 测试矩阵-向量乘法性能
/// 
/// 运行命令：cargo test --release benchmark_spmv_scaling -- --ignored --nocapture
/// 预期：>5 GFLOPS (AVX2)，>10 GFLOPS (AVX-512)
#[test]
#[ignore = "性能基准测试：需 --release -- --ignored --nocapture"]
fn benchmark_spmv_scaling() {
    let sizes = [32, 64, 128, 256, 512];
    let iterations = 1000;
    
    println!("\n=== SpMV Scaling Benchmark ===\n");
    println!("{:>10} {:>12} {:>15} {:>15} {:>15}", "N", "NNZ", "Time/iter(us)", "GFLOPS", "GB/s");
    println!("{}", "-".repeat(70));
    
    for &n in &sizes {
        let matrix = generate_laplacian_5pt(n);
        let x = generate_rhs(n * n, 42);
        let mut y = vec![0.0; n * n];
        
        // 预热
        for _ in 0..10 {
            matrix.mul_vec(&x, &mut y);
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            matrix.mul_vec(&x, &mut y);
            y.fill(0.0); // 重置避免依赖
        }
        let elapsed = start.elapsed();
        
        let per_iter_us = elapsed.as_secs_f64() * 1e6 / iterations as f64;
        let flops = 2.0 * matrix.nnz() as f64; // 每次乘加2 FLOPs
        let gflops = flops / per_iter_us / 1e3; // GFLOPS
        let memory_bytes = (matrix.nnz() * 16 + matrix.n_rows() * 8) as f64; // 矩阵+向量
        let memory_gb_s = memory_bytes * iterations as f64 / elapsed.as_secs_f64() / 1e9;
        
        println!("{:>10} {:>12} {:>15.2} {:>15.3} {:>15.1}", n * n, matrix.nnz(), per_iter_us, gflops, memory_gb_s);
    }
}

// ============================================================
// 内存使用估算
// ============================================================

#[test]
fn test_memory_estimate() {
    let sizes = [100, 1000, 10000, 100000];
    
    println!("\n=== Memory Usage Estimate ===\n");
    println!("{:>12} {:>12} {:>15} {:>15}", "Cells", "NNZ(est)", "CSR(MB)", "Vectors(MB)");
    println!("{}", "-".repeat(60));
    
    for &size in &sizes {
        // 估算非零元素数（五点格式，平均 ~5）
        let nnz = size * 5;
        
        // CSR 格式内存
        // - values: nnz * 8 bytes
        // - col_indices: nnz * 8 bytes (usize)
        // - row_ptr: (size+1) * 8 bytes
        let csr_bytes = nnz * 8 + nnz * 8 + (size + 1) * 8;
        let csr_mb = csr_bytes as f64 / 1024.0 / 1024.0;
        
        // 向量内存（x, b, r, p, Ap 等，约 6 个向量）
        let vec_bytes = size * 8 * 6;
        let vec_mb = vec_bytes as f64 / 1024.0 / 1024.0;
        
        println!("{:>12} {:>12} {:>15.2} {:>15.2}", size, nnz, csr_mb, vec_mb);
    }
}