// crates/mh_physics/tests/benchmark_implicit.rs

//! 隐式求解器性能基准测试
//! 
//! 测试不同规模问题的求解性能，验证算法可扩展性。
//! 需在release模式下运行：cargo test --release benchmark_ -- --ignored --nocapture

use mh_physics::numerics::linear_algebra::{
    CsrBuilder, CsrMatrix, ConjugateGradient, IterativeSolver, JacobiPreconditioner,
    IdentityPreconditioner, Preconditioner, SolverConfig, SolverResult,
};
use mh_runtime::{Backend, CpuBackend, DeviceBuffer};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

/// 全局Backend实例，确保所有测试共享同一上下文，避免重复分配开销
static BACKEND: LazyLock<CpuBackend<f64>> = LazyLock::new(|| CpuBackend::<f64>::new());

/// 基准测试结果
#[derive(Debug, Clone)]
struct BenchmarkResult {
    problem_size: usize,
    nnz: usize,
    assembly_time: Duration,
    solve_time: Duration,
    iterations: usize,
    residual: f64,
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

/// 生成Laplacian测试矩阵（五点差分格式）
fn generate_laplacian_5pt(n: usize) -> CsrMatrix<f64> {
    let size = n * n;
    let mut builder = CsrBuilder::<f64>::new(size, size);

    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            builder.set(idx, idx, 4.0);

            if j > 0 {
                builder.set(idx, idx - 1, -1.0);
            }
            if j < n - 1 {
                builder.set(idx, idx + 1, -1.0);
            }
            if i > 0 {
                builder.set(idx, idx - n, -1.0);
            }
            if i < n - 1 {
                builder.set(idx, idx + n, -1.0);
            }
        }
    }

    builder.build()
}

/// 生成对称正定随机右端向量
fn generate_rhs(size: usize, seed: u64, backend: &CpuBackend<f64>) -> <CpuBackend<f64> as Backend>::Buffer<f64> {
    let mut state = seed;
    let mut rhs = backend.alloc(size);
    
    for i in 0..size {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        rhs[i] = ((state >> 16) as i32 % 1000) as f64 / 1000.0;
    }
    
    rhs
}

/// 运行单个基准测试
fn run_benchmark<P: Preconditioner<CpuBackend<f64>>>(
    matrix: &CsrMatrix<f64>,
    rhs: &<CpuBackend<f64> as Backend>::Buffer<f64>,
    preconditioner: &P,
    config: &SolverConfig,
    backend: &CpuBackend<f64>,
) -> (Duration, SolverResult<f64>, <CpuBackend<f64> as Backend>::Buffer<f64>) {
    let n = matrix.n_rows();
    let mut x = backend.alloc_init(n, 0.0);
    
    let mut solver = ConjugateGradient::new(backend.clone(), config.clone());
    
    let start = Instant::now();
    let result = solver.solve(matrix, rhs, &mut x, preconditioner);
    let elapsed = start.elapsed();
    
    (elapsed, result, x)
}

#[test]
fn test_csr_matrix_generation() {
    let n = 10;
    let matrix = generate_laplacian_5pt(n);

    assert_eq!(matrix.n_rows(), n * n);
    assert_eq!(matrix.n_cols(), n * n);

    for i in 0..n * n {
        let diag = matrix.get(i, i);
        assert!((diag - 4.0).abs() < 1e-10);
    }
}

#[test]
fn test_solver_convergence_small() {
    let n = 10;
    let matrix = generate_laplacian_5pt(n);
    let backend = (*BACKEND).clone();
    let rhs = generate_rhs(n * n, 42, &backend);
    
    let config = SolverConfig::new(1e-10, 1000);
    
    let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(backend.clone(), &matrix).unwrap();
    let (elapsed, result, _x) = run_benchmark(&matrix, &rhs, &precond, &config, &backend);
    
    println!("小规模问题 ({}x{} = {} 单元): 求解时间 {:?}, 迭代次数 {}", n, n, n * n, elapsed, result.iterations);
    
    assert!(result.is_converged());
    assert!(result.residual_norm < 1e-8);
}

#[test]
fn test_preconditioner_comparison() {
    let n = 20;
    let matrix = generate_laplacian_5pt(n);
    let backend = (*BACKEND).clone();
    let rhs = generate_rhs(n * n, 42, &backend);
    
    let config = SolverConfig::new(1e-10, 1000);
    
    let no_precond = IdentityPreconditioner::<CpuBackend<f64>>::new(backend.clone());
    let (time_no, result_no, _) = run_benchmark(&matrix, &rhs, &no_precond, &config, &backend);
    
    let jacobi = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(backend.clone(), &matrix).unwrap();
    let (time_jacobi, result_jacobi, _) = run_benchmark(&matrix, &rhs, &jacobi, &config, &backend);
    
    println!("预条件器对比 ({}x{}): 无预条件器 {} 次迭代 {:?}, Jacobi {} 次迭代 {:?}", 
             n, n, result_no.iterations, time_no, result_jacobi.iterations, time_jacobi);
    
    assert!(result_no.is_converged());
    assert!(result_jacobi.is_converged());
}

/// 测试不同问题规模的求解性能，验证算法可扩展性
#[test]
#[ignore = "性能基准测试：需 --release -- --ignored --nocapture"]
fn benchmark_scaling() {
    let sizes = [16, 32, 64, 128, 256];
    let mut results = Vec::new();
    
    let config = SolverConfig::new(1e-10, 5000);
    
    println!("\n=== 规模扩展性测试 ===\n");
    println!("{:>10} {:>12} {:>15} {:>15} {:>10}", "N", "单元数", "非零元数", "求解时间(ms)", "迭代次数");
    println!("{}", "-".repeat(65));
    
    for &n in &sizes {
        let size = n * n;
        
        let start_asm = Instant::now();
        let matrix = generate_laplacian_5pt(n);
        let assembly_time = start_asm.elapsed();
        
        let backend = (*BACKEND).clone();
        let rhs = generate_rhs(size, 42, &backend);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(backend.clone(), &matrix).unwrap();
        
        let (solve_time, result, _) = run_benchmark(&matrix, &rhs, &precond, &config, &backend);
        
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
            n, size, matrix.nnz(), solve_time.as_secs_f64() * 1000.0, result.iterations);
        
        results.push(bench_result);
    }
    
    println!("\n=== 结果摘要 ===\n");
    for result in &results {
        result.print();
        println!();
    }
    
    // PCG 求解器的复杂度约为 O(n * iter)，其中 iter ~ O(√n)
    // 因此总体复杂度约 O(n^1.5)，但实际中由于缓存效应等可能更高
    // 这里使用 O(n^2) 作为保守上界
    if results.len() >= 3 {
        let time_16 = results[0].solve_time.as_secs_f64();
        let time_64 = results[2].solve_time.as_secs_f64();
        // 规模从 16x16 到 64x64，单元数增长 16 倍
        // 对于 O(n^2) 复杂度，时间应增长 16^2 = 256 倍
        // 使用更宽松的 O(n^1.5) 作为预期，允许 3 倍余量
        let scale_factor = ((64.0 / 16.0_f64).powi(2) as f64).powf(1.5); // n^1.5 增长
        let expected_time = time_16 * scale_factor * 3.0;
        
        assert!(time_64 < expected_time, 
            "求解时间增长异常：64x64 耗时 {:.4?}s，预期 < {:.4?}s", time_64, expected_time);
    }
}

/// 测试迭代次数与问题规模的关系
#[test]
#[ignore = "性能基准测试：需 --release -- --ignored --nocapture"]
fn benchmark_iteration_count() {
    let sizes = [16, 32, 64, 128];
    
    let config = SolverConfig::new(1e-12, 10000);
    
    println!("\n=== 迭代次数扩展性测试 ===\n");
    println!("{:>10} {:>12} {:>15} {:>15}", "N", "单元数", "迭代次数", "Iter/√N");
    println!("{}", "-".repeat(55));
    
    for &n in &sizes {
        let matrix = generate_laplacian_5pt(n);
        let backend = (*BACKEND).clone();
        let rhs = generate_rhs(n * n, 42, &backend);
        let precond = JacobiPreconditioner::<CpuBackend<f64>>::from_matrix(backend.clone(), &matrix).unwrap();
        
        let (_time, result, _) = run_benchmark(&matrix, &rhs, &precond, &config, &backend);
        
        let iter_per_sqrt_n = result.iterations as f64 / (n as f64).sqrt();
        
        println!("{:>10} {:>12} {:>15} {:>15.2}", n, n * n, result.iterations, iter_per_sqrt_n);
    }
}

#[test]
fn test_spmv_performance_small() {
    let n = 50;
    let matrix = generate_laplacian_5pt(n);
    let backend = (*BACKEND).clone();
    let x = generate_rhs(n * n, 42, &backend);
    let mut y = backend.alloc_init(n * n, 0.0);
    
    let iterations = 100;
    let start = Instant::now();
    
    for _ in 0..iterations {
        matrix.mul_vec(x.as_slice(), y.as_slice_mut());
    }
    
    let elapsed = start.elapsed();
    let per_iter = elapsed / iterations as u32;
    
    println!("SpMV性能 ({}x{} 矩阵，{} 非零元): {} 次迭代，单次 {:?}", n * n, n * n, matrix.nnz(), iterations, per_iter);
}

/// 测试矩阵-向量乘法性能
#[test]
#[ignore = "性能基准测试：需 --release -- --ignored --nocapture"]
fn benchmark_spmv_scaling() {
    let sizes = [32, 64, 128, 256, 512];
    let iterations = 1000;
    
    println!("\n=== SpMV性能测试 ===\n");
    println!("{:>10} {:>12} {:>15} {:>15} {:>15}", "N", "非零元", "单次时间(us)", "GFLOPS", "GB/s");
    println!("{}", "-".repeat(70));
    
    for &n in &sizes {
        let matrix = generate_laplacian_5pt(n);
        let backend = (*BACKEND).clone();
        let x = generate_rhs(n * n, 42, &backend);
        let mut y = backend.alloc_init(n * n, 0.0);
        
        for _ in 0..10 {
            matrix.mul_vec(x.as_slice(), y.as_slice_mut());
        }
        
        let start = Instant::now();
        for _ in 0..iterations {
            matrix.mul_vec(x.as_slice(), y.as_slice_mut());
            y.fill(0.0);
        }
        let elapsed = start.elapsed();
        
        let per_iter_us = elapsed.as_secs_f64() * 1e6 / iterations as f64;
        let flops = 2.0 * matrix.nnz() as f64;
        let gflops = flops / per_iter_us / 1e3;
        let memory_bytes = (matrix.nnz() * 16 + matrix.n_rows() * 8) as f64;
        let memory_gb_s = memory_bytes * iterations as f64 / elapsed.as_secs_f64() / 1e9;
        
        println!("{:>10} {:>12} {:>15.2} {:>15.3} {:>15.1}", n * n, matrix.nnz(), per_iter_us, gflops, memory_gb_s);
    }
}

#[test]
fn test_memory_estimate() {
    let sizes = [100, 1000, 10000, 100000];
    
    println!("\n=== 内存使用估算 ===\n");
    println!("{:>12} {:>12} {:>15} {:>15}", "单元数", "预估非零元", "CSR(MB)", "向量(MB)");
    println!("{}", "-".repeat(60));
    
    for &size in &sizes {
        let nnz = size * 5;
        let csr_bytes = nnz * 8 + nnz * 8 + (size + 1) * 8;
        let csr_mb = csr_bytes as f64 / 1024.0 / 1024.0;
        let vec_bytes = size * 8 * 6;
        let vec_mb = vec_bytes as f64 / 1024.0 / 1024.0;
        
        println!("{:>12} {:>12} {:>15.2} {:>15.2}", size, nnz, csr_mb, vec_mb);
    }
}