// marihydro\crates\mh_physics\src\engine\pcg.rs
//! 预处理共轭梯度法（PCG）求解器
//!
//! 该模块实现了用于求解稀疏对称正定线性系统 Ax = b 的 PCG 算法。
//! 主要用于半隐式时间积分中的压力泊松方程求解。
//!
//! # 算法概述
//!
//! PCG 算法是共轭梯度法（CG）的预处理版本，通过引入预处理矩阵 M
//! 来加速收敛。基本迭代格式为：
//!
//! 1. r_0 = b - A*x_0
//! 2. z_0 = M^{-1} * r_0
//! 3. p_0 = z_0
//! 4. 对于 k = 0, 1, 2, ...
//!    - α_k = (r_k, z_k) / (p_k, A*p_k)
//!    - x_{k+1} = x_k + α_k * p_k
//!    - r_{k+1} = r_k - α_k * A*p_k
//!    - 检查收敛: ||r_{k+1}|| < tol * ||b||
//!    - z_{k+1} = M^{-1} * r_{k+1}
//!    - β_k = (r_{k+1}, z_{k+1}) / (r_k, z_k)
//!    - p_{k+1} = z_{k+1} + β_k * p_k
//!
//! # 预处理器
//!
//! 目前支持的预处理器：
//! - 雅可比（对角）预处理：M = diag(A)
//! - 无预处理（单位矩阵）

use crate::core::{Backend, CpuBackend, Scalar};
use std::marker::PhantomData;

/// PCG 求解器配置
#[derive(Debug, Clone)]
pub struct PcgConfig {
    /// 相对容差
    pub rtol: f64,
    /// 绝对容差
    pub atol: f64,
    /// 最大迭代次数
    pub max_iter: usize,
    /// 预处理器类型
    pub preconditioner: PreconditionerType,
    /// 是否输出诊断信息
    pub verbose: bool,
}

impl Default for PcgConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-14,
            max_iter: 1000,
            preconditioner: PreconditionerType::Jacobi,
            verbose: false,
        }
    }
}

/// 预处理器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreconditionerType {
    /// 无预处理（单位矩阵）
    None,
    /// 雅可比（对角）预处理
    Jacobi,
}

/// PCG 求解结果
#[derive(Debug, Clone)]
pub struct PcgResult<S: Scalar> {
    /// 是否收敛
    pub converged: bool,
    /// 实际迭代次数
    pub iterations: usize,
    /// 最终残差范数
    pub residual_norm: S,
    /// 初始残差范数
    pub initial_residual_norm: S,
    /// 相对残差 (||r|| / ||b||)
    pub relative_residual: S,
}

/// 稀疏矩阵的矩阵-向量乘法 trait
/// 
/// 用于 PCG 求解器中的矩阵-向量乘积计算。
/// 实现者需要提供高效的 y = A*x 计算。
pub trait SparseMvp<B: Backend> {
    /// 计算矩阵-向量乘积: y = A * x
    /// 
    /// # 参数
    /// 
    /// - `x`: 输入向量
    /// - `y`: 输出向量（结果将写入此缓冲区）
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>);
    
    /// 获取矩阵维度
    fn dimension(&self) -> usize;
}

/// 对角矩阵（用于雅可比预处理）
pub struct DiagonalMatrix<B: Backend> {
    /// 对角元素
    pub diag: B::Buffer<B::Scalar>,
    /// 维度
    n: usize,
    /// 后端标记
    _marker: PhantomData<B>,
}

impl<B: Backend> DiagonalMatrix<B> {
    /// 从对角元素创建对角矩阵
    pub fn new(diag: B::Buffer<B::Scalar>, n: usize) -> Self {
        Self {
            diag,
            n,
            _marker: PhantomData,
        }
    }
    
    /// 获取维度
    pub fn dimension(&self) -> usize {
        self.n
    }
}

/// PCG 求解器工作区
/// 
/// 存储 PCG 迭代所需的所有临时向量，避免重复分配内存。
pub struct PcgWorkspace<B: Backend> {
    /// 残差向量 r
    pub r: B::Buffer<B::Scalar>,
    /// 预处理后的残差 z = M^{-1} * r
    pub z: B::Buffer<B::Scalar>,
    /// 搜索方向 p
    pub p: B::Buffer<B::Scalar>,
    /// 矩阵-向量乘积结果 Ap
    pub ap: B::Buffer<B::Scalar>,
    /// 已分配的维度
    n_allocated: usize,
}

impl<B: Backend> PcgWorkspace<B> {
    /// 创建新的工作区
    pub fn new_with_backend(backend: &B, n: usize) -> Self {
        Self {
            r: backend.alloc(n),
            z: backend.alloc(n),
            p: backend.alloc(n),
            ap: backend.alloc(n),
            n_allocated: n,
        }
    }
    
    /// 确保工作区容量足够
    pub fn ensure_capacity(&mut self, backend: &B, n: usize) {
        if n > self.n_allocated {
            self.r = backend.alloc(n);
            self.z = backend.alloc(n);
            self.p = backend.alloc(n);
            self.ap = backend.alloc(n);
            self.n_allocated = n;
        }
    }
    
    /// 获取已分配的维度
    pub fn capacity(&self) -> usize {
        self.n_allocated
    }
}

/// PCG 求解器
/// 
/// 泛型预处理共轭梯度法求解器，用于求解稀疏对称正定线性系统。
/// 
/// # 类型参数
/// 
/// - `B`: 计算后端类型
/// 
/// # 示例
/// 
/// ```ignore
/// let backend = CpuBackend::<f64>::new();
/// let config = PcgConfig::default();
/// let solver = PcgSolver::new_with_backend(backend, n, config);
/// 
/// let result = solver.solve(&matrix, &mut x, &b, Some(&precond));
/// if result.converged {
///     println!("求解成功，迭代次数: {}", result.iterations);
/// }
/// ```
pub struct PcgSolver<B: Backend> {
    /// 计算后端实例
    backend: B,
    /// 配置
    config: PcgConfig,
    /// 工作区
    workspace: PcgWorkspace<B>,
}

impl<B: Backend> PcgSolver<B> {
    /// 使用后端实例创建 PCG 求解器
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `n`: 问题维度（向量长度）
    /// - `config`: 求解器配置
    pub fn new_with_backend(backend: B, n: usize, config: PcgConfig) -> Self
    where
        B: Clone,
    {
        let workspace = PcgWorkspace::new_with_backend(&backend, n);
        Self {
            backend,
            config,
            workspace,
        }
    }
    
    /// 获取后端引用
    pub fn backend(&self) -> &B {
        &self.backend
    }
    
    /// 获取配置引用
    pub fn config(&self) -> &PcgConfig {
        &self.config
    }
    
    /// 更新配置
    pub fn set_config(&mut self, config: PcgConfig) {
        self.config = config;
    }
    
    /// 确保工作区容量
    pub fn ensure_capacity(&mut self, n: usize)
    where
        B: Clone,
    {
        self.workspace.ensure_capacity(&self.backend, n);
    }
}

/// CPU f64 后端的 PCG 求解器实现
impl PcgSolver<CpuBackend<f64>> {
    /// 求解线性系统 Ax = b
    /// 
    /// # 参数
    /// 
    /// - `matrix`: 系数矩阵（通过 SparseMvp trait 提供）
    /// - `x`: 解向量（输入初始猜测，输出解）
    /// - `b`: 右端向量
    /// - `precond`: 可选的雅可比预处理器（对角矩阵）
    /// 
    /// # 返回
    /// 
    /// 返回 PcgResult，包含收敛信息和迭代统计
    pub fn solve<M: SparseMvp<CpuBackend<f64>>>(
        &mut self,
        matrix: &M,
        x: &mut Vec<f64>,
        b: &Vec<f64>,
        precond: Option<&DiagonalMatrix<CpuBackend<f64>>>,
    ) -> PcgResult<f64> {
        let n = matrix.dimension();
        
        // 确保工作区容量
        self.workspace.ensure_capacity(&self.backend, n);
        
        // 步骤 1: 计算初始残差 r_0 = b - A*x_0
        matrix.apply(x, &mut self.workspace.r);  // r = A*x
        for i in 0..n {
            self.workspace.r[i] = b[i] - self.workspace.r[i];  // r = b - A*x
        }
        
        // 计算 ||b|| 用于相对收敛判断
        let b_norm = dot_product(b, b, n).sqrt();
        let initial_r_norm = dot_product(&self.workspace.r, &self.workspace.r, n).sqrt();
        
        // 如果 b 接近零，直接返回
        if b_norm < self.config.atol {
            return PcgResult {
                converged: true,
                iterations: 0,
                residual_norm: initial_r_norm,
                initial_residual_norm: initial_r_norm,
                relative_residual: 0.0,
            };
        }
        
        // 步骤 2: 应用预处理 z_0 = M^{-1} * r_0
        apply_preconditioner_static(
            &self.workspace.r,
            &mut self.workspace.z,
            &self.config,
            precond,
            n,
        );
        
        // 步骤 3: 初始化搜索方向 p_0 = z_0
        for i in 0..n {
            self.workspace.p[i] = self.workspace.z[i];
        }
        
        // rho = (r, z)
        let mut rho = dot_product(&self.workspace.r, &self.workspace.z, n);
        
        // 主迭代循环
        for iter in 0..self.config.max_iter {
            // 计算 Ap
            matrix.apply(&self.workspace.p, &mut self.workspace.ap);
            
            // alpha = rho / (p, Ap)
            let p_ap = dot_product(&self.workspace.p, &self.workspace.ap, n);
            if p_ap.abs() < 1e-30 {
                // 防止除零
                let r_norm = dot_product(&self.workspace.r, &self.workspace.r, n).sqrt();
                return PcgResult {
                    converged: false,
                    iterations: iter,
                    residual_norm: r_norm,
                    initial_residual_norm: initial_r_norm,
                    relative_residual: r_norm / b_norm,
                };
            }
            let alpha = rho / p_ap;
            
            // x = x + alpha * p
            // r = r - alpha * Ap
            for i in 0..n {
                x[i] += alpha * self.workspace.p[i];
                self.workspace.r[i] -= alpha * self.workspace.ap[i];
            }
            
            // 检查收敛
            let r_norm = dot_product(&self.workspace.r, &self.workspace.r, n).sqrt();
            let relative_residual = r_norm / b_norm;
            
            if self.config.verbose && (iter % 10 == 0 || iter < 5) {
                eprintln!("PCG 迭代 {}: 相对残差 = {:.6e}", iter, relative_residual);
            }
            
            if r_norm < self.config.atol || relative_residual < self.config.rtol {
                return PcgResult {
                    converged: true,
                    iterations: iter + 1,
                    residual_norm: r_norm,
                    initial_residual_norm: initial_r_norm,
                    relative_residual,
                };
            }
            
            // 应用预处理 z = M^{-1} * r
            apply_preconditioner_static(
                &self.workspace.r,
                &mut self.workspace.z,
                &self.config,
                precond,
                n,
            );
            
            // beta = (r_new, z_new) / (r_old, z_old)
            let rho_new = dot_product(&self.workspace.r, &self.workspace.z, n);
            let beta = rho_new / rho;
            rho = rho_new;
            
            // p = z + beta * p
            for i in 0..n {
                self.workspace.p[i] = self.workspace.z[i] + beta * self.workspace.p[i];
            }
        }
        
        // 达到最大迭代次数，未收敛
        let r_norm = dot_product(&self.workspace.r, &self.workspace.r, n).sqrt();
        PcgResult {
            converged: false,
            iterations: self.config.max_iter,
            residual_norm: r_norm,
            initial_residual_norm: initial_r_norm,
            relative_residual: r_norm / b_norm,
        }
    }
}

/// 静态预处理器应用函数（避免借用冲突）
fn apply_preconditioner_static(
    r: &[f64],
    z: &mut [f64],
    config: &PcgConfig,
    precond: Option<&DiagonalMatrix<CpuBackend<f64>>>,
    n: usize,
) {
    match (config.preconditioner, precond) {
        (PreconditionerType::Jacobi, Some(diag)) => {
            // 雅可比预处理: z_i = r_i / diag_i
            for i in 0..n {
                let d = diag.diag[i];
                if d.abs() > 1e-30 {
                    z[i] = r[i] / d;
                } else {
                    z[i] = r[i];
                }
            }
        }
        _ => {
            // 无预处理: z = r
            for i in 0..n {
                z[i] = r[i];
            }
        }
    }
}

/// 静态点积函数（避免借用冲突）
#[inline]
fn dot_product(x: &[f64], y: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i] * y[i];
    }
    sum
}

/// 简化的对角矩阵乘法（用于泊松方程）
/// 
/// 当系统矩阵近似为对角矩阵时使用。
impl SparseMvp<CpuBackend<f64>> for DiagonalMatrix<CpuBackend<f64>> {
    fn apply(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        for i in 0..self.n {
            y[i] = self.diag[i] * x[i];
        }
    }
    
    fn dimension(&self) -> usize {
        self.n
    }
}

/// 通用稀疏矩阵（CSR 格式）
/// 
/// 压缩稀疏行（Compressed Sparse Row）格式的稀疏矩阵，
/// 用于存储压力泊松方程的系数矩阵。
pub struct CsrMatrix<B: Backend> {
    /// 行指针数组（长度 n+1）
    pub row_ptr: Vec<usize>,
    /// 列索引数组
    pub col_idx: Vec<usize>,
    /// 非零元素值
    pub values: B::Buffer<B::Scalar>,
    /// 矩阵行数
    n_rows: usize,
    /// 矩阵列数
    n_cols: usize,
    /// 后端标记
    _marker: PhantomData<B>,
}

impl<B: Backend> CsrMatrix<B> {
    /// 创建新的 CSR 矩阵
    /// 
    /// # 参数
    /// 
    /// - `n_rows`: 行数
    /// - `n_cols`: 列数
    /// - `row_ptr`: 行指针数组
    /// - `col_idx`: 列索引数组
    /// - `values`: 非零元素值
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: B::Buffer<B::Scalar>,
    ) -> Self {
        Self {
            row_ptr,
            col_idx,
            values,
            n_rows,
            n_cols,
            _marker: PhantomData,
        }
    }
    
    /// 获取行数
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }
    
    /// 获取列数
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
    
    /// 获取非零元素个数
    pub fn nnz(&self) -> usize {
        self.col_idx.len()
    }
}

impl SparseMvp<CpuBackend<f64>> for CsrMatrix<CpuBackend<f64>> {
    /// 计算 CSR 矩阵-向量乘积: y = A * x
    fn apply(&self, x: &Vec<f64>, y: &mut Vec<f64>) {
        // 清零输出向量
        for i in 0..self.n_rows {
            y[i] = 0.0;
        }
        
        // 逐行计算
        for row in 0..self.n_rows {
            let row_start = self.row_ptr[row];
            let row_end = self.row_ptr[row + 1];
            
            let mut sum = 0.0;
            for j in row_start..row_end {
                let col = self.col_idx[j];
                sum += self.values[j] * x[col];
            }
            y[row] = sum;
        }
    }
    
    fn dimension(&self) -> usize {
        self.n_rows
    }
}

/// 压力泊松矩阵构建器
/// 
/// 用于从网格拓扑构建压力泊松方程的系数矩阵。
pub struct PoissonMatrixBuilder {
    /// 单元数量
    n_cells: usize,
}

impl PoissonMatrixBuilder {
    /// 创建矩阵构建器
    pub fn new(n_cells: usize) -> Self {
        Self { n_cells }
    }
    
    /// 构建压力泊松矩阵（对角部分）
    /// 
    /// 对于压力校正方程，对角项与单元面积和时间步长相关：
    /// A_ii = Σ_f (h_f * L_f / d_f) + ε
    /// 
    /// 其中 ε 是小的正则化项，用于处理干单元。
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端
    /// - `cell_volumes`: 单元体积（面积）
    /// - `dt`: 时间步长
    /// - `gravity`: 重力加速度
    /// - `h`: 水深
    /// - `h_min`: 最小水深阈值
    /// 
    /// # 返回
    /// 
    /// 返回对角矩阵的对角元素向量
    pub fn build_diagonal(
        &self,
        backend: &CpuBackend<f64>,
        cell_areas: &[f64],
        dt: f64,
        gravity: f64,
        h: &[f64],
        h_min: f64,
    ) -> DiagonalMatrix<CpuBackend<f64>> {
        let mut diag = backend.alloc(self.n_cells);
        
        for i in 0..self.n_cells {
            let area = cell_areas[i];
            let h_eff = h[i].max(h_min);
            
            // 对角项系数：用于简化的对角近似
            // 实际的泊松方程对角项应包含相邻单元的贡献
            // 这里使用 A_ii ≈ area / (g * θ * dt² * h)
            // 其中 θ 是隐式因子（通常取 0.5-1.0）
            let theta = 0.5;
            diag[i] = area / (gravity * theta * dt * dt * h_eff);
        }
        
        DiagonalMatrix::new(diag, self.n_cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// 测试简单的对角系统求解
    #[test]
    fn test_pcg_diagonal_system() {
        let backend = CpuBackend::<f64>::new();
        let n = 10;
        
        // 创建对角矩阵 A = diag([1, 2, 3, ..., 10])
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let matrix = DiagonalMatrix::new(diag.clone(), n);
        
        // 右端向量 b = [1, 1, ..., 1]
        let b: Vec<f64> = vec![1.0; n];
        
        // 初始猜测 x = [0, 0, ..., 0]
        let mut x: Vec<f64> = vec![0.0; n];
        
        // 创建求解器
        let config = PcgConfig::default();
        let mut solver = PcgSolver::new_with_backend(backend, n, config);
        
        // 求解
        let result = solver.solve(&matrix, &mut x, &b, None);
        
        assert!(result.converged, "PCG 应该收敛");
        
        // 验证解：x_i = 1/i
        for i in 0..n {
            let expected = 1.0 / ((i + 1) as f64);
            assert!((x[i] - expected).abs() < 1e-6, 
                "x[{}] = {}, 期望 {}", i, x[i], expected);
        }
    }
}
