// crates/mh_physics/src/engine/pcg.rs

//! 预处理共轭梯度法（PCG）求解器
//!
//! 泛型预处理共轭梯度法，支持任意Backend（f32/f64/GPU）。
//! 用于求解稀疏对称正定线性系统，主要用于半隐式时间积分中的压力泊松方程。

use crate::core::{Backend, CpuBackend};
use mh_runtime::{RuntimeScalar, DeviceBuffer};
use num_traits::{FromPrimitive, Float};
use std::marker::PhantomData;

/// PCG 求解器配置（Layer 4，保持 f64）
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
pub struct PcgResult<S: RuntimeScalar> {
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
pub trait SparseMvp<B: Backend> {
    /// 计算矩阵-向量乘积: y = A * x
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
    /// Backend 标记
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
pub struct PcgWorkspace<B: Backend> {
    /// 残差向量 r
    pub r: B::Buffer<B::Scalar>,
    /// 预处理后的残差 z = M⁻¹ * r
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
        let zero = B::Scalar::ZERO;
        Self {
            r: backend.alloc_init(n, zero),
            z: backend.alloc_init(n, zero),
            p: backend.alloc_init(n, zero),
            ap: backend.alloc_init(n, zero),
            n_allocated: n,
        }
    }
    
    /// 确保工作区容量足够
    pub fn ensure_capacity(&mut self, backend: &B, n: usize) {
        if n > self.n_allocated {
            let zero = B::Scalar::ZERO;
            self.r = backend.alloc_init(n, zero);
            self.z = backend.alloc_init(n, zero);
            self.p = backend.alloc_init(n, zero);
            self.ap = backend.alloc_init(n, zero);
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
/// 泛型预处理共轭梯度法求解器，支持任意 Backend。
pub struct PcgSolver<B: Backend> {
    /// Backend 实例
    backend: B,
    /// 配置
    config: PcgConfig,
    /// 工作区
    workspace: PcgWorkspace<B>,
}

impl<B: Backend> PcgSolver<B> {
    /// 使用 Backend 实例创建 PCG 求解器
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
    
    /// 获取 Backend 引用
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
    pub fn ensure_capacity(&mut self, n: usize) {
        self.workspace.ensure_capacity(&self.backend, n);
    }
}

/// 泛型点积函数
#[inline]
fn dot_product<S: RuntimeScalar>(x: &[S], y: &[S], n: usize) -> S {
    let mut sum = S::ZERO;
    for i in 0..n {
        sum = sum + x[i] * y[i];
    }
    sum
}

/// 泛型预处理器应用函数
fn apply_preconditioner<B: Backend>(
    r: &B::Buffer<B::Scalar>,
    z: &mut B::Buffer<B::Scalar>,
    config: &PcgConfig,
    precond: Option<&DiagonalMatrix<B>>,
    n: usize,
) {
    match (config.preconditioner, precond) {
        (PreconditionerType::Jacobi, Some(diag)) => {
            for i in 0..n {
                let d = diag.diag[i];
                let eps = B::Scalar::from_f64(1e-30).unwrap_or(B::Scalar::ZERO);
                if d.abs() > eps {
                    z[i] = r[i] / d;
                } else {
                    z[i] = r[i];
                }
            }
        }
        _ => {
            for i in 0..n {
                z[i] = r[i];
            }
        }
    }
}

/// 统一的 PCG 求解器实现（支持任意 Backend）
impl<B: Backend> PcgSolver<B>
where
    B::Scalar: RuntimeScalar + FromPrimitive + Float,
{
    /// 求解线性系统 Ax = b
    pub fn solve<M: SparseMvp<B>>(
        &mut self,
        matrix: &M,
        x: &mut B::Buffer<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        precond: Option<&DiagonalMatrix<B>>,
    ) -> PcgResult<B::Scalar> {
        let n = matrix.dimension();
        
        // 确保工作区容量
        self.ensure_capacity(n);
        
        let workspace = &mut self.workspace;
        
        // 步骤 1: 计算初始残差 r₀ = b - A·x₀
        matrix.apply(x, &mut workspace.ap);  // 临时使用 ap 存储 A·x
        for i in 0..n {
            workspace.r[i] = b[i] - workspace.ap[i];
        }
        
        // 计算 ||b|| 用于相对收敛判断
        let b_norm = dot_product(&b.as_slice()[..n], &b.as_slice()[..n], n).sqrt();
        let initial_r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();
        
        // 如果 b 接近零，直接返回
        let eps = B::Scalar::from_f64(self.config.atol).unwrap_or(B::Scalar::ZERO);
        if b_norm < eps {
            return PcgResult {
                converged: true,
                iterations: 0,
                residual_norm: initial_r_norm,
                initial_residual_norm: initial_r_norm,
                relative_residual: B::Scalar::ZERO,
            };
        }
        
        // 步骤 2: 应用预处理 z₀ = M⁻¹·r₀
        apply_preconditioner(&workspace.r, &mut workspace.z, &self.config, precond, n);
        
        // 步骤 3: 初始化搜索方向 p₀ = z₀
        for i in 0..n {
            workspace.p[i] = workspace.z[i];
        }
        
        // ρ = (r, z)
        let mut rho = dot_product(&workspace.r.as_slice()[..n], &workspace.z.as_slice()[..n], n);
        
        // 主迭代循环
        for iter in 0..self.config.max_iter {
            // 计算 Ap
            matrix.apply(&workspace.p, &mut workspace.ap);
            
            // α = ρ / (p, Ap)
            let p_ap = dot_product(&workspace.p.as_slice()[..n], &workspace.ap.as_slice()[..n], n);
            let eps = B::Scalar::from_f64(1e-30).unwrap_or(B::Scalar::ZERO);
            if p_ap.abs() < eps {
                return PcgResult {
                    converged: false,
                    iterations: iter,
                    residual_norm: dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt(),
                    initial_residual_norm: initial_r_norm,
                    relative_residual: B::Scalar::ZERO,
                };
            }
            let alpha = rho / p_ap;
            
            // x = x + α·p
            // r = r - α·Ap
            for i in 0..n {
                x[i] = x[i] + alpha * workspace.p[i];
                workspace.r[i] = workspace.r[i] - alpha * workspace.ap[i];
            }
            
            // 检查收敛
            let r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();
            let relative_residual = r_norm / b_norm;
            
            if r_norm < eps || relative_residual < B::Scalar::from_f64(self.config.rtol).unwrap_or(B::Scalar::ZERO) {
                return PcgResult {
                    converged: true,
                    iterations: iter + 1,
                    residual_norm: r_norm,
                    initial_residual_norm: initial_r_norm,
                    relative_residual,
                };
            }
            
            // 应用预处理 z = M⁻¹·r
            apply_preconditioner(&workspace.r, &mut workspace.z, &self.config, precond, n);
            
            // β = (rₙₑ𝓌, zₙₑ𝓌) / (rₒₗ𝒹, zₒₗ𝒹)
            let rho_new = dot_product(&workspace.r.as_slice()[..n], &workspace.z.as_slice()[..n], n);
            let beta = rho_new / rho;
            rho = rho_new;
            
            // p = z + β·p
            for i in 0..n {
                workspace.p[i] = workspace.z[i] + beta * workspace.p[i];
            }
        }
        
        // 达到最大迭代次数，未收敛
        let r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();
        PcgResult {
            converged: false,
            iterations: self.config.max_iter,
            residual_norm: r_norm,
            initial_residual_norm: initial_r_norm,
            relative_residual: r_norm / b_norm,
        }
    }
}

/// 稀疏矩阵（CSR 格式）
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

impl<B: Backend> SparseMvp<B> for CsrMatrix<B>
where
    B::Scalar: RuntimeScalar + FromPrimitive,
{
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        // 清零输出向量
        for i in 0..self.n_rows {
            y[i] = B::Scalar::ZERO;
        }
        
        // 逐行计算
        for row in 0..self.n_rows {
            let row_start = self.row_ptr[row];
            let row_end = self.row_ptr[row + 1];
            
            let mut sum = B::Scalar::ZERO;
            for j in row_start..row_end {
                let col = self.col_idx[j];
                sum = sum + self.values[j] * x[col];
            }
            y[row] = sum;
        }
    }
    
    fn dimension(&self) -> usize {
        self.n_rows
    }
}

/// 对角矩阵的稀疏矩阵-向量乘法实现
impl<B: Backend> SparseMvp<B> for DiagonalMatrix<B>
where
    B::Scalar: RuntimeScalar,
{
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        for i in 0..self.n {
            y[i] = self.diag[i] * x[i];
        }
    }
    
    fn dimension(&self) -> usize {
        self.n
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
            
            let theta = 0.5;
            diag[i] = area / (gravity * theta * dt * dt * h_eff);
        }
        
        DiagonalMatrix::new(diag, self.n_cells)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mh_runtime::CpuBackend;

    #[test]
    fn test_pcg_diagonal_system() {
        let backend = CpuBackend::<f64>::new();
        let n = 10;
        
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let matrix = DiagonalMatrix::new(diag, n);
        
        let mut b = vec![0.0_f64; n];
        b.fill(1.0);
        let mut x = vec![0.0_f64; n];
        
        let config = PcgConfig::default();
        let mut solver = PcgSolver::new_with_backend(backend, n, config);
        
        let result = solver.solve(&matrix, &mut x, &b, None);
        
        assert!(result.converged);
        
        for i in 0..n {
            let expected = 1.0 / ((i + 1) as f64);
            assert!((x[i] - expected).abs() < 1e-6);
        }
    }
}