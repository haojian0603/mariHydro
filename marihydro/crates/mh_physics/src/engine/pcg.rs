//! 预处理共轭梯度法求解器
//!
//! 实现泛型PCG算法，支持任意Backend（f32/f64/GPU），用于求解稀疏对称正定线性系统。
//! 主要用于半隐式时间积分中的压力泊松方程求解。

use crate::core::Backend;
use mh_runtime::{DeviceBuffer, RuntimeScalar};
use num_traits::{FromPrimitive, Float};
use std::marker::PhantomData;

/// PCG 求解器配置（Layer 4，保持f64）
#[derive(Debug, Clone)]
pub struct PcgConfig {
    pub rtol: f64,
    pub atol: f64,
    pub max_iter: usize,
    pub preconditioner: PreconditionerType,
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
    None,
    Jacobi,
}

/// PCG 求解结果
#[derive(Debug, Clone)]
pub struct PcgResult<S: RuntimeScalar> {
    pub converged: bool,
    pub iterations: usize,
    pub residual_norm: S,
    pub initial_residual_norm: S,
    pub relative_residual: S,
}

/// 稀疏矩阵-向量乘法trait
pub trait SparseMvp<B: Backend> {
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>);
    fn dimension(&self) -> usize;
}

/// 对角矩阵（用于Jacobi预处理）
pub struct DiagonalMatrix<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub diag: B::Buffer<B::Scalar>,
    n: usize,
    _marker: PhantomData<B>,
}

impl<B: Backend> DiagonalMatrix<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn new(diag: B::Buffer<B::Scalar>, n: usize) -> Self {
        Self {
            diag,
            n,
            _marker: PhantomData,
        }
    }

    pub fn dimension(&self) -> usize {
        self.n
    }

    /// 使用闭包批量生成对角线元素
    pub fn from_fn<F>(backend: &B, n: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> B::Scalar,
    {
        let mut diag = backend.alloc(n);
        for i in 0..n {
            diag[i] = f(i);
        }
        Self::new(diag, n)
    }
}

impl<B: Backend> SparseMvp<B> for DiagonalMatrix<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
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

/// PCG 求解器工作区
pub struct PcgWorkspace<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub r: B::Buffer<B::Scalar>,
    pub z: B::Buffer<B::Scalar>,
    pub p: B::Buffer<B::Scalar>,
    pub ap: B::Buffer<B::Scalar>,
    n_allocated: usize,
}

impl<B: Backend> PcgWorkspace<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
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

    pub fn capacity(&self) -> usize {
        self.n_allocated
    }
}

/// PCG 求解器
pub struct PcgSolver<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    backend: B,
    config: PcgConfig,
    workspace: PcgWorkspace<B>,
}

impl<B: Backend> PcgSolver<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
    B: Clone,
{
    pub fn new_with_backend(backend: B, n: usize, config: PcgConfig) -> Self {
        let workspace = PcgWorkspace::new_with_backend(&backend, n);
        Self {
            backend,
            config,
            workspace,
        }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn config(&self) -> &PcgConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: PcgConfig) {
        self.config = config;
    }

    pub fn ensure_capacity(&mut self, n: usize) {
        self.workspace.ensure_capacity(&self.backend, n);
    }
}

fn dot_product<S: RuntimeScalar>(x: &[S], y: &[S], n: usize) -> S {
    let mut sum = S::ZERO;
    for i in 0..n {
        sum = sum + x[i] * y[i];
    }
    sum
}

fn apply_preconditioner<B: Backend>(
    r: &B::Buffer<B::Scalar>,
    z: &mut B::Buffer<B::Scalar>,
    config: &PcgConfig,
    precond: Option<&DiagonalMatrix<B>>,
    n: usize,
) where
    B::Buffer<B::Scalar>: Send + Sync,
{
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

impl<B: Backend> PcgSolver<B>
where
    B::Scalar: RuntimeScalar + FromPrimitive + Float,
    B::Buffer<B::Scalar>: Send + Sync,
{
    pub fn solve<M: SparseMvp<B>>(
        &mut self,
        matrix: &M,
        x: &mut B::Buffer<B::Scalar>,
        b: &B::Buffer<B::Scalar>,
        precond: Option<&DiagonalMatrix<B>>,
    ) -> PcgResult<B::Scalar> {
        let n = matrix.dimension();
        self.ensure_capacity(n);
        let workspace = &mut self.workspace;

        matrix.apply(x, &mut workspace.ap);
        for i in 0..n {
            workspace.r[i] = b[i] - workspace.ap[i];
        }

        let b_norm = dot_product(&b.as_slice()[..n], &b.as_slice()[..n], n).sqrt();
        let initial_r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();

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

        apply_preconditioner(&workspace.r, &mut workspace.z, &self.config, precond, n);

        for i in 0..n {
            workspace.p[i] = workspace.z[i];
        }

        let mut rho = dot_product(&workspace.r.as_slice()[..n], &workspace.z.as_slice()[..n], n);

        for iter in 0..self.config.max_iter {
            matrix.apply(&workspace.p, &mut workspace.ap);

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

            for i in 0..n {
                x[i] = x[i] + alpha * workspace.p[i];
                workspace.r[i] = workspace.r[i] - alpha * workspace.ap[i];
            }

            let r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();
            let relative_residual = r_norm / b_norm;
            let rtol = B::Scalar::from_f64(self.config.rtol).unwrap_or(B::Scalar::ZERO);

            if r_norm < eps || relative_residual < rtol {
                return PcgResult {
                    converged: true,
                    iterations: iter + 1,
                    residual_norm: r_norm,
                    initial_residual_norm: initial_r_norm,
                    relative_residual,
                };
            }

            apply_preconditioner(&workspace.r, &mut workspace.z, &self.config, precond, n);

            let rho_new = dot_product(&workspace.r.as_slice()[..n], &workspace.z.as_slice()[..n], n);
            let beta = rho_new / rho;
            rho = rho_new;

            for i in 0..n {
                workspace.p[i] = workspace.z[i] + beta * workspace.p[i];
            }
        }

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

/// 压力泊松矩阵构建器
///
/// 从网格拓扑构建压力泊松方程的系数矩阵
pub struct PoissonMatrixBuilder {
    n_cells: usize,
}

impl PoissonMatrixBuilder {
    pub fn new(n_cells: usize) -> Self {
        Self { n_cells }
    }

    pub fn build_diagonal<B: Backend>(
        &self,
        backend: &B,
        cell_areas: &[B::Scalar],
        dt: B::Scalar,
        gravity: B::Scalar,
        theta: B::Scalar,
        h: &[B::Scalar],
        h_min: B::Scalar,
    ) -> DiagonalMatrix<B>
    where
        B::Buffer<B::Scalar>: Send + Sync,
        B::Scalar: RuntimeScalar + FromPrimitive + Float,
    {
        assert_eq!(cell_areas.len(), self.n_cells, "cell_areas 长度不匹配");
        assert_eq!(h.len(), self.n_cells, "h 长度不匹配");

        if let Err((idx, val)) = B::Scalar::validate_slice(cell_areas) {
            panic!("cell_areas 第 {} 项为非法值: {:?}", idx, val);
        }
        if let Err((idx, val)) = B::Scalar::validate_slice(h) {
            panic!("h 第 {} 项为非法值: {:?}", idx, val);
        }

        let g_min = B::Scalar::from_f64(1e-6).unwrap_or(B::Scalar::MIN_POSITIVE);
        assert!(gravity > g_min, "重力加速度过小，可能导致数值不稳定");

        let eps = B::Scalar::from_f64(1e-30).unwrap_or(B::Scalar::MIN_POSITIVE);
        let theta_safe = if theta.abs() > eps { theta } else { B::Scalar::HALF };

        DiagonalMatrix::from_fn(backend, self.n_cells, |i| {
            let area = cell_areas[i];
            let h_eff = h[i].max(h_min);
            let denom = gravity * theta_safe * dt * dt * h_eff;
            if denom.abs() > eps {
                area / denom
            } else {
                B::Scalar::ZERO
            }
        })
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