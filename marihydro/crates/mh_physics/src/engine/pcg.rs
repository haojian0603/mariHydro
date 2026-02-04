//! 预处理共轭梯度法求解器
//!
//! 实现泛型PCG算法，支持任意Backend（f32/f64/GPU），用于求解稀疏对称正定线性系统。
//! 主要用于半隐式时间积分中的压力泊松方程求解。

use crate::core::Backend;
use crate::mesh::{MeshGeometry, MeshTopology};
use crate::numerics::linear_algebra::csr::{CsrBuilder, CsrMatrix};
use crate::prelude::*;
use mh_foundation::{MhError, MhResult};

/// PCG 求解器配置（Layer 4，保持 f64）
/// 
/// 配置层使用 f64 类型，在 Layer 3 引擎层使用时通过 `to_runtime()` 转换。
/// 这确保配置文件的可移植性和精度一致性。
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
    /// 是否输出详细信息
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

impl PcgConfig {
    /// 转换为运行时配置
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端引用，用于标量转换
    /// 
    /// # 返回
    /// 
    /// 返回使用 Backend 标量类型的运行时配置
    pub fn to_runtime<B: Backend>(&self, backend: &B) -> PcgRuntimeConfig<B> {
        PcgRuntimeConfig {
            rtol: backend.scalar_from_f64(self.rtol),
            atol: backend.scalar_from_f64(self.atol),
            max_iter: self.max_iter,
            preconditioner: self.preconditioner,
            verbose: self.verbose,
        }
    }
}

/// PCG 运行时配置（Backend 泛型）
/// 
/// 从 `PcgConfig` 转换而来，所有标量类型使用 Backend 类型。
/// 仅在 Layer 3 引擎层内部使用。
#[derive(Debug, Clone)]
pub struct PcgRuntimeConfig<B: Backend> {
    /// 相对容差
    pub rtol: B::Scalar,
    /// 绝对容差
    pub atol: B::Scalar,
    /// 最大迭代次数
    pub max_iter: usize,
    /// 预处理器类型
    pub preconditioner: PreconditionerType,
    /// 是否输出详细信息
    pub verbose: bool,
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

impl<B: Backend> SparseMvp<B> for CsrMatrix<B::Scalar>
where
    B::Buffer<B::Scalar>: Send + Sync,
    B::Scalar: RuntimeScalar,
{
    fn apply(&self, x: &B::Buffer<B::Scalar>, y: &mut B::Buffer<B::Scalar>) {
        let x_slice = x.as_slice();
        let y_slice = y.as_slice_mut();
        y_slice.fill(B::Scalar::ZERO);

        let row_ptr = self.row_ptr();
        let col_idx = self.pattern().col_idx();
        let values = self.values();

        for row in 0..self.n_rows() {
            let start = row_ptr[row];
            let end = row_ptr[row + 1];
            let mut sum = B::Scalar::ZERO;
            for idx in start..end {
                let col = col_idx[idx];
                sum = sum + values[idx] * x_slice[col];
            }
            y_slice[row] = sum;
        }
    }

    fn dimension(&self) -> usize {
        self.n_rows()
    }
}

/// 对角矩阵（用于 Jacobi 预处理）
/// 
/// 存储对角线元素，支持高效的对角矩阵-向量乘法。
/// 通过 Backend 泛型实现精度切换和内存管理。
/// 
/// # 数学表示
/// 
/// $$D = \text{diag}(d_0, d_1, ..., d_{n-1})$$
/// 
/// # 性能
/// 
/// - 存储复杂度: O(n)
/// - 乘法复杂度: O(n)
#[derive(Clone)]
pub struct DiagonalMatrix<B: Backend>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    /// 对角线元素
    pub diag: B::Buffer<B::Scalar>,
    /// 矩阵维度
    n: usize,
    /// 计算后端实例
    backend: B,
}

impl<B: Backend> DiagonalMatrix<B>
where
    B::Buffer<B::Scalar>: Send + Sync,
{
    /// 从对角线缓冲区创建对角矩阵
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `diag`: 对角线元素缓冲区
    /// - `n`: 矩阵维度
    pub fn new(backend: B, diag: B::Buffer<B::Scalar>, n: usize) -> Self {
        Self { diag, n, backend }
    }
    
    /// 获取矩阵维度
    #[inline]
    pub fn dimension(&self) -> usize {
        self.n
    }
    
    /// 获取后端引用
    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// 使用闭包批量生成对角线元素
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `n`: 矩阵维度
    /// - `f`: 生成函数，接收索引返回对角元素
    pub fn from_fn<F>(backend: B, n: usize, mut f: F) -> Self
    where
        F: FnMut(usize) -> B::Scalar,
    {
        let mut diag = backend.alloc(n);
        for i in 0..n {
            diag[i] = f(i);
        }
        Self::new(backend, diag, n)
    }
    
    /// 从切片创建对角矩阵
    pub fn from_slice(backend: B, slice: &[B::Scalar]) -> Self {
        let n = slice.len();
        let mut diag = backend.alloc(n);
        diag.copy_from_slice(slice);
        Self::new(backend, diag, n)
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
    B::Scalar: RuntimeScalar,
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
    backend: &B,
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
                let eps = backend.scalar_from_f64(1e-30);
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
    B::Scalar: RuntimeScalar,
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

        let mut x_buf = self.backend.alloc(n);
        x_buf.copy_from_slice(x.as_slice());

        matrix.apply(&x_buf, &mut workspace.ap);
        for i in 0..n {
            workspace.r[i] = b[i] - workspace.ap[i];
        }

        let b_norm = dot_product(&b.as_slice()[..n], &b.as_slice()[..n], n).sqrt();
        let initial_r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();

        let eps = self.backend.scalar_from_f64(self.config.atol);
        if b_norm < eps {
            x.copy_from_slice(x_buf.as_slice());
            return PcgResult {
                converged: true,
                iterations: 0,
                residual_norm: initial_r_norm,
                initial_residual_norm: initial_r_norm,
                relative_residual: B::Scalar::ZERO,
            };
        }

        apply_preconditioner(&self.backend, &workspace.r, &mut workspace.z, &self.config, precond, n);

        for i in 0..n {
            workspace.p[i] = workspace.z[i];
        }

        let mut rho = dot_product(&workspace.r.as_slice()[..n], &workspace.z.as_slice()[..n], n);

        for iter in 0..self.config.max_iter {
            matrix.apply(&workspace.p, &mut workspace.ap);

            let p_ap = dot_product(&workspace.p.as_slice()[..n], &workspace.ap.as_slice()[..n], n);
            let eps = self.backend.scalar_from_f64(1e-30);
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
                x_buf[i] = x_buf[i] + alpha * workspace.p[i];
                workspace.r[i] = workspace.r[i] - alpha * workspace.ap[i];
            }

            let r_norm = dot_product(&workspace.r.as_slice()[..n], &workspace.r.as_slice()[..n], n).sqrt();
            let relative_residual = r_norm / b_norm;
            let rtol = self.backend.scalar_from_f64(self.config.rtol);

            if r_norm < eps || relative_residual < rtol {
                x.copy_from_slice(x_buf.as_slice());
                return PcgResult {
                    converged: true,
                    iterations: iter + 1,
                    residual_norm: r_norm,
                    initial_residual_norm: initial_r_norm,
                    relative_residual,
                };
            }

            apply_preconditioner(&self.backend, &workspace.r, &mut workspace.z, &self.config, precond, n);

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
    /// 创建泊松矩阵构建器
    /// 
    /// # 参数
    /// 
    /// - `n_cells`: 单元数量
    pub fn new(n_cells: usize) -> Self {
        Self { n_cells }
    }

    /// 构建对角矩阵（用于 Jacobi 预处理）
    /// 
    /// # 参数
    /// 
    /// - `backend`: 计算后端实例
    /// - `cell_areas`: 单元面积数组
    /// - `dt`: 时间步长
    /// - `gravity`: 重力加速度
    /// - `theta`: 隐式权重因子
    /// - `h`: 水深数组
    /// - `h_min`: 最小水深阈值
    /// 
    /// # 返回
    /// 
    /// 返回对角矩阵，用于预处理
    pub fn build_diagonal<B: Backend + Clone>(
        &self,
        backend: &B,
        cell_areas: &[B::Scalar],
        dt: B::Scalar,
        gravity: B::Scalar,
        theta: B::Scalar,
        h: &[B::Scalar],
        h_min: B::Scalar,
    ) -> MhResult<DiagonalMatrix<B>>
    where
        B::Buffer<B::Scalar>: Send + Sync,
        B::Scalar: RuntimeScalar,
    {
        if cell_areas.len() != self.n_cells {
            return Err(MhError::size_mismatch("cell_areas", self.n_cells, cell_areas.len()));
        }
        if h.len() != self.n_cells {
            return Err(MhError::size_mismatch("h", self.n_cells, h.len()));
        }

        if let Err((idx, val)) = B::Scalar::validate_slice(cell_areas) {
            return Err(MhError::invalid_input(format!(
                "cell_areas 第 {} 项为非法值: {:?}", idx, val
            )));
        }
        if let Err((idx, val)) = B::Scalar::validate_slice(h) {
            return Err(MhError::invalid_input(format!(
                "h 第 {} 项为非法值: {:?}", idx, val
            )));
        }

        let g_min = backend.scalar_from_f64(1e-6);
        if gravity <= g_min {
            return Err(MhError::invalid_input(
                "重力加速度过小，可能导致数值不稳定".to_string(),
            ));
        }

        let eps = backend.scalar_from_f64(1e-30);
        let theta_safe = if theta.abs() > eps { theta } else { B::Scalar::HALF };

        Ok(DiagonalMatrix::from_fn(backend.clone(), self.n_cells, |i| {
            let area = cell_areas[i];
            let h_eff = h[i].max(h_min);
            let denom = gravity * theta_safe * dt * dt * h_eff;
            if denom.abs() > eps {
                area / denom
            } else {
                B::Scalar::ZERO
            }
        }))
    }

    /// 构建稀疏泊松矩阵（CSR）
    ///
    /// 使用面邻接离散拉普拉斯项并叠加质量项，形成对称正定矩阵。
    pub fn build_csr<B: Backend + Clone>(
        &self,
        mesh: &dyn MeshTopology<B>,
        cell_areas: &[B::Scalar],
        dt: B::Scalar,
        gravity: B::Scalar,
        theta: B::Scalar,
        h: &[B::Scalar],
        h_min: B::Scalar,
    ) -> CsrMatrix<B::Scalar>
    where
        B::Buffer<B::Scalar>: Send + Sync,
        B::Scalar: RuntimeScalar,
    {
        assert_eq!(cell_areas.len(), self.n_cells, "cell_areas 长度不匹配");
        assert_eq!(h.len(), self.n_cells, "h 长度不匹配");

        let eps = B::Scalar::MIN_POSITIVE;
        let theta_safe = if theta.abs() > eps { theta } else { B::Scalar::HALF };
        let alpha = gravity * theta_safe * dt * dt;

        let mut builder = CsrBuilder::<B::Scalar>::new_square(self.n_cells);

        for i in 0..self.n_cells {
            let area = cell_areas[i];
            let h_eff = h[i].max(h_min);
            let denom = alpha * h_eff;
            let diag_mass = if denom.abs() > eps { area / denom } else { B::Scalar::ZERO };
            builder.add(i, i, diag_mass);
        }

            for face_id in 0..mesh.n_faces() {
            let owner = mesh.face_owner(face_id);
            let neighbor = match mesh.face_neighbor(face_id) {
                Some(n) => n,
                None => continue,
            };

            let length = mesh.face_length(face_id);
            if !length.is_finite() || length <= B::Scalar::ZERO {
                continue;
            }

            let center_o = mesh.cell_center(owner);
            let center_n = mesh.cell_center(neighbor);
            let dist = MeshGeometry::distance(center_o, center_n);
            if !dist.is_finite() || dist <= B::Scalar::ZERO {
                continue;
            }
            let h_face = (h[owner] + h[neighbor]) * B::Scalar::HALF;
            let h_face = h_face.max(h_min);

            let coeff = alpha * h_face * length / dist;
            if coeff.abs() <= eps {
                continue;
            }

            builder.add(owner, owner, coeff);
            builder.add(neighbor, neighbor, coeff);
            builder.add(owner, neighbor, -coeff);
            builder.add(neighbor, owner, -coeff);
        }

        builder.build()
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
        let matrix = DiagonalMatrix::new(backend.clone(), diag, n);
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