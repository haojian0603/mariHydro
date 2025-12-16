// crates/mh_physics/src/numerics/linear_algebra/solver.rs
//! 迭代线性求解器
//!
//! 提供用于求解稀疏线性系统 Ax = b 的迭代方法：
//! 支持泛型标量类型 `S: RuntimeScalar`（f32 或 f64）。
//!
//! # 求解器类型
//!
//! - [`ConjugateGradient`]: 共轭梯度法（CG）
//! - [`PcgSolver`]: 预条件共轭梯度法（PCG）
//! - [`BiCgStabSolver`]: 双共轭梯度稳定法（BiCGStab）
//!
//! # 使用示例
//!
//! ```ignore
//! use mh_physics::numerics::linear_algebra::{
//!     CsrMatrix, PcgSolver, JacobiPreconditioner, SolverConfig,
//! };
//!
//! let matrix: CsrMatrix<f64> = /* ... */;
//! let b = vec![1.0, 2.0, 3.0];
//! let mut x = vec![0.0; 3];
//!
//! let precond = JacobiPreconditioner::from_matrix(&matrix);
//! let config = SolverConfig::new(1e-8, 100);
//! let mut solver = PcgSolver::<f64>::new(config);
//!
//! let result = solver.solve(&matrix, &b, &mut x, &precond);
//! println!("Converged in {} iterations", result.iterations);
//! ```

use super::csr::CsrMatrix;
use super::preconditioner::{Preconditioner, ScalarPreconditioner};
use super::vector_ops::{axpy, copy, dot, norm2};
use mh_runtime::RuntimeScalar;
use serde::{Deserialize, Serialize};

// ============================================================================
// 配置层 (Layer 4) - 允许使用 f64
// ============================================================================

/// 求解器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// 相对收敛容差
    pub rtol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 绝对收敛容差
    pub atol: f64, // ALLOW_F64: Layer 4 配置参数
    /// 最大迭代次数
    pub max_iter: usize,
    /// 是否打印迭代信息
    pub verbose: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-14,
            max_iter: 1000,
            verbose: false,
        }
    }
}

impl SolverConfig {
    /// 创建求解器配置
    // ALLOW_F64: Layer 4 配置参数构造方法
    pub fn new(rtol: f64, max_iter: usize) -> Self {
        Self {
            rtol,
            max_iter,
            ..Default::default()
        }
    }

    /// 设置绝对容差
    // ALLOW_F64: Layer 4 配置参数设置方法
    pub fn with_atol(mut self, atol: f64) -> Self {
        self.atol = atol;
        self
    }

    /// 启用详细输出
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

/// 求解器状态
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    /// 收敛
    Converged,
    /// 达到最大迭代次数
    MaxIterationsReached,
    /// 发散
    Diverged,
    /// 停滞
    Stagnated,
}

/// 求解器结果
#[derive(Debug, Clone)]
pub struct SolverResult<S: RuntimeScalar> {
    /// 求解状态
    pub status: SolverStatus,
    /// 迭代次数
    pub iterations: usize,
    /// 最终残差范数
    pub residual_norm: S,
    /// 初始残差范数
    pub initial_residual_norm: S,
    /// 相对残差
    pub relative_residual: S,
}

/// Legacy 类型别名，保持向后兼容
pub type SolverResultF64 = SolverResult<f64>;

impl<S: RuntimeScalar> SolverResult<S> {
    /// 是否成功收敛
    pub fn is_converged(&self) -> bool {
        self.status == SolverStatus::Converged
    }

    /// 返回是否收敛（别名）
    pub fn converged(&self) -> bool {
        self.is_converged()
    }
}

/// CG 求解器工作区
///
/// 预分配的工作向量，避免 solve 内部频繁分配
#[derive(Debug, Clone, Default)]
pub struct CgWorkspace<S: RuntimeScalar> {
    /// 残差向量
    pub r: Vec<S>,
    /// 搜索方向
    pub p: Vec<S>,
    /// A*p
    pub ap: Vec<S>,
    /// 预条件后的残差
    pub z: Vec<S>,
}

/// Legacy 类型别名，保持向后兼容
pub type CgWorkspaceF64 = CgWorkspace<f64>;

impl<S: RuntimeScalar> CgWorkspace<S> {
    /// 创建新的工作区
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![S::ZERO; n],
            p: vec![S::ZERO; n],
            ap: vec![S::ZERO; n],
            z: vec![S::ZERO; n],
        }
    }

    /// 调整工作区大小并清零
    ///
    /// 无论大小是否变化，均清零防止历史数据污染。
    pub fn resize(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![S::ZERO; n];
            self.p = vec![S::ZERO; n];
            self.ap = vec![S::ZERO; n];
            self.z = vec![S::ZERO; n];
        } else {
            // 即使大小不变也要清零
            self.r.fill(S::ZERO);
            self.p.fill(S::ZERO);
            self.ap.fill(S::ZERO);
            self.z.fill(S::ZERO);
        }
    }

    /// 清零工作区
    pub fn clear(&mut self) {
        self.r.fill(S::ZERO);
        self.p.fill(S::ZERO);
        self.ap.fill(S::ZERO);
        self.z.fill(S::ZERO);
    }
}

/// BiCGStab 求解器工作区
#[derive(Debug, Clone, Default)]
pub struct BiCgStabWorkspace<S: RuntimeScalar> {
    /// 残差向量
    pub r: Vec<S>,
    /// 影子残差，必须保持不变
    pub r0: Vec<S>,
    /// 搜索方向
    pub p: Vec<S>,
    /// A*p_hat
    pub v: Vec<S>,
    /// 中间残差
    pub s: Vec<S>,
    /// A*s_hat
    pub t: Vec<S>,
    /// 预条件后的向量
    pub p_hat: Vec<S>,
    /// 预条件后的向量
    pub s_hat: Vec<S>,
}

/// Legacy 类型别名，保持向后兼容
pub type BiCgStabWorkspaceF64 = BiCgStabWorkspace<f64>;

impl<S: RuntimeScalar> BiCgStabWorkspace<S> {
    /// 创建新的工作区
    pub fn new(n: usize) -> Self {
        Self {
            r: vec![S::ZERO; n],
            r0: vec![S::ZERO; n],
            p: vec![S::ZERO; n],
            v: vec![S::ZERO; n],
            s: vec![S::ZERO; n],
            t: vec![S::ZERO; n],
            p_hat: vec![S::ZERO; n],
            s_hat: vec![S::ZERO; n],
        }
    }

    /// 调整工作区大小并清零
    ///
    /// 无论大小是否变化，均清零防止历史数据污染。
    pub fn resize(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![S::ZERO; n];
            self.r0 = vec![S::ZERO; n];
            self.p = vec![S::ZERO; n];
            self.v = vec![S::ZERO; n];
            self.s = vec![S::ZERO; n];
            self.t = vec![S::ZERO; n];
            self.p_hat = vec![S::ZERO; n];
            self.s_hat = vec![S::ZERO; n];
        } else {
            self.clear();
        }
    }

    /// 清零工作区
    pub fn clear(&mut self) {
        self.r.fill(S::ZERO);
        self.r0.fill(S::ZERO);
        self.p.fill(S::ZERO);
        self.v.fill(S::ZERO);
        self.s.fill(S::ZERO);
        self.t.fill(S::ZERO);
        self.p_hat.fill(S::ZERO);
        self.s_hat.fill(S::ZERO);
    }
}

/// 迭代求解器 trait
pub trait IterativeSolver<S: RuntimeScalar> {
    /// 求解线性系统 Ax = b
    ///
    /// # 参数
    ///
    /// - `matrix`: 系数矩阵 A
    /// - `b`: 右端项向量
    /// - `x`: 解向量（输入初始猜测，输出解）
    /// - `precond`: 预条件器
    ///
    /// # 返回
    ///
    /// 求解结果
    fn solve<P: ScalarPreconditioner<S>>(
        &mut self,
        matrix: &CsrMatrix<S>,
        b: &[S],
        x: &mut [S],
        precond: &P,
    ) -> SolverResult<S>;

    /// 获取求解器名称
    fn name(&self) -> &'static str;
}

/// 共轭梯度法求解器
///
/// 适用于对称正定矩阵
pub struct ConjugateGradient<S: RuntimeScalar> {
    config: SolverConfig,
    // 工作向量
    r: Vec<S>,
    p: Vec<S>,
    ap: Vec<S>,
}

/// Legacy 类型别名，保持向后兼容
pub type ConjugateGradientF64 = ConjugateGradient<f64>;

impl<S: RuntimeScalar> ConjugateGradient<S> {
    /// 创建共轭梯度求解器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            r: Vec::new(),
            p: Vec::new(),
            ap: Vec::new(),
        }
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![S::ZERO; n];
            self.p = vec![S::ZERO; n];
            self.ap = vec![S::ZERO; n];
        }
    }
}

impl<S> IterativeSolver<S> for ConjugateGradient<S>
where
    S: RuntimeScalar,
{
    fn solve<P: ScalarPreconditioner<S>>(
        &mut self,
        matrix: &CsrMatrix<S>,
        b: &[S],
        x: &mut [S],
        _precond: &P,
    ) -> SolverResult<S> {
        let n = b.len();
        self.ensure_workspace(n);
        let rtol = S::from_f64(self.config.rtol).unwrap_or(S::EPSILON);
        let atol = S::from_f64(self.config.atol).unwrap_or(S::MIN_POSITIVE);
        let stag_tol = S::from_f64(1e-30).unwrap_or(S::MIN_POSITIVE);

        // r = b - A*x
        matrix.mul_vec(x, &mut self.r);
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = norm2(&self.r);
        if initial_norm < atol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: S::ZERO,
            };
        }

        // p = r
        copy(&self.r, &mut self.p);

        let mut rr = dot(&self.r, &self.r);

        for iter in 0..self.config.max_iter {
            // ap = A * p
            matrix.mul_vec(&self.p, &mut self.ap);

            // alpha = r'r / p'Ap
            let pap = dot(&self.p, &self.ap);
            if pap.abs() < stag_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }

            let alpha = rr / pap;

            // x = x + alpha * p
            axpy(alpha, &self.p, x);

            // r = r - alpha * ap
            axpy(-alpha, &self.ap, &mut self.r);

            let res_norm = norm2(&self.r);
            let rel_res = res_norm / initial_norm;

            if self.config.verbose {
                if let Some(res_val) = res_norm.to_f64() {
                    log::trace!("CG iter {}: residual = {:.6e}", iter + 1, res_val);
                }
            }

            // 检查收敛
            if res_norm < atol || rel_res < rtol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            // beta = r'r_new / r'r_old
            let rr_new = dot(&self.r, &self.r);
            let beta = rr_new / rr;
            rr = rr_new;

            // p = r + beta * p
            for i in 0..n {
                self.p[i] = self.r[i] + beta * self.p[i];
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: norm2(&self.r) / initial_norm,
        }
    }

    fn name(&self) -> &'static str {
        "CG"
    }
}

/// 预条件共轭梯度法求解器
///
/// 适用于对称正定矩阵，使用预条件器加速收敛
pub struct PcgSolver<S: RuntimeScalar> {
    config: SolverConfig,
    // 工作向量
    r: Vec<S>,
    z: Vec<S>,
    p: Vec<S>,
    ap: Vec<S>,
}

/// Legacy 类型别名，保持向后兼容
pub type PcgSolverF64 = PcgSolver<f64>;

impl<S: RuntimeScalar> PcgSolver<S> {
    /// 创建 PCG 求解器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            r: Vec::new(),
            z: Vec::new(),
            p: Vec::new(),
            ap: Vec::new(),
        }
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![S::ZERO; n];
            self.z = vec![S::ZERO; n];
            self.p = vec![S::ZERO; n];
            self.ap = vec![S::ZERO; n];
        }
    }

    /// 使用外部工作区求解（避免内部分配）
    ///
    /// # 参数
    ///
    /// - `matrix`: 系数矩阵
    /// - `b`: 右端项
    /// - `x`: 解向量
    /// - `precond`: 预条件器
    /// - `ws`: 外部工作区
    pub fn solve_with_workspace<P: ScalarPreconditioner<S>>(
        &self,
        matrix: &CsrMatrix<S>,
        b: &[S],
        x: &mut [S],
        precond: &P,
        ws: &mut CgWorkspace<S>,
    ) -> SolverResult<S> {
        let n = b.len();
        ws.resize(n);
        let rtol = S::from_f64(self.config.rtol).unwrap_or(S::EPSILON);
        let atol = S::from_f64(self.config.atol).unwrap_or(S::MIN_POSITIVE);
        let stag_tol = S::from_f64(1e-30).unwrap_or(S::MIN_POSITIVE);

        // r = b - A*x
        matrix.mul_vec(x, &mut ws.r);
        for i in 0..n {
            ws.r[i] = b[i] - ws.r[i];
        }

        let initial_norm = norm2(&ws.r);
        let b_norm = norm2(b);

        // 鲁棒的收敛判据：处理 b_norm ≈ 0 的情况
        let effective_tol = if b_norm < S::MIN_POSITIVE {
            atol
        } else {
            atol.max(rtol * b_norm)
        };

        if initial_norm < effective_tol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: S::ZERO,
            };
        }

        // z = M^{-1} * r
        precond.apply(&ws.r, &mut ws.z);

        // p = z
        copy(&ws.z, &mut ws.p);

        let mut rz = dot(&ws.r, &ws.z);

        for iter in 0..self.config.max_iter {
            // ap = A * p
            matrix.mul_vec(&ws.p, &mut ws.ap);

            // alpha = r'z / p'Ap
            let pap = dot(&ws.p, &ws.ap);
            if pap.abs() < stag_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&ws.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: if initial_norm > S::ZERO {
                        norm2(&ws.r) / initial_norm
                    } else {
                        S::ZERO
                    },
                };
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            axpy(alpha, &ws.p, x);

            // r = r - alpha * ap
            axpy(-alpha, &ws.ap, &mut ws.r);

            let res_norm = norm2(&ws.r);

            if self.config.verbose {
                if let Some(res_val) = res_norm.to_f64() {
                    log::trace!("PCG iter {}: residual = {:.6e}", iter + 1, res_val);
                }
            }

            // 检查收敛
            if res_norm < effective_tol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: if initial_norm > S::ZERO {
                        res_norm / initial_norm
                    } else {
                        S::ZERO
                    },
                };
            }

            // z = M^{-1} * r
            precond.apply(&ws.r, &mut ws.z);

            // beta = r'z_new / r'z_old
            let rz_new = dot(&ws.r, &ws.z);
            let beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for i in 0..n {
                ws.p[i] = ws.z[i] + beta * ws.p[i];
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: norm2(&ws.r),
            initial_residual_norm: initial_norm,
            relative_residual: if initial_norm > S::ZERO {
                norm2(&ws.r) / initial_norm
            } else {
                S::ZERO
            },
        }
    }
}

impl<S> IterativeSolver<S> for PcgSolver<S>
where
    S: RuntimeScalar,
{
    fn solve<P: ScalarPreconditioner<S>>(
        &mut self,
        matrix: &CsrMatrix<S>,
        b: &[S],
        x: &mut [S],
        precond: &P,
    ) -> SolverResult<S> {
        let n = b.len();
        self.ensure_workspace(n);
        let rtol = S::from_f64(self.config.rtol).unwrap_or(S::EPSILON);
        let atol = S::from_f64(self.config.atol).unwrap_or(S::MIN_POSITIVE);
        let stag_tol = S::from_f64(1e-30).unwrap_or(S::MIN_POSITIVE);

        // r = b - A*x
        matrix.mul_vec(x, &mut self.r);
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = norm2(&self.r);
        if initial_norm < atol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: S::ZERO,
            };
        }

        // z = M^{-1} * r
        precond.apply(&self.r, &mut self.z);

        // p = z
        copy(&self.z, &mut self.p);

        let mut rz = dot(&self.r, &self.z);

        for iter in 0..self.config.max_iter {
            // ap = A * p
            matrix.mul_vec(&self.p, &mut self.ap);

            // alpha = r'z / p'Ap
            let pap = dot(&self.p, &self.ap);
            if pap.abs() < stag_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }

            let alpha = rz / pap;

            // x = x + alpha * p
            axpy(alpha, &self.p, x);

            // r = r - alpha * ap
            axpy(-alpha, &self.ap, &mut self.r);

            let res_norm = norm2(&self.r);
            let rel_res = res_norm / initial_norm;

            if self.config.verbose {
                if let Some(res_val) = res_norm.to_f64() {
                    log::trace!("PCG iter {}: residual = {:.6e}", iter + 1, res_val);
                }
            }

            // 检查收敛
            if res_norm < atol || rel_res < rtol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            // z = M^{-1} * r
            precond.apply(&self.r, &mut self.z);

            // beta = r'z_new / r'z_old
            let rz_new = dot(&self.r, &self.z);
            let beta = rz_new / rz;
            rz = rz_new;

            // p = z + beta * p
            for i in 0..n {
                self.p[i] = self.z[i] + beta * self.p[i];
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: norm2(&self.r) / initial_norm,
        }
    }

    fn name(&self) -> &'static str {
        "PCG"
    }
}

/// 双共轭梯度稳定法求解器
///
/// 适用于非对称矩阵
pub struct BiCgStabSolver<S: RuntimeScalar> {
    config: SolverConfig,
    // 工作向量
    r: Vec<S>,
    r0: Vec<S>,
    p: Vec<S>,
    v: Vec<S>,
    s: Vec<S>,
    t: Vec<S>,
    z: Vec<S>,
}

/// Legacy 类型别名，保持向后兼容
pub type BiCgStabSolverF64 = BiCgStabSolver<f64>;

impl<S: RuntimeScalar> BiCgStabSolver<S> {
    /// 创建 BiCGStab 求解器
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            r: Vec::new(),
            r0: Vec::new(),
            p: Vec::new(),
            v: Vec::new(),
            s: Vec::new(),
            t: Vec::new(),
            z: Vec::new(),
        }
    }

    /// 确保工作向量大小正确
    fn ensure_workspace(&mut self, n: usize) {
        if self.r.len() != n {
            self.r = vec![S::ZERO; n];
            self.r0 = vec![S::ZERO; n];
            self.p = vec![S::ZERO; n];
            self.v = vec![S::ZERO; n];
            self.s = vec![S::ZERO; n];
            self.t = vec![S::ZERO; n];
            self.z = vec![S::ZERO; n];
        }
    }
}

impl<S> IterativeSolver<S> for BiCgStabSolver<S>
where
    S: RuntimeScalar,
{
    fn solve<P: ScalarPreconditioner<S>>(
        &mut self,
        matrix: &CsrMatrix<S>,
        b: &[S],
        x: &mut [S],
        precond: &P,
    ) -> SolverResult<S> {
        let n = b.len();
        self.ensure_workspace(n);
        let rtol = S::from_f64(self.config.rtol).unwrap_or(S::EPSILON);
        let atol = S::from_f64(self.config.atol).unwrap_or(S::MIN_POSITIVE);
        let stag_tol = S::from_f64(1e-30).unwrap_or(S::MIN_POSITIVE);
        let div_factor = S::from_f64(1e6).unwrap_or(S::MAX);

        // r = b - A*x
        matrix.mul_vec(x, &mut self.r);
        for i in 0..n {
            self.r[i] = b[i] - self.r[i];
        }

        let initial_norm = norm2(&self.r);
        if initial_norm < atol {
            return SolverResult {
                status: SolverStatus::Converged,
                iterations: 0,
                residual_norm: initial_norm,
                initial_residual_norm: initial_norm,
                relative_residual: S::ZERO,
            };
        }

        // r0 = r (shadow residual) - 固定为初始残差，在迭代中保持不变
        copy(&self.r, &mut self.r0);

        // 标准 BiCGStab: rho_old 用于计算 beta
        let mut rho_old = S::ONE;
        let mut alpha = S::ONE;
        let mut omega = S::ONE;

        self.v.fill(S::ZERO);
        self.p.fill(S::ZERO);

        for iter in 0..self.config.max_iter {
            // 计算 rho = (r0, r)
            let rho = dot(&self.r0, &self.r);
            
            // 检查 rho breakdown
            if rho.abs() < stag_tol {
                if iter == 0 {
                    // 初始残差与影子残差正交，已经收敛
                    return SolverResult {
                        status: SolverStatus::Converged,
                        iterations: 0,
                        residual_norm: initial_norm,
                        initial_residual_norm: initial_norm,
                        relative_residual: S::ZERO,
                    };
                }
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }

            // 计算 beta（第一次迭代时 beta = 0，因为 rho_old = 1, omega = 1）
            let beta = if iter == 0 {
                // 首次迭代: p = r
                S::ZERO
            } else {
                // 标准公式: beta = (rho / rho_old) * (alpha / omega)
                (rho / rho_old) * (alpha / omega)
            };
            
            // 保存 rho 供下次迭代使用
            rho_old = rho;

            // p = r + beta * (p - omega * v)
            for i in 0..n {
                self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i]);
            }

            // z = M^{-1} * p
            precond.apply(&self.p, &mut self.z);

            // v = A * z
            matrix.mul_vec(&self.z, &mut self.v);

            // alpha = rho / (r0, v)
            let r0v = dot(&self.r0, &self.v);
            if r0v.abs() < stag_tol {
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter,
                    residual_norm: norm2(&self.r),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.r) / initial_norm,
                };
            }
            alpha = rho / r0v;

            // s = r - alpha * v
            for i in 0..n {
                self.s[i] = self.r[i] - alpha * self.v[i];
            }

            // 检查 s 的范数
            let s_norm = norm2(&self.s);
            if s_norm < atol {
                // x = x + alpha * z
                axpy(alpha, &self.z, x);
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: s_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: s_norm / initial_norm,
                };
            }

            // z = M^{-1} * s
            precond.apply(&self.s, &mut self.z);

            // t = A * z
            matrix.mul_vec(&self.z, &mut self.t);

            // omega = (t, s) / (t, t)
            let tt = dot(&self.t, &self.t);
            if tt.abs() < stag_tol {
                omega = S::ONE;
            } else {
                omega = dot(&self.t, &self.s) / tt;
            }
            
            // 检查 omega breakdown（omega 过小会导致算法不稳定）
            if omega.abs() < stag_tol {
                // 只更新 x 的 alpha 部分后返回
                precond.apply(&self.p, &mut self.z);
                axpy(alpha, &self.z, x);
                return SolverResult {
                    status: SolverStatus::Stagnated,
                    iterations: iter + 1,
                    residual_norm: norm2(&self.s),
                    initial_residual_norm: initial_norm,
                    relative_residual: norm2(&self.s) / initial_norm,
                };
            }

            // x = x + alpha * (M^{-1} p) + omega * (M^{-1} s)
            precond.apply(&self.p, &mut self.z);
            axpy(alpha, &self.z, x);
            precond.apply(&self.s, &mut self.z);
            axpy(omega, &self.z, x);

            // r = s - omega * t
            for i in 0..n {
                self.r[i] = self.s[i] - omega * self.t[i];
            }

            let res_norm = norm2(&self.r);
            let rel_res = res_norm / initial_norm;

            if self.config.verbose {
                if let Some(res_val) = res_norm.to_f64() {
                    log::trace!("BiCGStab iter {}: residual = {:.6e}", iter + 1, res_val);
                }
            }

            // 检查收敛
            if res_norm < atol || rel_res < rtol {
                return SolverResult {
                    status: SolverStatus::Converged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }

            // 检查发散
            if res_norm > initial_norm * div_factor {
                return SolverResult {
                    status: SolverStatus::Diverged,
                    iterations: iter + 1,
                    residual_norm: res_norm,
                    initial_residual_norm: initial_norm,
                    relative_residual: rel_res,
                };
            }
        }

        SolverResult {
            status: SolverStatus::MaxIterationsReached,
            iterations: self.config.max_iter,
            residual_norm: norm2(&self.r),
            initial_residual_norm: initial_norm,
            relative_residual: norm2(&self.r) / initial_norm,
        }
    }

    fn name(&self) -> &'static str {
        "BiCGStab"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::linear_algebra::csr::CsrBuilder;
    use crate::numerics::linear_algebra::preconditioner::{
        IdentityPreconditioner, JacobiPreconditioner,
    };

    fn create_spd_matrix(n: usize) -> CsrMatrix<f64> {
        // 创建三对角对称正定矩阵
        let mut builder = CsrBuilder::<f64>::new_square(n);
        for i in 0..n {
            builder.set(i, i, 4.0);
            if i > 0 {
                builder.set(i, i - 1, -1.0);
            }
            if i < n - 1 {
                builder.set(i, i + 1, -1.0);
            }
        }
        builder.build()
    }

    #[test]
    fn test_cg_simple() {
        let matrix = create_spd_matrix(10);
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = ConjugateGradient::<f64>::new(config);
        let precond = IdentityPreconditioner::new();

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < 1e-8);
    }

    #[test]
    fn test_pcg_simple() {
        let matrix = create_spd_matrix(10);
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = PcgSolver::<f64>::new(config);
        let precond = JacobiPreconditioner::from_matrix(&matrix);

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < 1e-8);
    }

    #[test]
    fn test_pcg_faster_than_cg() {
        let matrix = create_spd_matrix(50);
        let b = vec![1.0; 50];

        // CG
        let mut x_cg = vec![0.0; 50];
        let config = SolverConfig::new(1e-10, 200);
        let mut cg_solver = ConjugateGradient::<f64>::new(config.clone());
        let ident = IdentityPreconditioner::new();
        let cg_result = cg_solver.solve(&matrix, &b, &mut x_cg, &ident);

        // PCG
        let mut x_pcg = vec![0.0; 50];
        let mut pcg_solver = PcgSolver::<f64>::new(config);
        let precond = JacobiPreconditioner::from_matrix(&matrix);
        let pcg_result = pcg_solver.solve(&matrix, &b, &mut x_pcg, &precond);

        // PCG 应该更快收敛
        assert!(pcg_result.is_converged());
        assert!(cg_result.is_converged());
        assert!(pcg_result.iterations <= cg_result.iterations);
    }

    #[test]
    fn test_bicgstab_simple() {
        let matrix = create_spd_matrix(10);
        let b = vec![1.0; 10];
        let mut x = vec![0.0; 10];

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = BiCgStabSolver::<f64>::new(config);
        let precond = JacobiPreconditioner::from_matrix(&matrix);

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert!(result.relative_residual < 1e-8);
    }

    #[test]
    fn test_already_converged() {
        let matrix = create_spd_matrix(3);
        // b = A * x_exact
        let x_exact = vec![0.25, 0.25, 0.25];
        let mut b = vec![0.0; 3];
        matrix.mul_vec(&x_exact, &mut b);

        let mut x = x_exact.clone();

        let config = SolverConfig::new(1e-10, 100);
        let mut solver = PcgSolver::<f64>::new(config);
        let precond = IdentityPreconditioner::new();

        let result = solver.solve(&matrix, &b, &mut x, &precond);

        assert!(result.is_converged());
        assert_eq!(result.iterations, 0);
    }

    #[test]
    fn test_solver_result() {
        let result = SolverResult::<f64> {
            status: SolverStatus::Converged,
            iterations: 10,
            residual_norm: 1e-12,
            initial_residual_norm: 1.0,
            relative_residual: 1e-12,
        };

        assert!(result.is_converged());
    }
}